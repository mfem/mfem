#include "mfem.hpp"
#include <iostream>

#include "hdiv_linear_solver.hpp"
#include "discrete_divergence.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);

double f(const Vector &xvec);
double g(const Vector &xvec);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;
   bool mt_value = true;
   bool darcy = true;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&mt_value, "-val", "--value", "-int", "--integral",
                  "Map type integral or value.");
   args.AddOption(&darcy, "-da", "--darcy", "-g", "--grad-div",
                  "Grad-div or Darcy problem");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   const int mt = mt_value ? FiniteElement::VALUE : FiniteElement::INTEGRAL;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nRT DOFs: " << ndofs_rt << "\nL2 DOFs: " << ndofs_l2 << endl;
   }

   Array<int> ess_rt_dofs;

   FunctionCoefficient a_coeff(f);
   FunctionCoefficient b_coeff(g);
   ConstantCoefficient one(1.0);

   Coefficient &div_coeff = darcy ? (Coefficient&)one : (Coefficient&)a_coeff;

   // Solve the system with the saddle-point solver
   const auto solver_mode = darcy ? HdivSaddlePointSolver::Mode::DARCY
                            : HdivSaddlePointSolver::Mode::GRAD_DIV;
   HdivSaddlePointSolver saddle_point_solver(
      mesh, fes_rt, fes_l2, a_coeff, b_coeff, ess_rt_dofs, solver_mode);

   const Array<int> &offsets = saddle_point_solver.GetOffsets();
   BlockVector X_block(offsets), B_block(offsets);

   saddle_point_solver.GetMINRES().SetAbsTol(1e-18);
   saddle_point_solver.GetMINRES().SetRelTol(1e-20);
   saddle_point_solver.GetMINRES().SetPrintLevel(
      IterativeSolver::PrintLevel().FirstAndLast());
   X_block = 0.0;
   B_block.Randomize(1);
   B_block.GetBlock(0) = 0.0;
   if (Mpi::Root()) { std::cout << "Saddle point solver... " << std::endl; }
   saddle_point_solver.Mult(B_block, X_block);

   // Form the matrix-based system
   ParBilinearForm w(&fes_l2);
   w.AddDomainIntegrator(new MassIntegrator(a_coeff));
   w.Assemble();
   w.Finalize();
   std::unique_ptr<HypreParMatrix> W(w.ParallelAssemble());

   ParMixedBilinearForm b(&fes_rt, &fes_l2);
   b.AddDomainIntegrator(new VectorFEDivergenceIntegrator(div_coeff));
   b.Assemble();
   b.Finalize();
   std::unique_ptr<HypreParMatrix> B(b.ParallelAssemble());
   std::unique_ptr<HypreParMatrix> Bt(B->Transpose());

   ParBilinearForm m(&fes_rt);
   m.AddDomainIntegrator(new VectorFEMassIntegrator(b_coeff));
   m.Assemble();
   m.Finalize();
   std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());

   BlockOperator A(offsets);
   A.SetBlock(0, 0, W.get());
   A.SetBlock(0, 1, B.get());
   A.SetBlock(1, 0, Bt.get());
   A.SetBlock(1, 1, M.get(), -1.0);

   // Compute the residual
   BlockVector Y_block(offsets);
   A.Mult(X_block, Y_block);
   Y_block -= B_block;

   auto nrm2 = [](const Vector &x)
   {
      return sqrt(InnerProduct(MPI_COMM_WORLD, x, x));
   };

   const double resnorm1 = nrm2(Y_block)/nrm2(B_block);
   if (Mpi::Root()) { std::cout << "Linear residual norm: " << resnorm1 << "\n\n"; }

   // Solve the system with a matrix-based solver (see ex5p)
   HypreParVector Md(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                     M->GetRowStarts());
   M->GetDiag(Md);
   std::unique_ptr<HypreParMatrix> MinvBt(B->Transpose());
   MinvBt->InvScaleRows(Md);
   std::unique_ptr<HypreParMatrix> S(ParMult(B.get(), MinvBt.get()));

   HypreDiagScale M_inv(*M);
   HypreBoomerAMG S_inv(*S);
   S_inv.SetPrintLevel(0);

   BlockDiagonalPreconditioner D(offsets);
   D.SetDiagonalBlock(0, &S_inv);
   D.SetDiagonalBlock(1, &M_inv);

   X_block = 0.0;
   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetAbsTol(1e-18);
   minres.SetRelTol(1e-20);
   minres.SetMaxIter(500);
   minres.SetOperator(A);
   minres.SetPreconditioner(D);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().FirstAndLast());
   if (Mpi::Root()) { std::cout << "Matrix-based solver... " << std::endl; }
   minres.Mult(B_block, X_block);

   A.Mult(X_block, Y_block);
   Y_block -= B_block;

   const double resnorm2 = nrm2(Y_block)/nrm2(B_block);
   if (Mpi::Root()) { std::cout << "Linear residual norm: " << resnorm2 << "\n\n"; }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

double f(const Vector &xvec)
{
   const int dim = xvec.Size();
   const double x = xvec[0], y = xvec[1];
   if (dim == 2)
   {
      return 2*(2.0 + sin(x)*sin(y));
   }
   else // dim == 3
   {
      const double z = xvec[2];
      return 3*(2.0 + sin(x)*sin(y)*sin(z));
   }
}

double g(const Vector &xvec)
{
   const int dim = xvec.Size();
   const double x = xvec[0], y = xvec[1];
   if (dim == 2)
   {
      return 2*(2.0 + cos(x)*cos(y));
   }
   else // dim == 3
   {
      const double z = xvec[2];
      return 3*(2.0 + cos(x)*cos(y)*cos(z));
   }
}
