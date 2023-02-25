#include "mfem.hpp"
#include <iostream>

#include "hdiv_linear_solver.hpp"
#include "discrete_divergence.hpp"
#include "../solvers/lor_mms.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);

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

   // Compute the residual of the linear system
   FunctionCoefficient a_coeff(u);
   FunctionCoefficient b_coeff([](const Vector &xvec)
   {
      const int dim = xvec.Size();
      const double x = pi*xvec[0], y = pi*xvec[1];
      if (dim == 2)
      {
         return 2*pi2*(2.0 + sin(x)*sin(y));
      }
      else // dim == 3
      {
         const double z = pi*xvec[2];
         return 3*pi2*(2.0 + sin(x)*sin(y)*sin(z));
      }
   });

   const auto solver_mode = HdivSaddlePointSolver::Mode::DARCY;
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

   ParBilinearForm w(&fes_l2);
   w.AddDomainIntegrator(new MassIntegrator(a_coeff));
   w.Assemble();
   w.Finalize();
   HypreParMatrix *W = w.ParallelAssemble();

   ParMixedBilinearForm b(&fes_rt, &fes_l2);
   b.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   b.Assemble();
   b.Finalize();
   HypreParMatrix *B = b.ParallelAssemble();
   HypreParMatrix *Bt = B->Transpose();

   ParBilinearForm m(&fes_rt);
   m.AddDomainIntegrator(new VectorFEMassIntegrator(b_coeff));
   m.Assemble();
   m.Finalize();
   HypreParMatrix *M = m.ParallelAssemble();

   BlockOperator A(offsets);
   A.SetBlock(0, 0, W);
   A.SetBlock(0, 1, B);
   A.SetBlock(1, 0, Bt);
   A.SetBlock(1, 1, M, -1.0);

   BlockVector Y_block(offsets);
   A.Mult(X_block, Y_block);
   Y_block -= B_block;

   auto nrm2 = [](const Vector &x)
   {
      return sqrt(InnerProduct(MPI_COMM_WORLD, x, x));
   };

   const double resnorm1 = nrm2(Y_block)/nrm2(B_block);
   if (Mpi::Root()) { std::cout << "Linear residual norm: " << resnorm1 << "\n\n"; }

   std::cout << Y_block.GetBlock(0).Norml2() << '\n';
   std::cout << Y_block.GetBlock(1).Norml2() << '\n' << '\n';

   HypreParVector Md(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                     M->GetRowStarts());
   M->GetDiag(Md);
   HypreParMatrix *MinvBt = B->Transpose();
   MinvBt->InvScaleRows(Md);
   HypreParMatrix *S = ParMult(B, MinvBt);

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

   {
      ChangeOfBasis_RT basis_rt(fes_rt);
      ChangeOfBasis_L2 basis_l2(fes_l2);
      HypreParMatrix *D = FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs);

      BilinearForm W_mix(&fes_l2);
      // W_mix.AddDomainIntegrator(new ModifiedMassIntegrator);
      W_mix.AddDomainIntegrator(new MassIntegrator);
      W_mix.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      W_mix.Assemble();

      GridFunction x1(&fes_rt);
      Vector y1(B->Height());
      Vector x2(x1.Size());
      Vector y2(B->Height()), z1(B->Height()), z2(B->Height()), z3(B->Height());
      Vector w(x1.Size());

      x1.Randomize(3);
      B->Mult(x1, y1);

      basis_rt.MultInverse(x1, x2);
      D->Mult(x2, z1);
      basis_l2.Mult(z1, z2);
      W_mix.Mult(z2, y2);

      y2 -= y1;
      std::cout << "Difference: " << y2.Norml2() << '\n';

      //    y1.Randomize(1);
      //    B->MultTranspose(y1, w);
      //    basis_rt.MultTranspose(w, x1);

      //    W_mix.MultTranspose(y1, z1);
      //    basis_l2.MultTranspose(z1, z2);
      //    D->MultTranspose(z2, x2);

      //    x2 -= x1;
      //    std::cout << "Difference: " << x2.Norml2() << '\n';

      //    ///////
      //    y1.Randomize(1);
      //    D->MultTranspose(y1, x1);

      //    // ChangeMapType_L2 cmt(saddle_point_solver.fes_l2, fes_l2, saddle_point_solver.qs);

      //    saddle_point_solver.L_inv->Mult(y1, z1);
      //    // basis_l2.Mult(z1, z2);
      //    z2 = z1;
      //    W_mix.Mult(z2, z3);
      //    DGMassInverse W_inv(fes_l2);
      //    W_inv.Mult(z3, y2);

      //    // cmt.Mult(y1, y2);

      //    W_mix.Mult(y2, z3);

      //    z1.Randomize(1);
      //    std::cout << z3*z1 - y1*z1 << '\n';

      //    B->MultTranspose(y2, w);
      //    basis_rt.MultTranspose(w, x2);

      //    QuadratureFunction detJinv(saddle_point_solver.qs);
      //    auto *geom = mesh.GetGeometricFactors(
      //                    saddle_point_solver.qs.GetIntRule(0),
      //                    GeometricFactors::DETERMINANTS
      //                 );
      //    for (int i = 0; i < geom->detJ.Size(); ++i)
      //    {
      //       detJinv[i] = geom->detJ[i];
      //    }
      //    QuadratureFunctionCoefficient qf_coeff(detJinv);
      //    DGMassInverse W_mix_inv(saddle_point_solver.fes_l2, qf_coeff);
      //    W_mix_inv.Mult(y1, z2);
      //    basis_l2.Mult(z2, z3);

      //    W_mix.Mult(z3, z2);
      //    std::cout << z2*z1 - y1*z1 << '\n';

      //    x2 -= x1;
      //    std::cout << "Difference: " << x2.Norml2() << '\n';
   }

   delete MinvBt;
   delete S;

   delete W;
   delete B;
   delete Bt;
   delete M;

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
