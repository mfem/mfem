#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>

using namespace std;
using namespace mfem;

class OrthoSolver : public Solver
{
public:
   OrthoSolver();

   virtual void SetOperator(const Operator &op);

   void Mult(const Vector &b, Vector &x) const;

private:
   const Operator *oper;

   mutable Vector b_ortho;

   void Orthoganalize(const Vector &v, Vector &v_ortho) const;
};

OrthoSolver::OrthoSolver() : Solver(0, true) {}

void OrthoSolver::SetOperator(const Operator &op)
{
   width = op.Width();
   oper = &op;
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   // Orthoganlize input.
   Orthoganalize(b, b_ortho);

   // Apply operator.
   oper->Mult(b_ortho, x);

   // Orthoganlize output.
   Orthoganalize(x, x);
}

void OrthoSolver::Orthoganalize(const Vector &v, Vector &v_ortho) const
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   double ratio = global_sum / static_cast<double>(global_size);
   v_ortho.SetSize(v.Size());
   for (int i = 0; i < v_ortho.Size(); ++i)
   {
      v_ortho(i) = v(i) - ratio;
   }
}

double ComputeResidual(Operator &A, Vector &x, Vector &b)
{
   Vector r(x.Size());
   A.Mult(x, r);
   r -= b;
   r.HostRead();
   return GlobalLpNorm(infinity(), r.Normlinf(), MPI_COMM_WORLD);
}

void OrthoRHS(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}

void MkMeanZero(ParGridFunction &v)
{
   ConstantCoefficient one{1.0};
   ParLinearForm mass_lf{v.ParFESpace()};
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   mass_lf.Assemble();

   ParGridFunction one_gf(v.ParFESpace());
   one_gf.ProjectCoefficient(one);

   double volume = mass_lf(one_gf);
   double integ = mass_lf(v);

   v -= integ / volume;
}

double rhs(const Vector &xpt)
{
   int dim = xpt.Size();
   double x = xpt[0];
   double y = (dim >= 2) ? xpt[1] : 0.0;
   double z1 = ((dim >= 3) ? xpt[2] : 0.0) + 1.0;
   return sin(x)*cos(y)*z1*z1;
}

int driver(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   const char *device_config = "cpu";
   int order = 2;
   int npatches = 1;
   int ref_levels = 0;
   bool visualization = false;
   bool uniform_ref = false;
   bool run_amg = true;
   bool run_as = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&uniform_ref, "-u", "--uniform-refinement", "-no-u",
                  "--no-uniform-refinement", "Enable uniform refinement");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&npatches, "-n", "--npatches",
                  "Number of patches to use in additive Schwarz method");
   args.AddOption(&run_amg, "-amg", "--run-amg", "-no-amg", "--no-run-amg",
                  "Solve system using hypre AMG");
   args.AddOption(&run_as, "-as", "--run-as", "-no-as", "--no-run-as",
                  "Solve system using additive Schwarz");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   device.Print();

   // Create the serial mesh, and do some uniform refinement if requested.
   std::unique_ptr<Mesh> mesh_ho(new Mesh(mesh_file, 1, 1));
   mesh_ho->SetCurvature(order, true, -1, Ordering::byNODES);

   int dim = mesh_ho->Dimension();
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_ho->UniformRefinement();
   }

   // Define the parallel mesh
   ParMesh pmesh_ho(MPI_COMM_WORLD, *mesh_ho);
   int basis_lor = uniform_ref ? BasisType::ClosedUniform
                   : BasisType::GaussLobatto;
   ParMesh pmesh_lor(&pmesh_ho, order, basis_lor);
   // Output the meshes to files for visualization.
   // Delete the serial mesh
   mesh_ho.reset();

   H1_FECollection fec_ho(order, dim);
   H1_FECollection fec_lor(1, dim);
   ParFiniteElementSpace fespace_ho(&pmesh_ho, &fec_ho);
   ParFiniteElementSpace fespace_lor(&pmesh_lor, &fec_lor);
   ParFiniteElementSpace fespace_coarse(&pmesh_ho, &fec_lor);

   int nel_total = pmesh_ho.ReduceInt(pmesh_ho.GetNE());
   HYPRE_Int size_ho = fespace_ho.GlobalTrueVSize();
   HYPRE_Int size_lor = fespace_lor.GlobalTrueVSize();
   HYPRE_Int size_coarse = fespace_coarse.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of HO elements in mesh: " << nel_total << endl;
      cout << "Number of HO finite element unknowns:     " << size_ho << endl;
      cout << "Number of LOR finite element unknowns:    " << size_lor << endl;
      cout << "Number of coarse finite element unknowns: "
           << size_coarse << endl;
   }

   // Determine the list of true (i.e. parallel conforming) essential
   // boundary dofs. In this example, the boundary conditions are defined
   // by marking all the boundary attributes from the mesh as essential
   // (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list, ess_tdof_list_coarse;

   int nbdr;
   if (pmesh_ho.bdr_attributes.Size() > 0)
   {
      nbdr = pmesh_ho.bdr_attributes.Max();
   }
   else
   {
      nbdr = 0;
   }

   Array<int> ess_bdr(nbdr);

   // Pure Neumann...
   // ess_bdr = 0;
   // ess_bdr = 1;
   if (nbdr >= 2)
   {
      ess_bdr[1] = 0;
   }
   //ess_bdr = 1;
   ess_bdr = 0;

   if (pmesh_ho.bdr_attributes.Size())
   {
      fespace_coarse.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_coarse);
      fespace_ho.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coeff(rhs);

   HypreParMatrix A0;
   ParBilinearForm a_coarse(&fespace_coarse);
   a_coarse.AddDomainIntegrator(new DiffusionIntegrator);
   a_coarse.Assemble();
   a_coarse.FormSystemMatrix(ess_tdof_list_coarse, A0);

   ParBilinearForm a_lor(&fespace_lor);
   HypreParMatrix A_lor;
   a_lor.AddDomainIntegrator(new DiffusionIntegrator);
   a_lor.Assemble();
   a_lor.FormSystemMatrix(ess_tdof_list, A_lor);

   ParLinearForm b(&fespace_ho);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   b.Assemble();

   b.Randomize(3);

   OrthoRHS(b);

   ParBilinearForm a(&fespace_ho);
   OperatorHandle A;
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();

   ParGridFunction x(&fespace_ho);
   // Test out inhomogeneous (g=1) Dirichlet conditions
   // x.ProjectBdrCoefficient(one, ess_bdr);
   x = 0.0;

   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);


   // std::ofstream matout("a_poisson.txt");
   // A->PrintMatlab(matout);
   // matout.close();
   // return(0);

   CGSolver itsolv(MPI_COMM_WORLD);

   itsolv.SetPrintLevel(1);
   itsolv.SetMaxIter(500);
   itsolv.SetRelTol(1e-6);
   itsolv.SetAbsTol(0.0);
   itsolv.SetOperator(*A);

   // Solve 1. AMG:
   double amg_elapsed_solv = -1, amg_elapsed_setup = -1, amg_resnorm = -1;
   // The AMG preconditioner is defined in terms of the LOR matrix
   // since the goal is to avoid ever forming the high-order system matrix.
   if (run_amg)
   {
      tic_toc.Clear();
      tic_toc.Start();

      HypreBoomerAMG amg(A_lor);;
      // HYPRE_BoomerAMGSetAggNumLevels(amg, 0);
      // HYPRE_BoomerAMGSetRelaxType(amg, 3);
      // HYPRE_BoomerAMGSetRelaxType(amg, 18);

      // amg.SetPrintLevel(0); // 1

      OrthoSolver orth_amg;
      orth_amg.SetOperator(amg);

      itsolv.SetPreconditioner(orth_amg);
      // Force setup of AMG preconditioner. This is a stupid hack but necessary
      // because MFEM doesn't expose the setup_called member data.
      amg.Mult(B, X);
      tic_toc.Stop();
      amg_elapsed_setup = tic_toc.RealTime();
      tic_toc.Clear();
      X = 0.0;
      tic_toc.Start();
      itsolv.Mult(B, X);
      tic_toc.Stop();
      amg_elapsed_solv = tic_toc.RealTime();
      amg_resnorm = ComputeResidual(*A, X, B);
      if (myid == 0) { std::cout << std::endl; }
   }

   // Solve 2. AS:
   double as_elapsed_solv = -1, as_elapsed_setup = -1, as_resnorm = -1;
   if (myid == 0)
   {
      std::cout << "AS residual:            " << as_resnorm << '\n';
      std::cout << "AMG residual:           " << amg_resnorm << '\n';
      std::cout << '\n';
      std::cout << "AS elapsed setup time:  " << as_elapsed_setup << '\n';
      std::cout << "AS elapsed solve time:  " << as_elapsed_solv << '\n';
      std::cout << '\n';
      std::cout << "AMG elapsed setup time: " << amg_elapsed_setup << '\n';
      std::cout << "AMG elapsed solve time: " << amg_elapsed_solv << '\n';
      std::cout << '\n';
      std::cout << "AS elapsed total time:  "
                << as_elapsed_solv + as_elapsed_setup << '\n';
      std::cout << "AMG elapsed total time: "
                << amg_elapsed_solv + amg_elapsed_setup << '\n';
      std::cout << std::endl;
   }

   // Recover the parallel grid function corresponding to X. This is the local
   // finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   MkMeanZero(x);

   // Then send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh_ho << x << flush;
   }

   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int res = driver(argc, argv);
   MPI_Finalize();
   return res;
}
