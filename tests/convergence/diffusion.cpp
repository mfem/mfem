//               MFEM convergence test - diffusion
//
// Compile with: make diffusion
//
// Sample runs:
//
//    Linear elements:
//
//    mpirun -np 4 diffusion -vl 1 -m ../../data/star.mesh -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/square-disc.mesh -f '3.01 2.93' -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/star-mixed.mesh -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/inline-tet.mesh -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/escher.mesh -f '0.55 0.63 0.57' -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/fichera.mesh -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/inline-wedge.mesh -rp 4
//    mpirun -np 4 diffusion -vl 1 -m ../../data/fichera-mixed.mesh -rp 4
//
//    Cubic elements:
//
//    mpirun -np 4 diffusion -vl 1 -m ../../data/star.mesh -f '3.21 3.45' -rp 4 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/square-disc.mesh -f '9.01 8.93' -rp 4 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/star-mixed.mesh -f '3.21 3.45' -rp 4 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/inline-tet.mesh -f '3.21 3.45 3.37' -rp 3 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/escher.mesh -f '1.55 1.63 1.57' -rp 4 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/fichera.mesh -f '3.21 3.45 3.37' -rp 4 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/inline-wedge.mesh -f '3.21 3.45 3.37' -rp 3 -o 3
//    mpirun -np 4 diffusion -vl 1 -m ../../data/fichera-mixed.mesh -f '3.21 3.45 3.37' -rp 4 -o 3
//
// Description:  This test illustrates and verifies the convergence of the
//               continuous (H1) Galerkin method of a given order on a given
//               unstructured mesh using a manufactured solution. The
//               manufactured solution can be adjusted to a particular mesh by
//               setting frequencies along the coordinate axes.
//
//               This test is similar to MFEM Example 1.

#include <mfem.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// Exact solution parameters: shifts and frequencies
double sol_s[3] = { -0.32, 0.15, 0.24 };
double sol_k[3] = { 1.21, 1.45, 1.37 };

// Exact solution, its gradient, and minus Laplacian
double sol_func(const Vector &x);
void sol_grad(const Vector &x, Vector &grad);
double rhs_func(const Vector &x);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 1;
   Vector ks(sol_k, 3);
   bool static_cond = false;
   int verbosity = 0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of uniform refinements of the mesh before parallel"
                  " decomposition.");
   args.AddOption(&parallel_ref_levels, "-rp", "--refine-parallel",
                  "Number of uniform refinements to perform after parallel"
                  " decomposition.\n\t" "This is the number of levels used for"
                  " the convergence study.");
   args.AddOption(&ks, "-f", "--frequencies",
                  "Frequencies of the solution along the coordinate axes.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&verbosity, "-vl", "--verbosity-level",
                  "Verbosity level: 0, 1, 2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   MFEM_VERIFY(dim <= ks.Size() && ks.Size() <= 3,
               "invalid number of frequencies: " << ks.Size());

   // 4. Refine the mesh on all processors to increase the resolution.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the parallel mesh by partitioning the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 6. Define a parallel finite element space on the parallel mesh. If order
   //    is non-positive, an isoparametric space is used, i.e. the solution
   //    space order matches the mesh order.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 7. Define essential boundary conditions on the whole mesh boundary.
   MFEM_VERIFY(pmesh->bdr_attributes.Size() > 0,
               "the mesh has to define at least one boundary attribute");
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParLinearForm *b = new ParLinearForm(fespace);
   FunctionCoefficient rhs_coeff(rhs_func);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Also, define the exact solution and its
   //    gradient as Coefficient and VectorCoefficient, respectively.
   ParGridFunction x(fespace);
   x = 0.0;  // Initial guess for the interior (non-essential) DOFs.
   FunctionCoefficient sol_coeff(sol_func);
   VectorFunctionCoefficient sol_grad_coeff(dim, sol_grad);

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     using a DiffusionIntegrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (static_cond) { a->EnableStaticCondensation(); }

   double last_error_l2 = -1.0, last_error_h1 = -1.0;

   // 12. Refinement loop: discretize and solve the problem; then compute and
   //     print the discretization errors.
   for (int level = 0; true; level++)
   {
      HYPRE_Int num_elements = pmesh->GetGlobalNE();
      HYPRE_Int fespace_size = fespace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Refinement level   : " << level << '\n'
              << "number of elements : " << num_elements << '\n'
              << "FE space size      : " << fespace_size << '\n';
      }
      if (verbosity > 1)
      {
         pmesh->PrintInfo();
      }

      if (myid == 0 && verbosity > 0)
      {
         cout << "Assembly ..." << flush;
      }
      a->Assemble();
      b->Assemble();

      HypreParMatrix A;
      Vector X, B;

      Array<int> ess_tdof_list;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

#if 0
      x.ProjectBdrCoefficient(sol_coeff, ess_bdr);
#else
      x.ProjectCoefficient(sol_coeff);
      {
         Array<int> ess_vdofs_marker;
         fespace->GetEssentialVDofs(ess_bdr, ess_vdofs_marker);
         for (int i = 0; i < x.Size(); i++)
         {
            if (!ess_vdofs_marker[i]) { x(i) = 0.0; }
         }
      }
#endif

      const bool copy_sol_interior = true;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, copy_sol_interior);

      {
         if (myid == 0 && verbosity > 0)
         {
            cout << " preconditioner ..." << flush;
         }
         HypreBoomerAMG amg;
         amg.SetPrintLevel(0);
         CGSolver pcg(pmesh->GetComm());
         pcg.SetAbsTol(0.0);
         pcg.SetRelTol(1e-16);
         pcg.SetMaxIter(200);
         pcg.SetPrintLevel((verbosity == 0) ? 0 : (verbosity == 1) ? 3 : 1);
         pcg.SetPreconditioner(amg);
         pcg.SetOperator(A);
         if (myid == 0 && verbosity > 0)
         {
            cout << " solving the linear system:" << endl;
         }
         pcg.Mult(B, X);
      }

      a->RecoverFEMSolution(X, *b, x);

      const double error_l2 = x.ComputeL2Error(sol_coeff);
      const int h1_norm_type = 1;
      const double loc_h1_norm = x.ComputeH1Error(&sol_coeff, &sol_grad_coeff,
                                                  &one, 1.0, h1_norm_type);
      const double error_h1 = GlobalLpNorm(2.0, loc_h1_norm, MPI_COMM_WORLD);
      if (myid == 0)
      {
         streamsize prec1 = 8, prec2 = 6;
         streamsize old_prec = cout.precision(prec1);
         ios_base::fmtflags old_flags = cout.flags();
         cout.setf(ios_base::scientific);
         cout << "L2 error           : " << error_l2;
         if (last_error_l2 > 0.0 && error_l2 > 0.0)
         {
            cout << ",  ratio: "
                 << resetiosflags(ios_base::scientific)
                 << setiosflags(ios_base::fixed)
                 << setprecision(prec2) << setw(prec2+3)
                 << last_error_l2/error_l2
                 << ",  rate: "
                 << setw(prec2+3)
                 << log(last_error_l2/error_l2)/log(2.0)
                 << setiosflags(ios_base::scientific)
                 << resetiosflags(ios_base::fixed)
                 << setprecision(prec1);
         }
         cout << '\n';
         cout << "Gradient L2 error  : " << error_h1;
         if (last_error_h1 > 0.0)
         {
            cout << ",  ratio: "
                 << resetiosflags(ios_base::scientific)
                 << setiosflags(ios_base::fixed)
                 << setprecision(prec2) << setw(prec2+3)
                 << last_error_h1/error_h1
                 << ",  rate: "
                 << setw(prec2+3)
                 << log(last_error_h1/error_h1)/log(2.0)
                 << setiosflags(ios_base::scientific)
                 << resetiosflags(ios_base::fixed)
                 << setprecision(prec1);
         }
         cout << '\n';
         last_error_l2 = error_l2;
         last_error_h1 = error_h1;
         cout.flags(old_flags);
         cout.precision(old_prec);
      }

      if (level == parallel_ref_levels) { break; }

      // Refine the mesh and update fespace, a, b, and x.
      pmesh->UniformRefinement();
      fespace->Update();
      a->Update();
      b->Update();
      // Interpolates the last solution to the new mesh to use as an initial
      // guess in PCG.
      x.Update();
   }

   // 15. Send the last solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}


double sol_func(const Vector &x)
{
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      val *= sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
   }
   return val;
}

void sol_grad(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   double *g = grad.GetData();
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double y = M_PI*(sol_s[d]+sol_k[d]*x(d));
      const double f = sin(y);
      for (int j = 0; j < d; j++) { g[j] *= f; }
      g[d] = val*M_PI*sol_k[d]*cos(y);
      val *= f;
   }
}

double rhs_func(const Vector &x)
{
   double val = 1.0, lap = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double f = sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
      val *= f;
      lap = lap*f + val*M_PI*M_PI*sol_k[d]*sol_k[d];
   }
   return lap;
}
