//                        MFEM Analytic Convergence Example
//
// Compile with: make convergence
//
// Sample runs:  mpirun -np 4 convergence -m ../../../data/square-disc.mesh
//               mpirun -np 4 convergence -m ../../../data/star.mesh
//               mpirun -np 4 convergence -m ../../../data/escher.mesh
//               mpirun -np 4 convergence -m ../../../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 convergence -m ../../../data/square-disc-p3.mesh -o 3
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = f with exact sinusoidal solution under uniform
//               refinement. Convergence statistics are gathered for both
//               L2 and H1 error so various order methods can be compared.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact smooth analytic solution for convergence study
double u_exact(const Vector &);
void u_grad_exact(const Vector &, Vector &);
double f_exact(const Vector &);

// Setting the frequency for the exact solution
double freq = 1.0;
double kappa;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int total_refinements = 4;
   int max_serial_refinements = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&total_refinements, "-r", "--refinements",
                  "Number of total uniform refinements");
   args.AddOption(&max_serial_refinements, "-sr", "--serial-refinements",
                  "Maximum number of serial uniform refinements");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");

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
   kappa = freq * M_PI;

   // Set output options and print header
   cout.precision(4);

   if (myid == 0)
   {
      cout << "----------------------------------------------------------------------------------------"
           << endl;
      cout << left << setw(16) << "DOFs "<< setw(16) <<"h "<< setw(
              16) << "L^2 error "<< setw(16);
      cout << "L^2 rate "<< setw(16) << "H^1 error "<< setw(16) << "H^1 rate" << endl;
      cout << "----------------------------------------------------------------------------------------"
           << endl;
   }

   double l2_err_prev = 0.0;
   double h1_err_prev = 0.0;
   double h_prev = 0.0;

   // Loop over number of refinements for convergence study
   for (int ref = 0; ref < total_refinements; ref++)
   {
      // Read the (serial) mesh from the given mesh file on all processors.  We
      // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
      // and volume meshes with the same code.
      Mesh *mesh = new Mesh(mesh_file, 1, 1);
      int dim = mesh->Dimension();

      // Refine the serial mesh on all processors.
      for (int l = 0; l < min(ref, max_serial_refinements); l++)
      {
         mesh->UniformRefinement();
      }

      // Continue to refine the mesh in parallel if needed.
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      for (int l = 0; l < ref - max_serial_refinements; l++)
      {
         pmesh->UniformRefinement();
      }

      // Define a parallel finite element space on the parallel mesh. Here we
      // use continuous Lagrange finite elements of the specified order.
      FiniteElementCollection *fec = new H1_FECollection(order, dim);
      ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
      HYPRE_Int size = fespace->GlobalTrueVSize();

      // Determine the list of true (i.e. parallel conforming) essential
      // boundary dofs. In this example, the boundary conditions are defined by
      // marking all the boundary attributes from the mesh as essential
      // (Dirichlet) and converting them to a list of true dofs.
      Array<int> ess_tdof_list;
      if (pmesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // Set up the parallel linear form b(.) which corresponds to the
      // right-hand side of the FEM linear system, which in this case is (f,
      // phi_i) where phi_i are the basis functions in fespace and f is the
      // Laplacian of the exact solution.
      FunctionCoefficient f(f_exact);
      ParLinearForm *b = new ParLinearForm(fespace);
      b->AddDomainIntegrator(new DomainLFIntegrator(f));
      b->Assemble();

      // Define the solution vector x as a parallel finite element grid function
      // corresponding to fespace. Initialize x with the exact solution, which
      // satisfies the boundary conditions.
      ParGridFunction x(fespace);
      FunctionCoefficient u(u_exact);
      VectorFunctionCoefficient u_grad(dim, u_grad_exact);
      x.ProjectCoefficient(u);

      // Set up the parallel bilinear form a(.,.) on the finite element space
      // corresponding to the Laplacian operator -Delta, by adding the Diffusion
      // domain integrator.
      ParBilinearForm *a = new ParBilinearForm(fespace);
      ConstantCoefficient one(1.0);
      a->AddDomainIntegrator(new DiffusionIntegrator(one));

      // Assemble the parallel bilinear form and the corresponding linear
      // system, applying any necessary transformations such as: parallel
      // assembly, eliminating boundary conditions, applying conforming
      // constraints for non-conforming AMR, static condensation, etc.
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      // Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      // preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      amg->SetPrintLevel(0);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(200);
      pcg->SetPrintLevel(0);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // Recover the parallel grid function corresponding to X. This is the
      // local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);

      // Compute and print the L^2 and H^1 norms of the error.
      double l2_err = x.ComputeL2Error(u);
      double h1_err = x.ComputeH1Error(&u, &u_grad, &one, 1.0, 1);
      double h_min, h_max, kappa_min, kappa_max, l2_rate, h1_rate;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      if (ref != 0)
      {
         l2_rate = log(l2_err/l2_err_prev) / log(h_min/h_prev);
         h1_rate = log(h1_err/h1_err_prev) / log(h_min/h_prev);
      }
      else
      {
         l2_rate = 0.0;
         h1_rate = 0.0;
      }

      if (myid == 0)
      {
         cout << setw(16) << size << setw(16) << h_min << setw(16) << l2_err << setw(
                 16) << l2_rate;
         cout << setw(16) << h1_err << setw(16) << h1_rate << endl;
      }
      l2_err_prev = l2_err;
      h1_err_prev = h1_err;
      h_prev = h_min;

      // Free the used memory.
      delete pcg;
      delete amg;
      delete a;
      delete b;
      delete fespace;
      if (order > 0) { delete fec; }
      delete pmesh;
   }

   MPI_Finalize();

   return 0;
}

double u_exact(const Vector &x)
{
   double u = 0.0;
   if (x.Size() == 2)
   {
      u = sin(kappa * x(0)) * sin(kappa * x(1));
   }
   else
   {
      u = sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
   }

   return u;
}

void u_grad_exact(const Vector &x, Vector &u)
{
   if (x.Size() == 2)
   {
      u(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1));
      u(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   }
   else
   {
      u(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
      u(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1)) * sin(kappa * x(2));
      u(2) = kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * cos(kappa * x(2));
   }
}

double f_exact(const Vector &x)
{
   double f = 0.0;
   if (x.Size() == 2)
   {
      f = 2.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)));
   }
   else
   {
      f = 3.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)) * sin(
                                    kappa * x(2)));
   }

   return f;
}

