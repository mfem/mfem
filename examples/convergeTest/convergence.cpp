//                        MFEM Analytic Convergence Example
//
// *** Adapted from an MPI example in atpesc-dev branch
// *** find it at mfem/examples/atpesc/mfem/convergence.cpp
// *** Developed at LLNL
//
// Compile with: make convergence
//
// Sample runs:  convergence -m ../../../data/square-disc.mesh
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

void convergenceStudy(const char *mesh_file, int num_ref, int &order,
                      double &l2_err_prev, double &h1_err_prev, bool &visualization)
{
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh num_ref times

   for (int l = 0; l < num_ref; l++)
   {
      mesh->UniformRefinement();
      // cout << "Did a unif ref; GetNE= " << mesh->GetNE() << endl;
   }


   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.

   FiniteElementCollection *fec;
   if (order == 1)
   {
      // fec = new H1_FECollection(order, dim);
      fec = new H1_FECollection(2, 2);
   }
   else if (order == 2)
   {
      fec = new H1Ser_FECollection(2, 2);
   }
   else
   {
     cout << "Error - should not have gotten here" << endl;
     fec = NULL;
   }

   // Set exact solution
   FunctionCoefficient u(u_exact);
   VectorFunctionCoefficient u_grad(dim, u_grad_exact);

   // cout << "getNE= " << mesh->GetNE() << endl;

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   //      cout << "Number of finite element unknowns: "  << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.

   // this variable may not be right:
   int gotNdofs = fespace->GetNDofs();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   if (mesh->bdr_attributes.Size())
   {
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient zero(0.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(zero));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x=0.0;
   x.ProjectBdrCoefficient(u, ess_bdr);

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   DiffusionIntegrator *my_integrator = new DiffusionIntegrator;
   a->AddDomainIntegrator(my_integrator);

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   //  if (static_cond) { a->EnableStaticCondensation(); }

   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.

   // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
   GSSmoother M((SparseMatrix&)(*A));
   X = 0.0;
   PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }


   // Compute and print the L^2 and H^1 norms of the error.
   ConstantCoefficient one(1.0);
   double l2_err = x.ComputeL2Error(u);
   double h1_err = x.ComputeH1Error(&u, &u_grad, &one, 1.0, 1);
   double l2_rate, h1_rate;

   // cout << "l2_err=" << l2_err << " l2_err_prev=" << l2_err_prev << " num_ref=" << num_ref << endl;

   // NOTE: denominator here is log(2) because uniform refinement halves h each time.
   //       if this is to be generalized, the denominator would have to change, too.

   if (num_ref != 0)
   {
      l2_rate = -log(l2_err/l2_err_prev) / log(2);
      h1_rate = -log(h1_err/h1_err_prev) / log(2);
   }
   else
   {
      l2_rate = 0.0;
      h1_rate = 0.0;
   }

   int one_over_h = mesh->GetNE();
   one_over_h = sqrt(one_over_h);

   cout << setw(16) << gotNdofs << setw(16) << one_over_h << setw(
           16) << l2_err << setw( 16) << l2_rate;
   cout << setw(16) << h1_err << setw(16) << h1_rate << endl;

   l2_err_prev = l2_err;
   h1_err_prev = h1_err;

   // 15. Free the used memory.
   // delete pcg;
   // delete amg;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }

   return;
}


// u_exact is for the case \Delta u = 1; the solution is u(x,y)=sin(pi x) sin(pi y)
double u_exact(const Vector &x)
{
   return sin(x(0))*exp(x(1));
}

void u_grad_exact(const Vector &x, Vector &u)
{
   u(0) = cos(x(0))*exp(x(1));
   u(1) = sin(x(0))*exp(x(1));
}

int main(int argc, char *argv[])
{
   int total_refinements = 4;

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/singleSquare.mesh";
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   //args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
   //             "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (order == 1)
   {
      cout << "convergence: Using H1 quadratic tensor product elements" << endl;
   }
   else if (order == 2)
   {
      cout << "convergence: Using H1 quadratic serendipity elements!" << endl;
   }
   else
   {
      cout << "In this example, the order parameter is used as a proxy:\n order 1: quadratic tensor product elements\n order 2: quadratic quadratic serendipity elements"
           << endl;
      return 1;
   }

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Set output options and print header
   cout.precision(4);

   cout << "----------------------------------------------------------------------------------------"
        << endl;
   cout << left << setw(16) << "DOFs "<< setw(16) <<"1/h "<< setw(
           16) << "L^2 error "<< setw(16);
   cout << "L^2 rate "<< setw(16) << "H^1 error "<< setw(16) << "H^1 rate" << endl;
   cout << "----------------------------------------------------------------------------------------"
        << endl;

   double l2_err_prev = 0.0;
   double h1_err_prev = 0.0;

   // 3. Read the mesh from the given mesh file.
   // Run last round with vis, if desired.

   // can use this as a max DoF tolerance:  (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);

   // Loop over number of refinements for convergence study

   bool noVisYet = false;
   for (int i = 0; i < (total_refinements-1); i++)
     {
       convergenceStudy(mesh_file, i, order, l2_err_prev, h1_err_prev, noVisYet);
     }
   convergenceStudy(mesh_file, total_refinements-1, order, l2_err_prev, h1_err_prev, visualization);
   return 0;
}
