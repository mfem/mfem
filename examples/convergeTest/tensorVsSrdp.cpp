//      MFEM Comparing Tensor product and serendipity implementations
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact smooth analytic solution for convergence study
double u_exact(const Vector &);
void u_grad_exact(const Vector &, Vector &);
double u_exact_2(const Vector &);
void u_grad_exact_2(const Vector &, Vector &);

void compare(const char *mesh_file, int num_ref, int &order,
                      double &l2_err_prev, double &h1_err_prev, int &exact, int &solvePDE)
{
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh num_ref times

   for (int l = 1; l < num_ref+1; l++)
   {
      mesh->UniformRefinement();
   }

   // order > 1 --> serendipity
   // order < 0 --> tensor product

   FiniteElementCollection *fec;
   if (order > 1)
   {
      fec = new H1Ser_FECollection(order, 2);
   }
   else if (order < 0)
   {
      fec = new H1_FECollection(-order, 2);
   }
   else
   {
     cout << "Error - something went wrong in processing order input." << endl;
     fec = NULL;
   }

   // Set exact solution
   
   FunctionCoefficient *u;
   VectorFunctionCoefficient *u_grad;


   if (exact == 1)
   {
       u = new FunctionCoefficient(u_exact);
       u_grad = new VectorFunctionCoefficient(dim, u_grad_exact);
   }
   else if (exact == 2)
   {
      u = new FunctionCoefficient(u_exact_2);
      u_grad = new VectorFunctionCoefficient(dim, u_grad_exact_2);
   }
   else
   {
      cout << "Error - did not set exact solution" << endl;
      u = NULL;
      u_grad = NULL;
   }

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.

   // this variable may not be right:
   int gotNdofs = fespace->GetNDofs();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;


   if (solvePDE==1)
   {
      if (mesh->bdr_attributes.Size())
      {
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }
   // For L2 Projection:
   // Do not get boundary dofs


   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.

   LinearForm *b = new LinearForm(fespace);

   // // For solving PDE: (2 of 3 changes)
   if (solvePDE==1)
   {
      ConstantCoefficient zero(0.0);
      b->AddDomainIntegrator(new DomainLFIntegrator(zero));
   }
   else
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*u));
   }

   b->Assemble();
   
   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x=0.0;
   x.ProjectBdrCoefficient(*u, ess_bdr);

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);

   // DiffusionIntegrator *my_diff_integrator = new DiffusionIntegrator;
   // MassIntegrator *my_mass_integrator = new MassIntegrator;

   if (solvePDE==1)
   {   
      a->AddDomainIntegrator(new DiffusionIntegrator);
   }
   else
   {
      a->AddDomainIntegrator(new MassIntegrator);
   }

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
   // size of system is same as DOFs

   // 11. Solve the linear system A X = B.

   GSSmoother M((SparseMatrix&)(*A));
   X = 0.0;
   PCG(*A, M, B, X, 0, 200, 1e-24, 0.0);


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

   // Compute and print the L^2 and H^1 norms of the error.
   ConstantCoefficient one(1.0);
   double l2_err = x.ComputeL2Error(*u);
   double h1_err = x.ComputeH1Error(u, u_grad, &one, 1.0, 1);
   double l2_rate, h1_rate;

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

   cout << gotNdofs <<  ", " << l2_err <<  ", ";
   cout << h1_err;

   l2_err_prev = l2_err;
   h1_err_prev = h1_err;

   // 15. Free the used memory.
   // delete pcg;
   // delete amg;
   // delete my_diff_integrator;
   // delete my_mass_integrator;
   delete a;
   delete b;
   delete fespace;
   delete u_grad;
   delete u;
   delete fec;
   delete mesh;

   return;
}





double u_exact(const Vector &x)
{
   return(x(0)+x(1));
}

void u_grad_exact(const Vector &x, Vector &u)
{
   u(0) = 1;
   u(1) = 1;
}



double u_exact_2(const Vector &x)
{
   return sin(x(1))*exp(x(0));
}

void u_grad_exact_2(const Vector &x, Vector &u)
{
   u(0) = sin(x(1))*exp(x(0));
   u(1) = cos(x(1))*exp(x(0));
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int total_refinements = 0;

   // const char *mesh_file = "../../data/twoSquare.mesh";
   // const char *mesh_file = "../../data/star-q3.mesh";
   const char *mesh_file = "../../data/inline-oneQuad.mesh";
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   int exact = 1;
   int solvePDE = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&total_refinements, "-r", "--refine",
                  "Number of refinements to do.");   
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&exact, "-e", "--exact", 
                  "Choice of exact solution. 1=constant 1; 2=sin(x)e^y.");
   args.AddOption(&solvePDE, "-L", "--L2Project",
                  "Solve a PDE (1) or do L2 Projection (2)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   // args.PrintOptions(cout);

   if (order == 1)
   {
      // cout << "Using nodal H1 *** CUBIC *** tensor product elements (for testing / comparison)." << endl;
   }
   else if (order > 1)
   {
      // cout << "Using H1 serendipity elements of order " << order << "." << endl;
   }
   else if (order < 0)
   {
      // cout << "Using H1 positive (Bernstein) basis of order " << -order << "." << endl;
   }
   else
   {
      cout << "In this example, the order parameter is used as a proxy:\n" << 
      "order = 1: quadratic tensor product elements\n" <<
      "order = p>1: order p serendipity elements\n" <<
      "order = p<0: order -p Bernstein basis tensor product elements"  << endl;
      return 1;
   }

   if (solvePDE == 1)
   {
      // cout << "Approximating solution to Laplace problem with ";
      if (exact == 1)
      {
        // cout << "exact solution u(x,y)=x+y" << endl;
      }
      else if (exact == 2)
      {
         // cout << "exact solution u(x,y)=sin(y)e^x" << endl;
      }
      else
      {
         cout << endl << "*** Wrong usage of exact solution parameter (-e)"
              << endl;
         return 1;
      }
   }
   else if (solvePDE == 2)
   {
      // cout << "Doing L^2 projection of basis with right hand side ";
      if (exact == 1)
      {
        // cout << "u(x,y)=x+y" << endl;
      }
      else if (exact == 2)
      {
         // cout << "u(x,y)=sin(y)e^x" << endl;
      }
      else
      {
         cout << endl << "*** Wrong usage of exact solution parameter (-e)"
              << endl;
         return 1;
      }
   }
   else
   {
      cout << "Wrong usage of solve vs. L2 Projection option -L."
           << endl;
      return 1;
   }



   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   // device.Print();

   // Set output options and print header
   cout.precision(10);


   cout << left << "q3dofs, " << "q3eru, " << "q3erDu, ";
   cout << "s3dofs, " << "s3eru, " << "s3erDu";
   // cout << "q4dofs, " << "q4eru, " << "q4erDu";
   // cout << "s4dofs, " << "s4eru, " << "s4erDu";
   // cout << "q5dofs, " << "q5eru, " << "q5erDu";
   // cout << "s5dofs, " << "s5eru, " << "s5erDu";
   cout << endl;

   double l2_err_prev = 0.0;
   double h1_err_prev = 0.0;

   int o1 = -3;
   int o2 = 3;
   // int o3 = -4;
   // int o4 = 4;
   // int o5 = -5;
   // int o6 = 5;

   for (int i = 0; i < (total_refinements)+1; i++)
   {
       compare(mesh_file, i, o1, l2_err_prev, h1_err_prev, exact, solvePDE);
       cout << ", ";
       compare(mesh_file, i, o2, l2_err_prev, h1_err_prev, exact, solvePDE);
      //  cout << ", ";
      //  compare(mesh_file, i, o3, l2_err_prev, h1_err_prev, exact, solvePDE);
      //  cout << ", ";
      //  compare(mesh_file, i, o4, l2_err_prev, h1_err_prev, exact, solvePDE);
       // cout << ", ";
       // compare(mesh_file, i, o5, l2_err_prev, h1_err_prev, exact, solvePDE);
       // cout << ", ";
      //  compare(mesh_file, i, o6, l2_err_prev, h1_err_prev, exact, solvePDE);
       cout << endl;
   }
   return 0;
}
