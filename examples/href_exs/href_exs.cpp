// h-refinement examples
//
// Compile with: make href_exs
//
// Sample runs:  
//
// Description: Make h-refinement data sets for neural net
//              comparison



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
double u_exact_3(const Vector &);
void u_grad_exact_3(const Vector &, Vector &);

void convergenceStudy(const char *mesh_file, int num_ref, int &order,
                      double &l2_err_prev, double &h1_err_prev, bool &visualization, 
                      int &exact, int &solvePDE, bool static_cond)
{
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh num_ref times


    for (int l = 1; l < num_ref+1; l++)
    {
        mesh->UniformRefinement();
    }


   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.

   FiniteElementCollection *fec;
   if (order == 1)
   {
      fec = new H1_FECollection(1, 2);
   }
   else if (order > 1)
   {
      fec = new H1_FECollection(order, 2);
   }
   else if (order < 0)
   {
      // fec = new H1_FECollection(-order, 2, BasisType::Positive);
      fec = new H1_FECollection(-order, 2);
   }
   else
   {
     cout << "Error - something went wrong in processing order input." << endl;
     fec = NULL;
   }

   // Set exact solution

   // exact == 1 case:
   FunctionCoefficient *u1 = new FunctionCoefficient(u_exact);
   VectorFunctionCoefficient *(u1_grad) = new VectorFunctionCoefficient(dim, u_grad_exact);
   
   // exact == 2 case:
   FunctionCoefficient *u2 = new FunctionCoefficient(u_exact_2);
   VectorFunctionCoefficient *u2_grad = new VectorFunctionCoefficient(dim, u_grad_exact_2);

   // exact == 3 case:
   FunctionCoefficient *u3 = new FunctionCoefficient(u_exact_3);
   VectorFunctionCoefficient *u3_grad = new VectorFunctionCoefficient(dim, u_grad_exact_3);

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.

   // this variable may not be right for more general meshes than lattices?
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

   if (solvePDE==1)
   {
      ConstantCoefficient zero(0.0);
      b->AddDomainIntegrator(new DomainLFIntegrator(zero));
   }
   else // L2 Projection
   {
      if (exact == 1)
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(*u1));
      }
      else if (exact == 2) 
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(*u2));
      }
      else // exact == 3
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(*u3));
      }
   }

   b->Assemble();
   
   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x=0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);

   // DiffusionIntegrator *my_diff_integrator = new DiffusionIntegrator;
   // MassIntegrator *my_mass_integrator = new MassIntegrator;

   if (solvePDE==1)
   {   
      if (exact == 1)
      {
         x.ProjectBdrCoefficient(*u1, ess_bdr);
      }
      else if (exact == 2)
      {
         x.ProjectBdrCoefficient(*u2, ess_bdr);
      }
      else if (exact == 3)
      {
         x.ProjectBdrCoefficient(*u3, ess_bdr);
      }
      a->AddDomainIntegrator(new DiffusionIntegrator);
   }
   else // L2 Projection
   {
      a->AddDomainIntegrator(new MassIntegrator);
   }
   
   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   
   // cout << "Size of linear system: " << A->Height() << endl;
  
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
   
   double l2_err = 0;
   double h1_err = 0;

   if (exact == 1)
   {
      l2_err = x.ComputeL2Error(*u1);
      h1_err = x.ComputeH1Error(u1, u1_grad, &one, 1.0, 1);
   }
   else if (exact == 2)
   {
      l2_err = x.ComputeL2Error(*u2);
      h1_err = x.ComputeH1Error(u2, u2_grad, &one, 1.0, 1);
   }
   else if (exact == 3)
   {
      l2_err = x.ComputeL2Error(*u3);
      h1_err = x.ComputeH1Error(u3, u3_grad, &one, 1.0, 1);
   }

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

   cout << setw(16) << gotNdofs << setw(16) << one_over_h << setw(
           16) << l2_err << setw( 16) << l2_rate;
   cout << setw(16) << h1_err << setw(16) << h1_rate << endl;

   l2_err_prev = l2_err;
   h1_err_prev = h1_err;


   // Save the data to pass to neural network
   cout << "Mesh vertex array = " << endl;
   cout <<  mesh->GetVertex(0) << endl;

   // 15. Free the used memory.
   // delete pcg;
   // delete amg;
   // delete my_diff_integrator;
   // delete my_mass_integrator;
   delete a; 
   delete b;
   delete fespace;
   delete u1;
   delete u1_grad;
   delete u2;
   delete u2_grad;
   delete u3;
   delete u3_grad;
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

double u_exact_3(const Vector &x)
{
    // return (x(0)*x(0) + (0.5)*x(1)*x(1));
    int m=10;
    double total = sin(x(0))* pow( sin(  x(0)*x(0) / M_PI  ), 2*m  );
    total += sin(x(1))* pow( sin( 2*x(1)*x(1) / M_PI  ), 2*m  );
    total *= -1;
    return total;
}

void u_grad_exact_3(const Vector &x, Vector &u)
{
    // presumes m=10
    u(0)  = - (40.0 * x(0) * cos( x(0)*x(0)/M_PI ) * sin(x(0)) * pow(sin( x(0)*x(0)/M_PI ),19)) / M_PI;
    u(0) += - cos(x(0))*pow(sin( x(0)*x(0)/M_PI ),20);
    u(1)  = - (80.0 * x(1) * cos( 2.0 * x(1)*x(1)/M_PI ) * sin(x(1)) * pow(sin( 2.0 * x(1)*x(1)/M_PI ),19)   ) / M_PI;
    u(1) += - cos(x(1))*pow(sin( 2.0 * x(1)*x(1)/M_PI ),20);
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int total_refinements = 0;

   // const char *mesh_file = "../../data/twoSquare.mesh";
   // const char *mesh_file = "../../data/star-q3.mesh";
   const char *mesh_file = "../../data/inline-oneTri.mesh";
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = false;
   int exact = 3;
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
   //args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
   //             "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&exact, "-e", "--exact", 
                  "Choice of exact solution. 1=constant 1; 2=sin(x)e^y; 3=michalewicz.");
   args.AddOption(&solvePDE, "-L", "--L2Project",
                  "Solve a PDE (1) or do L2 Projection (2)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (order != 1)
   {
      cout << "Only allowing order 1 triangle elements for now." << endl;
      return 1;
   }
   else if (order == 1)
   {
      cout << "Using H1 triangular elements of order " << order << "." << endl;
   }

   if (solvePDE == 1)
   {
      cout << "Approximating solution to Laplace problem with ";
      if (exact == 1)
      {
        cout << "exact solution u(x,y)=x+y" << endl;
      }
      else if (exact == 2)
      {
         cout << "exact solution u(x,y)=sin(y)e^x" << endl;
      }
      else if (exact == 3)
      {
         cout << endl << "Michalewicz is not harmonic. Use option -L 2 to do L2 projection instead."
              << endl;
         return 1;
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
      cout << "Doing L^2 projection of basis with right hand side ";
      if (exact == 1)
      {
        cout << "u(x,y)=x+y" << endl;
      }
      else if (exact == 2)
      {
         cout << "u(x,y)=sin(y)e^x" << endl;
      }
      else if (exact == 3)
      {
         cout << "u(x,y)=michalewicz function" << endl;
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
    for (int i = 0; i < (total_refinements); i++)
    {
        convergenceStudy(mesh_file, i, order, l2_err_prev, h1_err_prev, noVisYet, 
        exact, solvePDE, static_cond);
    }

    convergenceStudy(mesh_file, total_refinements, order, l2_err_prev, h1_err_prev, visualization, 
            exact, solvePDE, static_cond);

    return 0;
}
