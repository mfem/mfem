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
double u_exact_2(const Vector &);
void u_grad_exact_2(const Vector &, Vector &);

void convergenceStudy(const char *mesh_file, int num_ref, int &order,
                      double &l2_err_prev, double &h1_err_prev, bool &visualization, 
                      int &exact, int &dof2view, int &solvePDE, bool static_cond, bool &use_serendip)
{
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh num_ref times

   if (dof2view == -1)
   {
      for (int l = 1; l < num_ref+1; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.

   FiniteElementCollection *fec;
   if (order == 1)
   {
      fec = new H1_FECollection(1, 2);
   }
   else 
   {
      if (use_serendip)
      {
         fec = new H1Ser_FECollection(order, 2);
      }
      else
      {
         fec = new H1_FECollection(order, 2);
      }
   }

   // Set exact solution

   // exact == 1 case:
   FunctionCoefficient *u1 = new FunctionCoefficient(u_exact);
   VectorFunctionCoefficient *(u1_grad) = new VectorFunctionCoefficient(dim, u_grad_exact);
   
   // exact == 2 case:
   FunctionCoefficient *u2 = new FunctionCoefficient(u_exact_2);
   VectorFunctionCoefficient *u2_grad = new VectorFunctionCoefficient(dim, u_grad_exact_2);

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
      else // exact == 2
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(*u2));
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
      else
      {
         x.ProjectBdrCoefficient(*u2, ess_bdr);
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


   // Hack to work on interior nodal interpolation
   if (dof2view == -3)
   {
      x=0;
      if (order <0)
      {
         order = -order;
      }
      int firstIntDof = 4 + 4*(order-1);
      cout << "dof2view = -3 ==> Setting all Dofs to 1 up to Dof " << firstIntDof << endl;
      // int numIntDofs = (order - 3)*(order - 2)/2; // assumes serendipity case

      // can go up to x.Size()

      for (int i = 0; i < firstIntDof; i++)
      {
         x(i) = 3.14159;
      }
   }
   else if (dof2view == -2)     // Hack to project something
   {
      // void H1Ser_QuadrilateralElement::Project(Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
      // fe->Project(coeff, Trans, X)
      // fes->GetFE(i)-> Project(*src.fes->GetFE(i), *mesh->GetElementTransformation(i), P);
      const FiniteElement *feholder = fespace->GetFE(0);
      ElementTransformation *trans = mesh->GetElementTransformation(0);
      DenseMatrix temporary;
      feholder->Project(*feholder, *trans, temporary);
      // temporary.Print();
   }
   else if (dof2view != -1)    // Hack to viusalize a single basis function   
   {
      x=0;
      x(dof2view) = 1;
   }

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
   else
   {
      l2_err = x.ComputeL2Error(*u2);
      h1_err = x.ComputeH1Error(u2, u2_grad, &one, 1.0, 1);
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
   // return sin(28*x(0))*sin(28*x(1));
}

void u_grad_exact_2(const Vector &x, Vector &u)
{
   u(0) = sin(x(1))*exp(x(0));
   u(1) = cos(x(1))*exp(x(0));
   // u(0) = 28*sin(28*x(1))*cos(28*x(0));
   // u(1) = 28*sin(28*x(0))*cos(28*x(1));
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
   bool visualization = false;
   bool use_serendip = false;
   int exact = 2;
   int dof2view = -1;
   int solvePDE = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&total_refinements, "-r", "--refine",
                  "Number of refinements to do.");   
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_serendip, "-ser", "--use-serendipity", 
                  "-no-ser", "--not-serendipity", 
                  "Use serendipity element collection.");
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
                  "Choice of exact solution. 1=constant 1; 2=sin(x)e^y.");
   args.AddOption(&dof2view, "-dof", "--dof2view",
                  "DEBUG option: viewing a single dof:");
   args.AddOption(&solvePDE, "-L", "--L2Project",
                  "Solve a PDE (1) or do L2 Projection (2)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (use_serendip)
   {
      cout << "Using H1 serendipity elements of order " << order << "." << endl;
   }
   else
   {
      cout << "Using H1 tensor product basis of order " << order << "." << endl;
   }
   
   if (order <1)
   {
      cout << "Order must be >0." << endl;
      return 1;
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

   if (dof2view != -1)
   {
      cout << "DEBUG option: solution deleted; just viewing dof # " << dof2view << endl;
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

   if (dof2view == -1)
   {
      bool noVisYet = false;
      for (int i = 0; i < (total_refinements); i++)
      {
         convergenceStudy(mesh_file, i, order, l2_err_prev, h1_err_prev, noVisYet, 
            exact, dof2view, solvePDE, static_cond, use_serendip);
      }
   }
   convergenceStudy(mesh_file, total_refinements, order, l2_err_prev, h1_err_prev, visualization, 
            exact, dof2view, solvePDE, static_cond, use_serendip);

   return 0;
}
