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
double f_exact(const Vector &);

// Setting the frequency for the exact solution
double freq = 1.0;
double kappa = freq * M_PI;


int main(int argc, char *argv[])
{
   int total_refinements = 4;
   
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/singleSquare.mesh";
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = true;

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
   else if(order == 2)
     {
       cout << "convergence: Using H1 quadratic serendipity elements!" << endl;
     }
   else
     {
       cout << "In this example, the order parameter is used as a proxy:\n order 1: quadratic tensor product elements\n order 2: quadratic quadratic serendipity elements" << endl;
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
   cout << left << setw(16) << "DOFs "<< setw(16) <<"h "<< setw(16) << "L^2 error "<< setw(16);
   cout << "L^2 rate "<< setw(16) << "H^1 error "<< setw(16) << "H^1 rate" << endl;
   cout << "----------------------------------------------------------------------------------------"
	<< endl;
   
   double l2_err_prev = 0.0;
   double h1_err_prev = 0.0;
   double h_prev = 0.0;

   cout << "Vis type= " << typeid(visualization).name() << endl;

   // convergenceStudy(0, &order, &l2_err_prev, &h1_err_prev, &h_prev, &visualization);

   // Loop over number of refinements for convergence study
   for (int ref = 0; ref < total_refinements; ref++)
   {
     // asdf call function here
   }
   return 0;
}




void convergenceStudy(int num_ref, int &order, double &l2_err_prev, double &h1_err_prev, double &h_prev, int &visualization)
{
  // should really pass this as a parameter:
  const char *mesh_file = "../../data/singleSquare.mesh";
  
  // 3. Read the mesh from the given mesh file.
  // can use this as a max DoF tolerance:  (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
  
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();
  
  // 4. Refine the mesh num_ref times
  
  for (int l = 0; l < num_ref; l++)
    {
      mesh->UniformRefinement();
      // cout << "Num of elements is " << mesh->GetNE() << endl;
    }
  
  
  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order.
  
  FiniteElementCollection *fec;
  if (order == 1)
    {
      // fec = new H1_FECollection(order, dim);
      fec = new H1_FECollection(3, 2);
    }
  else if(order == 2)
    {
      fec = new H1Ser_FECollection(2, 2);
    }
  else
    {
      cout << "Error - should not have gotten here" << endl;
    }
  // cout << "Done making the FE collection" << endl;
  
  // Set exact solution
  FunctionCoefficient f(f_exact);
  FunctionCoefficient u(u_exact);
  VectorFunctionCoefficient u_grad(dim, u_grad_exact);
  
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  //      cout << "Number of finite element unknowns: "  << fespace->GetTrueVSize() << endl;
  
  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  
  // this variable may not be right:
  int size = fespace->GetTrueVSize();  
  
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
  
  // 7. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  LinearForm *b = new LinearForm(fespace);
  ConstantCoefficient one(1.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(one));
  b->Assemble();
  
  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction x(fespace);
  x = 0.0;
  
  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.
  BilinearForm *a = new BilinearForm(fespace);
  
  a->AddDomainIntegrator(new DiffusionIntegrator(one));
  
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
#ifndef MFEM_USE_SUITESPARSE
  // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
  GSSmoother M((SparseMatrix&)(*A));
  PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);
  // Note: the 5th parameter is print_level; set to 1 for some Iteration informatio
#else
  // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*A);
  umf_solver.Mult(B, X);
#endif
  
  
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
  double l2_err = x.ComputeL2Error(u);
  double h1_err = x.ComputeH1Error(&u, &u_grad, &one, 1.0, 1);
  double h_min, h_max, kappa_min, kappa_max, l2_rate, h1_rate;
  //pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
  
  // h_min = 1; // What is h_min and how should it be initialized???
  
  // cout << "l2_err=" << l2_err << " l2_err_prev=" << l2_err_prev << " h_min=" << h_min << " h_prev=" << h_prev << " num_ref=" << num_ref << endl; 
  
  if (num_ref != 0)
    {
      l2_rate = log(l2_err/l2_err_prev) / log(h_min/h_prev);
      h1_rate = log(h1_err/h1_err_prev) / log(h_min/h_prev);
    }
  else
    {
      l2_rate = 0.0;
      h1_rate = 0.0;
    }
  
  cout << setw(16) << size << setw(16) << h_min << setw(16) << l2_err << setw( 16) << l2_rate;
  cout << setw(16) << h1_err << setw(16) << h1_rate << endl;
  
  l2_err_prev = l2_err;
  h1_err_prev = h1_err;
  h_prev = h_min;
  
  // 15. Free the used memory.
  // delete pcg;
  // delete amg;
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
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

