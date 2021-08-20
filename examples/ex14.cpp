//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 2 -o 2 -k 0 -e 1
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

// ./ex14 -pa -m ../data/inline-hex.mesh  --lob -i 2 -k -1 -s -0 -b 0 --noint --bdy -r 1 -o 1 &> ex14out.log


#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "../config/config.hpp"
#include "../fem/nonlininteg.hpp"
#include "../fem/fespace.hpp"
#include "../fem/libceed/ceed.hpp"
#include "../fem/fem.hpp"

#include "../fem/restriction.hpp"
#include "../fem/gridfunc.hpp"
#include "../fem/fespace.hpp"
#include "../general/forall.hpp"

//#include </usr/local/include/gperftools/profiler.h>  

using namespace std;
using namespace mfem;

double x_exact_approx(const Vector &);
double x1(const Vector &);
double x2(const Vector &);

int main(int argc, char *argv[])
{
   //ProfilerStart("/tmp/data.prof");

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   std::setprecision(18);

   // 1. Parse command-line options.
   //const char *mesh_file = "../data/inline-hex.mesh";
   const char *mesh_file = "../data/inline-quad.mesh"; 
   int ref_levels = 2;
   int order = 6;
   double sigma = -1.0;
   double kappa = -1.0;
   double beta = 1;
   double eta = 0.0;
   bool visualization = 1;
   bool pa = true;
   bool set_bc = true;
   bool lob = true;

   int initial = 0;
   bool bdyf = true;
   bool intf = true;
   
   OptionsParser args(argc, argv);

   args.AddOption(&initial, "-i", "--initial",
                  "nondesc");

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&beta, "-b", "--beta",
                  " bleh ."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");

   args.AddOption(&lob, "-lob", "--lob", "--pos", "--pos",
                  "Impose or not essential boundary conditions.");

   args.AddOption(&bdyf, "-bdy", "--bdy", "--nobdy", "--nobdy",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&intf, "-int", "--int", "--noint", "--noint",
                  "Impose or not essential boundary conditions.");

   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   //args.PrintOptions(cout);

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec;
   if(lob)
   {
      fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      fec = new DG_FECollection(order, dim, BasisType::Positive);
   }
   //fec = new DG_FECollection(order, dim, BasisType::Positive);
   if(pa)
   {
      // Only Gauss-Lobatto and Bernstein basis are supported in L2FaceRestriction.
      //fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
      //fec = new DG_FECollection(order, dim, BasisType::Positive);
   }
   else
   {
      //fec = new DG_FECollection(order, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "% Number of unknowns: " << fespace->GetVSize() << endl;

   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b->Assemble();

   // 6. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(fespace);
   x = 0.0;
   FunctionCoefficient f(x_exact_approx);
   FunctionCoefficient f1(x1);
   FunctionCoefficient f2(x2);

   x.ProjectCoefficient(f);

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

   // Test the full operator
   LinearForm *bfull = new LinearForm(fespace);
   bfull->AddDomainIntegrator(new DomainLFIntegrator(one));
   bfull->AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   bfull->Assemble();
   BilinearForm *afull = new BilinearForm(fespace);
   GridFunction xfull(fespace);
   xfull.ProjectCoefficient(f);

   afull->SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
   afull->AddDomainIntegrator(new DiffusionIntegrator(one));   
   if(intf)
   afull->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa, beta));
   if(bdyf)
   afull->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa, beta));
   afull->Assemble();
   afull->Finalize();

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   std::cout << "%-----------------------------------------------------------------------" << std::endl;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (pa)
   {
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      if(intf)
      a->AddInteriorNormalDerivativeFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa, beta));
      if(bdyf)
      a->AddBdrNormalDerivativeFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa, beta));
   }
   else if (eta > 0)
   {
      /*
      a->AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a->AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      */
   }
   else
   {
      /*
      // Default setting
      if(intf)
      {
         a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
      }
      else if(bdyf)
      {
         a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
      }
      else
      {
         std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
         exit(0);
      }
      */
         std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
         exit(0);
   }
   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   
   a->Assemble();

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   std::cout << "% done assembling " << std::endl;
   //exit(1);
   a->Finalize();

   std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // -------------------------------------------
   // Sanity checks (temporary, for debugging)
   // -------------------------------------------
   // -------------------------------------------
   // Diff the operators
   {
      Vector yout;
      Vector youtfull;
      yout = xfull;
      youtfull = xfull;

      auto test = Reshape(x.ReadWrite(), x.Size());
      auto testfull = Reshape(xfull.ReadWrite(), xfull.Size());

      switch (initial)
      {
         case 0:
            // f = constant = 1
            std::cout << "init = @(x,y) 1.0;  " << std::endl;
            x = 1.0;
            xfull = 1.0;
            break;
         case 1:
            // plateaus
            x.ProjectCoefficient(f1);
            xfull.ProjectCoefficient(f1);
            break;
         case 2:
            // x^2*exp(y)
            x.ProjectCoefficient(f2);
            xfull.ProjectCoefficient(f2);
            break;
         case 3:
            // exact solution for square
            x.ProjectCoefficient(f);
            xfull.ProjectCoefficient(f);
            break;
         case 4:
            for(int i = 0; i < x.Size() ; i++ )
            {
               test(i) = i;
               testfull(i) = i;
            }
            break;
         default:
            std::cout << "no case for initial " << initial << std::endl;
            exit(0);
      }

      std::chrono::time_point<std::chrono::system_clock> StartTime;
      std::chrono::time_point<std::chrono::system_clock> EndTime;

      std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      StartTime = std::chrono::system_clock::now();
      afull->Mult(xfull,youtfull);
      EndTime = std::chrono::system_clock::now();

      youtfull += 1000.0;
      youtfull -= 1000.0;
      std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      auto timefull = std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count();

      std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      StartTime = std::chrono::system_clock::now();
      a->Mult(x,yout);
      EndTime = std::chrono::system_clock::now();

      yout += 1000.0;
      yout -= 1000.0;

      auto timex = std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count();

      std::cout << "% " << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      Vector ydiff;
      ydiff = yout;
      ydiff -= youtfull;

      Vector xdiff;
      xdiff = x;
      xdiff -= xfull;

      if( true )
      {
         //std::cout << "x" << std::endl;
         //x.Print(std::cout,1);
         //std::cout << "xfull" << std::endl;
         //xfull.Print(std::cout,1);

         int N = yout.Size();
         auto yout_ = Reshape(yout.Read(), N);
         auto youtfull_ = Reshape(youtfull.Read(), N );

/*
         for(int i = 0 ; i < N ; i++ )
         {
            std::cout << "yout(" << i+1 << ",1) = " << yout_(i) << ";" << std::endl;
            std::cout << "youtfull(" << i+1 << ",1) = " << youtfull_(i) << ";" << std::endl;
         }
*/

#ifdef MFEM_DEBUG
         std::cout << " %{ " << std::endl;
         std::cout << "               yout" << std::endl;
         yout.Print(mfem::out,1);
         std::cout << "               youtfull"  << std::endl;
         youtfull.Print(mfem::out,1);
         std::cout << "               ydiff" << std::endl;
         ydiff.Print(mfem::out,1);
#endif

      }


      std::cout << " Timing full = " << double(timefull) << std::endl; 
      std::cout << " Timing pa   = " << double(timex) << std::endl; 

      double errnorm = ydiff.Normlinf();
      std::cout << "               ||xdiff|| = " << xdiff.Normlinf() << std::endl;
      std::cout << "               ||ydiff|| = " << errnorm << std::endl;
      std::cout << "----------------------------------" << std::endl;
      std::cout << " %} " << std::endl;
      //exit(1);
   }

   int print_iter = 3;
   int max_num_iter = 800;
   double rtol = 1.0e-12;
   double atol = 1.0e-14;
   rtol = 1.0e-8; 
   atol = 1.0e-8;
   // -------------------------------------------
   {
      std::cout << "full solver " << std::endl;
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = set_bc ? 1 : 0;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      OperatorPtr Afull;
      Vector Bfull, Xfull;
      Xfull = 1;
      
      afull->FormLinearSystem(ess_tdof_list, xfull, *bfull, Afull, Xfull, Bfull);
      OperatorJacobiSmoother Mfull(*afull, ess_tdof_list);

      std::chrono::time_point<std::chrono::system_clock> StartTime;
      std::chrono::time_point<std::chrono::system_clock> EndTime;


      CGSolver pcg;
      pcg.SetPrintLevel(print_iter);
      pcg.SetMaxIter(max_num_iter);
      pcg.SetRelTol(sqrt(rtol));
      pcg.SetAbsTol(sqrt(atol));
      pcg.SetOperator(*Afull);
      pcg.SetPreconditioner(Mfull);

      StartTime = std::chrono::system_clock::now();

      pcg.Mult(Bfull, Xfull);

      EndTime = std::chrono::system_clock::now();

      auto timefull = std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count();

      std::cout << "full solver time (ms): " << double(timefull)/1000.0 << std::endl;
      double iters = pcg.GetNumIterations();
      std::cout << "update rate (iters/ms): " << iters/timefull << std::endl;

      std::cout << "----------------------------------" << std::endl;
   }
   // -------------------------------------------
   // Test the invoked operator
   {
      std::cout << "pa solver " << std::endl;
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = set_bc ? 1 : 0;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      OperatorPtr A;
      Vector B, X;
      X = 1;
      
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      OperatorJacobiSmoother M(*a, ess_tdof_list);
      // M
      std::chrono::time_point<std::chrono::system_clock> StartTime;
      std::chrono::time_point<std::chrono::system_clock> EndTime;



      CGSolver pcg2;
      pcg2.SetPrintLevel(print_iter);
      pcg2.SetMaxIter(max_num_iter);
      pcg2.SetRelTol(sqrt(rtol));
      pcg2.SetAbsTol(sqrt(atol));
      pcg2.SetOperator(*A);
      pcg2.SetPreconditioner(M);

      StartTime = std::chrono::system_clock::now();

      pcg2.Mult(B, X);
      //auto solver = PCGr(*A, M, B, X, print_iter, max_num_iter, rtol, atol );

      EndTime = std::chrono::system_clock::now();

      auto timepa = std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count();

      std::cout << "pa   solver time (ms): " << double(timepa)/1000.0 << std::endl;
      double iters = pcg2.GetNumIterations();
      std::cout << "update rate (iters/ms): " << iters/timepa << std::endl;

      std::cout << "----------------------------------" << std::endl;
   }   
   // -------------------------------------------
   exit(0);

   std::chrono::time_point<std::chrono::system_clock> StartTime;
   std::chrono::time_point<std::chrono::system_clock> EndTime;

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   /*
   int print_iter = 1;
   int max_num_iter = 500;
   double rtol = 1.0e-12;
   double atol = 0.0; 
   */
   if(pa)
   {
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = set_bc ? 1 : 0;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      OperatorPtr A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      StartTime = std::chrono::system_clock::now();

      OperatorJacobiSmoother M(*a, ess_tdof_list);
      PCG(*A, M, B, X, print_iter, max_num_iter, rtol, atol );

      
      EndTime = std::chrono::system_clock::now();
      
   }
   else if (sigma == -1.0)
   {
      const SparseMatrix &A = a->SpMat();
      GSSmoother M(A);

      StartTime = std::chrono::system_clock::now();

      PCG(A, M, *b, x, print_iter, max_num_iter, rtol, atol);

      EndTime = std::chrono::system_clock::now();
   
   }
   else
   {
      const SparseMatrix &A = a->SpMat();
      GSSmoother M(A);
      GMRES(A, M, *b, x, print_iter, max_num_iter, 10, rtol, atol);
   }

   auto time = std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count();

   std::cout << " Timing = " << time << std::endl; 

#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   //ProfilerStop();
   return 0;
}

// Temporary, 
double x_exact_approx(const Vector &x)
{
   int inf = 2;
   double sum = 0;
   for( int i = 1 ; i < inf ; i = i+2 )
      for( int j = 1 ; j < inf ; j = j+2  )
      {
         double term = sin(x(0)*M_PI*i)*sin(x(1)*M_PI*j);
         double coeff = 8.0/(i*i+j*j)/i/j/M_PI/M_PI;
         sum += coeff*term;
      }
   return sum;
}

double x1(const Vector &x)
{
   return x(0) + 0.2*(x(0) > 0.3333 ) + 0.2*(x(0) > 0.666) ;
}

double x2(const Vector &x)
{
   return 2.0*x(1);
}