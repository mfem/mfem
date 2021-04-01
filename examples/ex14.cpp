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

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double x_exact_approx(const Vector &);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   double eta = 0.0;
   bool visualization = 1;
   bool pa = false;

   OptionsParser args(argc, argv);
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
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
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
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec;
   if(pa)
   {
      // Only Gauss-Lobatto and Bernstein basis are supported in L2FaceRestriction.
      fec = new DG_FECollection(order, dim, BasisType::Positive);
   }
   else
   {
      fec = new DG_FECollection(order, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

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
   x.ProjectCoefficient(f);

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
      a->AddInteriorNormalDerivativeFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
      a->AddBdrNormalDerivativeFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   }
   else
   {
      // Default setting
      a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
      a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   }
   /*
   if (eta > 0)
   {
      a->AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a->AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
   }
   */
   a->Assemble();
   a->Finalize();

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   int print_iter = 1;
   int max_num_iter = 500;
   double rtol = 1.0e-12;
   double atol = 0.0; 
   if(pa)
   {
      bool set_bc = true;
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

      OperatorJacobiSmoother M(*a, ess_tdof_list);
      PCG(*A, M, B, X, print_iter, max_num_iter, rtol, atol );

   }
   else if (sigma == -1.0)
   {
      const SparseMatrix &A = a->SpMat();
      GSSmoother M(A);
      PCG(A, M, *b, x, print_iter, max_num_iter, rtol, atol);
   }
   else
   {
      const SparseMatrix &A = a->SpMat();
      GSSmoother M(A);
      GMRES(A, M, *b, x, print_iter, max_num_iter, 10, rtol, atol);
   }

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

   return 0;
}

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
   //sum = x(0) + 2.0*x(1); 
   //sum = 1.0;
   //sum = exp(-10.0*((x(1)-0.55)*(x(1)-0.55))-5.0*((x(0)-0.35)*(x(0)-0.35)));
   return sum;
}
