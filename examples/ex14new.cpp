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

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   bool set_bc = true;
   double sigma = -1.0;
   double kappa = -1.0;
   double eta = 0.0;
   bool visualization = 1;
   bool pa = false;
   bool ea = false;
   bool fa = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
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

   std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
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

   std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = NULL;
   if(pa)
   {
      // Only Gauss-Lobatto and Bernstein basis are supported in L2FaceRestriction.
      fec = new DG_FECollection(order, dim,BasisType::GaussLobatto);
   }
   else
   {
      fec = new DG_FECollection(order, dim);
   }   
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
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

   std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   if (pa)
   {
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      a->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      a->SetAssemblyLevel(AssemblyLevel::FULL);
   }
   else
   {
      // Default setting
      a->SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
   }
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   if (eta > 0)
   {
      std::cout << "Whoops we aren't ready for this case yet.  " << std::endl;      
      exit(1);
   }
   a->Assemble();
   a->Finalize();

   // 8. Solve the linear system A X = B.
   // If not using partial assembly,
   // define a simple symmetric Gauss-Seidel preconditioner and use it to
   // solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   // non-symmetric one.
   // If using partial assembly , use Jacobi preconditioning
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = set_bc ? 1 : 0;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   OperatorPtr A;
   Vector B, X;
   X = 1.0;

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      if (sigma == -1.0)
      {
         PCG(*A, M, *b, x, 1, 500, 1e-12, 0.0);
      }
      else
      {
         GMRES(*A, M, *b, x, 1, 500, 10, 1e-12, 0.0);
      }
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      if (UsesTensorBasis(*fespace))
      {
         OperatorJacobiSmoother M(*a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 10000, 1e-20, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 10000, 1e-20, 0.0);
      }
   }

   // Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

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
