//                                MFEM Example 2
//
// Compile with: make ex2
//
// Sample runs:  ex2 -m ../data/beam-tri.mesh
//               ex2 -m ../data/beam-quad.mesh
//               ex2 -m ../data/beam-tet.mesh
//               ex2 -m ../data/beam-hex.mesh
//               ex2 -m ../data/beam-wedge.mesh
//               ex2 -m ../data/beam-quad.mesh -o 3 -sc
//               ex2 -m ../data/beam-quad-nurbs.mesh
//               ex2 -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               [TODO]
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   real_t force_mag = 1.0;
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   ConstantCoefficient lambda(1.0);
   ConstantCoefficient mu(1.0);
   real_t beta = -1.0;
   real_t kappa = 100.0;

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda,mu));
   a->AddBdrFaceIntegrator(
      new NitscheElasticityIntegrator(lambda, mu, beta, kappa),
      ess_bdr);

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component. [FIXME]
   ConstantCoefficient f(force_mag);

   LinearForm *b = new LinearForm(fespace);
   b->AddBdrFaceIntegrator(
         new NitscheElasticityDirichletLFIntegrator(
            f, lambda, mu, beta, kappa), ess_bdr);
   b->Assemble();

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   cout << "matrix ... " << flush;
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
   // 11. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }

   // 14. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
