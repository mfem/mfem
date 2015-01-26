//                                MFEM Example 7
//
// Compile with: make ex7
//
// Sample runs:  ex7 -e 0 -o 2 -r 4
//               ex7 -e 1 -o 2 -r 4 -snap
//
// Description:  This example code demonstrates the use of MFEM to define a
//               triangulation of a unit sphere and a simple isoparametric
//               finite element discretization of the Laplace problem with mass
//               term, -Delta u + u = f.
//
//               The example highlights mesh generation, the use of mesh
//               refinement, high-order meshes and finite elements, as well as
//               surface-based linear and bilinear forms corresponding to the
//               left-hand side and right-hand side of the discrete linear
//               system.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double analytic_solution(Vector &x);
double analytic_rhs(Vector &x);
void SnapNodes(Mesh &mesh);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int elem_type = 1;
   int ref_levels = 2;
   int order = 2;
   bool always_snap = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&elem_type, "-e", "--elem",
                  "Type of elements to use: 0 - triangles, 1 - quads.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&always_snap, "-snap", "--always-snap", "-no-snap",
                  "--snap-at-the-end",
                  "If true, snap nodes to the sphere initially and after each refinement "
                  "otherwise, snap only after the last refinement");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Generate an initial high-order (surface) mesh on the unit sphere. The
   //    Mesh object represents a 2D mesh in 3 spatial dimensions. We first add
   //    the elements and the vertices of the mesh, and then make it high-order
   //    by specifying a finite element space for its nodes.
   int Nvert = 8, Nelem = 6;
   if (elem_type == 0)
   {
      Nvert = 6;
      Nelem = 8;
   }
   Mesh *mesh = new Mesh(2, Nvert, Nelem, 0, 3);

   if (elem_type == 0) // inscribed octahedron
   {
      const double tri_v[6][3] =
         {{ 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
          { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}};
      const int tri_e[8][3] =
         {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
          {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(tri_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddTriangle(tri_e[j], attribute);
      }
      mesh->FinalizeTriMesh(1, 1, true);
   }
   else // inscribed cube
   {
      const double quad_v[8][3] =
         {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
          {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};
      const int quad_e[6][4] =
         {{3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
          {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddQuad(quad_e[j], attribute);
      }
      mesh->FinalizeQuadMesh(1, 1, true);
   }

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);

   // 3. Refine the mesh while snapping nodes to the sphere.
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
         mesh->UniformRefinement();

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap || l == ref_levels)
         SnapNodes(*mesh);
   }

   // 4. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements -- the same as the mesh nodes.
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, &fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef (analytic_rhs);
   FunctionCoefficient sol_coef (analytic_solution);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b->Assemble();

   // 6. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 1, 200, 1e-12, 0.0);
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // 9. Compute and print the L^2 norm of the error.
   cout<<"\nL2 norm of error: " << x.ComputeL2Error(sol_coef) << endl;

   // 10. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m sphere_refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("sphere_refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 12. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete mesh;

   return 0;
}

double analytic_solution(Vector &x)
{
   double l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return x(0)*x(1)/l2;
}

double analytic_rhs(Vector &x)
{
   double l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return 7*x(0)*x(1)/l2;
}

void SnapNodes(Mesh &mesh)
{
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));

      node /= node.Norml2();

      for (int d = 0; d < mesh.SpaceDimension(); d++)
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
   }
}
