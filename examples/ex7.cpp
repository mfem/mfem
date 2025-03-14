//                                MFEM Example 7
//
// Compile with: make ex7
//
// Sample runs:  ex7 -e 0 -o 2 -r 4
//               ex7 -e 1 -o 2 -r 4 -snap
//               ex7 -e 0 -amr 1
//               ex7 -e 1 -amr 2 -o 2
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
//               system. Simple local mesh refinement is also demonstrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
real_t analytic_solution(const Vector &x);
real_t analytic_rhs(const Vector &x);
void SnapNodes(Mesh &mesh);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int elem_type = 1;
   int ref_levels = 2;
   int amr = 0;
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
   args.AddOption(&amr, "-amr", "--refine-locally",
                  "Additional local (non-conforming) refinement:"
                  " 1 = refine around north pole, 2 = refine randomly.");
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
      const real_t tri_v[6][3] =
      {
         { 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
         { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}
      };
      const int tri_e[8][3] =
      {
         {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
         {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}
      };

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
      const real_t quad_v[8][3] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[6][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
      };

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
      {
         mesh->UniformRefinement();
      }

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap || l == ref_levels)
      {
         SnapNodes(*mesh);
      }
   }

   if (amr == 1)
   {
      Vertex target(0.0, 0.0, 1.0);
      for (int l = 0; l < 5; l++)
      {
         mesh->RefineAtVertex(target);
      }
      SnapNodes(*mesh);
   }
   else if (amr == 2)
   {
      for (int l = 0; l < 4; l++)
      {
         mesh->RandomRefinement(0.5); // 50% probability
      }
      SnapNodes(*mesh);
   }

   // 4. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements -- the same as the mesh nodes.
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, &fec);
   cout << "Number of unknowns: " << fespace->GetTrueVSize() << endl;

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
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    and Mass domain integrators.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));

   // 8. Assemble the linear system, apply conforming constraints, etc.
   a->Assemble();
   SparseMatrix A;
   Vector B, X;
   Array<int> empty_tdof_list;
   a->FormLinearSystem(empty_tdof_list, x, *b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
   // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system AX=B with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
   // 9. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 10. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 11. Compute and print the L^2 norm of the error.
   cout<<"\nL2 norm of error: " << x.ComputeL2Error(sol_coef) << endl;

   // 12. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m sphere_refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("sphere_refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 14. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete mesh;

   return 0;
}

real_t analytic_solution(const Vector &x)
{
   real_t l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return x(0)*x(1)/l2;
}

real_t analytic_rhs(const Vector &x)
{
   real_t l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return 7*x(0)*x(1)/l2;
}

void SnapNodes(Mesh &mesh)
{
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }

      node /= node.Norml2();

      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }
   if (mesh.Nonconforming())
   {
      // Snap hanging nodes to the master side.
      Vector tnodes;
      nodes.GetTrueDofs(tnodes);
      nodes.SetFromTrueDofs(tnodes);
   }
}
