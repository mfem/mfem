//                       MFEM Example 7 - Parallel Version
//
// Compile with: make ex7p
//
// Sample runs:  mpirun -np 4 ex7p -e 0 -o 2 -r 4
//               mpirun -np 4 ex7p -e 1 -o 2 -r 4 -snap
//               mpirun -np 4 ex7p -e 0 -amr 1
//               mpirun -np 4 ex7p -e 1 -amr 2 -o 2
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
double analytic_solution(const Vector &x);
double analytic_rhs(const Vector &x);
void SnapNodes(Mesh &mesh);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Generate an initial high-order (surface) mesh on the unit sphere. The
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
      const double quad_v[8][3] =
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

   // 4. Refine the mesh while snapping nodes to the sphere. Number of parallel
   //    refinements is fixed to 2.
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         mesh->UniformRefinement();
      }

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap)
      {
         SnapNodes(*mesh);
      }
   }

   if (amr == 1)
   {
      Vertex target(0.0, 0.0, 1.0);
      for (int l = 0; l < 3; l++)
      {
         mesh->RefineAtVertex(target);
      }
      SnapNodes(*mesh);
   }
   else if (amr == 2)
   {
      for (int l = 0; l < 2; l++)
      {
         mesh->RandomRefinement(0.5); // 50% probability
      }
      SnapNodes(*mesh);
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();

         // Snap the nodes of the refined mesh back to sphere surface.
         if (always_snap)
         {
            SnapNodes(*pmesh);
         }
      }
      if (!always_snap || par_ref_levels < 1)
      {
         SnapNodes(*pmesh);
      }
   }

   if (amr == 1)
   {
      Vertex target(0.0, 0.0, 1.0);
      for (int l = 0; l < 2; l++)
      {
         pmesh->RefineAtVertex(target);
      }
      SnapNodes(*pmesh);
   }
   else if (amr == 2)
   {
      for (int l = 0; l < 2; l++)
      {
         pmesh->RandomRefinement(0.5); // 50% probability
      }
      SnapNodes(*pmesh);
   }

   // 5. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements -- the same as the mesh nodes.
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, &fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef (analytic_rhs);
   FunctionCoefficient sol_coef (analytic_solution);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   ParGridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    and Mass domain integrators.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));

   // 9. Assemble the parallel linear system, applying any transformations
   //    such as: parallel assembly, applying conforming constraints, etc.
   a->Assemble();
   HypreParMatrix A;
   Vector B, X;
   Array<int> empty_tdof_list;
   a->FormLinearSystem(empty_tdof_list, x, *b, A, X, B);

   // 10. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre. Extract the parallel grid function x
   //     corresponding to the finite element approximation X. This is the local
   //     solution on each processor.
   HypreSolver *amg = new HypreBoomerAMG(A);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);
   a->RecoverFEMSolution(X, *b, x);

   delete a;
   delete b;

   // 11. Compute and print the L^2 norm of the error.
   double err = x.ComputeL2Error(sol_coef);
   if (myid == 0)
   {
      cout << "\nL2 norm of error: " << err << endl;
   }

   // 12. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -np <np> -m sphere_refined -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "sphere_refined." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 14. Free the used memory.
   delete pcg;
   delete amg;
   delete fespace;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double analytic_solution(const Vector &x)
{
   double l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return x(0)*x(1)/l2;
}

double analytic_rhs(const Vector &x)
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
