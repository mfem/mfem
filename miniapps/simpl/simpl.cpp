//                       MFEM Example 0 - Parallel Version
//
// Compile with: make ex0p
//
// Sample runs:  mpirun -np 4 ex0p
//               mpirun -np 4 ex0p -m ../data/fichera.mesh
//               mpirun -np 4 ex0p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with zero Dirichlet boundary
//              conditions. General 2D/3D serial mesh files and finite element
//              polynomial degrees can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "linear_solver.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;
   int ref_levels = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-rs", "--refine-serial",
                  "Serial refinement level");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh = Mesh::MakeCartesian2D(4, 4, Element::Type::QUADRILATERAL,
                                            false);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   for (int i=0; i<ref_levels; i++) {mesh.UniformRefinement();}

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> ess_bdr(4); ess_bdr = 1;

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   EllipticSolver solver(a, ess_bdr);
   solver.Solve(b, x);

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   x.Save("sol");
   mesh.Save("mesh");

   return 0;
}
