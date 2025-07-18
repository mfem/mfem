//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   // 1. Parse command line options.
   string mesh_file = "../../data/star.mesh";
   int order = 1;
   if (myid) { out.Disable(); }

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh ser_mesh(mesh_file);
   for (int i=0; i<5; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   ser_mesh.Clear();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   const HYPRE_BigInt n_dofs = fespace.GlobalTrueVSize();
   out << "Number of unknowns: " << n_dofs << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   std::unique_ptr<HypreParVector> hypre_b(b.ParallelAssemble());

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   HypreParMatrix A;
   HypreParVector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Create an Objective, (grad u, grad u)/2 - (f, u)
   QuadraticFunctional J(comm, &A, &B, -1.0);
   HypreBoomerAMG M;
   M.SetPrintLevel(0);
   CGSolver cg_solver(comm);
   cg_solver.SetPreconditioner(M);
   cg_solver.SetAbsTol(1e-12);
   cg_solver.SetRelTol(1e-12);
   cg_solver.SetMaxIter(1e04);
   NewtonSolver solver(comm);
   solver.SetPreconditioner(cg_solver);
   solver.SetOperator(J.GetGradient());
   solver.SetMaxIter(2); // We only need one iteration as it is quadratic.
   Vector dummy(0);
   solver.Mult(dummy, X);


   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << x << flush;

   return 0;
}
