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
   int solver_type = 0;
   int integrator_type = 0;
   double p_order = 1.0;
   double q_order = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&solver_type, "-s", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Stationary Linear Iteration"
                  "\n\t1: Preconditioned Conjugate Gradient"
                  "\n\tTODO");
   args.AddOption(&integrator_type, "-i", "--integrator",
                  "Integrators to be considered:"
                  "\n\t0: MassIntegrator"
                  "\n\t1: DiffusionIntegrator"
                  "\n\tTODO");
   args.AddOption(&p_order, "-p", "--p-order",
                  "P-order for L(p,q)-Jacobi preconditioner");
   args.AddOption(&q_order, "-q", "--q-order",
                  "Q-order for L(p,q)-Jacobi preconditioner");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh(mesh_file);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   mesh.UniformRefinement();

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
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

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
   switch (integrator_type)
   {
      case 0:
         a.AddDomainIntegrator(new MassIntegrator);
         break;
      case 1:
         a.AddDomainIntegrator(new DiffusionIntegrator);
         break;
   }
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 11. Solve the system using PCG with Lp-Jacobi preconditioner
   // D_{p,q} = diag( D^{1+q-p} |A|^p D^{-q} 1) , where D = diag(A)
   Vector first_diag(A.Height());  // right
   Vector temp(A.Height());
   Vector second_diag(A.Height());  // left

   A.GetDiag(first_diag);
   second_diag = first_diag;

   first_diag.PowerAbs(-q_order);
   A.PowAbsMult(p_order, 1.0, first_diag, 0.0, temp);
   second_diag.PowerAbs(1.0 + q_order - p_order);
   temp *= second_diag;

   auto lpq_jacobi = new OperatorJacobiSmoother(temp, boundary_dofs);

   Solver *solver = nullptr;
   switch (solver_type)
   {
      case 0:
         solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case 1:
         solver = new CGSolver(MPI_COMM_WORLD);
         break;
   }
   IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(solver);
   if (it_solver)
   {
      it_solver->SetRelTol(1e-12);
      it_solver->SetMaxIter(2000);
      it_solver->SetPrintLevel(1);
      it_solver->SetPreconditioner(*lpq_jacobi);
   }
   solver->SetOperator(A);
   solver->Mult(B, X);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol");
   mesh.Save("mesh");

   delete solver;
   delete lpq_jacobi;
   return 0;
}
