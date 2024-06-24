//                       MFEM Example 00 - Parallel Version
//
// Compile with: make ex00p
//
// Sample runs:  mpirun -np 4 ex00p
//               mpirun -np 4 ex00p -s -p 2.0
//               mpirun -np 4 ex00p -m ../data/fichera.mesh
//               mpirun -np 4 ex00p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with zero Dirichlet boundary
//              conditions. General 2D/3D serial mesh files and finite element
//              polynomial degrees can be specified by command line options.
//              This example makes emphasis on the utilization of a family of
//              L^p-Jacobi preconditioners.

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
   double p = 1.0;
   bool type_pc = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&p, "-p", "--p-order", "Order Lp-Jacobi preconditioner");
   args.AddOption(&type_pc, "-ns", "--non-symmetric", "-s", "--symmetric",
                  "Non-symmetric Lp-Jacobi (-ns) or symmetrized Lp-Jacobi (-s)");
   args.ParseCheck();

   MFEM_ASSERT(p>0, "p must be strictly-positive");

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
   Array<int> ess_tdofs_list;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      boundary_dofs = 1;
      fespace.GetEssentialTrueDofs(boundary_dofs, ess_tdofs_list);
   }


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

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 11. Solve the system using PCG with Lp-Jacobi preconditioner
   Solver* M = NULL;
   if (type_pc)
   {
      // diag(A)^(1-p) |A|^p 1
      Vector ones(A.Height());
      Vector temp(A.Height());
      Vector diag(A.Height());

      ones = 1.0;
      A.PowAbsMult(p, 1.0, ones, 0.0, temp);
      A.GetDiag(diag);
      diag.PowerAbs(1.0-p);
      temp *= diag;

      auto lp_jacobi_pc = new OperatorJacobiSmoother(temp, ess_tdofs_list);

      M = lp_jacobi_pc;
   }
   else
   {
      // diag(A)^(1-p/2) |A|^p diag(A)^(-p/2) 1
      Vector first_diag(A.Height());
      Vector temp(A.Height());
      Vector second_diag(A.Height());

      A.GetDiag(first_diag);
      second_diag = first_diag;
      first_diag.PowerAbs(-p/2.0);
      A.PowAbsMult(p, 1.0, first_diag, 0.0, temp);
      second_diag.PowerAbs(1.0-p/2.0);
      temp *= second_diag;

      auto lp_jacobi_pc = new OperatorJacobiSmoother(temp, ess_tdofs_list);

      M = lp_jacobi_pc;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol");
   mesh.Save("mesh");

   return 0;
}
