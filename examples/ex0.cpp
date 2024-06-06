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
#include "../linalg/dtensor.hpp"
#include <fstream>
#include <iostream>
#define IDX2C(i,j,k,inc) ((i)+(j*inc)+(k*inc*inc))
#define IDXV(i,j,inc) ((i)+(j*inc))

using namespace std;
using namespace mfem;

/// @brief 
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   cout << "Let's see if this works \n";

   ConstantCoefficient q(111.0);
   Coefficient *Q(&q); 
   Q->SetTime(222.);
   real_t time = Q->GetTime();


   cout << "Pointer Q is " << Q << "\n";
   cout << "This matches with the address of q, which is " << &q << "\n";
   // cout << "Note that the value that Q points to is " << *Q.constant << "\n";
   cout << "This matches with the value of q, which is " << q.constant << endl; 
   cout << "Pointer Q has the time of " << time << "\n";



   // Check operators are not issues

   // QuadratureFunctionCoefficient q();
   const real_t detJ = 3.141;
   const real_t d_D = detJ * q.constant;  // auto d_D is a DeviceTensor made up of real_t elements; ERRORS

   // cout << "d_D = " << d_D << endl;


   // Check pointer behavior: 
   double *f = new double[4];
   double *F = f;
   cout << "Size of F is " << sizeof (*F) << endl;

   int i, j, k;
   for (k=0;k<2;k++) {
      for (j=0;j<6;j++) {
         for (i=0;i<5;i++) {
            cout << IDX2C(i,j,k,6) << " ";
         }
      }
   }
   cout << endl;

   DenseTensor G(6,6,2);
   Vector g(6);
   Vector gy(6);

   cout << "At first, G is " << endl;
   for (k = 0; k < 2; k++) {
      for (j = 0; j < 6; j++) {
         for (i = 0; i < 6; i++) {
            printf ("%7.0f", G.Data()[IDX2C(i,j,k,6)]);
         }
         printf("\n");
      }
      printf("\n");
   }

   for (k = 0; k < 2; k++) {
      for (j = 0; j < 6; j++) {
         for (i = 0; i < 6; i++) {
            g.GetData()[i] = i;
            G.Data()[IDX2C(i,j,k,6)] = (double)(IDX2C(i,j,k,6));
         }
         printf("\n");
      }
      printf("\n");
   }
   g.Print();
   cout << g.GetData() << endl;

   cout << "G is " << endl;
   for (k = 0; k < 2; k++) {
      for (j = 0; j < 6; j++) {
         for (i = 0; i < 6; i++) {
            printf ("%7.0f", G.Data()[IDX2C(i,j,k,6)]);
         }
         printf("\n");
      }
      printf("\n");
   }
   cout << "G at (6,6,2) is " << G.Data()[IDX2C(5,5,1,6)] << endl;
   cout << &G.Data()[IDX2C(5,5,1,6)] << endl;

   auto d_G = Reshape(G.Read(), 6, 6, 1);

   gy.Print();
   cout << endl;


   double* y = 0;  // host device pointer
   cout << "Size of y: " << sizeof(y) << endl;
   y = (double *)malloc (6 * 1 * sizeof(*y));

   for (j = 0; j < 6; j++) {
      cout << y[j] << " ";
   }
   cout << endl;

   Vector Y(y,6);
   Y.SetData(y);
   cout << "Vector Y is "; Y.Print();

   cout << "Size of G: " << sizeof(*G.Data()) << endl;
   cout << "Size of g: " << sizeof(g.GetData()) << endl;
   cout << "Size of gy: " << sizeof(gy.GetData()) << endl;
   cout << "Size of y: " << sizeof(y) << endl;


   return 0;
}
