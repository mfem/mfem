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
   // 1. Parse command line options
   const char *mesh_file = "../data/star.mesh";
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
   int ndofs = fespace.GetTrueVSize();
   cout << "Number of unknowns: " << ndofs << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   LinearForm b(&fespace);
   int seed = 4000;
   GaussianWhiteNoiseDomainLFIntegrator *WhiteNoise = new
   GaussianWhiteNoiseDomainLFIntegrator(fespace, seed);
   b.AddDomainIntegrator(WhiteNoise);
   b.Assemble();

   b.Print();
   mfem::out << endl;

   // WhiteNoise->Reset();
   // b.Assemble();

   // b.Print();

   Vector bmean(ndofs);
   bmean = 0.0;
   int N = 1000000;
   for (int i = 0; i < N; i++)
   {
      WhiteNoise->Reset();
      b.Assemble();
      bmean += b;
   }
   bmean *= 1.0/(double)N;

   double diff = bmean.Normlinf();
   mfem::out << "mean error = " << diff << "\n" << endl;

   DenseMatrix C(ndofs);
   C = 0.0;

   for (int i = 0; i < N; i++)
   {
      WhiteNoise->Reset();
      b.Assemble();
      AddMultVVt(b, C);
   }
   C *= 1.0/(double)N;

   mfem::out << "C" << endl;
   C.PrintMatlab();
   mfem::out << endl;

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new MassIntegrator());
   a.Assemble();

   SparseMatrix M;
   Array<int> empty;
   a.FormSystemMatrix(empty,M);
   DenseMatrix Mdense;
   M.ToDenseMatrix(Mdense);

   mfem::out << "M" << endl;
   Mdense.PrintMatlab();
   mfem::out << endl;

   Mdense -= C;

   mfem::out << "M - C" << endl;
   Mdense.PrintMatlab();
   mfem::out << endl;

   diff = Mdense.MaxMaxNorm();

   mfem::out << "covariance error = " << diff << endl;

   return 0;
}
