
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mesh_partition.hpp"

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

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n" << mesh << "keys n \n" << flush;
   }

   // mesh.EnsureNodes();

   // Array<int> elems0({0,4,8,12,16});
   // Array<int> elems0({0,4,8,12,16});
   Array<int> elems0({8,9});
   // Array<int> elems0({7,6,17,20,21,22});
   // Array<int> elems0({0,1,2,3,4});
   // Array<int> elems0({104,103,86,109});
   elems0.Print(cout, elems0.Size());
   Subdomain subdomain0(mesh);
   Mesh * submesh0 = subdomain0.GetSubMesh(elems0);

   Array<int> bdrelems0({0,1});
   Mesh * bdrmesh0 = subdomain0.GetBdrSurfaceMesh(bdrelems0);

   // Array<int> elems1({0,16,19,76,77,20,1,2,3});
   // elems1.Print(cout, elems1.Size());
   // SubMesh submesh1(mesh, elems1);
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (submesh0)
      {
         socketstream mesh0_sock(vishost, visport);
         mesh0_sock.precision(8);
         mesh0_sock << "mesh\n" << *submesh0 << "keys n \n" << flush;
      }
      if (bdrmesh0)
      {
         socketstream mesh1_sock(vishost, visport);
         mesh1_sock.precision(8);
         mesh1_sock << "mesh\n" << *bdrmesh0 << "keys n \n" << flush;
      }
   }

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   // H1_FECollection fec(order, mesh.Dimension());
   // FiniteElementSpace fespace(&mesh, &fec);
   // cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // // 4. Extract the list of all the boundary DOFs. These will be marked as
   // //    Dirichlet in order to enforce zero boundary conditions.
   // Array<int> boundary_dofs;
   // fespace.GetBoundaryTrueDofs(boundary_dofs);

   // // 5. Define the solution x as a finite element grid function in fespace. Set
   // //    the initial guess to zero, which also sets the boundary conditions.
   // GridFunction x(&fespace);
   // x = 0.0;

   // // 6. Set up the linear form b(.) corresponding to the right-hand side.
   // ConstantCoefficient one(1.0);
   // LinearForm b(&fespace);
   // b.AddDomainIntegrator(new DomainLFIntegrator(one));
   // b.Assemble();

   // // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   // BilinearForm a(&fespace);
   // a.AddDomainIntegrator(new DiffusionIntegrator);
   // a.Assemble();

   // // 8. Form the linear system A X = B. This includes eliminating boundary
   // //    conditions, applying AMR constraints, and other transformations.
   // SparseMatrix A;
   // Vector B, X;
   // a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   // GSSmoother M(A);
   // PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // // 10. Recover the solution x as a grid function and save to file. The output
   // //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   // a.RecoverFEMSolution(X, b, x);

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream sol_sock(vishost, visport);
   // sol_sock.precision(8);
   // sol_sock << "solution\n" << mesh << x << flush;

   return 0;
}
