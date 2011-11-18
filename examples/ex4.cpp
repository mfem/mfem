//                                MFEM Example 4
//
// Compile with: make ex4
//
// Sample runs:  ex4 ../data/square-disc.mesh
//               ex4 ../data/star.mesh
//               ex4 ../data/beam-tet.mesh
//               ex4 ../data/beam-hex.mesh
//               ex4 ../data/escher.mesh
//               ex4 ../data/fichera.mesh
//               ex4 ../data/fichera-q2.vtk
//               ex4 ../data/fichera-q3.mesh
//               ex4 ../data/square-disc-nurbs.mesh
//               ex4 ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with the
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include <fstream>
#include "mfem.hpp"

// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex4 <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the 2D or 3D mesh from the given mesh file. In this example, we
   //    can handle triangular, quadrilateral, tetrahedral or hexahedral meshes
   //    with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   const int dim = mesh->Dimension();

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 25,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(25000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 3. Define a finite element space on the mesh. Here we use the lowest order
   //    Raviart-Thomas finite elements, but we can easily swich to higher-order
   //    spaces by changing the value of p.
   int p = 1;
   FiniteElementCollection *fec = new RT_FECollection(p-1, mesh -> Dimension());
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(dim, f_exact);
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogenious boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   VectorFunctionCoefficient F(dim, F_exact);
   x.ProjectCoefficient(F);

   // 6. Set up the bilinear form corresponding to the H(div) diffusion operator
   //    grad alpha div + beta I, by adding the div-div and the mass domain
   //    integrators and finally imposing the non-homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   Coefficient *alpha = new ConstantCoefficient(1.0);
   Coefficient *beta  = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   x = 0.0;
   PCG(A, M, *b, x, 1, 10000, 1e-20, 0.0);

   // 8. Compute and print the L^2 norm of the error.
   cout << "\n|| F_h - F ||_{L^2} = " << x.ComputeL2Error(F) << '\n' << endl;

   // 9. Save the refined mesh and the solution. This output can be viewed
   //    later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 11. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 12. Free the used memory.
   delete a;
   delete alpha;
   delete beta;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}


// The exact solution
void F_exact(const Vector &p, Vector &F)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   F(0) = cos(M_PI*x)*sin(M_PI*y);
   F(1) = cos(M_PI*y)*sin(M_PI*x);
   if(dim == 3)
      F(2) = 0.0;
}

// The right hand side
void f_exact(const Vector &p, Vector &f)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   double temp = 1 + 2*M_PI*M_PI;

   f(0) = temp*cos(M_PI*x)*sin(M_PI*y);
   f(1) = temp*cos(M_PI*y)*sin(M_PI*x);
   if(dim == 3)
      f(2) = 0;
}

