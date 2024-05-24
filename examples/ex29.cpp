//                                MFEM Example 29
//
// Compile with: make ex29
//
// Sample runs:  ex29
//               ex29 -r 2 -sc
//               ex29 -mt 3 -o 4 -sc
//               ex29 -mt 3 -r 2 -o 4 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               finite element discretization of a PDE on a 2 dimensional
//               surface embedded in a 3 dimensional domain. In this case we
//               solve the Laplace problem -Div(sigma Grad u) = 1, with
//               homogeneous Dirichlet boundary conditions, where sigma is an
//               anisotropic diffusion constant defined as a 3x3 matrix
//               coefficient.
//
//               This example demonstrates the use of finite element integrators
//               on 2D domains with 3D coefficients.
//
//               We recommend viewing examples 1 and 7 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

Mesh * GetMesh(int type);

void trans(const Vector &x, Vector &r);

void sigmaFunc(const Vector &x, DenseMatrix &s);

real_t uExact(const Vector &x)
{
   return (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]));
}

void duExact(const Vector &x, Vector &du)
{
   du.SetSize(3);
   du[0] = 0.125 * (2.0 + x[0]) * x[1] * x[1];
   du[1] = -0.125 * (2.0 + x[0]) * x[0] * x[1];
   du[2] = -2.0 * x[2];
}

void fluxExact(const Vector &x, Vector &f)
{
   f.SetSize(3);

   DenseMatrix s(3);
   sigmaFunc(x, s);

   Vector du(3);
   duExact(x, du);

   s.Mult(du, f);
   f *= -1.0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 3;
   int mesh_type = 4; // Default to Quadrilateral mesh
   int mesh_order = 3;
   int ref_levels = 0;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_type, "-mt", "--mesh-type",
                  "Mesh type: 3 - Triangular, 4 - Quadrilateral.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Geometric order of the curved mesh.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // 2. Construct a quadrilateral or triangular mesh with the topology of a
   //    cylindrical surface.
   Mesh *mesh = GetMesh(mesh_type);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Transform the mesh so that it has a more interesting geometry.
   mesh->SetCurvature(mesh_order);
   mesh->Transform(trans);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   MatrixFunctionCoefficient sigma(3, sigmaFunc);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma);
   a.AddDomainIntegrator(integ);

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   //     Use a simple symmetric Gauss-Seidel preconditioner with PCG.
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Compute error in the solution and its flux
   FunctionCoefficient uCoef(uExact);
   real_t error = x.ComputeL2Error(uCoef);

   cout << "|u - u_h|_2 = " << error << endl;

   FiniteElementSpace flux_fespace(mesh, &fec, 3);
   GridFunction flux(&flux_fespace);
   x.ComputeFlux(*integ, flux); flux *= -1.0;

   VectorFunctionCoefficient fluxCoef(3, fluxExact);
   real_t flux_err = flux.ComputeL2Error(fluxCoef);

   cout << "|f - f_h|_2 = " << flux_err << endl;

   // 14. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x
               << "window_title 'Solution'\n" << flush;

      socketstream flux_sock(vishost, visport);
      flux_sock.precision(8);
      flux_sock << "solution\n" << *mesh << flux
                << "keys vvv\n"
                << "window_geometry 402 0 400 350\n"
                << "window_title 'Flux'\n"  << flush;
   }

   // 16. Free the used memory.
   delete mesh;

   return 0;
}

// Defines a mesh consisting of four flat rectangular surfaces connected to form
// a loop.
Mesh * GetMesh(int type)
{
   Mesh * mesh = NULL;

   if (type == 3)
   {
      mesh = new Mesh(2, 12, 16, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);
      mesh->AddVertex( 0.0, -1.0, 0.5);
      mesh->AddVertex( 1.0,  0.0, 0.5);
      mesh->AddVertex( 0.0,  1.0, 0.5);
      mesh->AddVertex(-1.0,  0.0, 0.5);

      mesh->AddTriangle(0, 1, 8);
      mesh->AddTriangle(1, 5, 8);
      mesh->AddTriangle(5, 4, 8);
      mesh->AddTriangle(4, 0, 8);
      mesh->AddTriangle(1, 2, 9);
      mesh->AddTriangle(2, 6, 9);
      mesh->AddTriangle(6, 5, 9);
      mesh->AddTriangle(5, 1, 9);
      mesh->AddTriangle(2, 3, 10);
      mesh->AddTriangle(3, 7, 10);
      mesh->AddTriangle(7, 6, 10);
      mesh->AddTriangle(6, 2, 10);
      mesh->AddTriangle(3, 0, 11);
      mesh->AddTriangle(0, 4, 11);
      mesh->AddTriangle(4, 7, 11);
      mesh->AddTriangle(7, 3, 11);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else if (type == 4)
   {
      mesh = new Mesh(2, 8, 4, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);

      mesh->AddQuad(0, 1, 5, 4);
      mesh->AddQuad(1, 2, 6, 5);
      mesh->AddQuad(2, 3, 7, 6);
      mesh->AddQuad(3, 0, 4, 7);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else
   {
      MFEM_ABORT("Unrecognized mesh type " << type << "!");
   }
   mesh->FinalizeTopology();

   return mesh;
}

// Transforms the four-sided loop into a curved cylinder with skewed top and
// base.
void trans(const Vector &x, Vector &r)
{
   r.SetSize(3);

   real_t tol = 1e-6;
   real_t theta = 0.0;
   if (fabs(x[1] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (x[0] - 2.0);
   }
   else if (fabs(x[0] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * x[1];
   }
   else if (fabs(x[1] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * (2.0 - x[0]);
   }
   else if (fabs(x[0] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (4.0 - x[1]);
   }
   else
   {
      cerr << "side not recognized "
           << x[0] << " " << x[1] << " " << x[2] << endl;
   }

   r[0] = cos(theta);
   r[1] = sin(theta);
   r[2] = 0.25 * (2.0 * x[2] - 1.0) * (r[0] + 2.0);
}

// Anisotropic diffusion coefficient
void sigmaFunc(const Vector &x, DenseMatrix &s)
{
   s.SetSize(3);
   real_t a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
   s(0,0) = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
   s(0,1) = x[0] * x[1] * (8.0 / a - 0.5);
   s(0,2) = 0.0;
   s(1,0) = s(0,1);
   s(1,1) = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
   s(1,2) = 0.0;
   s(2,0) = 0.0;
   s(2,1) = 0.0;
   s(2,2) = a / 32.0;
}
