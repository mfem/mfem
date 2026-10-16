//                                MFEM Example 31
//
// Compile with: make ex31
//
// Sample runs:  ex31 -m ../data/inline-segment.mesh -o 2
//               ex31 -m ../data/hexagon.mesh -o 2
//               ex31 -m ../data/star.mesh -o 2
//               ex31 -m ../data/fichera.mesh -o 3 -r 1
//               ex31 -m ../data/square-disc-nurbs.mesh -o 3
//               ex31 -m ../data/amr-quad.mesh -o 2 -r 1
//               ex31 -m ../data/amr-hex.mesh -r 1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + sigma E = f with boundary condition
//               E x n = <given tangential field>. In this example sigma is an
//               anisotropic 3x3 tensor. Here, we use a given exact solution E
//               and compute the corresponding r.h.s. f.  We discretize with
//               Nedelec finite elements in 1D, 2D, or 3D.
//
//               The example demonstrates the use of restricted H(curl) finite
//               element spaces with the curl-curl and the (vector finite
//               element) mass bilinear form, as well as the computation of
//               discretization error when the exact solution is known. These
//               restricted spaces allow the solution of 1D or 2D
//               electromagnetic problems which involve 3D field vectors.  Such
//               problems arise in plasma physics and crystallography.
//
//               We recommend viewing example 3 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void CurlE_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
real_t freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   kappa = freq * M_PI;

   // 2. Read the mesh from the given mesh file.  We can handle triangular,
   //    quadrilateral, or mixed meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement (2 by default, or specified on
   //    the command line with -r).
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order restricted to 1D, 2D, or 3D
   //    depending on the dimension of the given mesh file.
   FiniteElementCollection *fec = NULL;
   if (dim == 1)
   {
      fec = new ND_R1D_FECollection(order, dim);
   }
   else if (dim == 2)
   {
      fec = new ND_R2D_FECollection(order, dim);
   }
   else
   {
      fec = new ND_FECollection(order, dim);
   }
   FiniteElementSpace fespace(&mesh, fec);
   int size = fespace.GetTrueVSize();
   cout << "Number of H(Curl) unknowns: " << size << endl;

   // 5. Determine the list of true essential boundary dofs. In this example,
   //    the boundary conditions are defined by marking all the boundary
   //    attributes from the mesh as essential (Dirichlet) and converting them
   //    to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in
   //    the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b.Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction sol(&fespace);
   VectorFunctionCoefficient E(3, E_exact);
   VectorFunctionCoefficient CurlE(3, CurlE_exact);
   sol.ProjectCoefficient(E);

   // 8. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2; // 1/sqrt(2) in cmath
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   ConstantCoefficient muinv(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    etc.
   a.Assemble();

   OperatorPtr A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // 10. Solve the system A X = B.

#ifndef MFEM_USE_SUITESPARSE
   // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
#else
   // 11. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
   //     system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
#endif

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, sol);

   // 13. Compute and print the H(Curl) norm of the error.
   {
      real_t error = sol.ComputeHCurlError(&E, &CurlE);
      cout << "\n|| E_h - E ||_{H(Curl)} = " << error << '\n' << endl;
   }


   // 14. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      sol.Save(sol_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      VectorGridFunctionCoefficient solCoef(&sol);
      CurlGridFunctionCoefficient dsolCoef(&sol);

      if (dim ==1)
      {
         socketstream x_sock(vishost, visport);
         socketstream y_sock(vishost, visport);
         socketstream z_sock(vishost, visport);
         socketstream dy_sock(vishost, visport);
         socketstream dz_sock(vishost, visport);
         x_sock.precision(8);
         y_sock.precision(8);
         z_sock.precision(8);
         dy_sock.precision(8);
         dz_sock.precision(8);

         Vector xVec(3); xVec = 0.0; xVec(0) = 1;
         Vector yVec(3); yVec = 0.0; yVec(1) = 1;
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient xVecCoef(xVec);
         VectorConstantCoefficient yVecCoef(yVec);
         VectorConstantCoefficient zVecCoef(zVec);

         H1_FECollection fec_h1(order, dim);
         L2_FECollection fec_l2(order-1, dim);

         FiniteElementSpace fes_h1(&mesh, &fec_h1);
         FiniteElementSpace fes_l2(&mesh, &fec_l2);

         GridFunction xComp(&fes_l2);
         GridFunction yComp(&fes_h1);
         GridFunction zComp(&fes_h1);

         GridFunction dyComp(&fes_l2);
         GridFunction dzComp(&fes_l2);

         InnerProductCoefficient xCoef(xVecCoef, solCoef);
         InnerProductCoefficient yCoef(yVecCoef, solCoef);
         InnerProductCoefficient zCoef(zVecCoef, solCoef);

         xComp.ProjectCoefficient(xCoef);
         yComp.ProjectCoefficient(yCoef);
         zComp.ProjectCoefficient(zCoef);

         x_sock << "solution\n" << mesh << xComp << flush
                << "window_title 'X component'" << endl;
         y_sock << "solution\n" << mesh << yComp << flush
                << "window_geometry 403 0 400 350 "
                << "window_title 'Y component'" << endl;
         z_sock << "solution\n" << mesh << zComp << flush
                << "window_geometry 806 0 400 350 "
                << "window_title 'Z component'" << endl;

         InnerProductCoefficient dyCoef(yVecCoef, dsolCoef);
         InnerProductCoefficient dzCoef(zVecCoef, dsolCoef);

         dyComp.ProjectCoefficient(dyCoef);
         dzComp.ProjectCoefficient(dzCoef);

         dy_sock << "solution\n" << mesh << dyComp << flush
                 << "window_geometry 403 375 400 350 "
                 << "window_title 'Y component of Curl'" << endl;
         dz_sock << "solution\n" << mesh << dzComp << flush
                 << "window_geometry 806 375 400 350 "
                 << "window_title 'Z component of Curl'" << endl;
      }
      else if (dim == 2)
      {
         socketstream xy_sock(vishost, visport);
         socketstream z_sock(vishost, visport);
         socketstream dxy_sock(vishost, visport);
         socketstream dz_sock(vishost, visport);

         DenseMatrix xyMat(2,3); xyMat = 0.0;
         xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
         MatrixConstantCoefficient xyMatCoef(xyMat);
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient zVecCoef(zVec);

         MatrixVectorProductCoefficient xyCoef(xyMatCoef, solCoef);
         InnerProductCoefficient zCoef(zVecCoef, solCoef);

         H1_FECollection fec_h1(order, dim);
         ND_FECollection fec_nd(order, dim);
         RT_FECollection fec_rt(order-1, dim);
         L2_FECollection fec_l2(order-1, dim);

         FiniteElementSpace fes_h1(&mesh, &fec_h1);
         FiniteElementSpace fes_nd(&mesh, &fec_nd);
         FiniteElementSpace fes_rt(&mesh, &fec_rt);
         FiniteElementSpace fes_l2(&mesh, &fec_l2);

         GridFunction xyComp(&fes_nd);
         GridFunction zComp(&fes_h1);

         GridFunction dxyComp(&fes_rt);
         GridFunction dzComp(&fes_l2);

         xyComp.ProjectCoefficient(xyCoef);
         zComp.ProjectCoefficient(zCoef);

         xy_sock.precision(8);
         xy_sock << "solution\n" << mesh << xyComp
                 << "window_title 'XY components'\n" << flush;
         z_sock << "solution\n" << mesh << zComp << flush
                << "window_geometry 403 0 400 350 "
                << "window_title 'Z component'" << endl;

         MatrixVectorProductCoefficient dxyCoef(xyMatCoef, dsolCoef);
         InnerProductCoefficient dzCoef(zVecCoef, dsolCoef);

         dxyComp.ProjectCoefficient(dxyCoef);
         dzComp.ProjectCoefficient(dzCoef);

         dxy_sock << "solution\n" << mesh << dxyComp << flush
                  << "window_geometry 0 375 400 350 "
                  << "window_title 'XY components of Curl'" << endl;
         dz_sock << "solution\n" << mesh << dzComp << flush
                 << "window_geometry 403 375 400 350 "
                 << "window_title 'Z component of Curl'" << endl;
      }
      else
      {
         socketstream sol_sock(vishost, visport);
         socketstream dsol_sock(vishost, visport);

         RT_FECollection fec_rt(order-1, dim);

         FiniteElementSpace fes_rt(&mesh, &fec_rt);

         GridFunction dsol(&fes_rt);

         dsol.ProjectCoefficient(dsolCoef);

         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << sol
                  << "window_title 'Solution'" << flush << endl;
         dsol_sock << "solution\n" << mesh << dsol << flush
                   << "window_geometry 0 375 400 350 "
                   << "window_title 'Curl of solution'" << endl;
      }
   }

   // 16. Free the used memory.
   delete fec;

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 1)
   {
      E(0) = 1.1 * sin(kappa * x(0) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * x(0) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * x(0) + 0.9 * M_PI);
   }
   else if (dim == 2)
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
   }
   else
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      E *= cos(kappa * x(2));
   }
}

void CurlE_exact(const Vector &x, Vector &dE)
{
   if (dim == 1)
   {
      real_t c4 = cos(kappa * x(0) + 0.4 * M_PI);
      real_t c9 = cos(kappa * x(0) + 0.9 * M_PI);

      dE(0) =  0.0;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4;
      dE *= kappa;
   }
   else if (dim == 2)
   {
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      dE(0) =  1.3 * c9;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4 - 1.1 * c0;
      dE *= kappa * M_SQRT1_2;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      dE(0) =  1.2 * s4 * sk + 1.3 * M_SQRT1_2 * c9 * ck;
      dE(1) = -1.1 * s0 * sk - 1.3 * M_SQRT1_2 * c9 * ck;
      dE(2) = -M_SQRT1_2 * (1.1 * c0 - 1.2 * c4) * ck;
      dE *= kappa;
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 1)
   {
      real_t s0 = sin(kappa * x(0) + 0.0 * M_PI);
      real_t s4 = sin(kappa * x(0) + 0.4 * M_PI);
      real_t s9 = sin(kappa * x(0) + 0.9 * M_PI);

      f(0) = 2.2 * s0 + 1.2 * M_SQRT1_2 * s4;
      f(1) = 1.2 * (2.0 + kappa * kappa) * s4 +
             M_SQRT1_2 * (1.1 * s0 + 1.3 * s9);
      f(2) = 1.3 * (2.0 + kappa * kappa) * s9 + 1.2 * M_SQRT1_2 * s4;
   }
   else if (dim == 2)
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      f(0) = 0.55 * (4.0 + kappa * kappa) * s0 +
             0.6 * (M_SQRT2 - kappa * kappa) * s4;
      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 +
             0.6 * (4.0 + kappa * kappa) * s4 +
             0.65 * M_SQRT2 * s9;
      f(2) = 0.6 * M_SQRT2 * s4 + 1.3 * (2.0 + kappa * kappa) * s9;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      f(0) = 0.55 * (4.0 + 3.0 * kappa * kappa) * s0 * ck +
             0.6 * (M_SQRT2 - kappa * kappa) * s4 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 * ck +
             0.6 * (4.0 + 3.0 * kappa * kappa) * s4 * ck +
             0.65 * M_SQRT2 * s9 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(2) = 0.6 * M_SQRT2 * s4 * ck -
             M_SQRT2 * kappa * kappa * (0.55 * c0 + 0.6 * c4) * sk
             + 1.3 * (2.0 + kappa * kappa) * s9 * ck;
   }
}
