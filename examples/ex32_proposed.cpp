//                                MFEM Example 32
//
// Compile with: make ex32
//
// Sample runs:  ex32 -m ../data/hexagon.mesh -o 2
//               ex32 -m ../data/star.mesh -o 3
//               ex32 -m ../data/amr-quad.mesh
//               ex32 -m ../data/amr-quad.mesh -o 2
//
// Description:  This is a version of Example 30 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the
//               electromagnetic diffusion problem with an anisotropic
//               conductivity coefficient. The problem is solved on a sequence
//               of 2D meshes which are locally refined in a conforming
//               (triangles) or non-conforming (quadrilaterals) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements on 2D meshes.
//               Interpolation of functions from coarse to fine meshes, as well
//               as persistent GLVis visualization are also illustrated.
//
//               We recommend viewing Examples 6 and 30 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Vector bb_min;
static Vector bb_max;
void f_func(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_amr_its = 100;
   int max_dofs = 20000;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_amr_its, "-mx", "--max-amr-its",
                  "Maximum number of AMR iterations.");
   args.AddOption(&max_dofs, "-md", "--max-amr-dofs",
                  "Maximum number of degrees of freedom.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file.  We can handle triangular
   //    or quadrilateral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.GetBoundingBox(bb_min, bb_max);

   MFEM_VERIFY(dim == 2, "This exmaple requires a 2D mesh.");

   // 3. Refine the mesh to increase the resolution. Also project a NURBS mesh
   //    to a piecewise-quadratic curved mesh. Make sure that the mesh is
   //    non-conforming.
   if (mesh.NURBSext)
   {
      mesh.UniformRefinement();
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh();

   // 4. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   ND_R2D_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 5. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_func and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(3, f_func);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));

   // 6. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);
   GridFunction sol(&fespace);
   sol = 0;

   // 7. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (ess_bdr.Size() > 0) { ess_bdr = 1; }

   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2; // 1/sqrt(2) in cmath
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   ConstantCoefficient muinv(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   BilinearForm a(&fespace);
   BilinearFormIntegrator * integ = new CurlCurlIntegrator(muinv);
   a.AddDomainIntegrator(integ);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

   // 8. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream xy_sock, z_sock;
   if (visualization)
   {
      xy_sock.open(vishost, visport);
      z_sock.open(vishost, visport);
      if (!xy_sock && !z_sock)
      {
	cout << "Unable to connect to GLVis server at "
	     << vishost << ':' << visport << endl;
	cout << "GLVis visualization disabled.\n";

         visualization = false;
      }

      xy_sock.precision(8);
      z_sock.precision(8);
   }

   // 9. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //    that uses the ComputeElementFlux method of the CurlCurlIntegrator to
   //    recover a smoothed flux (curl) that is subtracted from the element
   //    flux to get an error indicator. We need to supply a space for the
   //    discontinuous flux (RT) and a space for the smoothed flux (H(curl) is
   //    used here).
   //RT_R2D_FECollection flux_fec(order-1, dim);
   //FiniteElementSpace flux_fes(&mesh, &flux_fec);
   L2_FECollection flux_fec(order, dim);
   FiniteElementSpace flux_fes(&mesh, &flux_fec, 3);
   // ND_R2D_FECollection smooth_flux_fec(order, dim);
   // FiniteElementSpace smooth_flux_fes(&mesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   H1_FECollection smooth_flux_fec(order, dim);
   FiniteElementSpace smooth_flux_fes(&mesh, &smooth_flux_fec, 3);
   ZienkiewiczZhuEstimator estimator(*integ, sol, flux_fes);

   // 10. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.6);

   // 11. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; it <= max_amr_its; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 12. Assemble the right-hand side and determine the list of true
      //     essential boundary dofs.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      b.Assemble();

      // 13. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming.  The FE space is considered
      //     'cut' along hanging edges/faces.
      a.Assemble();

      // 14. Create the linear system: eliminate boundary conditions. The
      //     system will be solved for true (unconstrained/unique) DOFs only.
      OperatorPtr A;
      Vector B, X;

      sol.ProjectBdrCoefficientTangent(zeroCoef, ess_bdr);

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);

      // 15. Solve the linear system A X = B.
#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif

      // 16. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained edges are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, sol);

      // 17. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         DenseMatrix xyMat(2,3); xyMat = 0.0;
         xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
         MatrixConstantCoefficient xyMatCoef(xyMat);
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient zVecCoef(zVec);

         VectorGridFunctionCoefficient solCoef(&sol);
         MatrixVectorProductCoefficient xyCoef(xyMatCoef, solCoef);
         InnerProductCoefficient zCoef(zVecCoef, solCoef);

         H1_FECollection fec_h1(order, dim);
         ND_FECollection fec_nd(order, dim);

         FiniteElementSpace fes_h1(&mesh, &fec_h1);
         FiniteElementSpace fes_nd(&mesh, &fec_nd);

         GridFunction xyComp(&fes_nd);
         GridFunction zComp(&fes_h1);

         xyComp.ProjectCoefficient(xyCoef);
         zComp.ProjectCoefficient(zCoef);

         xy_sock << "solution\n" << mesh << xyComp << flush;
         if (it == 0)
         {
            xy_sock << "keys vvv "
                    << "window_geometry 0 0 400 350 "
                    << "window_title 'XY components'\n";
         }

         z_sock << "solution\n" << mesh << zComp << flush;
         if (it == 0)
         {
            z_sock << "window_geometry 403 0 400 350 "
                   << "window_title 'Z component'\n";
         }
      }

      if (cdofs > max_dofs)
      {
	cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
	 cout << "Stopping criterion satisfied. Stop." << endl;
	 break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      fespace.Update();
      sol.Update();

      // 22. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }
   if (visualization)
   {
      xy_sock.close();
      z_sock.close();
   }

   return 0;
}

void f_func(const Vector &x, Vector &f)
{
   double xc = 0.5 * (bb_min[0] + bb_max[0]);
   double yc = 0.5 * (bb_min[1] + bb_max[1]);
   double dx = bb_max[0] - bb_min[0];
   double dy = bb_max[1] - bb_min[1];

   f = 0.0;
   if (fabs(x[0] - xc) < 0.2 * dx && fabs(x[1] - yc) < 0.2 * dy)
   {
      double a = pow(cos(2.5 * M_PI * (x[0] - xc) / dx) *
                     cos(2.5 * M_PI * (x[1] - yc) / dy), 2);
      f(0) = a * sin(2.5 * M_PI * (x[1] - yc) / dy);
      f(1) = a * sin(5.0 * M_PI * (x[0] - xc) / dx);
      f(2) = a * cos(5.0 * M_PI * (x[0] - xc) / dx);
   }
}
