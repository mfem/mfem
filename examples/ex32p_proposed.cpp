//                       MFEM Example 32 - Parallel Version
//
// Compile with: make ex32p
//
// Sample runs:  mpirun -np 4 ex32p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex32p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex32p -m ../data/square-disc.mesh -o 3
//               mpirun -np 4 ex32p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex32p -m ../data/star.mesh -o 3
//               mpirun -np 4 ex32p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex32p -m ../data/amr-quad.mesh -o 2
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

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void CurlE_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_amr_its = 100;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_amr_its, "-mx", "--max-amr-its",
                  "Maximum number of AMR iterations.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   kappa = freq * M_PI;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   MFEM_VERIFY(dim == 2, "This exmaple requires a 2D mesh.");

   // 4. Refine the serial mesh on all processors to increase the resolution.
   //    Also project a NURBS mesh to a piecewise-quadratic curved mesh. Make
   //    sure that the mesh is non-conforming.
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      mesh->SetCurvature(2);
   }
   mesh->EnsureNCMesh();

   // 5. Define a parallel mesh by partitioning the serial mesh.
   //    Once the parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 6. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   ND_R2D_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));

   // 8. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   VectorFunctionCoefficient E(3, E_exact);
   VectorFunctionCoefficient CurlE(3, CurlE_exact);
   ParGridFunction sol(&fespace);
   sol = 0;

   // 9. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl + sigma I, by adding the curl-curl and the
   //    mass domain integrators.
   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2;
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   ConstantCoefficient muinv(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   ParBilinearForm a(&fespace);
   BilinearFormIntegrator * integ = new CurlCurlIntegrator(muinv);
   a.AddDomainIntegrator(integ);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

   // 10. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream xy_sock, z_sock;
   if (visualization)
   {
      xy_sock.open(vishost, visport);
      z_sock.open(vishost, visport);
      if (!xy_sock && !z_sock)
      {
         if (mpi.Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         visualization = false;
      }

      xy_sock.precision(8);
      z_sock.precision(8);
   }

   // 11. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (RT) and a space for the smoothed flux (H(curl) is
   //     used here).
   RT_R2D_FECollection flux_fec(order-1, dim);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec);
   ND_R2D_FECollection smooth_flux_fec(order, dim);
   ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, 3);
   L2ZienkiewiczZhuEstimator estimator(*integ, sol, flux_fes, smooth_flux_fes);

   // 12. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.6);

   // 13. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 20000;
   for (int it = 0; it <= max_amr_its; it++)
   {
      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      if (mpi.Root())
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 14. Assemble the right-hand side and determine the list of true
      //     (i.e. parallel conforming) essential boundary dofs.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      b.Assemble();

      // 15. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming and parallel.  The FE space is
      //     considered 'cut' along hanging edges/faces, and also across
      //     processor boundaries.
      a.Assemble();

      // 16. Create the parallel linear system: eliminate boundary conditions.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      OperatorPtr A;
      Vector B, X;

      sol.ProjectBdrCoefficientTangent(E, ess_bdr);

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use a diagonal preconditioner.
      HypreAMS ams(*A.As<HypreParMatrix>(), &fespace);
      ams.SetPrintLevel(0);

      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(3);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);

      // 18. Switch back to the host and extract the parallel grid function
      //     corresponding to the finite element approximation X. This is the
      //     local solution on each processor.
      a.RecoverFEMSolution(X, b, sol);

      // 19. Compute and print the H(Curl) norm of the error.
      {
         double err = sol.ComputeHCurlError(&E, &CurlE);
         if (mpi.Root())
         {
            cout << "\n|| E_h - E ||_{H(Curl)} = " << err << '\n' << endl;
         }
      }

      // 20. Send the solution by socket to a GLVis server.
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

         ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);
         ParFiniteElementSpace fes_nd(&pmesh, &fec_nd);

         ParGridFunction xyComp(&fes_nd);
         ParGridFunction zComp(&fes_h1);

         xyComp.ProjectCoefficient(xyCoef);
         zComp.ProjectCoefficient(zCoef);

         xy_sock << "parallel " << num_procs << " " << myid << "\n";
         xy_sock << "solution\n" << pmesh << xyComp << flush;
         if (it == 0)
         {
            xy_sock << "keys vvv "
                    << "window_geometry 0 0 400 350 "
                    << "window_title 'XY components'\n";
         }

         z_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << zComp << flush;
         if (it == 0)
         {
            z_sock << "window_geometry 403 0 400 350 "
                   << "window_title 'Z component'\n";
         }
      }

      if (global_dofs > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 21. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (mpi.Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 22. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      fespace.Update();
      sol.Update();

      // 23. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         sol.Update();
      }

      // 24. Inform also the bilinear and linear forms that the space has
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
      double c4 = cos(kappa * x(0) + 0.4 * M_PI);
      double c9 = cos(kappa * x(0) + 0.9 * M_PI);

      dE(0) =  0.0;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4;
      dE *= kappa;
   }
   else if (dim == 2)
   {
      double c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      dE(0) =  1.3 * c9;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4 - 1.1 * c0;
      dE *= kappa * M_SQRT1_2;
   }
   else
   {
      double s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      double sk = sin(kappa * x(2));
      double ck = cos(kappa * x(2));

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
      double s0 = sin(kappa * x(0) + 0.0 * M_PI);
      double s4 = sin(kappa * x(0) + 0.4 * M_PI);
      double s9 = sin(kappa * x(0) + 0.9 * M_PI);

      f(0) = 2.2 * s0 + 1.2 * M_SQRT1_2 * s4;
      f(1) = 1.2 * (2.0 + kappa * kappa) * s4 +
             M_SQRT1_2 * (1.1 * s0 + 1.3 * s9);
      f(2) = 1.3 * (2.0 + kappa * kappa) * s9 + 1.2 * M_SQRT1_2 * s4;
   }
   else if (dim == 2)
   {
      double s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      f(0) = 0.55 * (4.0 + kappa * kappa) * s0 +
             0.6 * (M_SQRT2 - kappa * kappa) * s4;
      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 +
             0.6 * (4.0 + kappa * kappa) * s4 +
             0.65 * M_SQRT2 * s9;
      f(2) = 0.6 * M_SQRT2 * s4 + 1.3 * (2.0 + kappa * kappa) * s9;
   }
   else
   {
      double s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      double c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      double sk = sin(kappa * x(2));
      double ck = cos(kappa * x(2));

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
