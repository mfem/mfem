//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs:  mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/star.mesh -o 3
//               mpirun -np 4 ex6p -m ../data/escher.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/amr-quad.mesh
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilateral, hexahedrons) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

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
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 7. As in Example 1p, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   ParBilinearForm a(&fespace);
   ParLinearForm b(&fespace);

   ConstantCoefficient one(1.0);

   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 8. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   ParGridFunction x(&fespace);
   x = 0;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (visualization)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         visualization = false;
      }

      sout.precision(8);
   }

   // 10. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, estimate the error on all
   //     elements, refine the worst elements and update all objects to work
   //     with the new mesh.
   const int max_dofs = 100000;
   for (int it = 0; ; it++)
   {
      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nIteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 11. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point that the mesh is nonconforming
      //     and parallel. The FE space is considered 'cut' along hanging
      //     edges/faces, and also across processor boundaries.
      a.Assemble();
      b.Assemble();

      // 12. Set the initial estimate of the solution and the Dirichlet DOFs,
      //     here we just use zero everywhere.
      x = 0.0;

      // 13. Create the parallel linear system: eliminate boundary conditions,
      //     constrain hanging nodes and nodes across processor boundaries.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      HypreParMatrix A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 14. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg(A);
      amg.SetPrintLevel(0);
      HyprePCG pcg(A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);

      // 15. Extract the parallel grid function corresponding to the finite element
      //     approximation X. This is the local solution on each processor.
      a.RecoverFEMSolution(X, b, x);

      // 16. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << pmesh << x << flush;
      }

      if (global_dofs > max_dofs)
      {
         break;
      }

      // 17. Estimate element errors using the Zienkiewicz-Zhu error estimator.
      //     The bilinear form integrator must have the 'ComputeElementFlux'
      //     method defined.
      Vector errors(pmesh.GetNE());
      {
         // Space for the discontinuous (original) flux
         DiffusionIntegrator flux_integrator(one);
         L2_FECollection flux_fec(order, dim);
         ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);

         // Space for the smoothed (conforming) flux
         double norm_p = 1;
         RT_FECollection smooth_flux_fec(order-1, dim);
         ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);

         // Another possible set of options for the smoothed flux space:
         // norm_p = 1;
         // H1_FECollection smooth_flux_fec(order, dim);
         // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);

         L2ZZErrorEstimator(flux_integrator, x,
                            smooth_flux_fes, flux_fes, errors, norm_p);
      }
      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // 18. Make a list of elements whose error is larger than a fraction
      //     of the maximum element error. These elements will be refined.
      Array<int> ref_list;
      const double frac = 0.7;
      double threshold = frac * global_max_err;
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold) { ref_list.Append(i); }
      }

      // 19. Refine the selected elements. Since we are going to transfer the
      //     grid function x from the coarse mesh to the new fine mesh in the
      //     next step, we need to request the "two-level state" of the mesh.
      pmesh.GeneralRefinement(ref_list);

      // 20. Inform the space, grid function and also the bilinear and linear
      //     forms that the space has changed.
      fespace.Update();
      x.Update();
      a.Update();
      b.Update();
   }

   MPI_Finalize();
   return 0;
}
