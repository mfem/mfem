//                       MFEM Example 6 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex6p
//
// Sample runs:
//    mpirun -np 4 ex6p -m ../../data/amr-quad.mesh
//    mpirun -np 4 ex6p -m ../../data/amr-quad.mesh -nonoverlapping
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               PETSc assembly timings can be benchmarked if requested by
//               command line.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool visualization = true;
   int max_dofs = 100000;
   bool use_petsc = true;
   const char *petscrc_file = "";
   bool use_nonoverlapping = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&max_dofs, "-md", "--max_dofs",
                  "Maximum number of dofs.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&use_nonoverlapping, "-nonoverlapping", "--nonoverlapping",
                  "-no-nonoverlapping", "--no-nonoverlapping",
                  "Use or not the block diagonal PETSc's matrix format "
                  "for non-overlapping domain decomposition.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   // 2b. We initialize PETSc
   if (use_petsc) { MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
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

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
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

   // 10. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);
   RT_FECollection smooth_flux_fec(order-1, dim);
   ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, smooth_flux_fes);

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
   {
      HYPRE_BigInt global_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 13. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point that the mesh is nonconforming
      //     and parallel. The FE space is considered 'cut' along hanging
      //     edges/faces, and also across processor boundaries.
      a.Assemble();
      b.Assemble();

      // 14. Create the parallel linear system: eliminate boundary conditions,
      //     constrain hanging nodes and nodes across processor boundaries.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      double time;
      const int copy_interior = 1;

      if (use_petsc)
      {
         a.SetOperatorType(use_nonoverlapping ?
                           Operator::PETSC_MATIS : Operator::PETSC_MATAIJ);
         PetscParMatrix pA;
         Vector pX,pB;
         MPI_Barrier(MPI_COMM_WORLD);
         time = -MPI_Wtime();
         a.FormLinearSystem(ess_tdof_list, x, b, pA, pX, pB, copy_interior);
         MPI_Barrier(MPI_COMM_WORLD);
         time += MPI_Wtime();
         if (myid == 0) { cout << "PETSc assembly timing : " << time << endl; }
      }

      a.Assemble();
      b.Assemble();
      a.SetOperatorType(Operator::Hypre_ParCSR);
      HypreParMatrix A;
      Vector B, X;
      MPI_Barrier(MPI_COMM_WORLD);
      time = -MPI_Wtime();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      MPI_Barrier(MPI_COMM_WORLD);
      time += MPI_Wtime();
      if (myid == 0) { cout << "HYPRE assembly timing : " << time << endl; }

      // 15. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);
      CGSolver pcg(A.GetComm());
      pcg.SetPreconditioner(amg);
      pcg.SetOperator(A);
      pcg.SetRelTol(1e-6);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(3); // print the first and the last iterations only
      pcg.Mult(B, X);

      // 16. Extract the parallel grid function corresponding to the finite element
      //     approximation X. This is the local solution on each processor.
      a.RecoverFEMSolution(X, b, x);

      // 17. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << pmesh << x << flush;
      }

      if (global_dofs > max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         // we need to call Update here to delete any internal PETSc object that
         // have been created by the ParBilinearForm; otherwise, these objects
         // will be destroyed at the end of the main scope, when PETSc has been
         // already finalized.
         a.Update();
         b.Update();
         break;
      }

      // 18. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         a.Update();
         b.Update();
         break;
      }

      // 19. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      fespace.Update();
      x.Update();

      // 20. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         x.Update();
      }

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   // We finalize PETSc
   if (use_petsc) { MFEMFinalizePetsc(); }

   return 0;
}
