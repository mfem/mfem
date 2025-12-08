//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs:  mpirun -np 4 ex6p -m ../data/star-hilbert.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/star-hilbert.mesh -pref
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -rm 1 -o 1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -rm 1 -o 2 -h1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 2 -cs
//               mpirun -np 4 ex6p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/escher.mesh -rm 2 -o 2
//               mpirun -np 4 ex6p -m ../data/escher.mesh -o 2 -cs
//               mpirun -np 4 ex6p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-surf.mesh -rm 2 -o 2
//               mpirun -np 4 ex6p -m ../data/inline-segment.mesh -o 1 -md 200
//               mpirun -np 4 ex6p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex6p --restart
//
// Device sample runs:
//               mpirun -np 4 ex6p -pa -d cuda
//               mpirun -np 4 ex6p -pa -d occa-cuda
//               mpirun -np 4 ex6p -pa -d raja-omp
//               mpirun -np 4 ex6p -pa -d ceed-cpu
//             * mpirun -np 4 ex6p -pa -d ceed-cuda
//               mpirun -np 4 ex6p -pa -d ceed-cuda:/gpu/cuda/shared
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Poisson
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, restarting from a checkpoint, as
//               well as persistent GLVis visualization are also illustrated.
//
//               There is also the option to use hp-refinement. Real
//               applications should use some problem-dependent criteria for
//               selecting between h- and p-refinement, but in this example, we
//               simply alternate between refinement types to demonstrate the
//               capabilities.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

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
   bool dg = false;
   bool brt = false;
   real_t td = 0.5;
   bool trace_h1 = false;
   const char *device_config = "cpu";
   bool nc_simplices = true;
   int reorder_mesh = 0;
   int max_dofs = 100000;
   bool restart = false;
   bool visualization = true;
   bool rebalance = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&reorder_mesh, "-rm", "--reorder-mesh",
                  "Reorder elements of the coarse mesh to improve "
                  "dynamic partitioning: 0=none, 1=hilbert, 2=gecko.");
   args.AddOption(&nc_simplices, "-ns", "--nonconforming-simplices",
                  "-cs", "--conforming-simplices",
                  "For simplicial meshes, enable/disable nonconforming"
                  " refinement");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
   args.AddOption(&rebalance, "-reb", "--rebalance", "-no-reb",
                  "--no-rebalance", "Load balance the nonconforming mesh.");
   args.AddOption(&restart, "-res", "--restart", "-no-res", "--no-restart",
                  "Restart computation from the last checkpoint.");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   ParMesh *pmesh;
   if (!restart)
   {
      // 4. Read the (serial) mesh from the given mesh file on all processors.
      //    We can handle triangular, quadrilateral, tetrahedral, hexahedral,
      //    surface and volume meshes with the same code.
      Mesh mesh(mesh_file, 1, 1);

      // 5. A NURBS mesh cannot be refined locally so we refine it uniformly
      //    and project it to a standard curvilinear mesh of order 2.
      if (mesh.NURBSext)
      {
         mesh.UniformRefinement();
         mesh.SetCurvature(2);
      }

      // 6. MFEM supports dynamic partitioning (load balancing) of parallel non-
      //    conforming meshes based on space-filling curve (SFC) partitioning.
      //    SFC partitioning is extremely fast and scales to hundreds of
      //    thousands of processors, but requires the coarse mesh to be ordered,
      //    ideally as a sequence of face-neighbors. The mesh may already be
      //    ordered (like star-hilbert.mesh) or we can order it here. Ordering
      //    type 1 is a fast spatial sort of the mesh, type 2 is a high quality
      //    optimization algorithm suitable for ordering general unstructured
      //    meshes.
      if (reorder_mesh)
      {
         Array<int> ordering;
         switch (reorder_mesh)
         {
            case 1: mesh.GetHilbertElementOrdering(ordering); break;
            case 2: mesh.GetGeckoElementOrdering(ordering); break;
            default: MFEM_ABORT("Unknown mesh reodering type " << reorder_mesh);
         }
         mesh.ReorderElements(ordering);
      }

      // 7. Make sure the mesh is in the non-conforming mode to enable local
      //    refinement of quadrilaterals/hexahedra, and the above partitioning
      //    algorithm. Simplices can be refined either in conforming or in non-
      //    conforming mode. The conforming mode however does not support
      //    dynamic partitioning.
      mesh.EnsureNCMesh(nc_simplices);

      // 8. Define a parallel mesh by partitioning the serial mesh.
      //    Once the parallel mesh is defined, the serial mesh can be deleted.
      pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   }
   else
   {
      // 9. We can also restart the computation by loading the mesh from a
      //    previously saved check-point.
      string fname(MakeParFilename("ex6p-checkpoint.", myid));
      ifstream ifs(fname);
      MFEM_VERIFY(ifs.good(), "Checkpoint file " << fname << " not found.");
      pmesh = new ParMesh(MPI_COMM_WORLD, ifs);
   }

   int dim = pmesh->Dimension();

   // 10. Define a finite element space on the mesh. The polynomial order is
   //     one (linear) by default, but this can be changed on the command line.
   FiniteElementCollection *R_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      R_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      R_coll = new BrokenRT_FECollection(order, dim);
   }
   else
   {
      R_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim);

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, R_coll,
                                                              (dg)?(dim):(1));
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, W_coll);

   // 11. As in Example 1p, we set up bilinear and linear forms corresponding to
   //     the Poisson problem -\Delta u = 1. We don't assemble the discrete
   //     problem yet, this will be done in the main loop.
   ParDarcyForm darcy(R_space, W_space);

   MemoryType mt = device.GetMemoryType();
   BlockVector b(darcy.GetOffsets(), mt);

   b.GetBlock(0) = 0.0;

   ConstantCoefficient one(1.0), negone(-1.0);

   ParLinearForm f;
   f.Update(W_space, b.GetBlock(1), 0);
   f.AddDomainIntegrator(new DomainLFIntegrator(negone));

   ParBilinearForm *Mq = darcy.GetParFluxMassForm();
   ParMixedBilinearForm *Bq = darcy.GetParFluxDivForm();
   ParBilinearForm *Mu = (dg)?(darcy.GetParPotentialMassForm()):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator());
      Bq->AddDomainIntegrator(new VectorDivergenceIntegrator());
      Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      Mu->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator());
      Bq->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      if (brt)
      {
         Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                          new DGNormalTraceIntegrator(-1.)));
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   ParFiniteElementSpace *trace_space = NULL;

   if (trace_h1)
   {
      trace_coll = new H1_Trace_FECollection(order+1, dim);
   }
   else
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
   }
   trace_space = new ParFiniteElementSpace(pmesh, trace_coll);
   darcy.EnableHybridization(trace_space,
                             new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs_list);

   // 12. The solution vector x and the associated finite element grid function
   //     will be maintained over the AMR iterations. We initialize it to zero.
   BlockVector x(darcy.GetOffsets(), mt);
   x = 0.0;

   ParGridFunction u_h, uhat_h;
   u_h.MakeRef(W_space, x.GetBlock(1), 0);

   // 13. Connect to GLVis.
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

   // 14. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux.
   HDGDiffusionIntegrator estimator_integ(one, td);
   HDGErrorEstimator estimator(estimator_integ, uhat_h, u_h);

   // 15. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 16. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
   {
      const HYPRE_BigInt q_dofs = R_space->GlobalTrueVSize();
      const HYPRE_BigInt u_dofs = W_space->GlobalTrueVSize();
      const HYPRE_BigInt uhat_dofs = trace_space->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of flux unknowns: " << q_dofs << endl;
         cout << "Number of potential unknowns: " << u_dofs << endl;
         cout << "Number of trace unknowns: " << uhat_dofs << endl;
      }

      // 17. Assemble the right-hand side and determine the list of true
      //     (i.e. parallel conforming) essential boundary dofs.
      f.Assemble();

      // 18. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming and parallel.  The FE space is
      //     considered 'cut' along hanging edges/faces, and also across
      //     processor boundaries.
      darcy.Assemble();

      // 19. Create the parallel linear system: eliminate boundary conditions.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      OperatorPtr A;
      Vector B, X;

      darcy.FormLinearSystem(ess_flux_tdofs_list, x, b, A, X, B);

      // 20. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use a diagonal preconditioner.
      HypreBoomerAMG M;
      M.SetPrintLevel(0);

      GMRESSolver solver(MPI_COMM_WORLD);
      solver.SetRelTol(1e-6);
      solver.SetMaxIter(2000);
      solver.SetPrintLevel(3); // print the first and the last iterations only
      solver.SetPreconditioner(M);
      solver.SetOperator(*A);
      solver.Mult(B, X);

      // 21. Switch back to the host and extract the parallel grid function
      //     corresponding to the finite element approximation X. This is the
      //     local solution on each processor.
      darcy.RecoverFEMSolution(X, b, x);
      uhat_h.MakeTRef(trace_space, X, 0);
      uhat_h.SetFromTrueVector();

      // 22. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << *pmesh << u_h << flush;
      }

      if (u_dofs >= max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 23. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(*pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 24. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      R_space->Update();
      W_space->Update();
      trace_space->Update();

      // 25. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh->Nonconforming() && rebalance)
      {
         pmesh->Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         R_space->Update();
         W_space->Update();
         trace_space->Update();
      }

      // 26. Inform also the bilinear and linear forms that the space has
      //     changed.
      darcy.Update();
      x.Update(darcy.GetOffsets(), mt);
      b.Update(darcy.GetOffsets(), mt);

      x = 0.;
      u_h.MakeRef(W_space, x.GetBlock(1), 0);

      b.GetBlock(0) = 0.;
      f.Update(W_space, b.GetBlock(1), 0);

      darcy.EnableHybridization(trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);

      // 27. Save the current state of the mesh every 5 iterations. The
      //     computation can be restarted from this point. Note that unlike in
      //     visualization, we need to use the 'ParPrint' method to save all
      //     internal parallel data structures.
      if ((it + 1) % 5 == 0)
      {
         ofstream ofs(MakeParFilename("ex6p-checkpoint.", myid));
         ofs.precision(8);
         pmesh->ParPrint(ofs);

         if (myid == 0)
         {
            cout << "\nCheckpoint saved." << endl;
         }
      }
   }

   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;
   delete pmesh;

   return 0;
}
