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
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool nc_simplices = true;
   int reorder_mesh = 0;
   int max_dofs = 100000;
   bool smooth_rt = true;
   bool restart = false;
   bool visualization = true;
   bool rebalance = true;
   bool usePRefinement = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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
   args.AddOption(&smooth_rt, "-rt", "--smooth-rt", "-h1", "--smooth-h1",
                  "Represent the smooth flux in RT or vector H1 space.");
   args.AddOption(&usePRefinement, "-pref", "--p-refine", "-no-pref",
                  "--no-p-refine", "Alternate between h- and p-refinement.");
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

   if (usePRefinement && rebalance)
   {
      rebalance = false;
      if (myid == 0)
      {
         cout << "Load balancing is not performed with p-refinements.\n";
      }
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
   int sdim = pmesh->SpaceDimension();

   MFEM_VERIFY(pmesh->bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 10. Define a finite element space on the mesh. The polynomial order is
   //     one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fec);

   // 11. As in Example 1p, we set up bilinear and linear forms corresponding to
   //     the Poisson problem -\Delta u = 1. We don't assemble the discrete
   //     problem yet, this will be done in the main loop.
   ParBilinearForm a(&fespace);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.SetDiagonalPolicy(Operator::DIAG_ONE);
   }
   ParLinearForm b(&fespace);

   ConstantCoefficient one(1.0);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 12. The solution vector x and the associated finite element grid function
   //     will be maintained over the AMR iterations. We initialize it to zero.
   ParGridFunction x(&fespace);
   x = 0;

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
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
   FiniteElementCollection *smooth_flux_fec = NULL;
   ParFiniteElementSpace *smooth_flux_fes = NULL;
   if (smooth_rt && dim > 1)
   {
      // Use an H(div) space for the smoothed flux (this is the default).
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec, 1);
   }
   else
   {
      // Another possible option for the smoothed flux space: H1^dim space
      smooth_flux_fec = new H1_FECollection(order, dim);
      smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec, dim);
   }
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, *smooth_flux_fes);

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
      HYPRE_BigInt global_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 17. Assemble the right-hand side and determine the list of true
      //     (i.e. parallel conforming) essential boundary dofs.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      b.Assemble();

      // 18. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming and parallel.  The FE space is
      //     considered 'cut' along hanging edges/faces, and also across
      //     processor boundaries.
      a.Assemble();

      // 19. Create the parallel linear system: eliminate boundary conditions.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 20. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use a diagonal preconditioner.
      Solver *M = NULL;
      if (pa)
      {
         M = new OperatorJacobiSmoother(a, ess_tdof_list);
      }
      else
      {
         HypreBoomerAMG *amg = new HypreBoomerAMG;
         amg->SetPrintLevel(0);
         M = amg;
      }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(3); // print the first and the last iterations only
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      // 21. Switch back to the host and extract the parallel grid function
      //     corresponding to the finite element approximation X. This is the
      //     local solution on each processor.
      a.RecoverFEMSolution(X, b, x);

      // 22. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         if (usePRefinement)
         {
            std::unique_ptr<GridFunction> vis_x = x.ProlongateToMaxOrder();
            sout << "solution\n" << *pmesh << *vis_x << flush;
         }
         else
         {
            sout << "solution\n" << *pmesh << x << flush;
         }
      }

      if (global_dofs >= max_dofs)
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

      // Simply alternate between h- and p-refinement.
      const bool pRefine = usePRefinement && ((it % 2) == 1);
      bool stop = false;
      Array<pRefinement> prefinements;
      if (pRefine)
      {
         Array<Refinement> refinements;
         refiner.MarkWithoutRefining(*pmesh, refinements);
         stop = pmesh->ReduceInt(refinements.Size()) == 0LL;

         prefinements.SetSize(refinements.Size());
         for (int i=0; i<refinements.Size(); ++i)
         {
            prefinements[i].index = refinements[i].index;
            prefinements[i].delta = 1;  // Increase the element order by 1
         }
      }
      else
      {
         refiner.Apply(*pmesh);
         stop = refiner.Stop();
      }

      if (stop)
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
      if (pRefine)
      {
         fespace.PRefineAndUpdate(prefinements);
      }
      else
      {
         fespace.Update();
      }

      x.Update();

      // 25. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh->Nonconforming() && rebalance)
      {
         pmesh->Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         x.Update();
      }

      // 26. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();

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

   // Save result
   if (usePRefinement)
   {
      L2_FECollection fecL2(0, dim);
      ParFiniteElementSpace l2fespace(pmesh, &fecL2);
      ParGridFunction xo(&l2fespace); // Element order field
      xo = 0.0;

      for (int e=0; e<pmesh->GetNE(); ++e)
      {
         const int p_elem = fespace.GetElementOrder(e);
         Array<int> dofs;
         l2fespace.GetElementDofs(e, dofs);
         xo[dofs[0]] = p_elem;
      }

      ostringstream mesh_name, sol_name, order_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      order_name << "order." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->ParPrint(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);

      std::unique_ptr<ParGridFunction> vis_x = x.ProlongateToMaxOrder();
      vis_x->Save(sol_ofs);

      ofstream order_ofs(order_name.str().c_str());
      order_ofs.precision(8);
      xo.Save(order_ofs);
   }

   delete smooth_flux_fes;
   delete smooth_flux_fec;
   delete pmesh;

   return 0;
}
