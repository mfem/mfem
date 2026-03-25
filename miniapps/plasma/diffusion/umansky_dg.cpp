// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "umansky.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class CustomSolverMonitor : public IterativeSolverMonitor
{
private:
   const ParMesh &pmesh;
   ParGridFunction &pgf;
public:
   CustomSolverMonitor(const ParMesh &pmesh_,
                       ParGridFunction &pgf_) :
      pmesh(pmesh_),
      pgf(pgf_) {}

   void MonitorSolution(int i, real_t norm, const Vector &x, bool final) override
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      int  num_procs, myid;

      MPI_Comm_size(pmesh.GetComm(), &num_procs);
      MPI_Comm_rank(pmesh.GetComm(), &myid);

      pgf.SetFromTrueDofs(x);

      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << pgf
               << "window_title 'Iteration no " << i << "'"
               << "keys rRjlc\n" << flush;
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   int order = 1;
   int nx = 8, ny = 8;
   int el_type_arg = 1;
   real_t w = 1.0, h = 1.0;
   real_t Ak = 10.0;
   int max_iter = 100;
   int max_dofs = 1000000;
   real_t min_err = 1e-6;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   const char *device_config = "cpu";
   bool nc = true;
   real_t sigma = -1.0;
   real_t kappa = -1.0;
   real_t eta = 0.0;
   bool pa = false;
   bool visualization = true;
   bool visit = false;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&el_type_arg, "-e", "--element-type",
                  "Element type: 0 - Triangle, 1 - Quadrilateral.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&max_iter, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&max_dofs, "-maxdofs", "--max-amr-dofs",
                  "Max number of degrees of freedom in the main AMR loop.");
   args.AddOption(&min_err, "-minerr", "--min-amr-err",
                  "Min error target in the main AMR loop.");
   args.AddOption(&nx, "-nx", "--num-elem-x",
                  "Number of elements in x direction.");
   args.AddOption(&ny, "-ny", "--num-elem-y",
                  "Number of elements in y direction.");
   args.AddOption(&w, "-w", "--width",
                  "Width of domain.");
   args.AddOption(&h, "-h", "--height",
                  "Height of domain.");
   args.AddOption(&Ak, "-Ak", "--diff-ratio",
                  "Diffusion Coefficient Ratio.");
   args.AddOption(&nc, "-nc", "--non-conforming-amr", "-c",
                  "--conforming-amr",
                  "Enable or disable conforming adaptive mesh refinement.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-eta", "--eta", "BR2 penalty parameter.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }



   // The output mesh could be triangles or quadrilaterals
   int el_type = (el_type_arg == 0) ?
                 Element::TRIANGLE : Element::QUADRILATERAL;

   // Create a Cartesian mesh consisting of triangular or quadrilateral
   // elements.
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::Type(el_type), 1, w, h);
   int dim = mesh.Dimension();

   if (nc)
   {
      mesh.EnsureNCMesh();
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. NURBS meshes are
   // refined at least twice, as they are typically coarse.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh.Finalize(true);

   // Define a parallel finite element space on the parallel mesh. Here we
   // use discontinuous finite elements of the specified order >= 0.
   const auto bt = pa ? BasisType::GaussLobatto : BasisType::GaussLegendre;
   DG_FECollection fec(order, dim, bt);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // Determine the list of true (i.e. parallel conforming) essential
   // boundary dofs. In this example, the boundary conditions are defined
   // by marking all the external boundary attributes from the mesh as
   // essential (Dirichlet) and converting them to a list of true dofs.
   /*
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
   }
   */
   // Set up the parallel linear form b(.) which corresponds to the
   // right-hand side of the FEM linear system, which in this case is
   // (1,phi_i) where phi_i are the basis functions in fespace.
   AnisotropicDiffusionCoefficient anisoDiffCoef(w, h, Ak);
   UnitStepCoefficient unitStepCoef(w, h);
   ParLinearForm b(&fespace);
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(unitStepCoef, anisoDiffCoef, sigma, kappa));

   // Define the solution vector x as a parallel finite element grid
   // function corresponding to fespace. Initialize x with initial guess of
   // zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // Set up the parallel bilinear form a(.,.) on the finite element space
   // corresponding to the Laplacian operator -Delta, by adding the
   // Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(anisoDiffCoef));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(anisoDiffCoef,
                                                         sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(anisoDiffCoef,
                                                    sigma, kappa));
   if (eta > 0)
   {
      MFEM_VERIFY(!pa, "BR2 not yet compatible with partial assembly.");
      a.AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a.AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
   }
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Umansky-DG-AMR-Parallel", &pmesh);
   visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);

   if (visit)
   {
      visit_dc.RegisterField("solution", &x);
   }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh.  We
   // refine until the maximum number of dofs in the nodal finite element space
   // reaches 10 million.
   for (int it = 1; it <= max_iter; it++)
   {
      if (Mpi::Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Assemble the parallel bilinear form and the corresponding linear
      // system, applying any necessary transformations such as: parallel
      // assembly, eliminating boundary conditions, applying conforming
      // constraints for non-conforming AMR, static condensation, etc.

      x = 0.0;
      b = 0.0;
      b.Assemble();
      a.Assemble();
      a.Finalize();

      OperatorHandle A;
      // Vector B, X;
      // Array<int> ess_tdof_list;
      // cout << "calling form lin sys" << endl;
      // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      std::unique_ptr<HypreBoomerAMG> amg;
      if (pa)
      {
         A.Reset(&a, false);
      }
      else
      {
         A.SetType(Operator::Hypre_ParCSR);
         a.ParallelAssemble(A);
         amg.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));
      }

      // 11. Depending on the symmetry of A, define and apply a parallel PCG or
      //     GMRES solver for AX=B using the BoomerAMG preconditioner from hypre.
      if (sigma == -1.0)
      {
         cout << "using cg" << endl;
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(500);
         cg.SetPrintLevel(1);
         cg.SetOperator(*A);
         if (amg) { cg.SetPreconditioner(*amg); }
         cg.Mult(b, x);
      }
      else
      {
         cout << "using gmres" << endl;
         CustomSolverMonitor monitor(pmesh, x);
         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetAbsTol(0.0);
         gmres.SetRelTol(1e-12);
         gmres.SetMaxIter(500);
         gmres.SetKDim(10);
         gmres.SetPrintLevel(1);
         gmres.SetOperator(*A);
         if (amg) { gmres.SetPreconditioner(*amg); }
         gmres.SetMonitor(monitor);
         gmres.Mult(b, x);
      }

      // Recover the parallel grid function corresponding to X. This is the
      // local finite element solution on each processor.
      // a.RecoverFEMSolution(X, b, x);

      int prob_size = fespace.GetTrueVSize();

      if (visit)
      {
         visit_dc.SetCycle(it);
         visit_dc.SetTime(prob_size);
         visit_dc.Save();
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << pmesh << x
                  << " window_title 'Number of DoFs: " << prob_size << "'";
         if (it == 1)
         {
            sol_sock << " keys 'mmjR'\n";
         }
         sol_sock << flush;
      }

      if (Mpi::Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if (it == max_iter)
      {
         if (Mpi::Root())
         {
            cout << "Reached maximum number of iterations, exiting..." << endl;
         }
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (Mpi::Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      {
         DiffusionIntegrator flux_integrator(anisoDiffCoef);
         L2_FECollection flux_fec(order, dim);
         ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, 2);

         // Space for the smoothed (conforming) flux
         int norm_p = 1;
         RT_FECollection smooth_flux_fec(order-1, dim);
         ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);

         L2ZZErrorEstimator(flux_integrator, x,
                            smooth_flux_fes, flux_fes, errors, norm_p);

      }

      real_t local_max_err = errors.Max();
      real_t global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

      if (global_max_err < min_err)
      {
         if (Mpi::Root())
         {
            cout << "Reached minimum error target, exiting..." << endl;
         }
         break;
      }

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const real_t frac = 0.7;
      real_t threshold = frac * global_max_err;
      if (Mpi::Root()) { cout << "Refining ..." << endl; }
      if (pmesh.RefineByError(errors, threshold))
      {
         // Update the solver to reflect the new state of the mesh.
         fespace.Update();
         a.Update();
         b.Update();
         x.Update();

         if (pmesh.Nonconforming() && Mpi::WorldSize() > 1)
         {
            if (Mpi::Root()) { cout << "Rebalancing ..." << endl; }
            pmesh.Rebalance();

            // Update again after rebalancing
            fespace.Update();
            a.Update();
            b.Update();
            x.Update();
         }
      }
      else
      {
         break;
      }
   }

   return 0;
}
