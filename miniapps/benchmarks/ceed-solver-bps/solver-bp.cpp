// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// --------------------------------------------------------------
//    MFEM Implementation of the CEED Solver Bake-off Problems
// --------------------------------------------------------------
//
// Run a suite of benchmarks and view the results:
//
//   1. Edit 'run.sh' to adjust machine and size parameters.
//   2. Run 'run.sh' redirecting output to a file, e.g.:
//        bash run.sh > run-001.out
//   3. Extract the CSV output:
//        sed -n -e 's/^= CSV:\(.*\)$/\1/p' run-001.out > run-001.csv
//   3. Edit the script 'plot_csv.py' set the name of your CSV file and,
//      optionally, customize the plot it generates.
//   4. Process the CSV file:
//        python3 plot_csv.py
//
// Sample runs:
//
//   solver-bp -nx 6
//   solver-bp -nx 6 -mg "1 2 3"
//   solver-bp -nx 6 -mg "1 r r 2 3"
//   solver-bp -nx 6 -rp 2 -mg 3 -cs 1
//   solver-bp -nx 6 -rp 2 -mg 3 -cs 2
//
// Device sample runs:
//
//   solver-bp -d cuda -nx 6 -mg "1 r r 2 3" -cs 0
//   solver-bp -d cuda -nx 6 -rp 2 -mg 3 -cs 3
//   solver-bp -d cuda -nx 6 -rp 2 -mg 3 -cs 4
//

#include "mfem.hpp"
#include "kershaw.hpp"
#include "rhs.hpp"
#include "preconditioners.hpp"
#include <regex>
#include <fem/integ/bilininteg_diffusion_kernels.hpp>

using namespace std;
using namespace mfem;

struct MGRefinement
{
   enum Type { P_MG, H_MG };
   const Type type;
   const int order;
   MGRefinement(Type type_, int order_) : type(type_), order(order_) { }
   static MGRefinement p(int order_) { return MGRefinement(P_MG, order_); }
   static MGRefinement h() { return MGRefinement(H_MG, 0); }
};

struct CGMonitor : IterativeSolverMonitor
{
   const double tol;
   double initial_nrm, final_nrm, saved_nrm;
   int final_it, saved_it;

   CGMonitor(double tol_) : tol(tol_) { }

   void MonitorResidual(int it, double norm, const Vector &r, bool final)
   override
   {
      MFEM_CONTRACT_VAR(norm);
      // Avoid recomputing the norm if it was already computed -- this method
      // is called two times for the final iteration: once with final = false
      // (possibly triggering the monitor convergence criterion) and a second
      // time with final = true.
      bool init_call = (it == 0 && !final);
      const double nrm =
         (!init_call && it == saved_it) ?
         saved_nrm :
         sqrt(InnerProduct(iter_solver->GetComm(), r, r));
      if ((it == 0 || final) && Mpi::Root())
      {
         mfem::out << (final ? "Final" : "   Initial")
                   << " l2 norm of residual: " << nrm << '\n';
      }
      if (init_call)
      {
         initial_nrm = nrm;
         converged = false;
         final_nrm = -1.0;
         final_it = -1;
      }
      saved_nrm = nrm;
      saved_it = it;
      // Check for monitor-triggered convergence
      converged = (nrm <= tol*initial_nrm);
      if (final)
      {
         final_nrm = nrm;
         final_it = it;
      }
      if (final && Mpi::Root())
      {
         mfem::out << "Final relative l2 residual: ";
         if (initial_nrm == 0.0)
         {
            mfem::out << "N/A (initial norm is 0)" << endl;
         }
         else
         {
            const double rel_nrm = nrm/initial_nrm;
            mfem::out << rel_nrm << '\n';
            mfem::out << "Average l2 reduction factor: ";
            if (it == 0) { mfem::out << "N/A"; }
            else { mfem::out << pow(rel_nrm, 1.0/it); }
            mfem::out << " [" << it << " iterations]" << endl;
         }
      }
   }
};

void report_hypre_gpu_status(bool gpu_aware_mpi_requested);
void report_env_vars();
double verify_ess_bdr(const Vector &b, const Vector &x,
                      const Array<int> &ess_tdof_list);

template <typename T> void PrintPair(const string &name, T val)
{
   cout << setw(14) << left << name << val << '\n';
}


int main(int argc, char *argv[])
{
   DiffusionIntegrator::AddSpecialization<3,3,3>();
   DiffusionIntegrator::AddSpecialization<3,4,4>();
   DiffusionIntegrator::AddSpecialization<3,5,5>();
   DiffusionIntegrator::AddSpecialization<3,6,6>();

   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_config = "cpu";
   bool gpu_aware_mpi = false;
   int nx = 6, ny = -1, nz = -1;
   int rhs_n = -1;
   const char *mg_spec = "1";
   int q1d_inc = 0; // num 1D qpts = p + 1 + q1d_inc
   int smoothers_cheby_order = 1;
   double epsy = 1.0, epsz = -1;
   int ref_par = 0;
   bool glvis = false;
   bool paraview = false;
   SolverConfig coarse_solver(SolverConfig::JACOBI);

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&gpu_aware_mpi, "-g", "--gpu-aware-mpi", "-no-g",
                  "--no-gpu-aware-mpi", "Enable GPU-aware MPI.");
   args.AddOption(&mg_spec, "-mg", "--multigrid-spec",
                  "Multigrid specification. See README for description.");
   args.AddOption(&q1d_inc, "-qi", "--quadrature-points-increment",
                  "Increment for the 1D quadrature points relative to p + 1");
   args.AddOption(&smoothers_cheby_order, "-cb",
                  "--smoothers-chebyshev-order",
                  "Order of the Chebyshev smoothers for the multigrid.");
   args.AddOption((int*)&coarse_solver.type, "-cs", "--coarse-solver-config",
                  "Coarse solver configuration. 0: Jacobi, 1: FA-HYPRE, "
                  "2: LOR-HYPRE, 3: FA-AMGX, 4: LOR-AMGX.");
   args.AddOption(&coarse_solver.inner_cg, "-cg", "--inner-cg",
                  "-no-cg", "--no-inner-cg",
                  "Use inner CG iteration for the coarse solver.");
   args.AddOption(&coarse_solver.inner_sli, "-sli", "--inner-sli",
                  "-no-sli", "--no-inner-sli",
                  "Use inner SLI iteration for the coarse solver.");
   args.AddOption(&coarse_solver.inner_sli_iter, "-sli-it",
                  "--inner-sli-iterations",
                  "Number of iterations for the inner SLI solver.");
   args.AddOption(&coarse_solver.coarse_smooth, "-cls", "--coarse-level-smooth",
                  "-no-cls", "--no-coarse-level-smooth",
                  "Use coarse smoothing in addition to the coarse solver.");
   args.AddOption(&coarse_solver.amgx_config_file, "-amgx", "--amgx-config",
                  "AmgX config JSON file.");
   args.AddOption(&nx, "-nx", "--nx", "Number of elements in x direction.");
   args.AddOption(&ny, "-ny", "--ny", "Number of elements in y direction.");
   args.AddOption(&nz, "-nz", "--nz", "Number of elements in z direction.");
   args.AddOption(&epsy, "-ey", "--epsy", "Kershaw parameter epsilon y.");
   args.AddOption(&epsz, "-ez", "--epsz", "Kershaw parameter epsilon z.");
   args.AddOption(&rhs_n, "-rn", "--rhs-n",
                  "Parameter n in the RHS function; -1 for default.");
   args.AddOption(&ref_par, "-rp", "--ref-par",
                  "Number of uniform parallel refinements to perform.");
   args.AddOption(&glvis, "-gv", "--glvis", "-no-gv", "--no-glvis",
                  "Save the mesh and solution for GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Save data files for ParaView visualization.");
   args.ParseCheck();

   if (ny < 0) { ny = nx; }
   if (nz < 0) { nz = nx; }
   if (epsz < 0) { epsz = epsy; }
   // rhs_n default is handled later

   Device device(device_config);
   device.SetGPUAwareMPI(gpu_aware_mpi);
   if (Mpi::Root()) { device.Print(); }
   // Report HYPRE's GPU config and GPU-aware MPI config. Terminates if
   // GPU-aware MPI is requested but HYPRE's GPU-aware MPI support is disabled.
   report_hypre_gpu_status(gpu_aware_mpi);
   // Report environment variables like {CUDA,ROCR}_VISIBLE_DEVICES:
   report_env_vars();

   // Generate mesh
   ParMesh mesh_coarse = CreateKershawMesh(nx, ny, nz, epsy, epsz);
   const int dim = mesh_coarse.Dimension();
   for (int i=0; i<ref_par; ++i) { mesh_coarse.UniformRefinement(); }

   int coarse_order = 0, order = 0, h_ref = ref_par;
   // Parse order specification
   vector<MGRefinement> mg_refinements;
   {
      istringstream mg_stream(mg_spec);
      string ref;
      mg_stream >> coarse_order;
      int prev_order = order = coarse_order;
      if (Mpi::Root()) { cout << "\nCoarse order " << coarse_order << '\n'; }
      while (mg_stream >> ref)
      {
         if (ref == "r")
         {
            if (Mpi::Root()) { cout << "h-MG uniform refinement\n"; }
            mg_refinements.push_back(MGRefinement::h());
            ++h_ref;
         }
         else
         {
            try { order = stoi(ref); }
            catch (...)
            {
               MFEM_ABORT("Multigrid refinement must either be an integer or "
                          "the character `r`");
            }
            if (Mpi::Root()) { cout << "p-MG order   " << order << '\n'; }
            MFEM_VERIFY(order > 0, "Orders must be positive");
            MFEM_VERIFY(order > prev_order, "Orders must be increasing");
            mg_refinements.push_back(MGRefinement::p(order));
            prev_order = order;
         }
      }
   }

   if (order == 1 && coarse_solver.type == SolverConfig::LOR_HYPRE)
   {
      // Using ~10^7 elements with p=1 overflows a Vector in the LOR setup.
      // The Vector has size (3D): (p+1)^3 * 27 * num_elem_ho.
      // In 3D, for p > 1, the overflow will happen around:
      // - p=2: ~23.6 million dofs or 2,945,794 elements
      // - p=3: ~33.6 million dofs or 1,242,757 elements
      // - p=4: ~40.7 million dofs or   636,292 elements
      // - p=5: ~46.0 million dofs or   368,225 elements
      // - p=6: ~50.1 million dofs or   231,885 elements
      //
      // Note: the size of the Jacobians at quadrature points (with q1d=p+1) in
      // 3D is: (p+1)^3 * 9 * num_elem, so 3x smaller than the above Vector.
      //
      // For q1d=p+2, the overflow happens around:
      // - p=1: 8,837,382 elements or ~8.8 million dofs
      // - p=2: 3,728,271 elements or ~29.8 million dofs
      // - p=3: 1,908,875 elements or ~51.5 million dofs
      // - p=4: 1,104,673 elements or ~70.7 million dofs
      // - p=5:   695,654 elements or ~87.0 million dofs
      // - p=6:   466,034 elements or ~100.7 million dofs
      coarse_solver.type = SolverConfig::FA_HYPRE;
      if (Mpi::Root())
      {
         cout << "\nOrder is 1: switching from LOR-HYPRE to FA-HYPRE.\n";
      }
   }
#if 0
   if (order == 1 && coarse_solver.type == SolverConfig::FA_HYPRE &&
       coarse_solver.inner_sli)
   {
      coarse_solver.inner_sli = false;
      if (Mpi::Root())
      {
         cout << "\nOrder is 1: turning off the inner SLI.\n";
      }
   }
#endif

   vector<unique_ptr<FiniteElementCollection>> fe_collections;
   fe_collections.emplace_back(new H1_FECollection(coarse_order, dim));
   ParFiniteElementSpace fes_coarse(&mesh_coarse, fe_collections.back().get());
   ParFiniteElementSpaceHierarchy hierarchy(&mesh_coarse, &fes_coarse,
                                            false, false);

   for (MGRefinement ref : mg_refinements)
   {
      if (ref.type == MGRefinement::H_MG)
      {
         hierarchy.AddUniformlyRefinedLevel();
      }
      else // P_MG
      {
         fe_collections.emplace_back(new H1_FECollection(ref.order, dim));
         hierarchy.AddOrderRefinedLevel(fe_collections.back().get());
      }
   }

   const int nlevels = hierarchy.GetNumLevels();
   if (Mpi::Root())
   {
      if (nlevels == 1)
      {
         cout << "1 level in MG hierarchy. Using coarse solver only." << endl;
      }
      else
      {
         cout << nlevels << " levels in MG hierarchy." << endl;
      }
      coarse_solver.Print();
      cout << endl;
   }

   // Determine final nx, ny, nz and use them to determine the default rhs_n.
   const int ref_factor = pow(2, h_ref);
   nx *= ref_factor;
   ny *= ref_factor;
   nz *= ref_factor;
   if (rhs_n < 0)
   {
      int n_min = min(nx, ny);
      if (nz > 0) { n_min = min(n_min, nz); }
      // Find rhs_n such that 2*3^rhs_n <= (order*n_min) < 2*3^{rhs_n+1}
      rhs_n = 0;
      for (int l = 2*3; l <= order*n_min; l *= 3) { rhs_n++; }
      if (epsy < 0.8) { rhs_n--; }
      if (Mpi::Root()) { cout << "Using rhs_n = " << rhs_n << '\n' << endl; }
   }

   ParFiniteElementSpace &fes = hierarchy.GetFinestFESpace();
   ParMesh &mesh = *fes.GetParMesh();
   mesh.PrintInfo(cout);
   HYPRE_Int ndof = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "\nTotal number of DOFs: " << ndof << endl << endl;
   }

   // All Dirichlet boundaries
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient coeff(1.0); // Diffusion coefficient
   // Set up RHS
   if (Mpi::Root()) { cout << "Assembling right-hand side..." << endl; }
   RHS rhs_coeff(dim, rhs_n);
   ParLinearForm b(&fes);
   const int rhs_ir_inc = 2*q1d_inc+1;
   // --> ir_order = 2*(p+1+q1d_inc)-1 --> q1d = p+1+q1d_inc
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff, 2, rhs_ir_inc));
   b.UseFastAssembly(true);
   b.Assemble();
   if (Mpi::Root()) { cout << "Assembling right-hand side... Done." << endl; }

   // make sure the GPU is done with any previous tasks:
   if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
   // make sure all ranks are done with any previous tasks:
   MPI_Barrier(MPI_COMM_WORLD);
   tic();
   // Set up operators in the multigrid hierarchy
   DiffusionMultigrid MG(hierarchy, coeff, ess_bdr, coarse_solver, q1d_inc,
                         smoothers_cheby_order);
   MG.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   // make sure the GPU is done with all setup tasks:
   if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
   // make sure all ranks are done with all setup tasks:
   MPI_Barrier(MPI_COMM_WORLD);
   const double t_setup = tic_toc.RealTime();

   ParGridFunction x(&fes);
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   MG.FormFineLinearSystem(x, b, A, X, B);

   const double l2_tol = 1e-8;
   CGMonitor monitor(l2_tol);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(0.0); // use the 'monitor' for convergence
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(MG);
   cg.SetMonitor(monitor);
   // Run 2 CG iterations to ensure everything is allocated and initialized for
   // the full CG solve:
   if (Mpi::Root()) { cout << "Running 2 warm-up CG iterations ...\n"; }
   cg.SetMaxIter(2);
   {
      Vector X_save(X);
      cg.Mult(B, X);
      X = X_save;
   }
   if (coarse_solver.inner_sli &&
       ((coarse_solver.type == SolverConfig::FA_HYPRE /* && order > 1 */) ||
        coarse_solver.type == SolverConfig::LOR_HYPRE))
   {
      // timing data: (t-solve,sli-iter,cheby-order,pcg-iter)
      std::vector<std::tuple<double,int,int,int>> timings;
      Vector X_save(X);
      if (Mpi::Root()) { cout << "\nFinding optimal MG parameters ...\n"; }
      cg.SetMaxIter(500);
      for (int sli_it = 1; sli_it <= coarse_solver.inner_sli_iter; sli_it++)
      {
         MG.SetInnerSLINumIter(sli_it);
         for (int cheby_order = 1; cheby_order <= smoothers_cheby_order;
              cheby_order++)
         {
            MG.SetSmoothersChebyshevOrder(cheby_order);

            if (Mpi::Root())
            {
               cout << "\nRunning and timing parameters (sli iter, cheby order)"
                    << " = (" << sli_it << ',' << cheby_order << ") ...\n";
            }
            // make sure the GPU is done with any previous tasks:
            if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
            // make sure all ranks are done with any previous tasks:
            MPI_Barrier(MPI_COMM_WORLD);
            tic();
            cg.Mult(B, X);
            // make sure the GPU is done with all solve tasks:
            if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
            // make sure all ranks are done with all solve tasks:
            MPI_Barrier(MPI_COMM_WORLD);
            const double t_solve = tic_toc.RealTime();
            if (cg.GetConverged())
            {
               timings.emplace_back(t_solve, sli_it, cheby_order,
                                    cg.GetNumIterations());
            }
            X = X_save;
         }
      }
      std::sort(timings.begin(), timings.end());
      if (Mpi::Root())
      {
         cout << "\nSorted timings from rank 0:\n";
         const auto old_prec = cout.precision(6);
         const auto old_fmtflags = cout.flags();
         cout << std::fixed;
         for (size_t i = 0; i < timings.size(); i++)
         {
            cout << setw(2) << i << ": "
                 << 1e3*std::get<0>(timings[i]) << " ms: ("
                 << std::get<1>(timings[i]) << ','
                 << std::get<2>(timings[i]) << "): "
                 << setw(3) << std::get<3>(timings[i]) << " iter\n";
         }
         cout.flags(old_fmtflags);
         cout.precision(old_prec);
      }
      if (timings.size() > 0)
      {
         // Use the fastest parameters (as timed on rank 0) for the full solve:
         int si = std::get<1>(timings[0]);
         int co = std::get<2>(timings[0]);
         MPI_Bcast(&si, 1, MPI_INT, 0, MPI_COMM_WORLD);
         MPI_Bcast(&co, 1, MPI_INT, 0, MPI_COMM_WORLD);
         MG.SetInnerSLINumIter(si);
         MG.SetSmoothersChebyshevOrder(co);
         coarse_solver.inner_sli_iter = si;
         smoothers_cheby_order = co;
         if (Mpi::Root())
         {
            cout << "\nUsing the fastest option (sli iter, cheby order) = ("
                 << si << ',' << co << ")\n";
         }
      }
      else
      {
         MG.SetInnerSLINumIter(1);
         MG.SetSmoothersChebyshevOrder(1);
         coarse_solver.inner_sli_iter = 1;
         smoothers_cheby_order = 1;
         if (Mpi::Root())
         {
            cout << "\nAll options failed to converge!"
                 << " Using (sli iter, cheby order) = (1,1)\n";
         }
      }
   }
   if (Mpi::Root()) { cout << "\nRunning and timing the full CG solve ...\n"; }
   cg.SetMaxIter(500);
   // make sure the GPU is done with any previous tasks:
   if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
   // make sure all ranks are done with any previous tasks:
   MPI_Barrier(MPI_COMM_WORLD);
   tic();
   cg.Mult(B, X);
   // make sure the GPU is done with all solve tasks:
   if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_STREAM_SYNC; }
   // make sure all ranks are done with all solve tasks:
   MPI_Barrier(MPI_COMM_WORLD);
   const double t_solve = tic_toc.RealTime();

   const int niter = cg.GetConverged() ? cg.GetNumIterations() : -1;

   const double bdr_err = verify_ess_bdr(B, X, MG.GetFineEssentialTrueDofs());
   if (Mpi::Root())
   {
      MFEM_VERIFY(bdr_err == 0.0, "Incorrect boundary values in solution!"
                  " bdr_err = " << bdr_err);
   }

   MG.RecoverFineFEMSolution(X, b, x);

   ExactSolution exact_coeff(dim, rhs_n);
   // ExactGrad exact_grad_coeff(dim, rhs_n);
   double L2_err = x.ComputeL2Error(exact_coeff);
   // double grad_err = x.ComputeGradError(&exact_grad_coeff);
   if (Mpi::Root())
   {
      cout << "\nL2 Error: " << setprecision(10) << scientific
           << L2_err << '\n';
      // cout << "\nGrad Error: " << setprecision(10) << scientific
      //      << grad_err << '\n';
   }

   if (glvis)
   {
      ofstream mesh_ofs(MakeParFilename("mesh.", Mpi::WorldRank()));
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);

      ofstream sol_ofs(MakeParFilename("sol.", Mpi::WorldRank()));
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   if (paraview)
   {
      ParGridFunction rhs_gf(&fes), exact_gf(&fes), error_gf(&fes);
      rhs_gf.ProjectCoefficient(rhs_coeff);
      exact_gf.ProjectCoefficient(exact_coeff);

      subtract(exact_gf, x, error_gf);

      ParaViewDataCollection dc("SolverBP", &mesh);
      dc.RegisterField("u", &x);
      dc.RegisterField("rhs", &rhs_gf);
      dc.RegisterField("exact", &exact_gf);
      dc.RegisterField("error", &error_gf);
      dc.SetPrefixPath("ParaView");
      dc.SetLevelsOfDetail(order);
      dc.SetHighOrderOutput(true);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   const long long nel = mesh.GetGlobalNE();
   if (nz == 0) { MFEM_VERIFY(nel == nx*ny, "Wrong number of elements"); }
   else { MFEM_VERIFY(nel == nx*ny*nz, "Wrong number of elements"); }

   if (Mpi::Root())
   {
      cout << "\n= Results\n";
      PrintPair("nranks", Mpi::WorldSize());
      PrintPair("nx", nx);
      PrintPair("ny", ny);
      PrintPair("nz", nz);
      PrintPair("degree", order);
      PrintPair("rhs_n", rhs_n);
      PrintPair("epsy", epsy);
      PrintPair("epsz", epsz);
      PrintPair("ndof", ndof);
      PrintPair("niter", niter);

      // Should also output:
      // code id
      // prec id
      // machine id
      // number of supercomputer nodes
      // number of 1d quadrature points
      // initial and final residuals
      // error

      // Timings
      PrintPair("t_setup", t_setup);
      PrintPair("t_solve", t_solve);

      cout << "\nSolve MDOFs/rank/sec:   "
           << ndof/1e6/Mpi::WorldSize()/t_solve << '\n';

      // CSV fields:
      // 1. code ID
      // 2. preconditioner ID
      // 3. machine ID
      // 4. number of nodes
      // 5. number of MPI ranks
      // 6,7,8. n_x, n_y, n_z
      // 9. solution polynomial degree
      // 10. number of 1D quadrature points
      // 11,12. eps_y, eps_z
      // 13. ndofs (including Dirichlet boundary)
      // 14. niter
      // 15,16. initial and final residuals
      // 17. error
      // 18. t_setup (preconditioner setup)
      // 19. t_solve (total iter time)
      //
      // extract the CSV lines from the output with:
      //    grep "= CSV:" out.txt | sed -e 's/^= CSV://' > out.csv
      cout << "\n= CSV:"
           << "MFEM-" + string(device_config); // 1
      string hypre_str =
#if defined(HYPRE_USING_HIP)
         "hypre-hip"
#elif defined(HYPRE_USING_CUDA)
         "hypre-cuda"
#else
         "hypre-cpu"
#endif
         ;
      auto cs = coarse_solver.type;
      string prec_id;
      if (cs == SolverConfig::FA_HYPRE) // p-MG, add (sli-iter,cheby-order)
      {
         prec_id = hypre_str + "-pMG(";
      }
      else if (cs == SolverConfig::LOR_HYPRE) // LOR, add (sli-iter,cheby-order)
      {
         prec_id = hypre_str + "-LOR(";
      }
      else if (cs == SolverConfig::JACOBI)
      {
         prec_id = "diag(";
      }
      else
      {
         prec_id = "(unknown)(";
      }
      if (coarse_solver.inner_cg)
      {
         prec_id += "cg;";
      }
      if (coarse_solver.inner_sli)
      {
         prec_id += to_string(coarse_solver.inner_sli_iter) + ";";
      }
      prec_id += to_string(smoothers_cheby_order) +
                 (coarse_solver.coarse_smooth ? "c" : "") + ")";
      prec_id += "-" + regex_replace(mg_spec, regex(" "), "-");
      cout << ',' << prec_id; // 2
      const char *hostname = getenv("HOSTNAME");
      if (!hostname) { hostname = getenv("HOST"); }
      string host_id = regex_replace(hostname ? hostname : "(unknown)",
                                     regex("[0-9]*$"), "");
      cout << ',' << host_id; // 3
      cout << ',' << (fes.GetNRanks() + 7)/8; // 4 (assuming 8 ranks/node !!)
      cout << ',' << fes.GetNRanks(); // 5
      cout << ',' << nx << ',' << ny << ',' << nz; // 6,7,8
      cout << ',' << order; // 9
      // DiffusionMultigrid::ConstructBilinearForm p+1+q1d_inc 1D points
      double Q1D = order + 1 + q1d_inc;
      cout << ',' << defaultfloat << Q1D; // 10 (note: written as double)
      cout << ',' << scientific << epsy << ',' << epsz; // 11,12
      cout << ',' << ndof; // 13
      cout << ',' << niter; // 14
      cout << ',' << monitor.initial_nrm << ',' << monitor.final_nrm; // 15,16
      cout << ',' << L2_err; // 17
      // cout << ',' << grad_err; // 17 *** for testing ***
      cout << ',' << t_setup << ',' << t_solve; // 18,19
      cout << endl;
   }

   return 0;
}

void report_hypre_gpu_status(bool gpu_aware_mpi_requested)
{
#ifdef HYPRE_WITH_GPU_AWARE_MPI
   bool hypre_gpu_aware_mpi = true;
#else
   bool hypre_gpu_aware_mpi = false;
#endif
#if (MFEM_HYPRE_VERSION > 23000)
   hypre_gpu_aware_mpi = hypre_gpu_aware_mpi && hypre_GetGpuAwareMPI();
#endif
   if (Mpi::Root())
   {
      MFEM_VERIFY(!gpu_aware_mpi_requested || hypre_gpu_aware_mpi,
                  "GPU-aware MPI requested but HYPRE's GPU-aware MPI support"
                  " is not enabled");
      cout << "\nHYPRE GPU support: "
#ifdef HYPRE_USING_GPU
           << "enabled";
#else
           << "disabled";
#endif
      cout << "\nHYPRE GPU-aware MPI support: "
           << (hypre_gpu_aware_mpi ? "enabled" : "disabled") << endl;
   }
}

void report_env_vars()
{
   const int myid = Mpi::WorldRank();
   // const int lastid = min(Mpi::WorldSize(),4)-1; // show up to 4 ranks
   const int lastid = Mpi::WorldSize()-1;
   if (myid > lastid) { return; }
   Array<char> recv_buf;
   int buflen = -1, tag = 42;
   const char *env_vars[] =
   {
      "HOST", "HOSTNAME", "MPICH_GPU_SUPPORT_ENABLED", "CUDA_VISIBLE_DEVICES",
      "ROCR_VISIBLE_DEVICES"
   };
   const int num_env_vars = sizeof(env_vars)/sizeof(env_vars[0]);
   // Send strings to rank 0, so that they can be printed in order, guaranteed.
   // Every rank > 0 sends to rank 0:
   if (myid > 0)
   {
      for (int ev = 0; ev < num_env_vars; ev++)
      {
         const char *env_var_val = getenv(env_vars[ev]);
         buflen = env_var_val ? int(strlen(env_var_val)+1) : -1;
         MPI_Send(&buflen, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
         if (env_var_val)
         {
            MPI_Send(env_var_val, buflen, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
         }
      }
   }
   else // myid == 0
   {
      cout << "\nDefined environment variables:\n";
      for (int id = 0; id <= lastid; id++)
      {
         cout << "[rank " << id << "]:";
         for (int ev = 0, vars_shown = 0; ev < num_env_vars; ev++)
         {
            const char *env_var_val = nullptr;
            if (id == 0)
            {
               env_var_val = getenv(env_vars[ev]);
               buflen = env_var_val ? 0 : -1;
            }
            else
            {
               MPI_Recv(&buflen, 1, MPI_INT, id, tag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
            }
            if (buflen != -1)
            {
               if (id > 0)
               {
                  recv_buf.SetSize(buflen);
                  MPI_Recv(recv_buf.begin(), buflen, MPI_CHAR, id, tag,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                  env_var_val = recv_buf.begin();
               }
               if (vars_shown)
               {
                  cout << "\n[rank " << id << "]:";
               }
               cout << ' ' << env_vars[ev] << '=' << env_var_val;
               vars_shown++;
            }
         }
         cout << '\n';
      }
      if (lastid < Mpi::WorldSize()-1)
      {
         cout << "... [only " << lastid+1 << '/' << Mpi::WorldSize()
              << " ranks shown]\n";
      }
      cout << flush;
   }
}

double verify_ess_bdr(const Vector &b, const Vector &x,
                      const Array<int> &ess_tdof_list)
{
   Vector d(ess_tdof_list.Size());
   auto d_b = b.Read();
   auto d_x = x.Read();
   auto d_d = d.Write();
   auto d_ess_ind = ess_tdof_list.Read();
   mfem::forall(ess_tdof_list.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const int ind = d_ess_ind[i];
      d_d[i] = -fabs(d_b[ind] - d_x[ind]);
   });
   double d_max = -d.Min(); // max is not implemented on device
   MPI_Allreduce(MPI_IN_PLACE, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return d_max;
}
