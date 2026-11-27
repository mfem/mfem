// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "contact_driver.hpp"
#include "solver_utils.hpp"
#include "contact_output.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

using namespace std;
using namespace mfem;

#include "driver_output.hpp"

bool mfem::ContactOptionSupplied(int argc, char *argv[],
                                 std::initializer_list<const char *> names)
{
   for (int i = 1; i < argc; i++)
   {
      for (const char *name : names)
      {
         if (std::string(argv[i]) == name) { return true; }
      }
   }
   return false;
}

int mfem::RunContactDriver(int argc, char *argv[],
                           ContactDriverOptions &options,
                           ContactDriverProblem &scenario)
{
#ifdef MFEM_USE_SINGLE
   cout << "Contact miniapp is not supported in single precision.\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif
   Mpi::Init();
   const int myid = Mpi::WorldRank(), num_procs = Mpi::WorldSize();
   auto &sref = options.sref;
   auto &pref = options.pref;
   auto &load_steps = options.load_steps;
   auto &extra_load_steps = options.extra_load_steps;
   auto &opt_maxit = options.opt_maxit;
   auto &print_level = options.print_level;
   auto &contact_max_updates = options.contact_max_updates;
   auto &contact_update_tol = options.contact_update_tol;
   auto &tribol_ratio = options.tribol_ratio;
   auto &nonlinear = options.nonlinear;
   auto &elasticity_fa = options.elasticity_fa;
   auto &allow_inexact_contact_updates = options.allow_inexact_contact_updates;
   auto &visualization = options.visualization;
   auto &paraview = options.paraview;
   auto &amgf = options.amgf;
   auto &amgf_fsolver = options.amgf_fsolver;
   auto &linear_solver = options.linear_solver;
   auto &amg_coarsen_type = options.amg_coarsen_type;
   auto &amg_agg_levels = options.amg_agg_levels;
   auto &amg_interp_type = options.amg_interp_type;
   auto &amg_relax_type = options.amg_relax_type;
   auto &amg_print_options = options.amg_print_options;
   auto &hypre_gpu_aware_mpi = options.hypre_gpu_aware_mpi;
   auto &device_config = options.device_config;
   OptionsParser args(argc, argv);
   scenario.AddOptions(args);
   args.AddOption(&nonlinear, "-nonlin", "--nonlinear", "-lin",
                  "--linear",
                  "Elasticity model (nonlinear for beam-sphere; fixed by the other problems).");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Serial uniform refinements (default 0).");
   args.AddOption(&opt_maxit, "-opt-maxit", "--optimizer-max-iterations",
                  "Maximum number of iterations for the optimizer.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Output level: 0 = silent, 1 = final IP summary only, "
                  "2 = solver tables/results (default), 3 = debug diagnostics.");
   args.AddOption(&load_steps, "-load-steps", "--load-steps",
                  "Number of load increments.");
   args.AddOption(&contact_max_updates, "-contact-max-updates",
                  "--contact-max-updates",
                  "Maximum geometry/material updates per nonlinear contact load step (default 1).");
   args.AddOption(&contact_update_tol, "-contact-update-tol",
                  "--contact-update-tolerance",
                  "Absolute displacement tolerance for nonlinear contact updates.");
   args.AddOption(&allow_inexact_contact_updates,
                  "-allow-inexact-contact-updates", "--allow-inexact-contact-updates",
                  "-require-converged-contact-updates", "--require-converged-contact-updates",
                  "Accept the last converged IP solution at the geometry update limit; "
                  "the IP solve must still converge (default: continue at the update limit).");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&amgf, "-amgf", "--amgf", "-no-amgf",
                  "--no-amgf", "Enable or disable AMG with Filtering solver.");
   args.AddOption(&amgf_fsolver, "-amgf-fsolver", "--amgf-filtered-solver",
                  "Direct solver for AMGF filtered subspace.\n"
                  "Choices: auto, mumps, cpardiso, superlu, strumpack, cudss "
                  "(CUDA only; use auto on CPU or HIP).");
   args.AddOption(&linear_solver, "-linear-solver", "--linear-solver",
                  "Reduced-system solver: cg (default, using AMG/AMGF), or a sparse direct backend "
                  "auto, mumps, superlu, strumpack, cpardiso, cudss.");
   args.AddOption(&amg_coarsen_type, "-amg-coarsen-type",
                  "--amg-coarsen-type",
                  "BoomerAMG coarsening type.");
   args.AddOption(&amg_agg_levels, "-amg-agg-levels",
                  "--amg-agg-levels",
                  "BoomerAMG number of aggressive coarsening levels.");
   args.AddOption(&amg_interp_type, "-amg-interp-type",
                  "--amg-interpolation-type",
                  "BoomerAMG interpolation type.");
   args.AddOption(&amg_relax_type, "-amg-relax-type",
                  "--amg-relax-type",
                  "BoomerAMG smoother: 16 = Chebyshev (default), 18 = l1-Jacobi.");
   args.AddOption(&hypre_gpu_aware_mpi, "-hypre-gpu-aware-mpi",
                  "--hypre-gpu-aware-mpi", "-no-hypre-gpu-aware-mpi",
                  "--no-hypre-gpu-aware-mpi",
                  "Enable HYPRE GPU-aware MPI; requires GPU-aware MPI support "
                  "in the MPI library and launcher.");
   args.AddOption(&amg_print_options, "-amg-print-options",
                  "--amg-print-options", "-no-amg-print-options",
                  "--no-amg-print-options",
                  "Print BoomerAMG options configured by this miniapp.");
   args.AddOption(&elasticity_fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly",
                  "Use full assembly for the linear elasticity stiffness matrix.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&tribol_ratio, "-tr", "--tribol-proximity-parameter",
                  "Tribol-proximity-parameter.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return args.Help() ? 0 : 1;
   }
   if (!scenario.ResolveOptions(argc, argv)) { return 1; }
   if (sref < 0 || pref < 0 || load_steps < 1 || extra_load_steps < 0 ||
       opt_maxit < 1 || print_level < 0 || print_level > 3 ||
       contact_max_updates < 1 || !IsFinite(contact_update_tol) ||
       contact_update_tol <= 0.0 || !IsFinite(tribol_ratio) || tribol_ratio <= 0.0)
   {
      if (myid == 0)
      {
         mfem::err <<
                   "Require positive steps, solver limits/tolerances and proximity; "
                   << "nonnegative refinements/extra steps; print level 0..3.\n";
      }
      return 1;
   }
   const std::string linear_solver_name(linear_solver);
   const bool direct_linear_solve = linear_solver_name != "cg";
   if (direct_linear_solve && linear_solver_name != "auto" &&
       linear_solver_name != "mumps" && linear_solver_name != "superlu" &&
       linear_solver_name != "strumpack" && linear_solver_name != "cpardiso" &&
       linear_solver_name != "cudss")
   {
      if (myid == 0) { mfem::err << "Unknown linear solver: " << linear_solver_name << '\n'; }
      return 1;
   }
   const bool update_geometry = nonlinear;
   if (print_level < 2)
   {
      mfem::out.Disable();
      mfem::err.Disable();
   }
   // Keep help/option validation independent of GPU HYPRE initialization.
   Hypre::Init();
   Device device(device_config);
#if MFEM_HYPRE_VERSION >= 23100
   MFEM_VERIFY(!hypre_gpu_aware_mpi || HypreUsingGPU(),
               "HYPRE GPU-aware MPI requires an active GPU HYPRE backend.");
   MFEM_VERIFY(HYPRE_SetGpuAwareMPI(hypre_gpu_aware_mpi ? 1 : 0) == 0,
               "HYPRE MPI policy configuration failed.");
   if (myid == 0 && print_level >= 2)
   {
      cout << "HYPRE GPU-aware MPI: "
           << (hypre_GetGpuAwareMPI() ? "enabled" : "disabled") << endl;
   }
#else
   MFEM_VERIFY(!hypre_gpu_aware_mpi,
               "The HYPRE GPU-aware MPI option requires HYPRE 2.31 or newer.");
#endif
   if (myid == 0 && print_level >= 2)
   {
      args.PrintOptions(cout);
   }
   if (myid == 0 && print_level >= 2) { device.Print(); }

   if (nonlinear && elasticity_fa && myid == 0 && print_level >= 2)
   {
      cout << "Full assembly is only used by the linear elasticity model; "
           << "nonlinear reference assembly uses "
           << (Device::IsEnabled() ? "device kernels." : "CPU kernels.")
           << endl;
   }

   // Displacement bounds are problem-specific and independent of the
   // geometry/material refresh loop used by every nonlinear problem.
   const bool bound_constraints = nonlinear && scenario.UsesDisplacementBounds();


   // 2. Get the problem mesh.
   std::unique_ptr<Mesh> mesh = scenario.MakeMesh();

   // 3. Refine the mesh serially.
   for (int i = 0; i<sref; i++)
   {
      mesh->UniformRefinement();
   }

   // 4. Convert to ParMesh and refine in parallel.
   ParMesh pmesh(MPI_COMM_WORLD,*mesh);
   mesh.reset();
   for (int i = 0; i<pref; i++)
   {
      pmesh.UniformRefinement();
   }

   // Material parameters per attribute (Young’s modulus and Poisson ratio).
   Vector E(pmesh.attributes.Max());
   Vector nu(pmesh.attributes.Max());

   // Essential boundary specification:
   // - ess_bdr_attr: boundary attribute ids to constrain
   // - ess_bdr_attr_comp: component to constrain (0=x,1=y,2=z, -1=all)
   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;

   scenario.Configure(pmesh, E, nu, ess_bdr_attr, ess_bdr_attr_comp);

   // 6. Build the elasticity problem object (linear or non-linear).
   // It owns the FE space and will assemble the stiffness and RHS.
   ElasticityOperator prob(&pmesh, ess_bdr_attr,ess_bdr_attr_comp, E, nu,
                           nonlinear, elasticity_fa);

   ParFiniteElementSpace * fes = prob.GetFESpace();
   HYPRE_BigInt gndofs = fes->GlobalTrueVSize();
   const long long global_elements = pmesh.GetGlobalNE();
   if (myid == 0 && print_level >= 2)
   {
      mfem::out << "\n --------------------------------------" << endl;
      mfem::out << "  Global number of dofs = " << gndofs << endl;
      mfem::out << "  Bulk/geometry order   = 1" << endl;
      mfem::out << "  Mesh elements         = " << global_elements << endl;
      mfem::out << "  Elasticity assembly   = "
                << (nonlinear ? (Device::IsEnabled() ? "nonlinear device" :
                                 "nonlinear CPU") :
                    elasticity_fa ? "full" : "legacy") << endl;
      mfem::out << "  RHS assembly          = "
                << (prob.UsesFastRHSAssembly() ? "fast" : "legacy") << endl;
      mfem::out << " --------------------------------------" << endl;
      scenario.PrintParameters();
   }

   // 7. Set up the contact interfaces for Tribol. Each interface is handled
   // by its own coupling scheme.
   std::vector<ContactInterface> contact_interfaces = scenario.Interfaces();
   ParGridFunction x_gf(fes); x_gf = 0.0;
   scenario.Initialize(prob);
   ParMesh pmesh_copy(pmesh);
   ParFiniteElementSpace fes_copy(*fes,pmesh_copy);
   ParGridFunction xcopy_gf(&fes_copy); xcopy_gf = x_gf;

   // Optional ParaView data collection
   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (paraview)
   {
      std::string paraview_run_name;
      if (myid == 0)
      {
         const auto now = std::chrono::system_clock::now();
         const std::time_t run_time = std::chrono::system_clock::to_time_t(now);
         std::ostringstream paraview_file_name;
         paraview_file_name << scenario.OutputName() << '_'
                            << std::put_time(std::gmtime(&run_time), "%Y%m%d_%H%M%S")
                            << 'Z';
         const std::string base_name = paraview_file_name.str();
         paraview_run_name = base_name;

         // Reserve the directory atomically so simultaneous runs stay distinct.
         std::error_code error;
         std::filesystem::create_directories("ParaView", error);
         MFEM_VERIFY(!error, "Cannot create ParaView directory: " << error.message());
         int suffix = 0;
         while (!std::filesystem::create_directory("ParaView/" + paraview_run_name,
                                                   error))
         {
            MFEM_VERIFY(!error || error == std::errc::file_exists,
                        "Cannot create ParaView run directory: " << error.message());
            paraview_run_name = base_name + '_' + std::to_string(++suffix);
         }
         // Keep the directory name concise and record the resolved run settings.
         std::ofstream run_info("ParaView/" + paraview_run_name + "/run-info.txt");
         MFEM_VERIFY(run_info, "Cannot create ParaView run-info.txt.");
         run_info << "Executable: " << argv[0] << '\n'
                  << "Problem: " << scenario.Name() << '\n'
                  << "MPI ranks: " << num_procs << '\n'
                  << "Displacement order: 1\n"
                  << "Global elements: " << global_elements << '\n'
                  << "Global true DOFs: " << gndofs << '\n'
                  << "Arguments:";
         for (int arg = 1; arg < argc; ++arg)
         { run_info << ' ' << std::quoted(argv[arg]); }
         run_info << "\n\n";
         args.PrintOptions(run_info);
         run_info.close();
         MFEM_VERIFY(run_info, "Cannot write ParaView run-info.txt.");
      }
      int name_length = static_cast<int>(paraview_run_name.size());
      MPI_Bcast(&name_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
      paraview_run_name.resize(name_length);
      MPI_Bcast(paraview_run_name.data(), name_length, MPI_CHAR, 0, MPI_COMM_WORLD);
      paraview_dc = std::make_unique<ParaViewDataCollection>(
                       paraview_run_name, &pmesh_copy);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(1);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(false);
      paraview_dc->RegisterField("u", &xcopy_gf);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
      if (myid == 0 && print_level >= 2)
      {
         cout << "ParaView output written to ParaView/" << paraview_run_name << endl;
      }
   }

   // Optional GLVis connection.
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   // Reference vs current coordinates (used to visualize total deformation).
   ParGridFunction ref_coords(prob.GetFESpace());
   ParGridFunction new_coords(prob.GetFESpace());
   pmesh.GetNodes(new_coords);
   pmesh.GetNodes(ref_coords);

   // 8. Construct the contact optimization problem wrapper. It sets up the
   // contact system using Tribol and provides operators for objective evaluation,
   // gradients, Hessians, and constraints. It also provides the filtered subspace
   // transfer operator used by AMGF
   OptContactProblem contact(&prob, contact_interfaces, tribol_ratio,
                             bound_constraints);

   real_t initial_clearance = 0.0;
   if (scenario.InitialClearance(initial_clearance))
   {
      prob.FormLinearSystem();
      Vector initial_displacement;
      x_gf.GetTrueDofs(initial_displacement);
      contact.FormContactSystem(&new_coords, initial_displacement);
      Vector gap(contact.GetConstraintJacobian()->Height());
      contact.EvaluateGap(initial_displacement, gap);
      const real_t local_min = gap.Size() ? gap.Min() : infinity();
      real_t minimum_gap;
      MPI_Allreduce(&local_min, &minimum_gap, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
      if (myid == 0 && print_level >= 2)
      {
         cout << scenario.Name() << " initial contact: surface clearance = "
              << initial_clearance
              << ", gap rows = " << contact.GetGlobalNumGapRows()
              << ", minimum mortar gap = " << minimum_gap << endl;
      }
   }

   // 9. Time/load stepping loop.
   int total_steps = load_steps + extra_load_steps;
   std::vector<IPSolveSummary> solve_results;
   for (int i = 0; i<total_steps; i++)
   {
      if (Mpi::Root() && print_level >= 2)
      {
         contact_output::PrintBanner(mfem::out,
                                     "Solving optimization problem for time step: " +
                                     std::to_string(i));
      }
      scenario.ApplyLoad(i, prob, x_gf, ref_coords);
      add(ref_coords, x_gf, new_coords);

      // 9(b). Assemble the elasticity system with current BCs.
      prob.FormLinearSystem();

      // Store the current (reference) displacement in true dofs.
      Vector xref;
      xref.UseDevice(true);
      x_gf.GetTrueDofs(xref);
      Vector load_start(xref); load_start.UseDevice(true);
      const real_t nonlinear_initial_energy = update_geometry ?
                                              prob.GetEnergy(xref) : 0.0;

      std::unique_ptr<ParallelDirectSolver> subspacesolver;
      std::unique_ptr<Solver> prec;
      std::unique_ptr<IPSolver> optimizer_ptr;
      Vector x0(fes->GetTrueVSize()), xf(fes->GetTrueVSize());
      x0.UseDevice(true);
      xf.UseDevice(true);
      CGSolver cgsolver(MPI_COMM_WORLD);
      std::unique_ptr<ParallelDirectSolver> bulk_solver;
      if (direct_linear_solve)
      {
         bulk_solver = std::make_unique<ParallelDirectSolver>(MPI_COMM_WORLD,
                                                              linear_solver_name);
         bulk_solver->SetPrintLevel(0);
      }
      cgsolver.SetPrintLevel(0);
      cgsolver.SetRelTol(1e-10);
      cgsolver.SetMaxIter(10000);
      const int max_updates = update_geometry ? contact_max_updates : 1;
      bool geometry_converged = !update_geometry;
      int total_ip_steps = 0, total_newton_solves = 0;
      long long total_cg_iterations = 0;
      int geometry_updates = 0;
      real_t update_norm = infinity();
      constexpr int activation_step = 2;
      for (int update = 0; update < max_updates; update++)
      {
         // Destroy borrowers before rebuilding the contact layout/operators.
         optimizer_ptr.reset();
         prec.reset();
         subspacesolver.reset();
         // 9(c). Form the contact system. This also builds the filtered subspace
         // transfer used by AMGF.
         contact.FormContactSystem(&new_coords, xref);

         // 9(d). Activate bound constraints after a few steps in the non-linear case.
         if (bound_constraints && i>activation_step)
         {
            contact.ActivateBoundConstraints();
         }

         // 9(e). Choose preconditioner: AMGF (with direct solver for subspace) or AMG.
         // The preconditioner borrows the filtered solver, so it must die first.
         HypreParMatrix *contact_transfer = amgf && !direct_linear_solve ?
                                            contact.GetContactSubspaceTransferOperator() : nullptr;
         if (!direct_linear_solve && amgf && contact_transfer)
         {
            auto amgfprec = std::make_unique<AMGFSolver>();
            ConfigureAMG(amgfprec->GetAMG(), amg_coarsen_type, amg_agg_levels,
                         amg_interp_type, amg_relax_type);
            if (amg_print_options && myid == 0 && print_level >= 2)
            {
               PrintAMGOptions(amgfprec->GetAMG(), "AMGF", amg_agg_levels,
                               amgf_fsolver);
            }
            subspacesolver = std::make_unique<ParallelDirectSolver>(
                                MPI_COMM_WORLD, amgf_fsolver);
            subspacesolver->SetPrintLevel(0);
            amgfprec->SetFilteredSubspaceSolver(*subspacesolver);
            amgfprec->SetFilteredSubspaceTransferOperator(
               *contact_transfer);
            prec = std::move(amgfprec);
         }
         else if (!direct_linear_solve)
         {
            if (amgf && myid == 0 && print_level >= 2)
            {
               cout << "AMGF: global contact subspace is empty; using AMG." << endl;
            }
            auto amgprec = std::make_unique<HypreBoomerAMG>();
            ConfigureAMG(*amgprec, amg_coarsen_type, amg_agg_levels,
                         amg_interp_type, amg_relax_type);
            if (amg_print_options && myid == 0 && print_level >= 2)
            {
               PrintAMGOptions(*amgprec, "AMG", amg_agg_levels);
            }
            prec = std::move(amgprec);
         }

         // 9(f). Linear solver used inside the IP optimizer.
         if (!direct_linear_solve)
         {
            cgsolver.SetPreconditioner(*prec);
         }

         // 9(g). Interior-Point optimizer driving contact resolution.
         optimizer_ptr = std::make_unique<IPSolver>(&contact);
         IPSolver &optimizer = *optimizer_ptr;
         optimizer.SetTol(1e-6);
         optimizer.SetMaxIter(opt_maxit);
         optimizer.SetLinearSolver(direct_linear_solve ?
                                   static_cast<Solver *>(bulk_solver.get()) : &cgsolver);
         optimizer.SetPrintLevel(print_level == 3 ? 1 : 0);
         optimizer.SetPrintLinearSolverStatistics(print_level >= 2);

         // Initial guess = previous reference configuration.
         x_gf.SetTrueVector();
         x0.Set(1.0, xref);

         // Solve for the next step displacement (xf).
         xf = 0.0;
         optimizer.Mult(x0, xf);
         total_ip_steps += optimizer.GetNumIPSteps();
         total_newton_solves += optimizer.GetNumIterations();
         for (int iterations : optimizer.GetLinearSolverIterations())
         {
            if (iterations > 0) { total_cg_iterations += iterations; }
         }
         if (!optimizer.GetConverged() || !update_geometry) { break; }

         // Sequential quadratic contact solves refresh both the Neo-Hookean
         // tangent and the mortar search on the deformed configuration. Limit
         // updates to a third of the body's thickness to keep large initial
         // corrections from inverting elements before the next linearization.
         Vector correction(xf); correction.UseDevice(true); correction -= xref;
         update_norm = GlobalLpNorm(infinity(), correction.Normlinf(), MPI_COMM_WORLD);
         geometry_updates = update + 1;
         if (myid == 0 && print_level >= 2)
         {
            cout << "Contact geometry update " << update + 1
                 << ": |du|_inf = " << update_norm
                 << ", gap rows = " << contact.GetGlobalNumGapRows() << endl;
         }
         if (update_norm <= contact_update_tol)
         {
            geometry_converged = true;
            break;
         }
         if (allow_inexact_contact_updates && update + 1 == max_updates)
         {
            // Keep the converged IP solution and its linearized feasibility.
            // xref stays at the last linearization point for gap evaluation.
            break;
         }
         const real_t damping = std::min(scenario.MaxDamping(),
                                         scenario.MaxCorrection()/update_norm);
         xf.Set(1.0, xref);
         xf.Add(damping, correction);
         xref.Set(1.0, xf);
         x_gf.SetFromTrueDofs(xref);
         add(ref_coords, x_gf, new_coords);
      }
      IPSolver &optimizer = *optimizer_ptr;

      IPSolveSummary result;
      result.load_step = i;
      result.converged = optimizer.GetConverged() && geometry_converged;
      result.geometry_converged = geometry_converged;
      result.accepted = result.converged ||
                        (allow_inexact_contact_updates && optimizer.GetConverged() &&
                         geometry_updates == max_updates);
      result.termination_reason = optimizer.GetConverged() && !geometry_converged
                                  ? (result.accepted ? "accepted inexact contact geometry" :
                                     "maximum contact geometry updates")
                                  : optimizer.GetTerminationReason();
      result.ip_steps = total_ip_steps;
      result.newton_solves = total_newton_solves;
      result.cg_iterations = total_cg_iterations;
      result.optimality_error = optimizer.GetOptimalityError();
      result.geometry_updates = geometry_updates;
      result.geometry_error = update_norm;

      if (!result.accepted)
      {
         if (Mpi::Root() && print_level > 0)
         {
            if (print_level >= 2)
            {
               PrintIPSolveResult(optimizer, result);
               mfem::err << "Contact solve failed at load step " << i << "." << endl;
            }
            solve_results.push_back(result);
            PrintIPSummary(solve_results, total_steps);
         }
         return 2;
      }

      // Update bounds using the total load-step displacement, including all
      // geometry corrections rather than only the last quadratic subproblem.
      Vector dx(xf); dx.UseDevice(true); dx -= load_start;
      if (bound_constraints)
      {
         contact.SetDisplacement(dx, (i>=activation_step));
      }

      // 9(h). Report objective values, dofs and IP iteration counts.
      int eval_err;
      real_t Einitial = update_geometry ? nonlinear_initial_energy
                        : contact.GetEnergy(x0, eval_err);
      real_t Efinal = update_geometry ? prob.GetEnergy(xf)
                      : contact.GetEnergy(xf, eval_err);
      if (update_geometry)
      {
         // Size by the mortar gap rows only: once bounds are active the
         // constraint Jacobian also stacks 2*ndofs bound rows.
         Vector gap(contact.GetNumGapRows());
         gap.UseDevice(true);
         contact.EvaluateGap(xf, gap);
         const real_t local_min_gap = gap.Size() ? gap.Min() : infinity();
         real_t min_gap;
         MPI_Allreduce(&local_min_gap, &min_gap, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
         if (myid == 0 && print_level >= 2)
         {
            cout << scenario.Name() << ": gap rows = " <<
                 contact.GetGlobalNumGapRows()
                 << ", minimum mortar gap = ";
            if (IsFinite(min_gap)) { cout << min_gap; }
            else { cout << "n/a (no candidate pairs)"; }
            cout << endl;
         }
      }

      if (Mpi::Root() && print_level > 0)
      {
         result.has_energy = true;
         result.initial_energy = Einitial;
         result.final_energy = Efinal;
         if (print_level >= 2) { PrintIPSolveResult(optimizer, result); }
         solve_results.push_back(result);
      }

      // 9(i). Visualization outputs.
      x_gf.SetFromTrueDofs(xf);
      if (print_level >= 2) { scenario.PrintState(x_gf); }
      add(ref_coords,x_gf,new_coords);
      pmesh_copy.SetNodes(new_coords);
      xcopy_gf = x_gf;
      xcopy_gf.SetTrueVector();
      if (paraview)
      {
         paraview_dc->SetCycle(i+1);
         paraview_dc->SetTime(real_t(i+1));
         paraview_dc->Save();
      }

      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh_copy << x_gf << flush;

         if (i == total_steps - 1)
         {
            pmesh.MoveNodes(x_gf);
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock_final(vishost, visport);
            sol_sock_final << "parallel " << num_procs << " " << myid << "\n";
            sol_sock_final.precision(8);
            sol_sock_final << "solution\n" << pmesh << x_gf << flush;
         }
      }

      // 9(j). Update RHS/load for the next step (if any).
      if (i == total_steps-1) { break; }
      prob.UpdateRHS();
   }

   if (Mpi::Root() && print_level > 0)
   {
      PrintIPSummary(solve_results, total_steps);
   }
   return 0;
}
