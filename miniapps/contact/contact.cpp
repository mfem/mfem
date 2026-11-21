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

//                       MFEM Contact Miniapp

// Compile with: make contact

// Sample runs
// Problem 0: two-block (linear elasticity)
// mpirun -np 4 ./contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 0 -amgf -amgf-fsolver auto
// mpirun -np 4 ./contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 0 -no-amgf

// Problem 1: ironing (linear elasticity)
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -amgf -amgf-fsolver auto
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -no-amgf

// Problem 2: beam–sphere (linear or non-linear elasticity)
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -lin -amgf -amgf-fsolver auto
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -lin -no-amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -nonlin -amgf -amgf-fsolver auto
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -nonlin -no-amgf

// Problem 3: onion, 9 nested spherical shells (linear elasticity)
// mpirun -np 4 ./contact -prob 3 -sr 0 -pr 0 -tr 2 -nsteps 1 -msteps 0 -amgf -amgf-fsolver auto
// mpirun -np 4 ./contact -prob 3 -sr 0 -pr 0 -tr 2 -nsteps 1 -msteps 0 -no-amgf

// Problem 4: folded U-shaped strip (nonlinear self-contact, loading/unloading)
// mpirun -np 4 ./contact -prob 4 -sr 0 -pr 0 -tr 2 -nsteps 12 -msteps 12 -u-pressure 0.06 -amgf -amgf-fsolver auto -d cpu -no-vis
// ./run-contact-bound.sh --ranks 4 --cores 4 -- -prob 4 -sr 0 -pr 0 -tr 2 -nsteps 12 -msteps 12 -u-pressure 0.06 -amgf -amgf-fsolver mumps -d hip -hypre-gpu-aware-mpi -no-vis -paraview

// Problem 5: compressed elastic ring (nonlinear self-contact, loading/unloading)
// mpirun -np 4 ./contact -prob 5 -sr 0 -pr 0 -tr 2 -nsteps 4 -msteps 2 -ring-pressure 2.5 -amgf -amgf-fsolver auto -d cpu -no-vis
// ./run-contact-bound.sh --ranks 4 --cores 4 -- -prob 5 -sr 0 -pr 0 -tr 2 -nsteps 4 -msteps 2 -ring-pressure 2.5 -amgf -amgf-fsolver mumps -d hip -hypre-gpu-aware-mpi -no-vis -paraview

// Problem 6: C-clip snap-fit over a fixed pin (nonlinear, insertion/removal)
// ./run-contact-bound.sh --ranks 1 -- -prob 6 -sr 0 -nsteps 32 -msteps 32 -amgf -amgf-fsolver mumps -d cuda -no-vis -paraview

// Output: -pl 0 for silence, -pl 1 for the final IP summary,
//         -pl 2 for solver tables/results (default), -pl 3 for debug diagnostics.

// Description:
// This miniapp solves benchmark frictionless contact problems using a
// self-contained Interior Point (IP) optimizer. Contact constraints
// are supplied by Tribol. The linear systems inside the IP iterations
// are solved with PCG, preconditioned by either standard HypreBoomerAMG
// or AMG with Filtering (AMGF). AMGF enhances AMG by applying an additional
// subspace-correction step on a small subspace associated with the contact interface.

// Notes:
//  1. Non-linear elasticity is supported for -prob 2 and the self-contact
//     and snap-fit examples (-prob 4/5/6), which always use nonlinear elasticity.
//  2. The meshes for -prob 0,1,3,4,5,6 are generated on the fly. For -prob 2,
//     the mesh is read from ./meshes/beam-sphere.mesh.
//  3. AMGF requires a sparse direct solver for the filtered subspace; build
//     MFEM with MUMPS, SuperLU_DIST, STRUMPACK, MKL CPardiso, or cuDSS.
//     A direct solver is needed only when the global contact subspace is
//     nonempty; an empty subspace falls back to ordinary AMG.
//  4. This miniapp requires MFEM build with Tribol, an open source contact mechanics library
//     available at https://github.com/LLNL/Tribol.

#include "mfem.hpp"
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ip.hpp"
#include "solver_utils.hpp"
#include "contact_output.hpp"

using namespace std;
using namespace mfem;

enum problem_name
{
   twoblock,
   ironing,
   beamsphere,
   onion,
   foldedstrip,
   ring,
   snapfit
};

Mesh * GetProblemMesh(problem_name prob_name, bool snap_fit_hex);
Mesh * MakeOnionMesh();
Mesh * MakeFoldedStripMesh();
Mesh * MakeRingMesh();
Mesh * MakeSnapFitMesh(bool hexahedra);

namespace folded_strip
{
constexpr real_t length = 6.0;
constexpr real_t width = 0.9;
constexpr real_t thickness = 0.3;
constexpr real_t radius = 0.72;
constexpr real_t gap = 2 * radius - thickness;
constexpr int support = 1;
constexpr int load = 2;
constexpr int lower_contact = 3;
constexpr int upper_contact = 4;
constexpr int free_surface = 5;
}

namespace snap_fit
{
constexpr real_t inner_radius = 1.0;
constexpr real_t thickness = 0.2;
constexpr real_t width = 0.6;
constexpr real_t pin_radius = inner_radius;
constexpr real_t offset = 1.9;
constexpr int angular_cells = 64;
constexpr int mouth_cells = 16;
constexpr real_t tip_angle = 35.0 * M_PI / 180.0;
constexpr int clip_body = 1;
constexpr int pin_body = 2;
constexpr int grip = 1;
constexpr int clip_contact = 2;
constexpr int free_surface = 3;
constexpr int pin_ends = 4;
constexpr int pin_contact = 5;

real_t Angle(real_t index)
{
   if (index <= angular_cells)
   {
      return tip_angle + (2*M_PI - 2*tip_angle)*index/angular_cells;
   }
   return 2*M_PI - tip_angle + 2*tip_angle*(index - angular_cells)/mouth_cells;
}

void GetBodyDofs(ParFiniteElementSpace &fes, int attr, Array<int> &dofs);
void TranslateClip(ParGridFunction &displacement, const Array<int> &dofs,
                   real_t distance);
void PrintState(const ParGridFunction &displacement,
                const Array<int> &clip_dofs,
                const Array<int> &pin_dofs);
}

template <typename T>
void PrintAMGOption(const char *option, const T &value)
{
   const int option_width = 28;
   cout << "  " << left << setw(option_width) << option << value << endl;
}

void PrintAMGIntOption(const char *option, HYPRE_Solver solver,
                       HYPRE_Int (*getter)(HYPRE_Solver, HYPRE_Int *))
{
   HYPRE_Int value = 0;
   const HYPRE_Int ierr = getter(solver, &value);
   PrintAMGOption(option, ierr ? "unavailable" : to_string(value));
}

void PrintAMGRealOption(const char *option, HYPRE_Solver solver,
                        HYPRE_Int (*getter)(HYPRE_Solver, HYPRE_Real *))
{
   HYPRE_Real value = 0.0;
   const HYPRE_Int ierr = getter(solver, &value);
   PrintAMGOption(option, ierr ? "unavailable" : to_string(value));
}

void PrintAMGCycleIntOption(const char *option, HYPRE_Solver solver,
                            HYPRE_Int cycle,
                            HYPRE_Int (*getter)(HYPRE_Solver, HYPRE_Int *,
                                                HYPRE_Int))
{
   HYPRE_Int value = 0;
   const HYPRE_Int ierr = getter(solver, &value, cycle);
   PrintAMGOption(option, ierr ? "unavailable" : to_string(value));
}

void PrintAMGOptions(const HypreBoomerAMG &amg,
                     const char *preconditioner,
                     int amg_agg_levels,
                     const char *amgf_fsolver = nullptr)
{
   const int option_width = 28;
   HYPRE_Solver solver = amg;

   cout << "BoomerAMG options from configured object:" << endl
        << "  " << left << setw(option_width) << "option" << "value" << endl
        << "  " << left << setw(option_width) << "preconditioner"
        << preconditioner << endl;

   if (amgf_fsolver)
   {
      PrintAMGOption("filtered-solver", amgf_fsolver);
   }

   PrintAMGIntOption("num-functions", solver, HYPRE_BoomerAMGGetNumFunctions);
   PrintAMGIntOption("converge-type", solver, HYPRE_BoomerAMGGetConvergeType);
   PrintAMGRealOption("tol", solver, HYPRE_BoomerAMGGetTol);
   PrintAMGIntOption("max-iter", solver, HYPRE_BoomerAMGGetMaxIter);
   PrintAMGIntOption("max-coarse-size", solver,
                     HYPRE_BoomerAMGGetMaxCoarseSize);
   PrintAMGIntOption("min-coarse-size", solver,
                     HYPRE_BoomerAMGGetMinCoarseSize);
   PrintAMGIntOption("max-levels", solver, HYPRE_BoomerAMGGetMaxLevels);
   PrintAMGIntOption("coarsen-cut-factor", solver,
                     HYPRE_BoomerAMGGetCoarsenCutFactor);
   PrintAMGRealOption("strong-threshold", solver,
                      HYPRE_BoomerAMGGetStrongThreshold);
   PrintAMGRealOption("strong-threshold-r", solver,
                      HYPRE_BoomerAMGGetStrongThresholdR);
   PrintAMGRealOption("filter-threshold-r", solver,
                      HYPRE_BoomerAMGGetFilterThresholdR);
   PrintAMGRealOption("max-row-sum", solver, HYPRE_BoomerAMGGetMaxRowSum);
   PrintAMGIntOption("coarsen-type", solver, HYPRE_BoomerAMGGetCoarsenType);
   PrintAMGOption("aggressive-levels", amg_agg_levels);
   HYPRE_Int interp_type = 0;
   const HYPRE_Int interp_ierr =
      hypre_BoomerAMGGetInterpType((void *)solver, &interp_type);
   PrintAMGOption("interpolation-type",
                  interp_ierr ? "unavailable" : to_string(interp_type));
   PrintAMGIntOption("measure-type", solver, HYPRE_BoomerAMGGetMeasureType);
   PrintAMGRealOption("trunc-factor", solver, HYPRE_BoomerAMGGetTruncFactor);
   PrintAMGIntOption("p-max-elmts", solver, HYPRE_BoomerAMGGetPMaxElmts);
   PrintAMGIntOption("cycle-type", solver, HYPRE_BoomerAMGGetCycleType);
   PrintAMGIntOption("f-cycle", solver, HYPRE_BoomerAMGGetFCycle);
   PrintAMGIntOption("additive", solver, HYPRE_BoomerAMGGetAdditive);
   PrintAMGIntOption("mult-additive", solver, HYPRE_BoomerAMGGetMultAdditive);
   PrintAMGIntOption("simple", solver, HYPRE_BoomerAMGGetSimple);
   PrintAMGIntOption("seq-threshold", solver, HYPRE_BoomerAMGGetSeqThreshold);
   PrintAMGIntOption("redundant", solver, HYPRE_BoomerAMGGetRedundant);
   PrintAMGCycleIntOption("num-sweeps-down", solver, 1,
                          HYPRE_BoomerAMGGetCycleNumSweeps);
   PrintAMGCycleIntOption("num-sweeps-up", solver, 2,
                          HYPRE_BoomerAMGGetCycleNumSweeps);
   PrintAMGCycleIntOption("num-sweeps-coarse", solver, 3,
                          HYPRE_BoomerAMGGetCycleNumSweeps);
   PrintAMGCycleIntOption("relax-type-down", solver, 1,
                          HYPRE_BoomerAMGGetCycleRelaxType);
   PrintAMGCycleIntOption("relax-type-up", solver, 2,
                          HYPRE_BoomerAMGGetCycleRelaxType);
   PrintAMGCycleIntOption("relax-type-coarse", solver, 3,
                          HYPRE_BoomerAMGGetCycleRelaxType);
   PrintAMGIntOption("smooth-type", solver, HYPRE_BoomerAMGGetSmoothType);
   PrintAMGIntOption("smooth-num-levels", solver,
                     HYPRE_BoomerAMGGetSmoothNumLevels);
   PrintAMGIntOption("smooth-num-sweeps", solver,
                     HYPRE_BoomerAMGGetSmoothNumSweeps);
   PrintAMGIntOption("variant", solver, HYPRE_BoomerAMGGetVariant);
   PrintAMGIntOption("overlap", solver, HYPRE_BoomerAMGGetOverlap);
   PrintAMGIntOption("domain-type", solver, HYPRE_BoomerAMGGetDomainType);
   PrintAMGRealOption("schwarz-relax-weight", solver,
                      HYPRE_BoomerAMGGetSchwarzRlxWeight);
   PrintAMGIntOption("print-level", solver, HYPRE_BoomerAMGGetPrintLevel);
   PrintAMGIntOption("logging", solver, HYPRE_BoomerAMGGetLogging);
   PrintAMGIntOption("debug-flag", solver, HYPRE_BoomerAMGGetDebugFlag);
   PrintAMGIntOption("post-interp-type", solver,
                     HYPRE_BoomerAMGGetPostInterpType);
   PrintAMGRealOption("jacobi-trunc-threshold", solver,
                      HYPRE_BoomerAMGGetJacobiTruncThreshold);

   cout << right;
}

void ConfigureAMG(HypreBoomerAMG &amg, int coarsen_type, int agg_levels,
                  int interp_type, int relax_type)
{
   amg.SetSystemsOptions(3);
   amg.SetCoarsening(coarsen_type);
   amg.SetAggressiveCoarsening(agg_levels);
   amg.SetInterpolation(interp_type);
   amg.SetPrintLevel(0);
   amg.SetRelaxType(relax_type);
}

namespace
{

struct IPSolveSummary
{
   int load_step;
   bool converged;
   bool accepted = false;
   bool geometry_converged = true;
   const char *termination_reason = "not run";
   int ip_steps;
   int newton_solves;
   long long cg_iterations = 0;
   int geometry_updates = 0;
   real_t geometry_error = 0.0;
   real_t optimality_error;
   bool has_energy = false;
   real_t initial_energy = 0.0;
   real_t final_energy = 0.0;
};

template <typename T>
void PrintResultField(std::ostream &out, const char *label, const T &value)
{
   out << "  " << std::left << std::setw(32) << label << " = " << value << '\n';
}

void PrintIPSolveResult(const IPSolver &optimizer, const IPSolveSummary &result)
{
   using namespace contact_output;
   std::ostringstream out;
   PrintLine(out);
   out << "  " << Center("IP SOLVE RESULT - LOAD STEP " +
                         std::to_string(result.load_step), report_width) << '\n';
   PrintLine(out);
   PrintResultField(out, "Optimizer status", optimizer.GetTerminationReason());
   PrintResultField(out, "Accepted IP steps", result.ip_steps);
   PrintResultField(out, "Newton solves", result.newton_solves);
   PrintResultField(out, "CG iterations", result.cg_iterations);
   PrintResultField(out, "Final stationarity error",
                    optimizer.GetStationarityError());
   PrintResultField(out, "Final feasibility error",
                    optimizer.GetFeasibilityError());
   PrintResultField(out, "Final complementarity error",
                    optimizer.GetComplementarityError());
   PrintResultField(out, "Final optimality error", result.optimality_error);
   if (result.geometry_updates)
   {
      PrintResultField(out, "Contact geometry updates", result.geometry_updates);
      PrintResultField(out, "Final displacement update", result.geometry_error);
      PrintResultField(out, "Contact geometry status", result.geometry_converged ?
                       "converged" : result.accepted ?
                       "inexact (accepted at update limit)" : "unconverged");
   }
   if (result.has_energy)
   {
      PrintResultField(out, "Initial Energy objective", result.initial_energy);
      PrintResultField(out, "Final Energy objective", result.final_energy);
   }
   PrintLine(out, '=');
   mfem::out << out.str() << std::flush;
}

void PrintIPSummary(const std::vector<IPSolveSummary> &results, int total_steps)
{
   using namespace contact_output;
   const std::array<int, 7> widths = {4, 9, 8, 13, 8, 13, 13};
   std::ostringstream out;
   PrintBanner(out, "FINAL IP SUMMARY");
   PrintTableRow(out, widths,
   {
      "Step", "Status", "IP steps", "Newton solves", "CG iters",
      "Optimality", "Final energy"
   });
   PrintTableSeparator(out, widths);
   long long ip_steps = 0, newton_solves = 0, cg_iterations = 0;
   int converged_steps = 0, accepted_steps = 0, inexact_steps = 0;
   for (const auto &result : results)
   {
      PrintTableRow(out, widths,
      {
         std::to_string(result.load_step),
         result.converged ? "converged" : result.accepted ? "inexact" : "failed",
         std::to_string(result.ip_steps),
         std::to_string(result.newton_solves),
         std::to_string(result.cg_iterations),
         Scientific(result.optimality_error),
         result.has_energy ? Scientific(result.final_energy) : "n/a"
      });
      ip_steps += result.ip_steps;
      newton_solves += result.newton_solves;
      cg_iterations += result.cg_iterations;
      converged_steps += result.converged;
      accepted_steps += result.accepted;
      inexact_steps += result.accepted && !result.converged;
   }
   PrintTableSeparator(out, widths);
   PrintResultField(out, "Converged load steps",
                    std::to_string(converged_steps) + " / " + std::to_string(total_steps));
   if (inexact_steps)
   {
      PrintResultField(out, "Accepted load steps",
                       std::to_string(accepted_steps) + " / " + std::to_string(total_steps));
      PrintResultField(out, "Inexact contact steps", inexact_steps);
   }
   if (!results.empty() && !results.back().accepted)
   {
      PrintResultField(out, "Termination reason", results.back().termination_reason);
   }
   PrintResultField(out, "Total accepted IP steps", ip_steps);
   PrintResultField(out, "Total Newton solves", newton_solves);
   PrintResultField(out, "Total CG iterations", cg_iterations);
   PrintLine(out, '=');
   // Summary-only mode suppresses library output during the solve. Enable this
   // stream just for the summary, then restore its previous state.
   const bool output_enabled = mfem::out.IsEnabled();
   mfem::out.Enable();
   mfem::out << out.str() << std::flush;
   if (!output_enabled) { mfem::out.Disable(); }
}

}

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   cout << "Contact miniapp is not supported in single precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif
   // 0. Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();

   problem_name prob_name;

   // 1. Command-line options.
   // Number of serial uniform refinements
   int sref = 1;
   // Number of parallel uniform refinements
   int pref = 0;
   // Enable/disable GLVis visualization
   bool visualization = true;
   // Enable/disable ParaView output
   bool paraview = false;
   // Problem choice (0 through 6)
   int prob_no = 0;
   // number of times steps in the z-direction (problem 0)
   int nsteps = -1; // 32 for snap-fit, 20 for self-contact, 1 otherwise
   // number of times steps in the x-direction (problem 1)
   int msteps = -1; // resolved to 32 for snap-fit, 0 otherwise
   // Maximum number of iterations for the optimizer
   int opt_maxit = 100;
   // Driver output: silent, final summary, solver tables/results, or debug.
   int print_level = 2;
   // The self-contact and snap-fit examples always use nonlinear elasticity.
   bool nonlinear = false;
   real_t strip_pressure = 0.06;
   real_t ring_pressure = 2.5;
   real_t clip_travel = snap_fit::offset;
   bool snap_fit_hex = false;
   int contact_max_updates = 50;
   real_t contact_update_tol = 1.e-6;
   bool allow_inexact_contact_updates = false;
   // Tribol search proximity ratio
   real_t tribol_ratio = -1.0; // 2 for updated contact geometry, 8 otherwise
   // Enable/disable AMGF preconditioner (default is AMG)
   bool amgf = false;

   // Direct solver for AMGF filtered subspace
   // Choices:"auto", "mumps", "cpardiso", "superlu", "strumpack", "cudss"
   const char *amgf_fsolver = "auto";
   // GPU-oriented BoomerAMG coarsening/interpolation with Chebyshev smoothing.
   int amg_coarsen_type = 8;
   int amg_agg_levels = 0;
   int amg_interp_type = 6;
   int amg_relax_type = 16;
   bool amg_print_options = false;
   bool hypre_gpu_aware_mpi = false;
   // Device configuration string.
   const char *device_config = "cpu";
   // Use full assembly for the linear elasticity stiffness matrix.
   bool elasticity_fa = false;


   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&prob_no, "-prob", "--problem-number",
                  "Choice of problem:"
                  "0: two-block problem"
                  "1: ironing problem"
                  "2: beam-sphere problem"
                  "3: onion problem "
                  "4: folded U-shaped strip (nonlinear self-contact) "
                  "5: compressed ring (nonlinear self-contact) "
                  "6: C-clip snap-fit over a fixed pin (nonlinear)");
   args.AddOption(&nonlinear, "-nonlin", "--nonlinear", "-lin",
                  "--linear", "Choice between linear and non-linear Elasticiy model.");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&opt_maxit, "-opt-maxit", "--optimizer-max-iterations",
                  "Maximum number of iterations for the optimizer.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Output level: 0 = silent, 1 = final IP summary only, "
                  "2 = solver tables/results (default), 3 = debug diagnostics.");
   args.AddOption(&nsteps, "-nsteps", "--nsteps",
                  "Number of steps.");
   args.AddOption(&msteps, "-msteps", "--msteps",
                  "Number of extra steps (unloading/removal for problems 4/5/6).");
   args.AddOption(&strip_pressure, "-u-pressure", "--strip-pressure",
                  "Maximum downward pressure on the folded strip's upper arm.");
   args.AddOption(&ring_pressure, "-ring-pressure", "--ring-pressure",
                  "Maximum inward reference pressure on the ring's upper outer arc.");
   args.AddOption(&clip_travel, "-clip-travel", "--clip-travel",
                  "Maximum insertion displacement of the snap-fit clip (default 1.9).");
   args.AddOption(&snap_fit_hex, "-hex", "--hexahedra", "-tet",
                  "--tetrahedra",
                  "Use hexahedra or tetrahedra for both snap-fit bodies "
                  "(problem 6 only; default: tetrahedra).");
   args.AddOption(&contact_max_updates, "-contact-max-updates",
                  "--contact-max-updates",
                  "Maximum geometry/material updates per nonlinear contact load step.");
   args.AddOption(&contact_update_tol, "-contact-update-tol",
                  "--contact-update-tolerance",
                  "Absolute displacement tolerance for nonlinear contact updates.");
   args.AddOption(&allow_inexact_contact_updates,
                  "-allow-inexact-contact-updates", "--allow-inexact-contact-updates",
                  "-require-converged-contact-updates", "--require-converged-contact-updates",
                  "Accept the last converged IP solution at the geometry update limit; "
                  "the IP solve must still converge (default: require geometry convergence).");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&amgf, "-amgf", "--amgf", "-no-amgf",
                  "--no-amgf", "Enable or disable AMG with Filtering solver.");
   args.AddOption(&amgf_fsolver, "-amgf-fsolver", "--amgf-filtered-solver",
                  "Direct solver for AMGF filtered subspace.\n"
                  "Choices: auto, mumps, cpardiso, superlu, strumpack, cudss "
                  "(CUDA only; use auto on CPU or HIP).");
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (print_level < 0 || print_level > 3)
   {
      if (myid == 0)
      {
         mfem::err << "Print level must be 0, 1, 2, or 3." << endl;
      }
      return 1;
   }
   const bool self_contact = prob_no == foldedstrip || prob_no == ring;
   const bool snap_fit_case = prob_no == snapfit;
   const bool update_geometry = self_contact || snap_fit_case;
   const char *contact_name = snap_fit_case ? "Snap-fit" :
                              prob_no == ring ? "Ring" : "Folded strip";
   if (nsteps == -1) { nsteps = snap_fit_case ? 32 : self_contact ? 20 : 1; }
   if (msteps == -1) { msteps = snap_fit_case ? 32 : 0; }
   if (tribol_ratio == -1.0) { tribol_ratio = update_geometry ? 2.0 : 8.0; }
   if (update_geometry) { nonlinear = true; }
   if (nsteps < 1 || msteps < 0 || sref < 0 || pref < 0 ||
       contact_max_updates < 1 || !IsFinite(contact_update_tol) ||
       contact_update_tol <= 0.0 || !IsFinite(strip_pressure) ||
       strip_pressure < 0.0 || !IsFinite(ring_pressure) || ring_pressure < 0.0 ||
       !IsFinite(clip_travel) || clip_travel < 0.0 ||
       !IsFinite(tribol_ratio) || tribol_ratio <= 0.0)
   {
      if (myid == 0)
      {
         mfem::err << "Require nsteps > 0, msteps/refinements >= 0, "
                   << "contact updates/tolerance > 0 and pressures/travel >= 0.\n";
      }
      return 1;
   }
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

   // Validate selection and convert to enum.
   MFEM_VERIFY(prob_no >= 0 &&
               prob_no <= 6, "Unknown test problem number: " << prob_no);

   prob_name = (problem_name)prob_no;

   // Beam-sphere, self-contact and snap-fit support nonlinear elasticity.
   if (nonlinear && prob_name != beamsphere && !update_geometry)
   {
      if (myid == 0 && print_level >= 2)
      {
         cout << "Non-linear elasticity is supported for beam-sphere, folded strip, ring and snap-fit"
              << endl;
         cout << "Switching to the linear model ..." << endl;
      }
      nonlinear = false;
   }
   if (nonlinear && elasticity_fa && myid == 0 && print_level >= 2)
   {
      cout << "Full assembly is only used by the linear elasticity model; "
           << "nonlinear reference assembly uses "
           << (Device::IsEnabled() ? "device kernels." : "CPU kernels.")
           << endl;
   }

   // Bound constraints are used only for the non-linear case
   // (activated later after a few load steps).
   // Self-contact uses repeated geometry updates rather than the beam-sphere's
   // displacement bounds, which would impede unloading/contact release.
   bool bound_constraints = nonlinear && !update_geometry;


   // 2. Get the problem mesh.
   std::unique_ptr<Mesh> mesh(GetProblemMesh(prob_name, snap_fit_hex));

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

   // 5. Define problem-dependent BCs and material properties.
   switch (prob_name)
   {
      case twoblock:
      case ironing:
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
         ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
         E[0] = 1.0;  E[1] = 1e3;
         nu[0] = 0.499;  nu[1] = 0.0;
         break;
      case beamsphere:
         ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(0);
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(1);
         ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(2);
         ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.4;
         break;
      case onion:
         // symmetry planes x = 0, y = 0, z = 0
         ess_bdr_attr.Append(21); ess_bdr_attr_comp.Append(0);
         ess_bdr_attr.Append(22); ess_bdr_attr_comp.Append(1);
         ess_bdr_attr.Append(23); ess_bdr_attr_comp.Append(2);
         E = 1.e3;
         nu = 0.33;
         break;
      case foldedstrip:
         // Clamp a short outer bend patch, leaving both arm tips free.
         ess_bdr_attr.Append(folded_strip::support);
         ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.3;
         break;
      case ring:
         // Clamp a short lower outer arc, leaving the inner arc free to contact.
         ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.3;
         break;
      case snapfit:
         ess_bdr_attr.Append(snap_fit::grip); ess_bdr_attr_comp.Append(-1);
         // Plane-strain clip motion keeps the extruded contact faces planar.
         ess_bdr_attr.Append(snap_fit::clip_contact); ess_bdr_attr_comp.Append(1);
         ess_bdr_attr.Append(snap_fit::free_surface); ess_bdr_attr_comp.Append(1);
         // All pin boundary nodes are fixed, including its contact surface.
         ess_bdr_attr.Append(snap_fit::pin_ends); ess_bdr_attr_comp.Append(-1);
         ess_bdr_attr.Append(snap_fit::pin_contact); ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.3;
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   // 6. Build the elasticity problem object (linear or non-linear).
   // It owns the FE space and will assemble the stiffness and RHS.
   ElasticityOperator prob(&pmesh, ess_bdr_attr,ess_bdr_attr_comp, E, nu,
                           nonlinear, elasticity_fa);

   ParFiniteElementSpace * fes = prob.GetFESpace();
   HYPRE_BigInt gndofs = fes->GlobalTrueVSize();
   if (myid == 0 && print_level >= 2)
   {
      mfem::out << "\n --------------------------------------" << endl;
      mfem::out << "  Global number of dofs = " << gndofs << endl;
      mfem::out << "  Elasticity assembly   = "
                << (nonlinear ? (Device::IsEnabled() ? "nonlinear device" :
                                 "nonlinear CPU") :
                    elasticity_fa ? "full" : "legacy") << endl;
      mfem::out << "  RHS assembly          = "
                << (prob.UsesFastRHSAssembly() ? "fast" : "legacy") << endl;
      mfem::out << " --------------------------------------" << endl;
   }

   // 7. Set up the contact interfaces for Tribol. Each interface is handled
   // by its own coupling scheme.
   std::vector<ContactInterface> contact_interfaces;
   switch (prob_name)
   {
      case twoblock:
      case  ironing:
         contact_interfaces.push_back({{3}, {4}});
         break;
      case beamsphere:
         contact_interfaces.push_back({{6, 9}, {7, 8}});
         break;
      case onion:
         // interfaces between consecutive shells: outer surface of shell k
         // (mortar, attr 2k) and inner surface of shell k+1 (non-mortar,
         // attr 2k+1)
         for (int a = 2; a <= 16; a += 2)
         {
            contact_interfaces.push_back({{a}, {a+1}});
         }
         break;
      case foldedstrip:
         // Disjoint facing patches of ONE connected body. The bend is excluded
         // so neighboring faces can never become spurious self-contact pairs.
         contact_interfaces.push_back({{folded_strip::lower_contact},
            {folded_strip::upper_contact}});
         break;
      case ring:
         // Opposite inner arcs of one ring; lateral arcs separate the patches
         // so neighboring elements are never paired against themselves.
         contact_interfaces.push_back({{3}, {4}});
         break;
      case snapfit:
         // Put the pressure field on the fixed, smooth non-mortar surface.
         // The clip's lips slide across it without rotating the pressure basis.
         contact_interfaces.push_back({{snap_fit::clip_contact},
            {snap_fit::pin_contact}});
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   // Displacement field (current step), and a copy for visualization.
   ParGridFunction x_gf(fes); x_gf = 0.0;
   Array<int> clip_dofs, pin_dofs;
   if (snap_fit_case)
   {
      snap_fit::GetBodyDofs(*fes, snap_fit::clip_body, clip_dofs);
      snap_fit::GetBodyDofs(*fes, snap_fit::pin_body, pin_dofs);
   }
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
         const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
         const std::time_t run_time = std::chrono::system_clock::to_time_t(seconds);
         const auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - seconds).count();
         std::ostringstream paraview_file_name;
         paraview_file_name << "contact-problem_" << prob_no;
         if (snap_fit_case)
         {
            paraview_file_name << (snap_fit_hex ? "_hex" : "_tet");
         }
         paraview_file_name << "_par_ref_" << pref
                            << "_ser_ref_" << sref
                            << "_nonlinear_" << nonlinear << '_'
                            << std::put_time(std::gmtime(&run_time), "%Y%m%d_%H%M%S")
                            << '_' << std::setfill('0') << std::setw(9)
                            << nanoseconds << 'Z';
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
      }
      int name_length = static_cast<int>(paraview_run_name.size());
      MPI_Bcast(&name_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
      paraview_run_name.resize(name_length);
      MPI_Bcast(paraview_run_name.data(), name_length, MPI_CHAR, 0, MPI_COMM_WORLD);
      paraview_dc = std::make_unique<ParaViewDataCollection>(
                       paraview_run_name, &pmesh_copy);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(2);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
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

   // Load magnitude for problems 2 (beam-sphere) and 3 (onion).
   real_t p = (prob_name == problem_name::onion) ? 30.0 : 40.0;
   ConstantCoefficient f(p);

   // 8. Construct the contact optimization problem wrapper. It sets up the
   // contact system using Tribol and provides operators for objective evaluation,
   // gradients, Hessians, and constraints. It also provides the filtered subspace
   // transfer operator used by AMGF
   OptContactProblem contact(&prob, contact_interfaces, tribol_ratio,
                             bound_constraints);

   int dim = pmesh.Dimension();
   Vector ess_values(dim); ess_values = 0.0;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
   }

   // 9. Time/load stepping loop.
   int total_steps = nsteps + msteps;
   std::vector<IPSolveSummary> solve_results;
   real_t previous_clip_travel = 0.0;
   for (int i = 0; i<total_steps; i++)
   {
      if (Mpi::Root() && print_level >= 2)
      {
         contact_output::PrintBanner(mfem::out,
                                     "Solving optimization problem for time step: " +
                                     std::to_string(i));
      }
      ess_values = 0.0;
      // 9(a). Apply problem-specific boundary conditions / loads for this step.
      switch (prob_name)
      {
         case twoblock:
         case ironing:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[5] = 1;
            }
            if (i < nsteps)
            {
               ess_values[2] = -1.0/1.4*(i+1)/nsteps;
            }
            else
            {
               ess_values[0] = 3.0/1.4*(i+1-nsteps)/msteps;
               ess_values[2] = -1.0/1.4;
            }
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            break;
         case beamsphere:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[2] = 1;
            }
            f.constant = -p*(i+1)/nsteps;
            prob.SetNeumannPressureData(f,ess_bdr);
            break;
         case onion:
            // equal pressure on the innermost (attr 1) and outermost
            // (attr 18) surfaces
            f.constant = -p*(i+1)/nsteps;
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[0] = 1;
            }
            prob.SetNeumannPressureData(f,ess_bdr);
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[17] = 1;
            }
            prob.SetNeumannPressureData(f,ess_bdr);
            break;
         case foldedstrip:
         case ring:
         {
            const real_t load_fraction = i < nsteps ? real_t(i+1)/nsteps
                                         : 1.0 - real_t(i+1-nsteps)/msteps;
            f.constant = -(prob_name == ring ? ring_pressure : strip_pressure)
                         * load_fraction;
            const int load_attr = prob_name == ring ? 6 : folded_strip::load;
            ess_bdr = 0; ess_bdr[load_attr - 1] = 1;
            prob.SetNeumannPressureData(f, ess_bdr);
            if (myid == 0 && print_level >= 2)
            {
               cout << contact_name << " pressure = " << -f.constant << endl;
            }
            break;
         }
         case snapfit:
         {
            const real_t fraction = i < nsteps ? real_t(i+1)/nsteps
                                    : 1.0 - real_t(i+1-nsteps)/msteps;
            const real_t travel = clip_travel * fraction;
            ess_bdr = 0; ess_bdr[snap_fit::grip - 1] = 1;
            ess_values[0] = travel;
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            // Predict a rigid translation before linearizing the material and
            // contact. Updating only the grip would stretch/invert the rear
            // elements instead of carrying forward the previous clip shape.
            snap_fit::TranslateClip(x_gf, clip_dofs, travel - previous_clip_travel);
            add(ref_coords, x_gf, new_coords);
            previous_clip_travel = travel;
            if (myid == 0 && print_level >= 2)
            {
               cout << "Snap-fit clip travel = " << travel << endl;
            }
            break;
         }
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }

      // 9(b). Assemble the elasticity system with current BCs.
      prob.FormLinearSystem();

      // Store the current (reference) displacement in true dofs.
      Vector xref;
      xref.UseDevice(true);
      x_gf.GetTrueDofs(xref);
      const real_t nonlinear_initial_energy = update_geometry ?
                                              prob.GetEnergy(xref) : 0.0;

      std::unique_ptr<ParallelDirectSolver> subspacesolver;
      std::unique_ptr<Solver> prec;
      std::unique_ptr<IPSolver> optimizer_ptr;
      Vector x0(fes->GetTrueVSize()), xf(fes->GetTrueVSize());
      x0.UseDevice(true);
      xf.UseDevice(true);
      CGSolver cgsolver(MPI_COMM_WORLD);
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
         HypreParMatrix *contact_transfer = amgf ?
                                            contact.GetContactSubspaceTransferOperator() : nullptr;
         if (amgf && contact_transfer)
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
         else
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
         cgsolver.SetPreconditioner(*prec);

         // 9(g). Interior-Point optimizer driving contact resolution.
         optimizer_ptr = std::make_unique<IPSolver>(&contact);
         IPSolver &optimizer = *optimizer_ptr;
         optimizer.SetTol(1e-6);
         optimizer.SetMaxIter(opt_maxit);
         optimizer.SetLinearSolver(&cgsolver);
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
            total_cg_iterations += iterations;
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
                 << ", gap rows = " << contact.GetGlobalNumConstraints() << endl;
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
         const real_t max_correction = snap_fit_case ? snap_fit::thickness/3 :
                                       prob_name == ring ? 0.04 : 0.1;
         // Underrelax the clip's sliding contact updates across the pin facets.
         const real_t max_damping = snap_fit_case ? 0.5 : 1.0;
         const real_t damping = std::min(max_damping, max_correction/update_norm);
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

      // Update internal state for next step (for bound constraints).
      Vector dx(xf); dx.UseDevice(true); dx -= x0;
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
         Vector gap(contact.GetConstraintJacobian()->Height());
         gap.UseDevice(true);
         contact.EvaluateGap(xf, gap);
         const real_t local_min_gap = gap.Size() ? gap.Min() : infinity();
         real_t min_gap;
         MPI_Allreduce(&local_min_gap, &min_gap, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
         if (myid == 0 && print_level >= 2)
         {
            cout << contact_name << ": gap rows = " <<
                 contact.GetGlobalNumConstraints()
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
      if (snap_fit_case && print_level >= 2)
      {
         snap_fit::PrintState(x_gf, clip_dofs, pin_dofs);
      }
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

Mesh * GetProblemMesh(problem_name prob_name, bool snap_fit_hex)
{

   if (prob_name == problem_name::beamsphere)
   {
      return new Mesh("meshes/beam-sphere.mesh",1);
   }
   else if (prob_name == problem_name::onion)
   {
      return MakeOnionMesh();
   }
   else if (prob_name == foldedstrip)
   {
      return MakeFoldedStripMesh();
   }
   else if (prob_name == ring)
   {
      return MakeRingMesh();
   }
   else if (prob_name == snapfit)
   {
      return MakeSnapFitMesh(snap_fit_hex);
   }
   else // construct the two-block or ironing mesh
   {
      Mesh * combined_mesh = nullptr;
      constexpr real_t scale = 9.0/3.6;
      constexpr real_t lx0 = 3.6*scale, ly0 = 1.6*scale, lz0 = 1.0*scale;
      constexpr int nx0 = 24, ny0 = 12, nz0 = 6;
      Mesh mesh0 = Mesh::MakeCartesian3D(nx0, ny0, nz0, Element::HEXAHEDRON,
                                         lx0, ly0, lz0);

      // adjust boundary attributes
      for (int i = 0; i<mesh0.GetNBE(); i++)
      {
         int battr = mesh0.GetBdrElement(i)->GetAttribute();
         int new_battr;
         switch (battr)
         {
            case 1: new_battr = 2; break;
            case 6: new_battr = 3; break;
            default: new_battr = 1; break;
         }
         mesh0.SetBdrAttribute(i, new_battr);
      }
      mesh0.SetAttributes();
      if (prob_name == problem_name::twoblock)
      {
         constexpr real_t lx = 1.0, ly = 1.0, lz = 1.0;
         constexpr int nx = 5, ny = 5, nz = 5;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++)  { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 4; break;
               case 6: new_battr = 6; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         Vector shift(3);
         shift(0) = (ly0-ly)/3;
         shift(1) = (ly0-ly)/2;
         shift(2) = lz0;

         auto shift_map = [&](const Vector &x, Vector &y)
         {
            y.SetSize(3);
            y = x;
            y+=shift;
         };

         mesh.Transform(shift_map);

         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = new Mesh(mesh_array2, 2);


      }
      else if (prob_name == problem_name::ironing)
      {
         constexpr real_t lx = 1.6*scale, ly = 2.0*scale, lz = 0.2*scale;
         constexpr int nx = 16, ny = 20, nz = 2;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++) { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 6; break;
               case 6: new_battr = 4; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         constexpr real_t theta_arc = M_PI/2;
         constexpr real_t rin = lx / theta_arc - lz;
         constexpr real_t rout = lx / theta_arc + lz;

         constexpr real_t xshift = lx0/4;
         constexpr real_t yshift = 0.5*(ly0-ly);
         constexpr real_t zshift = rout + lz0;

         auto bend_map = [&](const Vector &x, Vector &y)
         {
            const real_t theta = -((x(0) / lx + 0.5) * theta_arc);
            const real_t r = rin + (x(2) + lz);
            y.SetSize(3);
            y(0) = xshift + r * std::cos(theta);
            y(1) = yshift + x(1);
            y(2) = zshift + r * std::sin(theta);
         };

         mesh.Transform(bend_map);
         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = new Mesh(mesh_array2, 2);
      }

      return combined_mesh;
   }
}

// Map one Cartesian solid along the lower arm, around the bend, and along the
// upper arm. Shared vertices at both transitions keep the body connected.
Mesh * MakeFoldedStripMesh()
{
   constexpr int arm_cells = 24, bend_cells = 24;
   constexpr int width_cells = 4, thickness_cells = 4;
   constexpr int path_cells = 2 * arm_cells + bend_cells;
   auto mesh = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                         path_cells, width_cells, thickness_cells,
                                         Element::HEXAHEDRON, real_t(path_cells),
                                         folded_strip::width,
                                         folded_strip::thickness));

   // Mark patches before mapping. The inner bend remains a free surface,
   // keeping the two contact patches disjoint.
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element *boundary = mesh->GetBdrElement(i);
      const int *vertices = boundary->GetVertices();
      real_t path = 0.0;
      for (int j = 0; j < boundary->GetNVertices(); j++)
      {
         path += mesh->GetVertex(vertices[j])[0];
      }
      path /= boundary->GetNVertices();
      int attr = folded_strip::free_surface;
      if (boundary->GetAttribute() == 1)
      {
         if (path < arm_cells) { attr = folded_strip::lower_contact; }
         if (path > arm_cells + bend_cells)
         {
            attr = folded_strip::upper_contact;
         }
      }
      if (boundary->GetAttribute() == 6)
      {
         if (path > arm_cells + bend_cells) { attr = folded_strip::load; }
         if (std::abs(path - (arm_cells + bend_cells / 2)) < 2.0)
         {
            attr = folded_strip::support;
         }
      }
      mesh->SetBdrAttribute(i, attr);
   }
   mesh->SetAttributes();

   auto bend_map = [](const Vector &logical, Vector &physical)
   {
      const real_t path = logical[0];
      const real_t radius = folded_strip::radius + logical[2] -
                            folded_strip::thickness / 2;
      physical.SetSize(3);
      physical[1] = logical[1] - folded_strip::width / 2;
      if (path <= arm_cells)
      {
         physical[0] = folded_strip::length * (1.0 - path / arm_cells);
         physical[2] = -radius;
      }
      else if (path <= arm_cells + bend_cells)
      {
         const real_t theta = -M_PI / 2 -
                              M_PI * (path - arm_cells) / bend_cells;
         physical[0] = radius * std::cos(theta);
         physical[2] = radius * std::sin(theta);
      }
      else
      {
         physical[0] = folded_strip::length *
                       (path - arm_cells - bend_cells) / arm_cells;
         physical[2] = radius;
      }
   };
   mesh->Transform(bend_map);
   return mesh.release();
}

// Circular elastic ring extruded across its width. Station indices wrap around
// to share the seam vertices, producing one closed connected body. Clockwise
// stations, increasing width and increasing radius give positive hex orientation.
Mesh * MakeRingMesh()
{
   constexpr int angular_cells = 64, width_cells = 2, radial_cells = 2;
   constexpr real_t radius = 1.0, thickness = 0.12, width = 0.6;
   auto mesh = std::make_unique<Mesh>(3, 0, 0);
   const auto vertex = [](int i, int j, int k)
   {
      return ((i % angular_cells) * (width_cells + 1) + j)
             * (radial_cells + 1) + k;
   };
   for (int i = 0; i < angular_cells; i++)
   {
      const real_t theta = 2.0 * M_PI * i / angular_cells;
      for (int j = 0; j <= width_cells; j++)
      {
         for (int k = 0; k <= radial_cells; k++)
         {
            const real_t r = radius + thickness * (real_t(k)/radial_cells - 0.5);
            mesh->AddVertex(r * std::cos(theta),
                            width * (real_t(j)/width_cells - 0.5),
                            -r * std::sin(theta));
         }
      }
   }
   for (int i = 0; i < angular_cells; i++)
   {
      for (int j = 0; j < width_cells; j++)
      {
         for (int k = 0; k < radial_cells; k++)
         {
            const int v[8] = {vertex(i,j,k), vertex(i+1,j,k),
                              vertex(i+1,j+1,k), vertex(i,j+1,k),
                              vertex(i,j,k+1), vertex(i+1,j,k+1),
                              vertex(i+1,j+1,k+1), vertex(i,j+1,k+1)
                             };
            mesh->AddHex(v, 1);
            if (k == 0)
            {
               const int attr = i >= 8 && i < 24 ? 3 :
                                i >= 40 && i < 56 ? 4 : 2;
               mesh->AddBdrQuad(v[3], v[2], v[1], v[0], attr);
            }
            if (k == radial_cells - 1)
            {
               const int attr = i >= 14 && i < 18 ? 1 :
                                i >= 44 && i < 52 ? 6 : 2;
               mesh->AddBdrQuad(v[4], v[5], v[6], v[7], attr);
            }
            if (j == 0) { mesh->AddBdrQuad(v[0], v[1], v[5], v[4], 2); }
            if (j == width_cells - 1)
            {
               mesh->AddBdrQuad(v[3], v[7], v[6], v[2], 2);
            }
         }
      }
   }
   mesh->FinalizeHexMesh(1, 1);
   return mesh.release();
}

// Open annular clip and a separate solid pin. Their vertices are deliberately
// disjoint: the pin surface is fixed and Tribol couples it to the moving clip.
Mesh * MakeSnapFitMesh(bool hexahedra)
{
   constexpr int angular_cells = snap_fit::angular_cells;
   constexpr int width_cells = 1, radial_cells = 4;
   constexpr real_t tip_angle = snap_fit::tip_angle;
   const Element::Type element_type = hexahedra ? Element::HEXAHEDRON :
                                      Element::TETRAHEDRON;
   auto clip = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                         angular_cells, width_cells, radial_cells,
                                         element_type, real_t(angular_cells),
                                         snap_fit::width, snap_fit::thickness));
   for (int i = 0; i < clip->GetNBE(); i++)
   {
      const Element &face = *clip->GetBdrElement(i);
      real_t path = 0.0;
      for (int j = 0; j < face.GetNVertices(); j++)
      {
         path += clip->GetVertex(face.GetVertices()[j])[0];
      }
      path /= face.GetNVertices();
      const real_t theta = snap_fit::Angle(path);
      int attr = snap_fit::free_surface;
      if (face.GetAttribute() == 1)
      {
         attr = snap_fit::clip_contact;
      }
      if (face.GetAttribute() == 3 || face.GetAttribute() == 5)
      {
         attr = snap_fit::clip_contact;
      }
      if (face.GetAttribute() == 6)
      {
         // The outer lip patches also touch the pin during initial entry.
         if (theta < M_PI/3 || theta > 5*M_PI/3)
         {
            attr = snap_fit::clip_contact;
         }
         if (std::abs(path - angular_cells/2) < 2.0)
         {
            attr = snap_fit::grip;
         }
      }
      clip->SetBdrAttribute(i, attr);
   }
   clip->SetAttributes();
   auto clip_map = [](const Vector &logical, Vector &physical)
   {
      const real_t theta = snap_fit::Angle(logical[0]);
      const real_t r = snap_fit::inner_radius + logical[2];
      // Round each lip with a semicircular end cap. Blend the cap into the
      // neighboring stations so contact normals turn smoothly around the nose.
      const real_t half = snap_fit::thickness/2;
      const real_t radial_offset = logical[2] - half;
      const real_t blend = std::max(real_t(0.0), real_t(1.0) -
                                    std::min(theta - tip_angle,
                                             2*M_PI - tip_angle - theta)/0.3);
      const real_t nose = std::sqrt(std::max(real_t(0.0),
                                             half*half - radial_offset*radial_offset))*blend;
      const real_t tangent = theta < M_PI ? -nose : nose;
      physical.SetSize(3);
      physical[0] = -snap_fit::offset + r*std::cos(theta) - tangent*std::sin(theta);
      physical[1] = logical[1] - snap_fit::width/2;
      physical[2] = -r*std::sin(theta) - tangent*std::cos(theta);
   };
   clip->Transform(clip_map);

   // Tetrahedra give triangular contact faces with Tribol's direct inverse map.
   // Hexahedral contact faces use its quadrilateral Newton iteration.
   // A square-to-disk map gives a solid pin without a singular axial cell.
   // End faces and the whole curved surface are fixed; no stiffness contrast
   // or penalty parameter is needed to hold the contact surface stationary.
   // Match the pin's boundary stations to the clip's inner arc. Equal radii
   // then give coincident facets when seated, including on the coarse mesh.
   constexpr int pin_cells = (snap_fit::angular_cells + snap_fit::mouth_cells)/4;
   auto pin = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                        pin_cells, width_cells, pin_cells,
                                        element_type, 2.0,
                                        snap_fit::width, 2.0));
   for (int i = 0; i < pin->GetNE(); i++)
   {
      pin->GetElement(i)->SetAttribute(snap_fit::pin_body);
   }
   for (int i = 0; i < pin->GetNBE(); i++)
   {
      const int attr = pin->GetBdrElement(i)->GetAttribute();
      pin->SetBdrAttribute(i, attr == 2 || attr == 4 ?
                           snap_fit::pin_ends : snap_fit::pin_contact);
   }
   pin->SetAttributes();
   auto pin_map = [](const Vector &logical, Vector &physical)
   {
      const real_t x = logical[0] - 1.0, z = logical[2] - 1.0;
      physical.SetSize(3);
      physical[1] = logical[1] - snap_fit::width/2;
      const real_t square_radius = std::max(std::abs(x), std::abs(z));
      if (square_radius == 0.0)
      {
         physical[0] = physical[2] = 0.0;
         return;
      }
      const real_t u = x/square_radius, v = z/square_radius;
      real_t station;
      if (u >= std::abs(v))
      {
         station = v <= 0.0 ? -pin_cells*v/2 : 4*pin_cells - pin_cells*v/2;
      }
      else if (-v >= std::abs(u)) { station = pin_cells - pin_cells*u/2; }
      else if (-u >= std::abs(v)) { station = 2*pin_cells + pin_cells*v/2; }
      else { station = 3*pin_cells + pin_cells*u/2; }
      // Boundary grid vertices have integer station indices. Snap roundoff
      // away so both bodies use exactly the same angle and radius there.
      if (square_radius == 1.0) { station = std::round(station); }
      const real_t theta = snap_fit::Angle(station);
      const real_t r = snap_fit::pin_radius * (square_radius == 1.0 ? 1.0 :
                                               std::sqrt(x*x + z*z - x*x*z*z));
      physical[0] = r*std::cos(theta);
      physical[2] = -r*std::sin(theta);
   };
   pin->Transform(pin_map);
   Mesh *parts[] = {clip.get(), pin.get()};
   return new Mesh(parts, 2);
}

void snap_fit::GetBodyDofs(ParFiniteElementSpace &fes, int attr,
                           Array<int> &dofs)
{
   Array<int> marker(fes.GetNDofs()), element_dofs;
   marker = 0;
   for (int i = 0; i < fes.GetNE(); i++)
   {
      if (fes.GetParMesh()->GetAttribute(i) != attr) { continue; }
      fes.GetElementDofs(i, element_dofs);
      for (int j : element_dofs) { marker[j] = 1; }
   }
   dofs.SetSize(0);
   for (int i = 0; i < marker.Size(); i++)
   {
      if (marker[i]) { dofs.Append(i); }
   }
}

void snap_fit::TranslateClip(ParGridFunction &displacement,
                             const Array<int> &dofs, real_t distance)
{
   const auto &fes = *displacement.ParFESpace();
   real_t *values = displacement.HostReadWrite();
   for (int dof : dofs) { values[fes.DofToVDof(dof, 0)] += distance; }
}

void snap_fit::PrintState(const ParGridFunction &displacement,
                          const Array<int> &clip_dofs, const Array<int> &pin_dofs)
{
   const auto &fes = *displacement.ParFESpace();
   const real_t *values = displacement.HostRead();
   real_t local[2] = {0.0, 0.0}, global[2];
   for (int dof : clip_dofs)
   {
      local[0] = std::max(local[0], std::abs(values[fes.DofToVDof(dof, 2)]));
   }
   for (int dof : pin_dofs)
   {
      for (int c = 0; c < 3; c++)
      {
         local[1] = std::max(local[1], std::abs(values[fes.DofToVDof(dof, c)]));
      }
   }
   MPI_Allreduce(local, global, 2, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 fes.GetComm());
   if (Mpi::Root())
   {
      cout << "Snap-fit: maximum transverse deflection = " << global[0]
           << ", fixed pin displacement = " << global[1] << endl;
   }
}

// Generate nine disconnected spherical octant shells. Distinct surface meshes
// at each shared radius are intentional: Tribol owns the eight interfaces.
Mesh * MakeOnionMesh()
{
   // Keep the element count close to the former mesh while giving neighboring
   // shells different surface resolutions for nonmatching mortar interfaces.
   constexpr int angular_cells[9] = {16, 17, 17, 17, 18, 18, 19, 19, 19};
   Mesh *mesh = new Mesh(3, 0, 0);

   for (int shell = 0; shell < 9; ++shell)
   {
      const int n = angular_cells[shell];
      const int radial_cells = shell < 4 ? 2 : 1;
      const int side = n + 1;
      const auto flat = [side](int i, int j, int k)
      {
         return (i * side + j) * side + k;
      };

      // Three square faces of the positive octant of a cube map to a sphere.
      // The shared edges use one vertex, while different shells do not share
      // vertices, including where their radii coincide.
      std::vector<int> surface_index(side * side * side, -1);
      std::vector<std::array<int, 3>> directions;
      for (int i = 0; i <= n; ++i)
      {
         for (int j = 0; j <= n; ++j)
         {
            for (int k = 0; k <= n; ++k)
            {
               if (i != n && j != n && k != n) { continue; }
               surface_index[flat(i, j, k)] = directions.size();
               directions.push_back({i, j, k});
            }
         }
      }

      std::vector<std::vector<int>> vertex(radial_cells + 1,
                                           std::vector<int>(directions.size()));
      for (int layer = 0; layer <= radial_cells; ++layer)
      {
         const real_t radius = 5.5 + 0.5 * shell +
                               0.5 * layer / radial_cells;
         for (int d = 0; d < static_cast<int>(directions.size()); ++d)
         {
            const auto &p = directions[d];
            const real_t scale = radius / std::sqrt(real_t(p[0] * p[0] +
                                                           p[1] * p[1] +
                                                           p[2] * p[2]));
            vertex[layer][d] = mesh->AddVertex(scale * p[0], scale * p[1],
                                               scale * p[2]);
         }
      }

      const auto face_point = [n](int face, int u, int v)
      {
         if (face == 0) { return std::array<int, 3> {n, u, v}; }
         if (face == 1) { return std::array<int, 3> {u, n, v}; }
         return std::array<int, 3> {u, v, n};
      };
      for (int face = 0; face < 3; ++face)
      {
         for (int u = 0; u < n; ++u)
         {
            for (int v = 0; v < n; ++v)
            {
               std::array<std::array<int, 3>, 4> p = {{
                     face_point(face, u, v), face_point(face, u + 1, v),
                     face_point(face, u + 1, v + 1), face_point(face, u, v + 1)
                  }
               };
               if (face == 1) { std::swap(p[1], p[3]); }

               for (int layer = 0; layer < radial_cells; ++layer)
               {
                  int inner[4], outer[4];
                  for (int c = 0; c < 4; ++c)
                  {
                     const int id = surface_index[flat(p[c][0], p[c][1], p[c][2])];
                     inner[c] = vertex[layer][id];
                     outer[c] = vertex[layer + 1][id];
                  }
                  mesh->AddHex(inner[0], inner[1], inner[2], inner[3],
                               outer[0], outer[1], outer[2], outer[3], shell + 1);
                  if (layer == 0)
                  {
                     mesh->AddBdrQuad(inner[3], inner[2], inner[1], inner[0],
                                      2 * shell + 1);
                  }
                  if (layer == radial_cells - 1)
                  {
                     mesh->AddBdrQuad(outer[0], outer[1], outer[2], outer[3],
                                      2 * shell + 2);
                  }
                  for (int edge = 0; edge < 4; ++edge)
                  {
                     const int next = (edge + 1) % 4;
                     for (int axis = 0; axis < 3; ++axis)
                     {
                        if (p[edge][axis] == 0 && p[next][axis] == 0)
                        {
                           mesh->AddBdrQuad(inner[edge], inner[next],
                                            outer[next], outer[edge], 21 + axis);
                        }
                     }
                  }
               }
            }
         }
      }
   }

   mesh->FinalizeHexMesh(1, 1);
   return mesh;
}
