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

#include "driver_output.hpp"
#include "contact_output.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace std;
namespace mfem
{
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
                     const char *amgf_fsolver)
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
