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

#ifndef MFEM_CONTACT_DRIVER_OUTPUT_HPP
#define MFEM_CONTACT_DRIVER_OUTPUT_HPP
#include "ip.hpp"
#include <vector>
namespace mfem
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
void ConfigureAMG(HypreBoomerAMG &, int, int, int, int);
void PrintAMGOptions(const HypreBoomerAMG &, const char *, int,
                     const char * = nullptr);
void PrintIPSolveResult(const IPSolver &, const IPSolveSummary &);
void PrintIPSummary(const std::vector<IPSolveSummary> &, int);
}
#endif
