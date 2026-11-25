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

#ifndef MFEM_CONTACT_SOLVER
#define MFEM_CONTACT_SOLVER

#include "mfem.hpp"
#include <limits>
#include <string>

namespace mfem
{

/// Outcome of solving one contact system with fixed geometry and boundary data.
struct ContactSolveResult
{
   bool converged = false; ///< Solver convergence and contact acceptance.
   /// Solution on success; initial displacement on failure.
   Vector displacement;
   int iterations = 0;           ///< Nonlinear corrections attempted.
   Array<int> linear_iterations; ///< Krylov iterations for each correction.
   /// Coupled residual norm, or NaN when the solver does not report this norm.
   real_t residual_norm = std::numeric_limits<real_t>::quiet_NaN();
   std::string message; ///< Failure reason; empty on success.
};

/// Solver interface shared by scalar IP and vector Coulomb contact.
class ContactSolver
{
public:
   virtual ~ContactSolver() = default;
   /** @brief Solve the contact problem formed by the caller.
       @param initial Last accepted displacement; boundary data are already set.
       @return Named outcome. Failure returns initial and restores any
       persistent multiplier input changed by the solver. Load and coordinate
       updates are the caller's responsibility. */
   virtual ContactSolveResult Solve(const Vector &initial) = 0;
};

} // namespace mfem
#endif
