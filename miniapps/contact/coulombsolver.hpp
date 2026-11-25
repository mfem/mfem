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

#ifndef MFEM_COULOMB_SOLVER
#define MFEM_COULOMB_SOLVER

#include "contactsolver.hpp"
#include "optcontactproblem.hpp"

namespace mfem
{

/// Damped Newton/FGMRES solver for a prepared frozen-geometry Coulomb system.
class CoulombSolver : public ContactSolver
{
private:
   /// Borrowed problem, which must outlive this solver.
   OptContactProblem &problem;
   /// Linear solve outcome; correction is empty on failure.
   struct LinearSolveResult
   {
      Vector correction;
      int iterations = 0;
      bool converged = false;
   };
   /// Global norm of locally owned residual entries.
   real_t Norm(const Vector &values) const;
   /** @brief Calculate a block-preconditioned Newton correction.
       @param displacement Current trial displacement.
       @param residual Coupled equations at that displacement and multiplier.
       @return Correction and iteration count. On failure, converged is false
       and correction is empty. */
   LinearSolveResult Correction(const Vector &displacement,
                                const Vector &residual);

public:
   /// Borrow a problem already configured for Coulomb mortar.
   explicit CoulombSolver(OptContactProblem &problem_) : problem(problem_)
   {
      MFEM_VERIFY(problem.UsesCoulomb(),
                  "CoulombSolver requires Coulomb mortar.");
   }
   /// Solve the current frozen system, restoring multiplier input on failure.
   ContactSolveResult Solve(const Vector &initial) override;
};

} // namespace mfem
#endif
