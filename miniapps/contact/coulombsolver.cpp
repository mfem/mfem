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

#include "coulombsolver.hpp"
#include <cmath>

namespace mfem
{

real_t CoulombSolver::Norm(const Vector &values) const
{
   return std::sqrt(InnerProduct(problem.GetComm(), values, values));
}

CoulombSolver::LinearSolveResult
CoulombSolver::Correction(const Vector &displacement, const Vector &residual)
{
   auto jacobian = problem.CoulombJacobian(displacement);
   auto &stiffness = static_cast<HypreParMatrix &>(jacobian->GetBlock(0, 0));
   auto &j01 = static_cast<HypreParMatrix &>(jacobian->GetBlock(0, 1));
   auto &j10 = static_cast<HypreParMatrix &>(jacobian->GetBlock(1, 0));
   auto &j11 = static_cast<HypreParMatrix &>(jacobian->GetBlock(1, 1));
   Vector diagonal;
   stiffness.GetDiag(diagonal);
   HypreParMatrix scaled(j01);
   scaled.InvScaleRows(diagonal);
   auto product = std::unique_ptr<HypreParMatrix>(ParMult(&j10, &scaled, true));
   auto schur = std::unique_ptr<HypreParMatrix>(Add(1.0, j11, -1.0, *product));
   HypreBoomerAMG bulk_inverse(stiffness);
   bulk_inverse.SetSystemsOptions(3);
   bulk_inverse.SetPrintLevel(0);
#if MFEM_HYPRE_VERSION >= 21900
   HypreILU contact_inverse;
   contact_inverse.SetLevelOfFill(1);
   contact_inverse.SetPrintLevel(0);
   contact_inverse.SetOperator(*schur);
#else
   HypreSmoother contact_inverse(*schur, HypreSmoother::l1Jacobi);
#endif
   BlockLowerTriangularPreconditioner preconditioner(jacobian->RowOffsets());
   preconditioner.SetDiagonalBlock(0, &bulk_inverse);
   preconditioner.SetDiagonalBlock(1, &contact_inverse);
   preconditioner.SetBlock(1, 0, &j10);
   FGMRESSolver solver(problem.GetComm());
   solver.SetPreconditioner(preconditioner);
   solver.SetOperator(*jacobian);
   solver.SetRelTol(1.0e-9);
   solver.SetAbsTol(1.0e-11);
   solver.SetMaxIter(2000);
   solver.SetKDim(200);
   solver.SetPrintLevel(0);
   Vector rhs(residual);
   rhs *= -1.0;
   LinearSolveResult result;
   result.correction.SetSize(residual.Size());
   result.correction = 0.0;
   solver.Mult(rhs, result.correction);
   result.iterations = solver.GetNumIterations();
   result.converged = solver.GetConverged();
   if (!result.converged)
   {
      result.correction.SetSize(0);
   }
   return result;
}

ContactSolveResult CoulombSolver::Solve(const Vector &initial)
{
   constexpr int maximum_corrections = 30;
   constexpr int maximum_stagnant_iterations = 5;
   constexpr real_t residual_tolerance = 1.0e-7;
   constexpr real_t improvement_factor = 0.99;
   constexpr real_t armijo_factor = 1.0e-4;
   constexpr real_t minimum_step = 1.0 / 1048576.0;
   ContactSolveResult result;
   Vector displacement(initial);
   const Vector accepted_multiplier(problem.GetContactMultipliers());
   try
   {
      Vector prescribed;
      problem.GetElasticityOperator()->Getxrefbc(prescribed);
      for (int i : problem.GetElasticityOperator()->GetEssentialDofs())
      {
         displacement[i] = prescribed[i];
      }
      real_t best_norm = std::numeric_limits<real_t>::infinity();
      int stagnant_iterations = 0;
      Vector residual = problem.CoulombResidual(displacement);
      result.residual_norm = Norm(residual);
      while (true)
      {
         if (!std::isfinite(result.residual_norm))
         {
            result.message = "Nonfinite Coulomb residual.";
            break;
         }
         if (result.residual_norm <= residual_tolerance)
         {
            if (problem.CheckCoulombContact(displacement))
            {
               result.converged = true;
               result.displacement = displacement;
               result.message.clear();
               return result;
            }
            result.message = "Coulomb contact acceptance checks failed.";
            break;
         }
         if (result.iterations == maximum_corrections)
         {
            result.message = "Coulomb Newton iteration limit reached.";
            break;
         }
         // Reduce the load rather than repeat almost unchanged Newton trials.
         if (result.residual_norm < improvement_factor * best_norm)
         {
            best_norm = result.residual_norm;
            stagnant_iterations = 0;
         }
         else if (++stagnant_iterations >= maximum_stagnant_iterations)
         {
            result.message = "Coulomb Newton residual stagnated.";
            break;
         }
         auto linear = Correction(displacement, residual);
         result.linear_iterations.Append(linear.iterations);
         if (!linear.converged)
         {
            result.message = "Coulomb Newton linear solve did not converge.";
            break;
         }
         ++result.iterations;
         auto &traction = problem.GetContactMultipliers();
         const Vector base(displacement), multiplier(traction);
         Vector du(linear.correction, 0, displacement.Size());
         Vector dl(linear.correction, displacement.Size(), traction.Size());
         bool reduced = false;
         for (real_t alpha = 1.0; alpha >= minimum_step; alpha *= 0.5)
         {
            add(base, alpha, du, displacement);
            add(multiplier, alpha, dl, traction);
            Vector trial_residual = problem.CoulombResidual(displacement);
            const real_t trial_norm = Norm(trial_residual);
            if (std::isfinite(trial_norm) &&
                trial_norm <= (1.0 - armijo_factor * alpha) * result.residual_norm)
            {
               residual = std::move(trial_residual);
               result.residual_norm = trial_norm;
               reduced = true;
               break;
            }
         }
         if (!reduced)
         {
            result.message =
                "Coulomb Newton line search did not reduce residual.";
            break;
         }
      }
   }
   catch (const std::exception &error)
   {
      result.message = error.what();
   }
   problem.GetContactMultipliers() = accepted_multiplier;
   result.displacement = initial;
   return result;
}

} // namespace mfem
