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

#include "ip.hpp"
#include "general/reducers.hpp"
#include "solver_utils.hpp"
#include "contact_output.hpp"

namespace mfem
{

namespace
{

void EnableDevice(BlockVector &v)
{
   v.UseDevice(true);
   for (int i = 0; i < v.NumBlocks(); i++)
   {
      v.GetBlock(i).UseDevice(true);
   }
}

void MultiplyWeights(const Vector &weights, const Vector &x, Vector &y)
{
   const real_t *weights_d = weights.Read();
   const real_t *x_d = x.Read();
   real_t *y_d = y.Write();
   mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y_d[i] = weights_d[i] * x_d[i];
   });
}

void UpdateDiagonal(HypreParMatrix &matrix, const Vector &diagonal)
{
   const int n = diagonal.Size();
   real_t *values = mfem::Write(matrix.GetDiagMemoryData(), n,
                                Device::IsEnabled());
   const real_t *diagonal_d = diagonal.Read();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      values[i] = diagonal_d[i];
   });
   matrix.HypreRead();
}

real_t FiniteNormlinf(const Vector &x, Array<real_t> &workspace)
{
   real_t norm = 0.0;
   const bool use_dev = x.UseDevice();
   const real_t *values = x.Read(use_dev);
   const real_t inf = infinity();
   reduce(x.Size(), norm, [=] MFEM_HOST_DEVICE (int i, real_t &value)
   {
      // fmax ignores NaNs, so map every nonfinite entry to infinity first.
      const real_t entry = values[i];
      value = fmax(value, entry > -inf && entry < inf ? fabs(entry) : inf);
   }, MaxReducer<real_t> {}, use_dev, workspace);
   return norm;
}

void AddBarrierGradient(const Vector &weights, const Vector &slack,
                        real_t mu, Vector &y)
{
   const real_t *weights_d = weights.Read();
   const real_t *slack_d = slack.Read();
   real_t *y_d = y.ReadWrite();
   mfem::forall(slack.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y_d[i] -= mu * ((1.0 / slack_d[i]) * weights_d[i]);
   });
}

void FillBarrierDiagonal(const BlockVector &x, const Vector &weights,
                         const Vector &z, real_t delta, Vector &diagonal)
{
   const int dim_u = x.GetBlock(0).Size();
   const real_t *weights_d = weights.Read();
   const real_t *z_d = z.Read();
   const real_t *x_d = x.Read();
   real_t *diagonal_d = diagonal.Write();
   mfem::forall(diagonal.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      diagonal_d[i] = (weights_d[i] * z_d[i]) / x_d[i + dim_u]
                      + delta * weights_d[i];
   });
}

void ComputeMultiplierDirection(const BlockVector &x,
                                const BlockVector &direction, const Vector &z,
                                real_t mu, Vector &z_direction)
{
   const int dim_u = x.GetBlock(0).Size();
   const real_t *z_d = z.Read();
   const real_t *direction_d = direction.Read();
   const real_t *x_d = x.Read();
   real_t *z_direction_d = z_direction.Write();
   mfem::forall(z.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      z_direction_d[i] = z_d[i]
                         + (z_d[i] * direction_d[i + dim_u] - mu)
                         / x_d[i + dim_u];
   });
   z_direction *= -1.;
}

real_t LocalMaxStepSize(const Vector &x, const Vector &direction, real_t tau,
                        Array<real_t> &workspace)
{
   real_t alpha_max = 1.0;
   const bool use_dev = x.UseDevice() || direction.UseDevice();
   const real_t *x_d = x.Read(use_dev);
   const real_t *direction_d = direction.Read(use_dev);
   reduce(x.Size(), alpha_max,
          [=] MFEM_HOST_DEVICE (int i, real_t &alpha)
   {
      if (direction_d[i] < 0.0)
      {
         alpha = fmin(alpha, -tau * x_d[i] / direction_d[i]);
      }
   },
   MinReducer<real_t> {}, use_dev, workspace);
   return alpha_max;
}

void ProjectMultipliers(const Vector &x, Vector &z, int dim_u,
                        real_t mu, real_t k_sig)
{
   const real_t *x_d = x.Read();
   real_t *z_d = z.ReadWrite();
   mfem::forall(z.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t zdual_i = z_d[i];
      const real_t zprimal_i = mu / x_d[i + dim_u];
      z_d[i] = fmax(fmin(zdual_i, k_sig * zprimal_i), zprimal_i / k_sig);
   });
}

real_t LocalWeightedLogSum(const BlockVector &x, const Vector &weights,
                           Array<real_t> &workspace)
{
   real_t log_sum = 0.0;
   const int dim_u = x.GetBlock(0).Size();
   const bool use_dev = x.UseDevice() || weights.UseDevice();
   const real_t *weights_d = weights.Read(use_dev);
   const real_t *x_d = x.Read(use_dev);
   reduce(weights.Size(), log_sum,
          [=] MFEM_HOST_DEVICE (int i, real_t &sum)
   {
      sum += weights_d[i] * log(x_d[dim_u + i]);
   },
   SumReducer<real_t> {}, use_dev, workspace);
   return log_sum;
}

real_t LocalComplementarityError(const BlockVector &x, const Vector &z,
                                 real_t mu, Array<real_t> &workspace)
{
   real_t max_error = 0.0;
   const int dim_u = x.GetBlock(0).Size();
   const bool use_dev = x.UseDevice() || z.UseDevice();
   const real_t *x_d = x.Read(use_dev);
   const real_t *z_d = z.Read(use_dev);
   const real_t inf = infinity();
   reduce(z.Size(), max_error,
          [=] MFEM_HOST_DEVICE (int i, real_t &error)
   {
      const real_t value = fabs(x_d[dim_u + i] * z_d[i] - mu);
      error = fmax(error, value < inf ? value : inf);
   },
   MaxReducer<real_t> {}, use_dev, workspace);
   return max_error;
}

}

IPSolver::IPSolver(OptContactProblem * problem_)
   : problem(problem_)
{
   // Options use the existing filter/barrier defaults from Wachter and Biegler,
   // and inertia-free regularization defaults from Chiang and Zavala.
   const auto &layout = problem->GetConstraintLayout();
   dimU = problem->GetDimU();
   dimM = layout.Size();
   dimC = dimM; // One slack variable per equality constraint.
   MFEM_VERIFY(layout.lumped_weights.Size() == dimM,
               "Constraint weights must match the explicit constraint layout.");

   comm = problem->GetComm();
   const Vector &volume_weights = problem->GetDisplacementWeights();
   Mvlump.SetSize(volume_weights.Size());
   Mvlump.UseDevice(true);
   Mvlump = 0.0;
   Mvlump.Set(1.0, volume_weights);

   block_offsetsumlz.SetSize(5);
   block_offsetsuml.SetSize(4);
   block_offsetsx.SetSize(3);

   block_offsetsumlz[0] = 0;
   block_offsetsumlz[1] = dimU; // u
   block_offsetsumlz[2] = dimM; // m
   block_offsetsumlz[3] = dimC; // lambda
   block_offsetsumlz[4] = dimM; // zl
   block_offsetsumlz.PartialSum();

   Mlump.SetSize(dimM); Mlump.UseDevice(true); Mlump = 0.0;
   Mlump.Set(1.0, layout.lumped_weights);

   for (int i = 0; i < block_offsetsuml.Size(); i++)
   {
      block_offsetsuml[i] = block_offsetsumlz[i];
   }
   for (int i = 0; i < block_offsetsx.Size(); i++)
   {
      block_offsetsx[i] = block_offsetsuml[i] ;
   }

   workspace.newton_rhs.Update(block_offsetsuml);
   EnableDevice(workspace.newton_rhs);
   for (BlockVector *v :
        {
           &workspace.objective_gradient, &workspace.lagrangian_gradient,
           &workspace.scaled_gradient, &workspace.hessian_action,
           &workspace.line_search_trial, &workspace.line_search_gradient
        })
   {
      v->Update(block_offsetsx);
      EnableDevice(*v);
   }
   for (Vector *v : {&workspace.regularization_diagonal, &workspace.reduced_rhs})
   {
      v->SetSize(dimU);
      v->UseDevice(true);
   }
   for (Vector *v :
        {
           &workspace.barrier_diagonal, &workspace.slack_work, &workspace.multiplier_sum,
           &workspace.constraint_residual, &workspace.weighted_constraints
        })
   {
      v->SetSize(dimM);
      v->UseDevice(true);
   }

   MPI_Comm_rank(comm, &myid);
}

IPSolver::PrimalDualVector::PrimalDualVector(
   const Array<int> &state_offsets, const Array<int> &primal_offsets,
   const Array<int> &newton_offsets)
   : values(state_offsets), primal(values, primal_offsets),
     newton(values, newton_offsets)
{
   EnableDevice(values);
   EnableDevice(primal);
   EnableDevice(newton);
   Reset();
}

void IPSolver::PrimalDualVector::Reset()
{
   values = 0.0;
   SyncToViews();
}

void IPSolver::PrimalDualVector::SyncToViews() const
{
   values.SyncToBlocks();
   primal.SyncMemory(values);
   primal.SyncToBlocks();
   newton.SyncMemory(values);
   newton.SyncToBlocks();
}

void IPSolver::PrimalDualVector::SyncFromBlocks() const
{
   values.SyncFromBlocks();
   SyncToViews();
}

void IPSolver::PrimalDualVector::SyncFromNewton() const
{
   // Propagate writes through the Newton view before refreshing other aliases.
   newton.SyncFromBlocks();
   newton.SyncAliasMemory(values);
   SyncToViews();
}

void IPSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); EnableDevice(x0block); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   x0block.GetBlock(1) = 1.0;
   x0block.SyncFromBlocks();
   BlockVector xfblock(block_offsetsx); EnableDevice(xfblock); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}

const char *IPSolver::GetTerminationReason() const
{
   switch (state.termination_status)
   {
      case TerminationStatus::NotRun: return "not run";
      case TerminationStatus::Converged: return "converged";
      case TerminationStatus::MaximumIterations:
         return "maximum optimization iterations";
      case TerminationStatus::NonfiniteOptimalityError:
         return "nonfinite optimality error";
      case TerminationStatus::LinearSolverFailed: return "linear solver failed";
      case TerminationStatus::CurvatureTestFailed: return "curvature test failed";
      case TerminationStatus::LineSearchFailed: return "line search failed";
   }
   return "not run";
}

void IPSolver::UpdateBarrierForIterate(PrimalDualVector &iterate,
                                       int max_updates)
{
   BlockVector &xk = iterate.Primal();
   Vector &lk = iterate.EqualityMultiplier();
   Vector &zlk = iterate.InequalityMultiplier();
   real_t OptErrSubproblem;
   for (int i = 0; i < max_updates; i++)
   {
      // Check convergence of the barrier subproblem
      OptErrSubproblem = OptimalityError(xk, lk, zlk, state.barrier_parameter);
      if (OptErrSubproblem < options.barrier.subproblem_tolerance_factor *
          state.barrier_parameter)
      {
         const real_t old_mu = state.barrier_parameter;
         UpdateBarrierSubProblem();
         if (myid == 0 && options.print_level > 0)
         {
            contact_output::PrintDebugField(mfem::out, "Barrier update",
                                            contact_output::Scientific(old_mu) + " -> " +
                                            contact_output::Scientific(
                                               state.barrier_parameter));
         }
      }
      else
      {
         break;
      }
   }
}

IPSolver::NewtonResult IPSolver::ComputeSearchDirection(
   PrimalDualVector &iterate, PrimalDualVector &direction, int ip_step)
{
   const auto &regularization = options.regularization;
   NewtonResult newton = IPNewtonSolve(iterate, direction, ip_step,
                                       state.barrier_parameter);
   if (!newton.converged) { return newton; }
   if (!newton.passed_curvature_test)
   {
      if (myid == 0 && options.print_level > 0)
      {
         contact_output::PrintDebugSection(mfem::out, "REGULARIZATION");
         contact_output::PrintDebugField(mfem::out, "Curvature test",
                                         "failed; retrying with regularization");
      }
      real_t deltaReg = 0.0;
      const int maxCurvatureTests = regularization.max_curvature_tests;

      // inertia regularization initialization
      if (state.last_regularization < regularization.minimum)
      {
         deltaReg = regularization.initial;
      }
      else
      {
         deltaReg = fmax(regularization.minimum,
                         regularization.decrease_factor * state.last_regularization);
      }

      // solve regularized IP-Newton linear system
      newton = IPNewtonSolve(iterate, direction, ip_step, state.barrier_parameter,
                             deltaReg);

      for (int numCurvatureTests = 0; numCurvatureTests < maxCurvatureTests;
           numCurvatureTests++)
      {
         if (!newton.converged) { break; }
         if (myid == 0 && options.print_level > 0)
         {
            contact_output::PrintDebugField(mfem::out, "Regularization shift",
                                            contact_output::Scientific(deltaReg));
            contact_output::PrintDebugField(mfem::out, "Curvature test",
                                            newton.passed_curvature_test ? "passed" : "failed");
         }
         if (newton.passed_curvature_test)
         {
            state.last_regularization = deltaReg;
            break;
         }
         else
         {
            if (state.last_regularization < regularization.minimum)
            {
               if (myid == 0 && options.print_level > 0)
               {
                  contact_output::PrintDebugField(mfem::out, "Shift growth factor",
                                                  contact_output::Scientific(
                                                     regularization.initial_increase_factor));
               }
               deltaReg *= regularization.initial_increase_factor;
            }
            else
            {
               if (myid == 0 && options.print_level > 0)
               {
                  contact_output::PrintDebugField(mfem::out, "Shift growth factor",
                                                  contact_output::Scientific(
                                                     regularization.increase_factor));
               }
               deltaReg *= regularization.increase_factor;
            }
         }
         // solve with regularization
         // Retain the solve after the final failed check in this loop.
         newton = IPNewtonSolve(iterate, direction, ip_step, state.barrier_parameter,
                                deltaReg);
      }
   }
   return newton;
}

void IPSolver::AcceptStep(PrimalDualVector &iterate,
                          const PrimalDualVector &direction,
                          const LineSearchResult &search)
{
   BlockVector &xk = iterate.Primal();
   Vector &lk = iterate.EqualityMultiplier();
   Vector &zlk = iterate.InequalityMultiplier();
   if (search.update_filter)
   {
      state.filter_theta.Append(search.filter_theta);
      state.filter_phi.Append(search.filter_phi);
   }
   // Accept the trial point
   iterate.Displacement().Add(search.primal_step,
                              direction.Displacement());
   iterate.Slack().Add(search.primal_step, direction.Slack());
   lk.Add(search.primal_step, direction.EqualityMultiplier());
   zlk.Add(search.multiplier_step, direction.InequalityMultiplier());
   iterate.SyncFromBlocks();
   ProjectZ(xk, zlk, state.barrier_parameter);
   iterate.SyncFromBlocks();
}

void IPSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   MFEM_VERIFY(solver, "Set a linear solver before calling IPSolver::Mult.");
   state.converged = false;
   state.accepted_steps = 0;
   state.newton_solves = 0;
   state.linear_iterations.SetSize(0);
   state.termination_status = TerminationStatus::MaximumIterations;
   x0.SyncToBlocks();
   PrimalDualVector iterate(block_offsetsumlz, block_offsetsx, block_offsetsuml);
   PrimalDualVector direction(block_offsetsumlz, block_offsetsx,
                              block_offsetsuml);
   BlockVector &xk = iterate.Primal();
   Vector &lk = iterate.EqualityMultiplier();
   Vector &zlk = iterate.InequalityMultiplier();

   iterate.Displacement().Set(1.0, x0.GetBlock(0));
   iterate.Slack().Set(1.0, x0.GetBlock(1));
   iterate.SyncFromBlocks();
   const int dim_u = dimU;
   const real_t mu = state.barrier_parameter;
   const real_t *xk_d = xk.Read();
   real_t *zlk_d = zlk.Write();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      zlk_d[i] = 1.e1 * mu / xk_d[i + dim_u];
   });

   iterate.SyncFromBlocks();

   /* set theta0 = theta(x0)
    *     thetaMin
    *     thetaMax
    * when theta(xk) < thetaMin and the switching condition holds
    * then we ask for the Armijo sufficient decrease of the barrier
    * objective to be satisfied, in order to accept the trial step length alphakl
    *
    * thetaMax controls how the filter is initialized for each log-barrier subproblem
    * F0 = {(th, phi) s.t. th > thetaMax}
    * that is the filter does not allow for iterates where the constraint violation
    * is larger than that of thetaMax
    */
   real_t theta0 = GetTheta(xk);
   state.filter_min_violation = 1.e-4 * std::max(1.0, theta0);
   state.filter_max_violation = 1.e8  * state.filter_min_violation;

   real_t OptErr;
   for (int j = 0; j < options.max_iterations; j++)
   {
      if (myid == 0 && options.print_level > 0)
      {
         contact_output::PrintDebugSection(mfem::out,
                                           "IP DEBUG - STEP " + std::to_string(j));
      }
      // Check convergence of optimization problem
      OptErr = OptimalityError(xk, lk, zlk);
      if (!IsFinite(OptErr))
      {
         state.termination_status = TerminationStatus::NonfiniteOptimalityError;
         break;
      }
      if (OptErr < options.abs_tol)
      {
         state.converged = true;
         state.termination_status = TerminationStatus::Converged;
         if (myid == 0 && options.print_level > 0)
         {
            contact_output::PrintDebugField(mfem::out, "Optimization status",
                                            "converged");
         }
         break;
      }

      UpdateBarrierForIterate(iterate, j == 0 ? 10 : 1);

      // Compute the search direction
      // solve for (uhat, mhat, lhat)
      const NewtonResult newton = ComputeSearchDirection(iterate, direction, j);
      if (!newton.converged || !newton.passed_curvature_test)
      {
         state.termination_status = newton.converged
                                    ? TerminationStatus::CurvatureTestFailed
                                    : TerminationStatus::LinearSolverFailed;
         break;
      }

      const LineSearchResult search = LineSearch(
                                         iterate, direction, state.barrier_parameter,
                                         state.last_armijo_sufficient_decrease);
      state.last_armijo_sufficient_decrease = search.armijo_sufficient_decrease;

      if (search.accepted)
      {
         if (myid == 0 && options.print_level > 0)
         {
            contact_output::PrintDebugField(mfem::out, "Step accepted", "yes");
            contact_output::PrintDebugField(mfem::out, "Primal step",
                                            contact_output::Scientific(search.primal_step));
            contact_output::PrintDebugField(mfem::out, "Multiplier step",
                                            contact_output::Scientific(search.multiplier_step));
         }
         AcceptStep(iterate, direction, search);
         state.accepted_steps++;
      }
      else
      {
         if (myid == 0 && options.print_level > 0)
         {
            contact_output::PrintDebugField(mfem::out, "Step accepted", "no");
         }
         state.converged = false;
         state.termination_status = TerminationStatus::LineSearchFailed;
         break;
      }
      if (j + 1 == options.max_iterations && myid == 0 && options.print_level > 0)
      {
         contact_output::PrintDebugField(mfem::out, "Optimization status",
                                         "maximum optimization iterations");
      }
   }
   // Check the accepted final iterate, including the last allowed IP step.
   OptErr = OptimalityError(xk, lk, zlk);
   if (IsFinite(OptErr) && OptErr < options.abs_tol)
   {
      state.converged = true;
      state.termination_status = TerminationStatus::Converged;
   }
   xf = 0.0;
   xf.SyncToBlocks();
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
   xf.SyncFromBlocks();
}

void IPSolver::FormNewtonBlocks(BlockVector &x, const Vector &zl,
                                real_t delta)
{
   x.SyncFromBlocks();
   workspace.Huu = problem->GetEnergyHessian(x.GetBlock(0));

   workspace.Ju = problem->GetConstraintJacobian();
   workspace.JuT = problem->GetConstraintJacobianTranspose();

   Vector &DiagLogBar = workspace.barrier_diagonal;
   FillBarrierDiagonal(x, Mlump, zl, delta, DiagLogBar);

   // The contact objective has no slack Hessian. Only the barrier and
   // slack regularization contribute to this diagonal block.
   if (!workspace.Wmm)
   {
      workspace.Wmm.reset(MakeDiagonalHypreParMatrix(
                             comm, problem->GetGlobalNumConstraints(),
                             problem->GetConstraintsStarts(), DiagLogBar));
   }
   else
   {
      UpdateDiagonal(*workspace.Wmm, DiagLogBar);
   }

   Vector &deltaDiagVec = workspace.regularization_diagonal;
   deltaDiagVec.Set(delta, Mvlump);

   workspace.Wuu.reset();
   if (workspace.Huu)
   {
      std::unique_ptr<HypreParMatrix> Duu(
         MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                    problem->GetDofStarts(), deltaDiagVec));
      workspace.Wuu.reset(ParAdd(workspace.Huu, Duu.get()));
   }
   else
   {
      workspace.Wuu.reset(MakeDiagonalHypreParMatrix(comm,
                                                     problem->GetGlobalNumDofs(),
                                                     problem->GetDofStarts(), deltaDiagVec));
   }

   // The contact KKT equations have blocks [Wuu, 0, Ju^T],
   // [0, Wmm, -I], [Ju, -I, 0]. IPNewtonSolve eliminates slack and
   // equality multipliers directly; no full KKT operator is needed.
}

// perturbed KKT system solve
// determine the search direction
IPSolver::NewtonResult IPSolver::IPNewtonSolve(
   PrimalDualVector &iterate, PrimalDualVector &direction, int ip_step,
   real_t mu, real_t delta)
{
   state.newton_solves++;
   iterate.SyncToViews();
   direction.Reset();
   BlockVector &x = iterate.Primal();
   Vector &l = iterate.EqualityMultiplier();
   Vector &zl = iterate.InequalityMultiplier();
   Vector &zlhat = direction.InequalityMultiplier();
   BlockVector &Xhat = direction.Newton();
   // Form the Hessian blocks and right-hand side for the reduced solve.
   BlockVector &b = workspace.newton_rhs;
   b.SyncToBlocks();
   FormNewtonBlocks(x, zl, delta);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector &gradphi = workspace.objective_gradient;
   GetDxphi(x, mu, gradphi);

   b.GetBlock(0).Set(1.0, gradphi.GetBlock(0));
   workspace.JuT->AddMult(l, b.GetBlock(0));
   b.GetBlock(1).Set(1.0, gradphi.GetBlock(1));
   // The slack Jacobian is -I: Jm^T l = -l.
   b.GetBlock(1).Add(-1.0, l);
   problem->EvaluateConstraints(x, b.GetBlock(2));
   b.SyncFromBlocks();
   b *= -1.0;
   b.SyncToBlocks();

   // Include the displacement regularization in the reduced system.
   std::unique_ptr<HypreParMatrix> JuTDJu(RAP(workspace.Wmm.get(),
                                              workspace.Ju)); // Ju^T D Ju
   std::unique_ptr<HypreParMatrix> Areduced(ParAdd(workspace.Wuu.get(),
                                                   JuTDJu.get()));

   /* compute the reduced rhs */
   // breduced = bu + Ju^T (bm + Wmm bl)
   Vector &breduced = workspace.reduced_rhs;
   Vector &tempVec = workspace.slack_work;
   workspace.Wmm->Mult(b.GetBlock(2), tempVec);
   tempVec.Add(1.0, b.GetBlock(1));
   workspace.JuT->Mult(tempVec, breduced);
   breduced.Add(1.0, b.GetBlock(0));

   // Solve the system with the given solver
   solver->SetOperator(*Areduced);
   solver->Mult(breduced, Xhat.GetBlock(0));
   auto itsolver = dynamic_cast<IterativeSolver *>(solver);
   int numit = (itsolver) ? itsolver->GetNumIterations() : -1;
   state.linear_iterations.Append(numit);
   if (options.print_linear_solver_statistics && itsolver && myid == 0)
   {
      PrintLinearSolverStatistics(ip_step, *itsolver);
   }
   int local_success = (!itsolver || itsolver->GetConverged()) &&
                       IsFinite(FiniteNormlinf(Xhat.GetBlock(0), workspace.reduction));
   int global_success;
   MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);
   if (!global_success)
   {
      direction.SyncFromNewton();
      return {};
   }

   // now propagate solved uhat to obtain mhat and lhat
   // xm = Ju xu - bl
   workspace.Ju->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
   Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

   // xl = Wmm xm - bm
   workspace.Wmm->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
   Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));
   direction.SyncFromNewton();

   JuTDJu.reset();
   Areduced.reset();

   const bool passed_curvature_test = CurvatureTest(*workspace.Wuu, *workspace.Wmm,
                                                    Xhat, l, b);

   /* backsolve to determine zlhat */
   x.SyncFromBlocks();
   ComputeMultiplierDirection(x, Xhat, zl, mu, zlhat);
   direction.SyncFromBlocks();
   local_success = IsFinite(FiniteNormlinf(Xhat, workspace.reduction)) &&
                   IsFinite(FiniteNormlinf(zlhat, workspace.reduction));
   MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);
   return {global_success != 0, passed_curvature_test};
}

void IPSolver::PrintLinearSolverStatistics(
   int ip_step, const IterativeSolver &linear_solver) const
{
   using namespace contact_output;
   const std::array<int, 5> widths = {8, 13, 21, 19, 13};
   std::ostringstream table;
   // Debug diagnostics separate CG rows, so repeat the header for each solve.
   if (state.linear_iterations.Size() == 1 || options.print_level > 0)
   {
      PrintTableRow(table, widths,
                    {"IP step", "Status", "Initial residual norm",
                     "Final residual norm", "Iterations"});
      PrintTableSeparator(table, widths);
   }
   PrintTableRow(table, widths,
                 {std::to_string(ip_step),
                  linear_solver.GetConverged() ? "Converged" : "Not converged",
                  Scientific(linear_solver.GetInitialNorm()),
                  Scientific(linear_solver.GetFinalNorm()),
                  std::to_string(linear_solver.GetNumIterations())});
   mfem::out << table.str() << std::flush;
}

real_t IPSolver::GetMaxStepSize(const Vector &x, const Vector &xhat,
                                real_t tau)
{
   real_t alphaMaxloc = LocalMaxStepSize(x, xhat, tau, workspace.reduction);

   real_t alphaMaxglb;
   MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, comm);
   return alphaMaxglb;
}

/* line-search from X0 along direction Xhat for the log-barrier
 * subproblem with barrier parameter \mu
 */
IPSolver::LineSearchResult IPSolver::LineSearch(
   const PrimalDualVector &X0, const PrimalDualVector &Xhat, real_t mu,
   bool previous_armijo_sufficient_decrease)
{
   using namespace contact_output;
   const auto &line_search = options.line_search;
   const bool debug = myid == 0 && options.print_level > 0;
   const std::array<int, 6> trial_widths = {6, 12, 13, 13, 10, 17};
   LineSearchResult result;
   result.armijo_sufficient_decrease = previous_armijo_sufficient_decrease;
   real_t &alpha = result.primal_step;
   bool &sufficientDecrease = result.armijo_sufficient_decrease;
   bool switchCondition = false;
   int eval_err = 0;
   real_t tau  = std::max(line_search.fraction_to_boundary, 1.0 - mu);
   X0.SyncToViews();
   Xhat.SyncToViews();
   const Vector &m0   = X0.Slack();
   const Vector &z0   = X0.InequalityMultiplier();
   const Vector &mhat = Xhat.Slack();
   const Vector &zhat = Xhat.InequalityMultiplier();
   real_t alphaMax  = GetMaxStepSize(m0, mhat, tau);
   real_t alphaMaxz = GetMaxStepSize(z0, zhat, tau);
   result.multiplier_step = alphaMaxz;

   const BlockVector &x0 = X0.Primal();
   const BlockVector &xhat = Xhat.Primal();

   BlockVector &xtrial = workspace.line_search_trial;
   xtrial = 0.0;
   BlockVector &Dxphi0 = workspace.line_search_gradient;
   Dxphi0 = 0.0;
   const int maxBacktrack = line_search.max_backtracks;
   alpha = alphaMax;

   GetDxphi(x0, mu, Dxphi0);

   Dxphi0.SyncFromBlocks();
   real_t Dxphi0_xhat = InnerProduct(comm, Dxphi0, xhat);
   bool descentDirection = (Dxphi0_xhat < 0.);


   const real_t thx0 = GetTheta(x0);
   const real_t phx0 = GetPhi(x0, mu, eval_err);

   const auto print_trial = [&](int trial, real_t violation, real_t barrier_value,
                                const char *filter, const char *decision)
   {
      if (!debug) { return; }
      std::ostringstream row;
      PrintTableRow(row, trial_widths,
                    {std::to_string(trial), Scientific(alpha), Scientific(violation),
                     Scientific(barrier_value), filter, decision});
      mfem::out << row.str() << std::flush;
   };
   if (debug)
   {
      std::ostringstream out;
      PrintDebugSection(out, "LINE SEARCH");
      PrintDebugField(out, "Barrier parameter", Scientific(mu));
      PrintDebugField(out, "Descent direction", descentDirection ? "yes" : "no");
      PrintDebugField(out, "Directional derivative", Scientific(Dxphi0_xhat));
      PrintDebugField(out, "Armijo factor", Scientific(line_search.armijo_factor));
      PrintDebugField(out, "Initial violation", Scientific(thx0));
      PrintDebugField(out, "Switching threshold",
                      Scientific(state.filter_min_violation));
      PrintDebugField(out, "Violation target",
                      Scientific((1. - line_search.violation_decrease) * thx0));
      PrintDebugField(out, "Initial barrier value", Scientific(phx0));
      PrintDebugField(out, "Barrier progress target",
                      Scientific(phx0 - line_search.objective_decrease * thx0));
      PrintDebugField(out, "Primal step limit", Scientific(alphaMax));
      PrintDebugField(out, "Multiplier step limit", Scientific(alphaMaxz));
      out << '\n';
      PrintTableRow(out, trial_widths,
                    {"Trial", "Step length", "Violation", "Barrier value",
                     "Filter", "Decision"});
      PrintTableSeparator(out, trial_widths);
      mfem::out << out.str() << std::flush;
   }
   if (eval_err || !IsFinite(thx0))
   {
      if (debug)
      {
         PrintDebugField(mfem::out, "Line-search status", "invalid initial point");
      }
      return result;
   }
   for (int i = 0; i < maxBacktrack; i++)
   {
      // ----- Compute trial point: xtrial = x0 + alpha * xhat
      xtrial.Set(1.0, x0);
      xtrial.Add(alpha, xhat);
      xtrial.SyncToBlocks();

      real_t thxtrial = GetTheta(xtrial);
      real_t phxtrial = GetPhi(xtrial, mu, eval_err);
      if (eval_err || !IsFinite(thxtrial))
      {
         print_trial(i, thxtrial, phxtrial, "n/a", "invalid value");
         alpha *= 0.5;
         continue;
      }

      auto inFilterRegion = FilterCheck(thxtrial, phxtrial);
      if (!inFilterRegion)
      {
         if (!descentDirection)
         {
            switchCondition = false;
         }
         else
         {
            switchCondition = (alpha * pow(abs(Dxphi0_xhat),
                                           line_search.objective_power) >
                               line_search.sufficient_progress_factor * pow(thx0,
                                                                            line_search.violation_power));
         }
         if (thx0 <= state.filter_min_violation && switchCondition)
         {
            // sufficient decrease of the log-barrier objective
            sufficientDecrease = (phxtrial <= phx0 + line_search.armijo_factor *
                                  alpha * Dxphi0_xhat);
            if (sufficientDecrease)
            {
               print_trial(i, thxtrial, phxtrial, "allowed", "Armijo accepted");
               // accept the trial step
               result.accepted = true;
               break;
            }
         }
         else
         {
            if (thxtrial <= (1. - line_search.violation_decrease) * thx0 ||
                phxtrial <= phx0 - line_search.objective_decrease * thx0)
            {
               print_trial(i, thxtrial, phxtrial, "allowed", "Progress accepted");
               // accept the trial step
               result.accepted = true;
               break;
            }
         }
      }
      print_trial(i, thxtrial, phxtrial,
                  inFilterRegion ? "blocked" : "allowed", "backtrack");
      alpha *= 0.5;
   }
   if (result.accepted)
   {
      result.update_filter = !switchCondition || !sufficientDecrease;
      if (result.update_filter)
      {
         result.filter_theta = (1. - line_search.violation_decrease) * thx0;
         result.filter_phi = phx0 - line_search.objective_decrease * thx0;
      }
   }
   return result;
}

bool IPSolver::FilterCheck(real_t thetax, real_t phix)
{
   bool inFilterRegion = false;
   if (thetax > state.filter_max_violation)
   {
      inFilterRegion = true;
   }
   else
   {
      for (int i = 0; i < state.filter_theta.Size(); i++)
      {
         if (thetax >= state.filter_theta[i] && phix >= state.filter_phi[i])
         {
            inFilterRegion = true;
            break;
         }
      }
   }
   return inFilterRegion;
}

void IPSolver::ProjectZ(const Vector &x, Vector &z, real_t mu)
{
   ProjectMultipliers(x, z, dimU, mu, options.barrier.multiplier_deviation);
}

real_t IPSolver::GetTheta(const BlockVector &x)
{
   x.SyncToBlocks();
   Vector &cx = workspace.constraint_residual;
   problem->EvaluateConstraints(x, cx);
   Vector &Mcx = workspace.weighted_constraints;
   MultiplyWeights(Mlump, cx, Mcx);
   return sqrt(InnerProduct(comm, Mcx, cx));
}

// log-barrier objective
real_t IPSolver::GetPhi(const BlockVector &x, real_t mu,
                        int &eval_err)
{
   x.SyncToBlocks();
   real_t fx = problem->GetEnergy(x.GetBlock(0), eval_err);
   real_t logBarrierLoc = LocalWeightedLogSum(x, Mlump, workspace.reduction);
   real_t logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   const real_t phi = fx - mu * logBarrierGlb;
   if (!IsFinite(phi)) { eval_err = 1; }
   return phi;
}

void IPSolver::GetObjectiveGradient(const BlockVector &x, BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   problem->GetEnergyGradient(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
   y.SyncFromBlocks();
}

// gradient of log-barrier objective with respect to x = (u, m)
void IPSolver::GetDxphi(const BlockVector &x, real_t mu,
                        BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   GetObjectiveGradient(x, y);
   AddBarrierGradient(Mlump, x.GetBlock(1), mu, y.GetBlock(1));
   y.SyncFromBlocks();
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
real_t IPSolver::EvalLagrangian(const BlockVector &x, const Vector &l,
                                const Vector &zl)
{
   x.SyncToBlocks();
   int eval_err = 0;
   real_t fx = problem->GetEnergy(x.GetBlock(0), eval_err);
   Vector &cx = workspace.constraint_residual;
   problem->EvaluateConstraints(x, cx);
   Vector &temp = workspace.slack_work;
   MultiplyWeights(Mlump, x.GetBlock(1), temp);
   return (fx + InnerProduct(comm, cx, l) - InnerProduct(comm, temp, zl));
}

// Gradient of the Lagrangian
// \nabla_x L = [ \nabla_u f + (\partial c / \partial u)^T l]
//              [ \nabla_m f + (\partial c / \partial m)^T l - zl]
void IPSolver::EvalLagrangianGradient(const BlockVector &x, const Vector &l,
                                      const Vector &zl, BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   y.GetBlock(1).Set(-1.0, zl);
   y.GetBlock(1) *= Mlump;

   problem->GetConstraintJacobianTranspose()->Mult(l, y.GetBlock(0));
   y.GetBlock(1).Add(-1.0, l);

   BlockVector &gradxf = workspace.objective_gradient;
   GetObjectiveGradient(x, gradxf);
   y.SyncFromBlocks();
   gradxf.SyncFromBlocks();
   y.Add(1.0, gradxf);
   y.SyncToBlocks();
}


// curvature test
// dk^T Wk dk + max{ -(lk + lhat)^T ck, 0.0} >= alpha * dk^T dk
// see "An Inertia-Free Filter Line-search Algorithm for
// Large-scale Nonlinear Programming" by Nai-Yuan Chiang and
// Victor M Zavala, Computational Optimization and Applications (2016)
bool IPSolver::CurvatureTest(const Operator &displacement_hessian,
                             const Operator &slack_hessian,
                             const BlockVector &Xhat, const Vector &l,
                             const BlockVector &b)
{
   Xhat.SyncToBlocks();
   b.SyncToBlocks();
   Vector &lplus = workspace.multiplier_sum;
   lplus.Set(1.0, l);
   lplus.Add(1.0, Xhat.GetBlock(2));


   real_t dWd = 0.0;
   real_t dd = 0.0;
   const Operator *hessians[] = {&displacement_hessian, &slack_hessian};
   for (int i = 0; i < 2; i++)
   {
      Vector &temp = workspace.hessian_action.GetBlock(i);
      hessians[i]->Mult(Xhat.GetBlock(i), temp);
      dWd += InnerProduct(comm, Xhat.GetBlock(i), temp);
      dd += InnerProduct(comm, Xhat.GetBlock(i), Xhat.GetBlock(i));
   }
   real_t lplusTck = -1.0 * InnerProduct(comm, lplus, b.GetBlock(2));

   bool passed = (dWd + fmax(-lplusTck,
                             0.0) >= options.regularization.curvature_tolerance * dd);
   return passed;
}


real_t IPSolver::OptimalityError(const BlockVector &x, const Vector &l,
                                 const Vector &zl, real_t mu)
{
   x.SyncToBlocks();
   /* stationarity, feasibility, and complementarity errors */
   real_t stationarityError, feasibilityError, complementarityError;
   real_t optimalityError;

   /* gradient of Lagrangian */
   BlockVector &gradL = workspace.lagrangian_gradient;
   EvalLagrangianGradient(x, l, zl, gradL);

   /* constraint function c(u, m) = 0 */
   Vector &cx = workspace.constraint_residual;
   problem->EvaluateConstraints(x, cx);

   /* regularized complementarity
    * |z_i * m_i - mu| */
   real_t complementarityLoc =
      LocalComplementarityError(x, zl, mu, workspace.reduction);

   BlockVector &MxinvgradL = workspace.scaled_gradient;
   MxinvgradL.Set(1.0, gradL);
   MxinvgradL.SyncToBlocks();
   MxinvgradL.GetBlock(0) /= Mvlump;
   MxinvgradL.GetBlock(1) /= Mlump;
   MxinvgradL.SyncFromBlocks();
   stationarityError = sqrt(InnerProduct(comm, gradL, MxinvgradL));
   feasibilityError = GlobalLpNorm(infinity(),
                                   FiniteNormlinf(cx, workspace.reduction), comm);
   complementarityError = GlobalLpNorm(infinity(), complementarityLoc, comm);

   state.stationarity_error = stationarityError;
   state.feasibility_error = feasibilityError;
   state.complementarity_error = complementarityError;


   optimalityError = std::max(std::max(stationarityError, feasibilityError),
                              complementarityError);
   if (!IsFinite(stationarityError) || !IsFinite(feasibilityError) ||
       !IsFinite(complementarityError))
   {
      optimalityError = infinity();
   }
   state.optimality_error = optimalityError;

   if (myid == 0 && options.print_level > 0)
   {
      using namespace contact_output;
      const std::array<int, 5> widths = {12, 16, 14, 17, 15};
      std::ostringstream table;
      table << '\n';
      PrintTableRow(table, widths,
                    {"Barrier mu", "Stationarity", "Feasibility",
                     "Complementarity", "Optimality"});
      PrintTableSeparator(table, widths);
      PrintTableRow(table, widths,
                    {Scientific(mu), Scientific(stationarityError),
                     Scientific(feasibilityError), Scientific(complementarityError),
                     Scientific(optimalityError)});
      mfem::out << table.str() << std::flush;
   }
   return optimalityError;
}

}
