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
#include <iomanip>
#include <sstream>

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
   abs_tol  = 1.e-2; // Tolerance of the optimizer
   max_iter = 20;   // Maximum iterations
   mu_k     = 1.0;  // Log-barrier penalty parameter

   /* The following constants follow that of
    * Wächter, Andreas, and Lorenz T. Biegler.
    * "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming."
    * Mathematical programming 106.1 (2006): 25-57.
    */

   /* line search constants */
   tauMin   =
      0.99;  // constant that controls the rate iterates approach boundary > 0, < 1
   eta      =
      1.e-4; // Armijo backtracking sufficient-decrease condition constant > 0, < 1
   thetaMin = 1.e-4; // allowed equality constraint violation
   delta    = 1.0;   // sufficient barrier objective progress constant > 0
   sTheta   = 1.1;   // sufficient barrier objective progress constant > 0
   sPhi     = 2.3;   // sufficient barrier objective progress constant >= 0
   gTheta = 1.e-5;   // sufficient constraint violation decrease constant > 0, < 1
   gPhi   = 1.e-5;   // sufficient barrier objective decrease constant > 0, < 1
   thetaMax = 1.e6; // maximum constraint violation (defines initial filter)

   /* interior-point continuation constants
    * \mu_new = max{abs_tol / 10, \min{ kMu * \mu_old, (\mu_old)^thetaMu}}
    * */
   kMu     = 0.2;
   thetaMu = 1.5;
   kEps   = 1.e1; // constant that determines when the log-barrier parameter is decreased

   kSig     = 1.e10; // primal-dual Hessian deviation constant

   /* The following constants follow that of
    * Chiang, Nai-Yuan, and Victor M. Zavala.
    * "An inertia-free filter line-search algorithm for large-scale nonlinear programming."
    * Computational Optimization and Applications 64.2 (2016): 327-354.
    * See Inertia-Free Regularizaton Algorithm (IFR).
    * We use the second variant (see Section 4.1 "Alternative Inertia-Free Tests")
    * which does not require solving for normal and tangential components.
    */

   /* inertia-regularization constants */
   alphaCurvatureTest = 1.e-11;
   deltaRegLast = 0.0;
   deltaRegMin = 1.e-20;
   deltaRegMax = 1.e40;
   deltaReg0 = 1.e-4;
   kRegMinus = 1. / 3.;
   kRegBarPlus = 1.e2;
   kRegPlus = 8.;

   dimU = problem->GetDimU();
   dimM = problem->GetDimM();
   dimC = problem->GetDimC();
   MFEM_VERIFY(dimM == dimC,
               "Expecting equal numbers of equality and inequality constraints");

   comm = problem->GetComm();
   Mcslump.UseDevice(true);
   Mvlump.UseDevice(true);
   problem->GetLumpedMassWeights(Mcslump, Mvlump);

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
   int dimG = 0; // number of gap constraints
   if (dimM < dimU)
   {
      dimG = dimM;
      constraint_offsets.SetSize(2);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimM;
      Mlump.Set(1.0, Mcslump);
   }
   else
   {
      dimG = dimM - 2 * dimU;
      constraint_offsets.SetSize(4);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimG;
      constraint_offsets[2] = dimU;
      constraint_offsets[3] = dimU;
      constraint_offsets.PartialSum();
      BlockVector Mlumpblock(constraint_offsets); EnableDevice(Mlumpblock);
      Mlumpblock = 0.0;
      Mlumpblock.GetBlock(0).Set(1.0, Mcslump);
      Mlumpblock.GetBlock(1).Set(1.0, Mvlump);
      Mlumpblock.GetBlock(2).Set(1.0, Mvlump);
      Mlumpblock.SyncFromBlocks();
      Mlump.Set(1.0, Mlumpblock);
   }

   for (int i = 0; i < block_offsetsuml.Size(); i++)
   {
      block_offsetsuml[i] = block_offsetsumlz[i];
   }
   for (int i = 0; i < block_offsetsx.Size(); i++)
   {
      block_offsetsx[i] = block_offsetsuml[i] ;
   }

   newton_rhs.Update(block_offsetsuml);
   EnableDevice(newton_rhs);
   for (BlockVector *v :
        {
           &objective_gradient, &lagrangian_gradient,
           &scaled_gradient, &hessian_action,
           &line_search_trial, &line_search_gradient
        })
   {
      v->Update(block_offsetsx);
      EnableDevice(*v);
   }
   for (Vector *v : {&regularization_diagonal, &reduced_rhs})
   {
      v->SetSize(dimU);
      v->UseDevice(true);
   }
   for (Vector *v :
        {
           &barrier_diagonal, &slack_work, &multiplier_sum,
           &constraint_residual, &weighted_constraints
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
   switch (termination_status)
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
      OptErrSubproblem = OptimalityError(xk, lk, zlk, mu_k);
      if (OptErrSubproblem < kEps * mu_k)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "solved mu = " << mu_k << " barrier subproblem\n";
         }
         UpdateBarrierSubProblem();
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
   NewtonResult newton = IPNewtonSolve(iterate, direction, ip_step, mu_k);
   if (!newton.converged) { return newton; }
   if (!newton.passed_curvature_test)
   {
      if (myid == 0 && print_level > 0)
      {
         mfem::out << "curvature test failed\n";
      }
      real_t deltaReg = 0.0;
      int maxCurvatureTests = 30;

      // inertia regularization initialization
      if (deltaRegLast < deltaRegMin)
      {
         deltaReg = deltaReg0;
      }
      else
      {
         deltaReg = fmax(deltaRegMin, kRegMinus * deltaRegLast);
      }

      // solve regularized IP-Newton linear system
      newton = IPNewtonSolve(iterate, direction, ip_step, mu_k, deltaReg);

      for (int numCurvatureTests = 0; numCurvatureTests < maxCurvatureTests;
           numCurvatureTests++)
      {
         if (!newton.converged) { break; }
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "deltaReg = " << deltaReg << std::endl;
         }
         if (newton.passed_curvature_test)
         {
            deltaRegLast = deltaReg;
            break;
         }
         else
         {
            if (deltaRegLast < deltaRegMin)
            {
               if (myid == 0 && print_level > 0)
               {
                  mfem::out << "delta *= " << kRegBarPlus << "\n";
               }
               deltaReg *= kRegBarPlus;
            }
            else
            {
               deltaReg *= kRegPlus;
            }
         }
         // solve with regularization
         // Retain the solve after the final failed check in this loop.
         newton = IPNewtonSolve(iterate, direction, ip_step, mu_k, deltaReg);
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
      F1.Append(search.filter_theta);
      F2.Append(search.filter_phi);
   }
   // Accept the trial point
   iterate.Displacement().Add(search.primal_step,
                              direction.Displacement());
   iterate.Slack().Add(search.primal_step, direction.Slack());
   lk.Add(search.primal_step, direction.EqualityMultiplier());
   zlk.Add(search.multiplier_step, direction.InequalityMultiplier());
   iterate.SyncFromBlocks();
   ProjectZ(xk, zlk, mu_k);
   iterate.SyncFromBlocks();
}

void IPSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   MFEM_VERIFY(solver, "Set a linear solver before calling IPSolver::Mult.");
   converged = false;
   iter = 0;
   lin_solver_iterations.SetSize(0);
   termination_status = TerminationStatus::MaximumIterations;
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
   const real_t mu = mu_k;
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
   thetaMin = 1.e-4 * std::max(1.0, theta0);
   thetaMax = 1.e8  * thetaMin;

   real_t OptErr;
   for (int j = 0; j < max_iter; j++)
   {
      if (myid == 0 && print_level > 0)
      {
         mfem::out << "\n" << std::string(50,'-') << std::endl;
         mfem::out << "interior-point solve step # " << j << std::endl;
      }
      // Check convergence of optimization problem
      OptErr = OptimalityError(xk, lk, zlk);
      if (!IsFinite(OptErr))
      {
         termination_status = TerminationStatus::NonfiniteOptimalityError;
         break;
      }
      if (OptErr < abs_tol)
      {
         converged = true;
         termination_status = TerminationStatus::Converged;
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "solved optimization problem\n";
         }
         break;
      }

      UpdateBarrierForIterate(iterate, j == 0 ? 10 : 1);

      // Compute the search direction
      // solve for (uhat, mhat, lhat)
      const NewtonResult newton = ComputeSearchDirection(iterate, direction, j);
      if (!newton.converged || !newton.passed_curvature_test)
      {
         termination_status = newton.converged
                              ? TerminationStatus::CurvatureTestFailed
                              : TerminationStatus::LinearSolverFailed;
         break;
      }

      if (myid == 0 && print_level > 0)
      {
         mfem::out << "mu = " << mu_k << std::endl;
      }
      const LineSearchResult search = LineSearch(
                                         iterate, direction, mu_k, last_armijo_sufficient_decrease);
      last_armijo_sufficient_decrease = search.armijo_sufficient_decrease;

      if (search.accepted)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "lineSearch successful\n";
         }
         AcceptStep(iterate, direction, search);
      }
      else
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "lineSearch not successful\n";
         }
         converged = false;
         termination_status = TerminationStatus::LineSearchFailed;
         break;
      }
      if (j + 1 == max_iter && myid == 0 && print_level > 0)
      {
         mfem::out << "maximum optimization iterations\n";
      }
   }
   // Check the accepted final iterate, including the last allowed IP step.
   OptErr = OptimalityError(xk, lk, zlk);
   if (IsFinite(OptErr) && OptErr < abs_tol)
   {
      converged = true;
      termination_status = TerminationStatus::Converged;
   }
   xf = 0.0;
   xf.SyncToBlocks();
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
   xf.SyncFromBlocks();
}

void IPSolver::FormIPNewtonMat(BlockVector & x, Vector & l,
                               Vector &zl,
                               BlockOperator &Ak, real_t delta)
{
   x.SyncFromBlocks();
   Huu = problem->Duuf(x);
   Hmm = problem->Dmmf(x);

   Ju = problem->Duc(x);
   JuT = problem->DucTranspose(x);
   Jm = problem->Dmc(x);

   Vector &DiagLogBar = barrier_diagonal;
   FillBarrierDiagonal(x, Mlump, zl, delta, DiagLogBar);

   if (Hmm)
   {
      Wmm.reset();
      std::unique_ptr<HypreParMatrix> D(
         MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumConstraints(),
                                    problem->GetConstraintsStarts(),
                                    DiagLogBar));
      Wmm.reset(ParAdd(Hmm, D.get()));
   }
   else
   {
      if (!Wmm)
      {
         Wmm.reset(MakeDiagonalHypreParMatrix(
                      comm, problem->GetGlobalNumConstraints(),
                      problem->GetConstraintsStarts(), DiagLogBar));
      }
      else
      {
         UpdateDiagonal(*Wmm, DiagLogBar);
      }
   }

   Vector &deltaDiagVec = regularization_diagonal;
   deltaDiagVec.Set(delta, Mvlump);

   Wuu.reset();
   if (Huu)
   {
      std::unique_ptr<HypreParMatrix> Duu(
         MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                    problem->GetDofStarts(), deltaDiagVec));
      Wuu.reset(ParAdd(Huu, Duu.get()));
   }
   else
   {
      Wuu.reset(MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                           problem->GetDofStarts(), deltaDiagVec));
   }

   //         IP-Newton system matrix
   //    Ak = [[W_(u,u)  H_(u,m)   J_u^T]
   //          [H_(m,u)  W_(m,m)   J_m^T]
   //          [ J_u      J_m       0  ]]

   //    Ak = [[K+Dreg  0      Jᵀ ]   [u]    [bᵤ]
   //          [0      D+Dreg  -I ]   [m]  = [bₘ]
   //          [J      -I      0  ]]  [λ]  = [bₗ ]
   Ak.SetBlock(0, 0, Wuu.get());                    Ak.SetBlock(0, 2, JuT);
   // The slack Jacobian is -I, so its transpose is the same operator.
   Ak.SetBlock(1, 1, Wmm.get()); Ak.SetBlock(1, 2, Jm);
   Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);
}

// perturbed KKT system solve
// determine the search direction
IPSolver::NewtonResult IPSolver::IPNewtonSolve(
   PrimalDualVector &iterate, PrimalDualVector &direction, int ip_step,
   real_t mu, real_t delta)
{
   iter++;
   iterate.SyncToViews();
   direction.Reset();
   BlockVector &x = iterate.Primal();
   Vector &l = iterate.EqualityMultiplier();
   Vector &zl = iterate.InequalityMultiplier();
   Vector &zlhat = direction.InequalityMultiplier();
   BlockVector &Xhat = direction.Newton();
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml);
   BlockVector &b = newton_rhs;
   b.SyncToBlocks();
   FormIPNewtonMat(x, l, zl, A, delta);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector &gradphi = objective_gradient;
   GetDxphi(x, mu, gradphi);

   b.GetBlock(0).Set(1.0, gradphi.GetBlock(0));
   JuT->AddMult(l, b.GetBlock(0));
   b.GetBlock(1).Set(1.0, gradphi.GetBlock(1));
   // The slack Jacobian is -I: Jm^T l = -l.
   b.GetBlock(1).Add(-1.0, l);
   problem->c(x, b.GetBlock(2));
   b.SyncFromBlocks();
   b *= -1.0;
   b.SyncToBlocks();

   // Include the displacement regularization in the reduced system.
   std::unique_ptr<HypreParMatrix> JuTDJu(RAP(Wmm.get(), Ju)); // Ju^T D Ju
   std::unique_ptr<HypreParMatrix> Areduced(ParAdd(Wuu.get(), JuTDJu.get()));

   /* compute the reduced rhs */
   // breduced = bu + Ju^T (bm + Wmm bl)
   Vector &breduced = reduced_rhs;
   Vector &tempVec = slack_work;
   Wmm->Mult(b.GetBlock(2), tempVec);
   tempVec.Add(1.0, b.GetBlock(1));
   JuT->Mult(tempVec, breduced);
   breduced.Add(1.0, b.GetBlock(0));

   // Solve the system with the given solver
   solver->SetOperator(*Areduced);
   solver->Mult(breduced, Xhat.GetBlock(0));
   auto itsolver = dynamic_cast<IterativeSolver *>(solver);
   int numit = (itsolver) ? itsolver->GetNumIterations() : -1;
   lin_solver_iterations.Append(numit);
   if (print_linear_solver_statistics && itsolver && myid == 0)
   {
      PrintLinearSolverStatistics(ip_step, *itsolver);
   }
   int local_success = (!itsolver || itsolver->GetConverged()) &&
                       IsFinite(FiniteNormlinf(Xhat.GetBlock(0), reduce_workspace));
   int global_success;
   MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);
   if (!global_success)
   {
      direction.SyncFromNewton();
      return {};
   }

   // now propagate solved uhat to obtain mhat and lhat
   // xm = Ju xu - bl
   Ju->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
   Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

   // xl = Wmm xm - bm
   Wmm->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
   Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));
   direction.SyncFromNewton();

   JuTDJu.reset();
   Areduced.reset();

   const bool passed_curvature_test = CurvatureTest(A, Xhat, l, b, delta);

   /* backsolve to determine zlhat */
   x.SyncFromBlocks();
   ComputeMultiplierDirection(x, Xhat, zl, mu, zlhat);
   direction.SyncFromBlocks();
   local_success = IsFinite(FiniteNormlinf(Xhat, reduce_workspace)) &&
                   IsFinite(FiniteNormlinf(zlhat, reduce_workspace));
   MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);
   return {global_success != 0, passed_curvature_test};
}

void IPSolver::PrintLinearSolverStatistics(
   int ip_step, const IterativeSolver &linear_solver) const
{
   std::ostringstream table;
   if (lin_solver_iterations.Size() == 1)
   {
      table << std::right << std::setw(8) << " IP step" << " | "
            << std::left << std::setw(13) << "Status" << " | "
            << std::right << std::setw(21) << "Initial residual norm" << " | "
            << std::setw(19) << "Final residual norm" << " | "
            << std::setw(10) << "Iterations" << '\n';
   }
   table << std::right << std::setw(8) << ip_step << " | "
         << std::left << std::setw(13)
         << (linear_solver.GetConverged() ? "Converged" : "Not converged")
         << " | " << std::right << std::scientific << std::setprecision(6)
         << std::setw(21) << linear_solver.GetInitialNorm() << " | "
         << std::setw(19) << linear_solver.GetFinalNorm() << " | "
         << std::setw(10) << linear_solver.GetNumIterations() << '\n';
   mfem::out << table.str() << std::flush;
}

real_t IPSolver::GetMaxStepSize(const Vector &x, const Vector &xhat,
                                real_t tau)
{
   real_t alphaMaxloc = LocalMaxStepSize(x, xhat, tau, reduce_workspace);

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
   LineSearchResult result;
   result.armijo_sufficient_decrease = previous_armijo_sufficient_decrease;
   real_t &alpha = result.primal_step;
   bool &sufficientDecrease = result.armijo_sufficient_decrease;
   bool switchCondition = false;
   int eval_err = 0;
   real_t tau  = std::max(tauMin, 1.0 - mu);
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

   BlockVector &xtrial = line_search_trial;
   xtrial = 0.0;
   BlockVector &Dxphi0 = line_search_gradient;
   Dxphi0 = 0.0;
   int maxBacktrack = 20;
   alpha = alphaMax;

   GetDxphi(x0, mu, Dxphi0);

   Dxphi0.SyncFromBlocks();
   real_t Dxphi0_xhat = InnerProduct(comm, Dxphi0, xhat);
   bool descentDirection = (Dxphi0_xhat < 0.);


   if (myid == 0 && print_level > 0)
   {
      mfem::out << "is";
      if (!descentDirection)
      {
         mfem::out << " not";
      }
      mfem::out << " a descent direction for the log-barrier objective\n";
   }

   const real_t thx0 = GetTheta(x0);
   const real_t phx0 = GetPhi(x0, mu, eval_err);

   if (eval_err || !IsFinite(thx0)) { return result; }
   for (int i = 0; i < maxBacktrack; i++)
   {
      if (myid == 0 && print_level > 0)
      {
         mfem::out << "\n--------- alpha = " << alpha << " ---------\n";
      }
      // ----- Compute trial point: xtrial = x0 + alpha * xhat
      xtrial.Set(1.0, x0);
      xtrial.Add(alpha, xhat);
      xtrial.SyncToBlocks();

      real_t thxtrial = GetTheta(xtrial);
      real_t phxtrial = GetPhi(xtrial, mu, eval_err);
      if (eval_err || !IsFinite(thxtrial))
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "bad log-barrier objective eval, reducing step length\n";
         }
         alpha *= 0.5;
         continue;
      }

      auto inFilterRegion = FilterCheck(thxtrial, phxtrial);
      if (!inFilterRegion)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "not in filter region\n";
         }
         if (!descentDirection)
         {
            switchCondition = false;
         }
         else
         {
            switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0,
                                                                                 sTheta));
         }
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "theta(x0) = "     << thx0     << ", thetaMin = "
                      <<
                      thetaMin             << std::endl;
            mfem::out << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "
                      <<
                      (1. - gTheta) * thx0 << std::endl;
            mfem::out << "phi(xtrial) = "   << phxtrial << ", phi(x0) - gPhi *theta(x0) = "
                      <<
                      phx0 - gPhi * thx0   << std::endl;
         }
         if (thx0 <= thetaMin && switchCondition)
         {
            // sufficient decrease of the log-barrier objective
            sufficientDecrease = (phxtrial <= phx0 + eta * alpha * Dxphi0_xhat);
            if (sufficientDecrease)
            {
               if (myid == 0 && print_level > 0)
               {
                  mfem::out <<
                            "Line search successful: sufficient decrease in log-barrier objective.\n";
               }
               // accept the trial step
               result.accepted = true;
               break;
            }
         }
         else
         {
            if (thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
            {
               if (myid == 0 && print_level > 0)
               {
                  mfem::out <<
                            "Line search successful: infeasibility or log-barrier objective decreased.\n";
               }
               // accept the trial step
               result.accepted = true;
               break;
            }
         }
      }
      else
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "in filter region\n";
         }
      }
      alpha *= 0.5;
   }
   if (result.accepted)
   {
      result.update_filter = !switchCondition || !sufficientDecrease;
      if (result.update_filter)
      {
         result.filter_theta = (1. - gTheta) * thx0;
         result.filter_phi = phx0 - gPhi * thx0;
      }
   }
   return result;
}

bool IPSolver::FilterCheck(real_t thetax, real_t phix)
{
   bool inFilterRegion = false;
   if (thetax > thetaMax)
   {
      inFilterRegion = true;
   }
   else
   {
      for (int i = 0; i < F1.Size(); i++)
      {
         if (thetax >= F1[i] && phix >= F2[i])
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
   ProjectMultipliers(x, z, dimU, mu, kSig);
}

real_t IPSolver::GetTheta(const BlockVector &x)
{
   x.SyncToBlocks();
   Vector &cx = constraint_residual;
   problem->c(x, cx);
   Vector &Mcx = weighted_constraints;
   MultiplyWeights(Mlump, cx, Mcx);
   return sqrt(InnerProduct(comm, Mcx, cx));
}

// log-barrier objective
real_t IPSolver::GetPhi(const BlockVector &x, real_t mu,
                        int &eval_err)
{
   x.SyncToBlocks();
   real_t fx = problem->CalcObjective(x, eval_err);
   real_t logBarrierLoc = LocalWeightedLogSum(x, Mlump, reduce_workspace);
   real_t logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   const real_t phi = fx - mu * logBarrierGlb;
   if (!IsFinite(phi)) { eval_err = 1; }
   return phi;
}

// gradient of log-barrier objective with respect to x = (u, m)
void IPSolver::GetDxphi(const BlockVector &x, real_t mu,
                        BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   problem->CalcObjectiveGrad(x, y);
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
   real_t fx = problem->CalcObjective(x, eval_err);
   Vector &cx = constraint_residual;
   problem->c(x, cx);
   Vector &temp = slack_work;
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

   problem->DucTranspose(x)->Mult(l, y.GetBlock(0));
   y.GetBlock(1).Add(-1.0, l);

   BlockVector &gradxf = objective_gradient;
   problem->CalcObjectiveGrad(x, gradxf);
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
bool IPSolver::CurvatureTest(const BlockOperator & A,
                             const BlockVector & Xhat, const Vector & l, const BlockVector & b,
                             const real_t & delta)
{
   Xhat.SyncToBlocks();
   b.SyncToBlocks();
   Vector &lplus = multiplier_sum;
   lplus.Set(1.0, l);
   lplus.Add(1.0, Xhat.GetBlock(2));


   real_t dWd = 0.0;
   real_t dd = 0.0;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         if (!A.IsZeroBlock(i, j))
         {
            Vector &temp = hessian_action.GetBlock(i);
            A.GetBlock(i, j).Mult(Xhat.GetBlock(j), temp);
            dWd += InnerProduct(comm, Xhat.GetBlock(i), temp);
         }
      }
      dd += InnerProduct(comm, Xhat.GetBlock(i), Xhat.GetBlock(i));
   }
   real_t lplusTck = -1.0 * InnerProduct(comm, lplus, b.GetBlock(2));

   bool passed = (dWd + fmax(-lplusTck,
                             0.0) >= alphaCurvatureTest * dd);
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
   BlockVector &gradL = lagrangian_gradient;
   EvalLagrangianGradient(x, l, zl, gradL);

   /* constraint function c(u, m) = 0 */
   Vector &cx = constraint_residual;
   problem->c(x, cx);

   /* regularized complementarity
    * |z_i * m_i - mu| */
   real_t complementarityLoc =
      LocalComplementarityError(x, zl, mu, reduce_workspace);

   BlockVector &MxinvgradL = scaled_gradient;
   MxinvgradL.Set(1.0, gradL);
   MxinvgradL.SyncToBlocks();
   MxinvgradL.GetBlock(0) /= Mvlump;
   MxinvgradL.GetBlock(1) /= Mlump;
   MxinvgradL.SyncFromBlocks();
   stationarityError = sqrt(InnerProduct(comm, gradL, MxinvgradL));
   feasibilityError = GlobalLpNorm(infinity(),
                                   FiniteNormlinf(cx, reduce_workspace), comm);
   complementarityError = GlobalLpNorm(infinity(), complementarityLoc, comm);

   stationarity_error = stationarityError;
   feasibility_error = feasibilityError;
   complementarity_error = complementarityError;


   optimalityError = std::max(std::max(stationarityError, feasibilityError),
                              complementarityError);
   if (!IsFinite(stationarityError) || !IsFinite(feasibilityError) ||
       !IsFinite(complementarityError))
   {
      optimalityError = infinity();
   }
   optimality_error = optimalityError;

   if (myid == 0 && print_level > 0)
   {
      mfem::out << "evaluating optimality error for mu = " << mu << std::endl;
      mfem::out << "stationarity error = " << stationarityError << std::endl;
      mfem::out << "feasibility error  = "    << feasibilityError << std::endl;
      mfem::out << "complimentarity error = " << complementarityError << std::endl;
      mfem::out << "optimality error = " << optimalityError << std::endl;
   }
   return optimalityError;
}

}
