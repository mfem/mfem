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

#ifndef MFEM_IPSOLVER
#define MFEM_IPSOLVER

#include "optcontactproblem.hpp"
#include <memory>

namespace mfem
{

/**
 * @class IPSolver
 * @brief Class for interior-point solvers of contact-problems
 *        described by a OptContactProblem
 *
 * IPSolver is an implementation of a primal-dual interior-point
 * algorithm as described in
 * Wächter, Andreas, and Lorenz T. Biegler.
 * "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming."
 * Mathematical programming 106.1 (2006): 25-57.
 *
 * With inertia-free regularization as described in
 * Chiang, Nai-Yuan, and Victor M. Zavala.
 * "An inertia-free filter line-search algorithm for large-scale nonlinear programming."
 * Computational Optimization and Applications 64.2 (2016): 327-354.
 *
 * This solver contains less solver options than say ipopt
 * but the user has full control over the linear solver.
 *
 * This solver is intended to solve problems of the form
 *
 * $$ \min_{u, m} f(u, m) \qquad \text{s.t.} \quad c(u, m) = 0,  m \geq 0 $$
 *
 * In the context of frictionless quasi-static
 * contact mechanics for the displacement variable $d$
 * and slack variable $s$ we have
 *
 * $$ \min_{d, s} E(d, s) \qquad \text{s.t.} \quad g(d) - s = 0, \; s \geq 0 $$
 *
 * The implementation is specialized to a displacement-only objective and
 * affine gap-minus-slack constraints. The reduced Newton solve and curvature
 * test rely on this contact structure.
 *
 * ## Typical usage
 * 1. Initialize IPSolver object.
 * 2. Set solver options e.g.,  SetTol, SetMaxIter, SetLinearSolver
 * 3. Use Mult to apply the solver.
 */
class IPSolver
{
public:
   /// Status of the most recent Mult call.
   enum class TerminationStatus
   {
      NotRun,
      Converged,
      MaximumIterations,
      NonfiniteOptimalityError,
      LinearSolverFailed,
      CurvatureTestFailed,
      LineSearchFailed
   };

   /// Construct interior-point solver
   IPSolver(OptContactProblem*);

   /// Apply the interior-point solver
   void Mult(const Vector&, Vector &);

   /// Apply the interior-point solver
   void Mult(const BlockVector&, BlockVector&);

   /// Set absolute tolerance
   void SetTol(real_t tol) {options.abs_tol = tol;}

   /// Set maximum number of interior-point steps
   void SetMaxIter(int max_it) {options.max_iterations = max_it;}

   /// Set linear solver
   void SetLinearSolver(Solver * solver_) { solver = solver_; };

   /// Set print level
   void SetPrintLevel(int print_level_) { options.print_level = print_level_; };

   /// Print one table row for each completed iterative linear solve.
   void SetPrintLinearSolverStatistics(bool print)
   { options.print_linear_solver_statistics = print; }

   /// Get convergence status of most recent Mult call
   bool GetConverged() const {return state.converged;}

   /// Reason the most recent solve stopped.
   const char *GetTerminationReason() const;

   /// Typed status of the most recent solve.
   TerminationStatus GetTerminationStatus() const { return state.termination_status; }

   /// Final stationarity, equality feasibility, and complementarity errors.
   real_t GetStationarityError() const { return state.stationarity_error; }
   real_t GetFeasibilityError() const { return state.feasibility_error; }
   real_t GetComplementarityError() const { return state.complementarity_error; }
   real_t GetOptimalityError() const { return state.optimality_error; }

   /// get number of interior-point iterations of most recent Mult call
   int GetNumIterations() {return state.newton_solves;};

   /// Get solver iteration counts
   Array<int> & GetLinearSolverIterations() {return state.linear_iterations;};

   virtual ~IPSolver() = default;
protected:
   /// OptContactProblem (not owned).
   OptContactProblem* problem = nullptr;

   /// Linear solver (not owned)
   Solver * solver = nullptr;

   /// Configuration remains fixed during a solve, except through setters.
   struct Options
   {
      real_t abs_tol = 1.e-2;
      int max_iterations = 20;
      int print_level = 0;
      bool print_linear_solver_statistics = false;

      struct LineSearch
      {
         real_t fraction_to_boundary = 0.99;
         real_t armijo_factor = 1.e-4;
         real_t sufficient_progress_factor = 1.0;
         real_t violation_power = 1.1;
         real_t objective_power = 2.3;
         real_t violation_decrease = 1.e-5;
         real_t objective_decrease = 1.e-5;
         int max_backtracks = 20;
      } line_search;

      struct Barrier
      {
         real_t reduction_factor = 0.2;
         real_t reduction_power = 1.5;
         real_t subproblem_tolerance_factor = 1.e1;
         real_t multiplier_deviation = 1.e10;
      } barrier;

      struct Regularization
      {
         real_t curvature_tolerance = 1.e-11;
         real_t minimum = 1.e-20;
         real_t maximum = 1.e40;
         real_t initial = 1.e-4;
         real_t decrease_factor = 1. / 3.;
         real_t initial_increase_factor = 1.e2;
         real_t increase_factor = 8.;
         int max_curvature_tests = 30;
      } regularization;
   } options;

   /// Evolving state. Barrier, filter, regularization and Armijo history retain
   /// their existing lifetime across Mult calls; per-call results reset in Mult.
   struct SolveState
   {
      int newton_solves = 0;
      real_t barrier_parameter = 1.0;
      real_t filter_min_violation = 1.e-4;
      real_t filter_max_violation = 1.e6;
      real_t last_regularization = 0.0;
      Array<real_t> filter_theta, filter_phi;
      bool last_armijo_sufficient_decrease = false;
      Array<int> linear_iterations;
      bool converged = false;
      TerminationStatus termination_status = TerminationStatus::NotRun;
      real_t stationarity_error = infinity();
      real_t feasibility_error = infinity();
      real_t complementarity_error = infinity();
      real_t optimality_error = infinity();
   } state;

   /// Problem dimensions, partitions and fixed lumped mass weights.
   int dimU, dimM, dimC;
   Array<int> constraint_offsets;
   Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
   Vector Mcslump, Mvlump, Mlump;

   int myid = -1;
   MPI_Comm comm;
private:
   /// One owning vector for displacement, slack and both multipliers. The
   /// primal and Newton vectors are views of its first two and three blocks.
   class PrimalDualVector
   {
      BlockVector values, primal, newton;

   public:
      PrimalDualVector(const Array<int> &state_offsets,
                       const Array<int> &primal_offsets,
                       const Array<int> &newton_offsets);
      PrimalDualVector(const PrimalDualVector&) = delete;
      PrimalDualVector &operator=(const PrimalDualVector&) = delete;

      Vector &Displacement() { return values.GetBlock(0); }
      Vector &Slack() { return values.GetBlock(1); }
      Vector &EqualityMultiplier() { return values.GetBlock(2); }
      Vector &InequalityMultiplier() { return values.GetBlock(3); }
      const Vector &Displacement() const { return values.GetBlock(0); }
      const Vector &Slack() const { return values.GetBlock(1); }
      const Vector &EqualityMultiplier() const { return values.GetBlock(2); }
      const Vector &InequalityMultiplier() const { return values.GetBlock(3); }
      BlockVector &Primal() { return primal; }
      const BlockVector &Primal() const { return primal; }
      BlockVector &Newton() { return newton; }

      void Reset();
      void SyncToViews() const;
      void SyncFromBlocks() const;
      void SyncFromNewton() const;
   };

   /// Outcome of the reduced solve and its curvature test.
   struct NewtonResult
   {
      bool converged = false;
      bool passed_curvature_test = false;
   };

   /// The returned filter entry is valid when update_filter is true.
   struct LineSearchResult
   {
      bool accepted = false;
      real_t primal_step = 0.0;
      real_t multiplier_step = 0.0;
      bool update_filter = false;
      real_t filter_theta = 0.0;
      real_t filter_phi = 0.0;
      // Retain the last evaluated Armijo test through non-Armijo trials.
      bool armijo_sufficient_decrease = false;
   };

   /// Reusable storage sized from the problem. Owned matrices are distinct
   /// from borrowed objective Hessians and constraint operators.
   struct Workspace
   {
      HypreParMatrix *Huu = nullptr;
      HypreParMatrix *Ju = nullptr;
      HypreParMatrix *JuT = nullptr;
      std::unique_ptr<HypreParMatrix> Wuu, Wmm;

      // OptimalityError evaluates both gradients; keep separate buffers.
      BlockVector newton_rhs, objective_gradient, lagrangian_gradient;
      BlockVector scaled_gradient, hessian_action;
      BlockVector line_search_trial, line_search_gradient;
      Vector barrier_diagonal, regularization_diagonal;
      Vector reduced_rhs, slack_work, multiplier_sum;
      Vector constraint_residual, weighted_constraints;
      Array<real_t> reduction;
   } workspace;

   /// Update the barrier until the subproblem needs another Newton step.
   void UpdateBarrierForIterate(PrimalDualVector&, int max_updates);

   /// Compute a direction, including the existing regularization retries.
   NewtonResult ComputeSearchDirection(PrimalDualVector&, PrimalDualVector&,
                                       int ip_step);

   /// Apply an accepted step and its filter update, then project multipliers.
   void AcceptStep(PrimalDualVector&, const PrimalDualVector&,
                   const LineSearchResult&);

   /// Form the displacement and diagonal slack Hessians for the reduced solve.
   void FormNewtonBlocks(BlockVector&, const Vector&, real_t delta = 0.0);

   /// Solve the (regularized) IP-Newton linear system
   NewtonResult IPNewtonSolve(PrimalDualVector&, PrimalDualVector&,
                              int ip_step, real_t, real_t delta = 0.0);

   /// Report the last linear solve without changing the output stream format.
   void PrintLinearSolverStatistics(int ip_step, const IterativeSolver&) const;

   /// Max step length that satisfies fraction-to-boundary rule
   real_t GetMaxStepSize(const Vector&, const Vector&, real_t);

   /// Globalizing line search
   LineSearchResult LineSearch(const PrimalDualVector&, const PrimalDualVector&,
                               real_t,
                               bool previous_armijo_sufficient_decrease);

   /// check if a point is acceptable to the filter
   bool FilterCheck(real_t, real_t);

   /// Project inequality constraint multiplier
   void ProjectZ(const Vector &, Vector &, real_t);

   /// Evaluate theta (equality constraint violation measure)
   real_t GetTheta(const BlockVector &);

   /// Evaluate log-barrier objective phi
   real_t GetPhi(const BlockVector &, real_t, int &eval_err);

   /// Gradient of log-barrier objective w.r.t. primal variables
   void GetDxphi(const BlockVector &, real_t, BlockVector &);

   /// Evaluate the primal-dual Lagrangian
   real_t EvalLagrangian(const BlockVector &, const Vector &, const Vector &);

   /// Gradient of the primal-dual Lagrangian w.r.t. primal variables
   void EvalLagrangianGradient(const BlockVector &, const Vector &, const Vector &,
                               BlockVector &);

   /// curvature test to detect negative-curvature
   bool CurvatureTest(const Operator &displacement_hessian,
                      const Operator &slack_hessian, const BlockVector &Xhat,
                      const Vector &l, const BlockVector &b);

   /// Compute the optimality error
   real_t OptimalityError(const BlockVector &, const Vector &, const Vector &,
                          real_t mu = 0.0);

   /// Update the barrier parameter
   void UpdateBarrierSubProblem()
   {
      const auto &barrier = options.barrier;
      const real_t mu = state.barrier_parameter;
      // reduced barrier parameter
      state.barrier_parameter = std::max(options.abs_tol / 10.,
                                         std::min(barrier.reduction_factor * mu,
                                                  pow(mu, barrier.reduction_power)));
      // clear subproblem filter
      state.filter_theta.DeleteAll();
      state.filter_phi.DeleteAll();
   }
};

}

#endif
