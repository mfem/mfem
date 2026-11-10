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
   void SetTol(real_t tol) {abs_tol = tol;}

   /// Set maximum number of interior-point steps
   void SetMaxIter(int max_it) {max_iter = max_it;}

   /// Set linear solver
   void SetLinearSolver(Solver * solver_) { solver = solver_; };

   /// Set print level
   void SetPrintLevel(int print_level_) { print_level = print_level_; };

   /// Print one table row for each completed iterative linear solve.
   void SetPrintLinearSolverStatistics(bool print)
   { print_linear_solver_statistics = print; }

   /// Get convergence status of most recent Mult call
   bool GetConverged() const {return converged;}

   /// Reason the most recent solve stopped.
   const char *GetTerminationReason() const;

   /// Typed status of the most recent solve.
   TerminationStatus GetTerminationStatus() const { return termination_status; }

   /// Final stationarity, equality feasibility, and complementarity errors.
   real_t GetStationarityError() const { return stationarity_error; }
   real_t GetFeasibilityError() const { return feasibility_error; }
   real_t GetComplementarityError() const { return complementarity_error; }
   real_t GetOptimalityError() const { return optimality_error; }

   /// get number of interior-point iterations of most recent Mult call
   int GetNumIterations() {return iter;};

   /// Get solver iteration counts
   Array<int> & GetLinearSolverIterations() {return lin_solver_iterations;};

   virtual ~IPSolver() = default;
protected:
   /// OptContactProblem (not owned).
   OptContactProblem* problem = nullptr;

   /// Linear solver (not owned)
   Solver * solver = nullptr;

   real_t abs_tol;
   int  max_iter;
   int  iter=0;
   real_t mu_k;

   /// interior-point algorithm parameters
   real_t kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
   real_t thetaMax, gTheta, gPhi, kEps;

   // filter
   Array<real_t> F1, F2;

   // The filter policy retains the last evaluated Armijo test, including
   // across Mult calls. Trials using the feasibility criterion do not change it.
   bool last_armijo_sufficient_decrease = false;

   int dimU, dimM, dimC;
   Array<int> constraint_offsets;
   Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;

   /// Objective Hessians (not owned).
   HypreParMatrix * Huu = nullptr;
   HypreParMatrix * Hmm = nullptr;

   /// Regularized Hessian blocks (owned).
   std::unique_ptr<HypreParMatrix> Wuu;
   std::unique_ptr<HypreParMatrix> Wmm;

   /// Constraint Jacobians and transpose (owned by the problem).
   HypreParMatrix * Ju = nullptr;
   HypreParMatrix * Jm = nullptr;
   HypreParMatrix * JuT = nullptr;

   /// Lumped masses
   Vector Mcslump;
   Vector Mvlump;
   Vector Mlump;

   /// inertia-regularization parameters
   real_t alphaCurvatureTest;
   real_t deltaRegLast;
   real_t deltaRegMin;
   real_t deltaRegMax;
   real_t deltaReg0;

   /// inertia-regularization rate parameters
   real_t kRegMinus;
   real_t kRegBarPlus;
   real_t kRegPlus;

   Array<int> lin_solver_iterations;

   bool converged = false;
   TerminationStatus termination_status = TerminationStatus::NotRun;
   real_t stationarity_error = infinity();
   real_t feasibility_error = infinity();
   real_t complementarity_error = infinity();
   real_t optimality_error = infinity();

   int myid = -1;

   /// print level, 0: no printing, > 0 various solver progress output is shown
   int print_level = 0;
   bool print_linear_solver_statistics = false;
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

   /// Scratch storage sized once from the problem dimensions. Keep the
   /// objective and Lagrangian gradients separate: OptimalityError evaluates both.
   BlockVector newton_rhs, objective_gradient, lagrangian_gradient;
   BlockVector scaled_gradient, hessian_action;
   BlockVector line_search_trial, line_search_gradient;
   Vector barrier_diagonal, regularization_diagonal;
   Vector reduced_rhs, slack_work, multiplier_sum;
   Vector constraint_residual, weighted_constraints;
   Array<real_t> reduce_workspace;

   /// Update the barrier until the subproblem needs another Newton step.
   void UpdateBarrierForIterate(PrimalDualVector&, int max_updates);

   /// Compute a direction, including the existing regularization retries.
   NewtonResult ComputeSearchDirection(PrimalDualVector&, PrimalDualVector&,
                                       int ip_step);

   /// Apply an accepted step and its filter update, then project multipliers.
   void AcceptStep(PrimalDualVector&, const PrimalDualVector&,
                   const LineSearchResult&);

   /// Form (regularized) IP-Newton linear system matrix
   void FormIPNewtonMat(BlockVector&, Vector&, Vector&, BlockOperator &,
                        real_t delta = 0.0);

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
   bool CurvatureTest(const BlockOperator & A, const BlockVector & Xhat,
                      const Vector &l, const BlockVector & b, const real_t & delta);

   /// Compute the optimality error
   real_t OptimalityError(const BlockVector &, const Vector &, const Vector &,
                          real_t mu = 0.0);

   /// Update the barrier parameter
   void UpdateBarrierSubProblem()
   {
      // reduced barrier parameter
      mu_k = std::max(abs_tol / 10., std::min(kMu * mu_k, pow(mu_k, thetaMu)));
      // clear subproblem filter
      F1.DeleteAll();
      F2.DeleteAll();
   }
};

}

#endif
