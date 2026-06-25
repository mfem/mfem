// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

// Forward declaration
class HypreSchwarz;

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

   /// Get convergence status of most recent Mult call
   bool GetConverged() const {return converged;}

   /// get number of interior-point iterations of most recent Mult call
   int GetNumIterations() {return iter;};

   /// Get solver iteration counts
   Array<int> & GetLinearSolverIterations() {return lin_solver_iterations;};

   /// Get linear solve wall times in seconds
   Array<real_t> & GetLinearSolverTimes() {return lin_solver_times;};

   virtual ~IPSolver();

   void SetLOBPCG(HypreLOBPCG * lobpcg_) {lobpcg = lobpcg_;};

   void SetSchwarzOptions(bool use_schwarz, bool expand, int cg_iters, real_t weight,
                          int variant = 2, real_t min_diag_val = 0.0, bool examine_diag = false, bool unweighted = false,
                          real_t uniform_weight = -1.0)
   {
      use_schwarz_subspace = use_schwarz;
      schwarz_expand = expand;
      schwarz_cg_iters = cg_iters;
      schwarz_relax_weight = weight;
      schwarz_variant = variant;
      schwarz_min_diag_value = min_diag_val;
      schwarz_examine_diagonal = examine_diag;
      schwarz_unweighted = unweighted;
      schwarz_uniform_weight = uniform_weight;
   }

   void SetSchwarzSolver(HypreSchwarz* schwarz) { schwarz_solver = schwarz; }

   void SetSchurComplementMode(bool use_schur) { use_schur_complement = use_schur; }

protected:
   /// OptContactProblem (not owned).
   OptContactProblem* problem = nullptr;

   /// Linear solver (not owned)
   Solver * solver = nullptr;

   real_t abs_tol;
   int  max_iter;
   int  iter=0;
   real_t mu_k;
   Vector lk, zlk;

   /// interior-point algorithm parameters
   real_t kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
   real_t thetaMax, gTheta, gPhi, kEps;

   // filter
   Array<real_t> F1, F2;

   // quantities computed in lineSearch
   real_t alpha, alphaz;
   real_t thx0;
   real_t phx0;
   bool switchCondition = false;
   bool sufficientDecrease = false;
   bool lineSearchSuccess = false;

   int dimU, dimM, dimC;
   Array<int> constraint_offsets;
   Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;

   /// Operators (not owned)
   HypreParMatrix * Huu = nullptr;
   HypreParMatrix * Hmm = nullptr;
   HypreParMatrix * Wuu = nullptr;
   HypreParMatrix * Wmm = nullptr;
   HypreParMatrix * Ju = nullptr;
   HypreParMatrix * Jm = nullptr;
   HypreParMatrix * JuT = nullptr;
   HypreParMatrix * JmT = nullptr;

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
   Array<real_t> lin_solver_times;

   bool converged = false;

   int myid = -1;

   /// print level, 0: no printing, > 0 various solver progress output is shown
   int print_level = 0;
   MPI_Comm comm;

   HypreLOBPCG * lobpcg = nullptr;

   // Schwarz solver options
   bool use_schwarz_subspace = false;
   bool schwarz_expand = false;
   int schwarz_cg_iters = 0;
   int schwarz_variant = 2;
   real_t schwarz_relax_weight = 1.0;
   real_t schwarz_min_diag_value = 0.0;
   bool schwarz_examine_diagonal = false;
   bool schwarz_unweighted = false;
   real_t schwarz_uniform_weight = -1.0;
   HypreSchwarz* schwarz_solver = nullptr;

   // Schur complement solver option
   bool use_schur_complement = false;

private:
   /// Form (regularized) IP-Newton linear system matrix
   void FormIPNewtonMat(BlockVector&, Vector&, Vector&, BlockOperator &,
                        real_t delta = 0.0);

   /// Solve the (regularized) IP-Newton linear system
   void IPNewtonSolve(BlockVector&, Vector&, Vector&, Vector&, BlockVector&,
                      bool &, real_t, real_t delta = 0.0);

   /// Max step length that satisfies fraction-to-boundary rule
   real_t GetMaxStepSize(Vector&, Vector&, real_t);

   /// Globalizing line search
   void LineSearch(BlockVector&, BlockVector&, real_t);

   /// check if a point is acceptable to the filter
   bool FilterCheck(real_t, real_t);

   /// Project inequality constraint multiplier
   void ProjectZ(const Vector &, Vector &, real_t);

   /// Build Schwarz subdomains from contact pairs
   void BuildSchwarzSubdomains(HypreParMatrix* Areduced, const BlockVector& x);

   /// Evaluate theta (equality constraint violation measure)
   real_t GetTheta(const BlockVector &);

   /// Evaluate log-barrier objective phi
   real_t GetPhi(const BlockVector &, real_t, int eval_err = 0);

   /// Gradient of log-barrier objective w.r.t. primal variables
   void GetDxphi(const BlockVector &, real_t, BlockVector &);

   /// Evaluate the primal-dual Lagrangian
   real_t EvalLangrangian(const BlockVector &, const Vector &, const Vector &);

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
