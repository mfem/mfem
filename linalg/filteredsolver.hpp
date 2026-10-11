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

#ifndef MFEM_FILTEREDSOLVER
#define MFEM_FILTEREDSOLVER

#include "../config/config.hpp"
#include "operator.hpp"
#include "handle.hpp"
#include <memory>
#include <functional>

namespace mfem
{
/**
* @class FilteredSolver
* @brief Base class for solvers with filtering (subspace correction).
*
* FilteredSolver is designed to augment an existing solver with an additional
* filtering step that targets small subspaces where the solver is less
* effective. The filtered subspace is defined by a transfer operator @p P,
* which maps the subspace into the full space.
*
* ### Typical usage
* 1. Call SetOperator() to define the operator @p A that acts on the full space.
* 2. Call SetSolver() to provide the underlying solver @p B for the full-space operator.
* 3. Call SetFilteredSubspaceTransferOperator() to set transfer operator @p P.
* 4. Call SetFilteredSubspaceSolver() to set the subspace solver @p S.
* 5. Use Mult() to apply the solver.
*
* ---
*
* The preconditioner applied by Mult() with the default Mode::MULTIPLICATIVE
* is
*  $$
* M = B + P S P^T (I - A B) + B (I - A P S P^T) (I- A B),
*  $$
* and the corresponding iteration matrix is
*  $$
* I - M A = (I - B A) (I - P S P^T A) (I - B A).
*  $$
*
* Two other combination modes are available via SetMode(): Mode::REVERSED
* applies the same three-stage combination with the roles of @p B and the
* subspace correction $ P S P^T $ swapped, and Mode::ADDITIVE applies a
* single application of each, summed, i.e. $ M = B + P S P^T $.
*/
class FilteredSolver : public Solver
{
public:
   /**
   * @brief Order in which the base solver @a B and the subspace correction
   * $ P S P^T $ are combined in Mult().
   *
   * - MULTIPLICATIVE (default): B, then the subspace correction, then B
   *   again, each stage using the residual updated by the previous one. This
   *   is the scheme documented in the class-level formula above.
   * - REVERSED: the same three-stage multiplicative combination, but with
   *   the roles of @a B and the subspace correction swapped, i.e. subspace
   *   correction, then B, then subspace correction again.
   * - ADDITIVE: a single application of @a B and a single application of
   *   the subspace correction, both using the original residual and summed,
   *   i.e. $ M = B + P S P^T $, with no intermediate residual update.
   */
   enum class Mode
   {
      MULTIPLICATIVE,
      REVERSED,
      ADDITIVE
   };

   /// Construct an empty filtered solver. Must set operator and solver before use.
   FilteredSolver() : Solver() { }

   /// Set the system operator @a A.
   virtual void SetOperator(const Operator &A) override;

   /// Set the solver @a B that operates on the full space.
   virtual void SetSolver(Solver &B);

   /// Set the transfer operator @a P from filtered subspace to the full space.
   void SetFilteredSubspaceTransferOperator(const Operator &P);

   /// Set a solver @a S that operates on the filtered subspace operator $ P^T A P $.
   void SetFilteredSubspaceSolver(Solver &S);

   /// Set the combination mode used by Mult(). Default is Mode::MULTIPLICATIVE.
   void SetMode(Mode mode_) { mode = mode_; }

   /// Get the combination mode used by Mult().
   Mode GetMode() const { return mode; }

   /// Apply the filtered solver
   void Mult(const Vector &x, Vector &y) const override;

   virtual ~FilteredSolver() = default;

   FilteredSolver(const FilteredSolver&) = delete;
   FilteredSolver& operator=(const FilteredSolver&) = delete;
   FilteredSolver(FilteredSolver&&) = default;
   FilteredSolver& operator=(FilteredSolver&&) = default;

protected:
   /// System operator (not owned).
   const Operator * A = nullptr;
   /// Transfer operator (not owned).
   const Operator * P = nullptr;
   /// Base solver (not owned).
   Solver * B = nullptr;
   /// Subspace solver (not owned).
   Solver * S = nullptr;
   /// Projected operator.
   mutable std::unique_ptr<const Operator> PtAP = nullptr;
   /// Initialize work vectors.
   void InitVectors() const;
   bool mutable solver_set = false;
   /// Combination mode used by Mult().
   Mode mode = Mode::MULTIPLICATIVE;
private:

   /// Build and/or return cached projected operator $ P^T A P $.
   std::unique_ptr<const Operator> GetPtAP(const Operator *Aop,
                                           const Operator *Pop) const;
   /// Finalize solver
   void MakeSolver() const;

   /// Mult() implementation for Mode::MULTIPLICATIVE: B, subspace, B.
   void MultMultiplicative(const Vector &b, Vector &x) const;
   /// Mult() implementation for Mode::REVERSED: subspace, B, subspace.
   void MultReversed(const Vector &b, Vector &x) const;
   /// Mult() implementation for Mode::ADDITIVE: B + subspace, applied once each.
   void MultAdditive(const Vector &b, Vector &x) const;

   // Work vectors used in Mult.
   mutable Vector z;
   mutable Vector rf;
   mutable Vector xf;
   mutable Vector r;

}; // mfem::FilteredSolver class


#ifdef MFEM_USE_MPI
/**
* @class AMGFSolver
* @brief AMG with Filtering: specialization of FilteredSolver.
*
* AMGFSolver is a convenience wrapper that fixes the base solver @a B of a
* FilteredSolver to HypreBoomerAMG.
* AMGF is particularly effective for constrained optimization
* problems such as frictionless contact. For more details, see:
* [AMG with Filtering: An Efficient Preconditioner for Interior Point Methods in Large-Scale Contact Mechanics Optimization](https://arxiv.org/abs/2505.18576)
*
* The internal HypreBoomerAMG instance can be accessed and configured via AMG().
*/
class AMGFSolver : public FilteredSolver
{
private:
   /// Owned HypreBoomerAMG instance.
   std::unique_ptr<HypreBoomerAMG> amg;
   /// System operator, cached as a HypreParMatrix (not owned).
   const HypreParMatrix *Ah = nullptr;
   /// Transfer operator built by GenerateFilteredSubspaceTransferOperator()
   /// (owned; null if it was never called or found no filtered subspace).
   std::unique_ptr<HypreParMatrix> generated_transfer;
   /// False once GenerateFilteredSubspaceTransferOperator() determines that
   /// the row-norm clusters are not well separated; Mult() then falls back
   /// to a single application of the base AMG solver.
   bool filtering_enabled = true;
   /// If true, SetOperator() calls GenerateFilteredSubspaceTransferOperator();
   /// see EnableAutoFilteredSubspace().
   bool auto_subspace = false;
   /// Maximum number of EM iterations passed to
   /// GenerateFilteredSubspaceTransferOperator() when auto_subspace is true.
   int auto_max_iter = 20;
   /// EM convergence tolerance passed to
   /// GenerateFilteredSubspaceTransferOperator() when auto_subspace is true.
   real_t auto_tol = 1e-3;
   /// Cluster-mean jump threshold passed to
   /// GenerateFilteredSubspaceTransferOperator() when auto_subspace is true.
   real_t auto_jump_threshold = 10.0;
   /// Builds a fresh solver for the auto-generated filtered subspace; see
   /// EnableAutoFilteredSubspace(). May be called more than once, since the
   /// subspace size can change every time it is regenerated.
   std::function<std::unique_ptr<Solver>()> auto_subspace_solver_factory;
   /// Solver most recently built by auto_subspace_solver_factory (owned).
   std::unique_ptr<Solver> auto_subspace_solver;
   /// Local (this rank's) width of the filtered subspace that
   /// auto_subspace_solver was built for, so a new one is only built when the
   /// subspace size actually changes on some rank; -1 if auto_subspace_solver
   /// has not been built yet.
   int auto_subspace_local_width = -1;
public:
   /// Construct AMGF solver with default HypreBoomerAMG.
   AMGFSolver() : FilteredSolver()
   {
      amg = std::make_unique<HypreBoomerAMG>();
      FilteredSolver::SetSolver(*amg);
   }

   /// Set the system operator @a A.
   virtual void SetOperator(const Operator &A) override;

   /// Access to the internal HypreBoomerAMG instance.
   HypreBoomerAMG& GetAMG() { return *amg; }

   /// Const access to the internal HypreBoomerAMG instance.
   const HypreBoomerAMG& GetAMG() const { return *amg; }

   void SetSolver(Solver &B) override
   {
      MFEM_ABORT("SetSolver is not supported in AMGFSolver. It is set to AMG by default");
   }

   /// Set the parallel transfer operator @a P for the filtered subspace.
   void SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop);

   /**
   * @brief Automatically build a filtered subspace transfer operator from
   * the row magnitudes of the system operator previously set via
   * SetOperator().
   *
   * This heuristic targets systems where a small number of rows have an L1
   * norm that is much larger than the rest, such as the penalty or
   * Lagrange-multiplier rows that appear in the interior-point formulation
   * used for frictionless contact. The procedure is:
   *  1. Compute the L1 norm of every row of the system operator (collected
   *     across all MPI ranks, since the operator is distributed).
   *  2. Fit a two-component 1D Gaussian Mixture Model (GMM) to the row
   *     norms using Expectation-Maximization; every step is computed in
   *     parallel across ranks (no row norm data is gathered to one rank).
   *  3. If the mean of the larger-norm cluster is more than @a
   *     jump_threshold times the mean of the smaller-norm cluster, build
   *     and install (via SetFilteredSubspaceTransferOperator()) a transfer
   *     operator whose subspace is the set of rows assigned to the
   *     larger-norm cluster.
   *  4. Otherwise, no clear separation was found: filtering is disabled,
   *     and Mult() falls back to a single application of the base AMG
   *     solver until this function is called again.
   *
   * This method is collective over the communicator of the operator, and
   * SetOperator() must have been called first. The operator is not owned and
   * must remain valid while this object uses it.
   *
   * The generated transfer operator is owned by this object and replaces any
   * transfer operator set previously via SetFilteredSubspaceTransferOperator().
   * Since the subspace size is not known in advance, the subspace solver must
   * be able to handle it: if a solver factory was given to
   * EnableAutoFilteredSubspace(), a new subspace solver is built and installed
   * whenever the subspace size changes on any rank; otherwise the solver set
   * via SetFilteredSubspaceSolver() is kept, and it is the caller's
   * responsibility that it supports the new subspace operator.
   *
   * See the `-amgf-auto-subspace` option of the contact miniapp
   * (miniapps/contact/contact.cpp) for an example of use.
   *
   * @param max_iter Maximum number of EM iterations for the GMM fit.
   * @param tol Relative tolerance on the change of the two cluster means,
   *            used as the EM convergence criterion.
   * @param jump_threshold The larger-norm cluster is used as the filtered
   *                        subspace only if its mean exceeds this many
   *                        times the smaller-norm cluster's mean.
   * @return true if a filtered subspace was generated and installed; false
   *         if filtering was disabled because no sufficient separation was
   *         found between the two clusters.
   */
   bool GenerateFilteredSubspaceTransferOperator(int max_iter = 20,
                                                 real_t tol = 1e-3,
                                                 real_t jump_threshold = 10.0);

   /**
   * @brief Enable or disable automatic filtered subspace generation.
   *
   * When enabled, every call to SetOperator() automatically calls
   * GenerateFilteredSubspaceTransferOperator() with the given parameters on
   * the newly installed operator, instead of requiring the caller to invoke
   * it manually. This is useful when AMGFSolver is driven indirectly, e.g.
   * as the preconditioner of an IterativeSolver whose SetOperator() cascades
   * to the preconditioner's SetOperator() on every linear solve (such as
   * every Newton iteration of an interior-point optimizer): the filtered
   * subspace is then automatically refreshed from the current operator each
   * time, with no other code changes required.
   *
   * Because the size of an automatically-generated subspace can change
   * every time it is regenerated, a fixed subspace solver (as set by
   * SetFilteredSubspaceSolver()) generally cannot be reused across calls:
   * most direct solvers assume a fixed local operator size for their
   * lifetime. Instead, @a solver_factory is called to build a fresh subspace
   * solver whenever the local subspace width changes on any rank (even if
   * the global width does not); AMGFSolver owns the resulting
   * solver and destroys it when it is replaced or when *this is destroyed.
   * Calling this function does not discard the transfer operator and subspace
   * solver generated so far, even when @a enable is false: they stay in use
   * until a later generation or one of the setters replaces them.
   *
   * While automatic generation is enabled, every SetOperator() call replaces
   * the transfer operator and, when the subspace size changes, the subspace
   * solver, so a transfer operator or subspace solver set manually via
   * SetFilteredSubspaceTransferOperator() or SetFilteredSubspaceSolver() is
   * overwritten.
   *
   * See the `-amgf-auto-subspace` option of the contact miniapp
   * (miniapps/contact/contact.cpp) for an example of use.
   *
   * @param enable Turn automatic generation on or off.
   * @param solver_factory Builds a new solver for the filtered subspace
   *        operator; called on the first use and again whenever the local
   *        subspace width changes on any rank. Required if @a enable is
   *        true.
   * @param max_iter,tol,jump_threshold Forwarded to
   *        GenerateFilteredSubspaceTransferOperator() on every SetOperator()
   *        call.
   */
   void EnableAutoFilteredSubspace(
      bool enable,
      std::function<std::unique_ptr<Solver>()> solver_factory,
      int max_iter = 20, real_t tol = 1e-3, real_t jump_threshold = 10.0);

   /// True if Mult() applies the filtered correction; false if the last
   /// call to GenerateFilteredSubspaceTransferOperator() found no clear
   /// separation, so Mult() falls back to a single AMG application.
   bool FilteringEnabled() const { return filtering_enabled; }

   /// Apply the solver, falling back to a single application of the base
   /// AMG solver when filtering has been disabled by
   /// GenerateFilteredSubspaceTransferOperator().
   void Mult(const Vector &b, Vector &x) const override;

   /// Destructor.
   ~AMGFSolver() override = default;

};
#endif


} // namespace mfem

#endif
