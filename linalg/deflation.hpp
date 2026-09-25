// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. See file CONTRIBUTING.md for details.

#ifndef MFEM_DEFLATION_HPP
#define MFEM_DEFLATION_HPP

#include "solvers.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace mfem
{

/// Select the projected solve or balanced coarse-correction preconditioner.
enum class CoarseCorrectionType { PROJECTED, BALANCED };
/// Select the default direct coarse solve or an opt-in LAPACK SVD solve.
enum class CoarseSolveMethod { DIRECT, SVD };
/// Select rejection or explicit projection of incompatible right-hand sides.
enum class IncompatibleRHSAction { REJECT, PROJECT };

/** A threshold of -1 selects the automatic, precision-aware default.
    Explicit values, including zero, override it. Automatic rank and coarse
    reciprocal-condition thresholds are respectively
    max(1e-10, 32*k*epsilon) and max(1e-12, 32*k*epsilon), where k is the
    relevant null/coarse column count and epsilon is real_t machine epsilon.
    DIRECT measures coarse conditioning in the 1-norm; SVD uses the 2-norm. */
struct DeflationSetupOptions
{
   /// Relative threshold for rank tests after global column normalization.
   real_t basis_rank_rtol = real_t(-1);
   /// Minimum acceptable reciprocal condition number of the coarse matrix.
   real_t coarse_rcond_min = real_t(-1);
   /// SVD requires MFEM_USE_LAPACK; DIRECT uses Cholesky or pivoted LU.
   CoarseSolveMethod coarse_solve_method = CoarseSolveMethod::DIRECT;
   /** Cache A^T V during a nonsymmetric setup so that P_R needs no operator
       application. This calls A.MultTranspose() once per coarse column and
       stores that many extra fine vectors. Symmetric setups always reuse the
       cached A U instead and ignore this option. */
   bool cache_transpose_images = false;
};

/// RHS compatibility policy and Euclidean norm tolerances.
struct DeflationRHSOptions
{
   /// Whether to reject or explicitly project an incompatible RHS.
   IncompatibleRHSAction action = IncompatibleRHSAction::REJECT;
   /// Absolute compatibility tolerance.
   real_t atol = real_t(0);
   /// Relative compatibility tolerance.
   real_t rtol = std::max(real_t(1e-10), real_t(32) *
                          std::numeric_limits<real_t>::epsilon());
};

/// Statistics for the most recent completed Mult(); setters clear validity.
struct DeflationSolveInfo
{
   /// True only after a completed Mult().
   bool valid = false;
   /// True when the original RHS passes the configured compatibility test.
   bool rhs_compatible = false;
   /// Physical convergence for the projected working RHS.
   bool converged_working_system = false;
   /// Physical convergence for the original RHS and its own initial scale.
   bool converged_original_system = false;
   /// Total Krylov iterations across all restarts.
   int iterations = 0;
   /// Norm of the removed left-null component of the RHS.
   real_t removed_rhs_norm = real_t(0);
   /// Working residual norm at the pre-coarse-correction x0.
   real_t initial_working_residual_norm = real_t(0);
   /// Working residual norm of the returned physical solution.
   real_t final_working_residual_norm = real_t(0);
   /// Original-system residual norm at x0.
   real_t initial_original_residual_norm = real_t(0);
   /// Original-system residual norm of the returned solution.
   real_t final_original_residual_norm = real_t(0);
   /// Norm of the returned solution's right-null component.
   real_t solution_null_component_norm = real_t(0);
};

/** An explicit set of vectors on this rank's owned true DOFs, used as a null
    or coarse basis. Coordinates are replicated: Mult() combines the stored
    vectors locally with the same coefficients on every rank, so every rank
    must hold the same number of vectors. DeflationSpaces copies the stored
    vectors directly instead of applying Mult() once per column. */
class VectorDeflationBasis : public Operator
{
private:
   std::vector<Vector> vectors;

public:
   /// @param local_size Number of owned true DOFs on this rank.
   explicit VectorDeflationBasis(int local_size) : Operator(local_size, 0) { }

   /// Append a copy of @a v, whose size must equal Height().
   void Add(const Vector &v);
   /// Append @a v, taking over its storage.
   void Add(Vector &&v);
   /// Remove every vector.
   void Clear();
   /// Return the number of stored vectors, equal to Width().
   int NumVectors() const { return width; }
   /// Return stored vector @a j.
   const Vector &GetVector(int j) const;
   /// y = sum_j x(j) v_j; @a x holds replicated coordinates.
   void Mult(const Vector &x, Vector &y) const override;
};

/** Common algebra for independent exact null and acceleration spaces.
    A and all input bases are borrowed. Calls are collective with the MPI
    constructor and this object is non-reentrant. Setup does not verify
    null-space completeness.

    Two coarse-space representations are supported:
    - Dense (SetCoarseSpace(Z)): Z maps replicated coordinates to owned
      true-DOF rows; its transpose is unnecessary. Setup materializes,
      null-projects, and orthonormalizes the columns, forms the replicated
      matrix V^T A U, and factors it. A VectorDeflationBasis is copied
      directly. This path suits tens to hundreds of global modes.
    - Operator (SetCoarseSpace(Z, S)): Z is a general operator from its own,
      possibly distributed, coarse space to owned true-DOF rows, and its
      MultTranspose() must be the global transpose, as for a HypreParMatrix.
      Nothing is materialized. Setup forms E = Z_L^T A Z_R, sparsely with
      RAP when A and Z are HypreParMatrix objects and as an RAPOperator
      otherwise, and binds the user solver S to it. Exact null spaces give
      Pi_L A Pi_R = A, so no candidate projection is needed; the correction
      is Q = Pi_R Z_R S Z_L^T Pi_L. This path suits large sparse subspaces.

    Null bases are always materialized and orthonormalized. Fine vectors use
    MFEM device-aware storage and operations; replicated coarse coordinates
    and dense factors remain on the host. */
class DeflationSpaces
{
private:
   /// Own materialized bases, replicated coarse factors, and MPI state.
   class Impl;
   std::unique_ptr<Impl> impl;

public:
   /// Construct a local, serial algebra object even in an MPI build.
   DeflationSpaces();
#ifdef MFEM_USE_MPI
   /// Borrow the communicator used for every global reduction.
   explicit DeflationSpaces(MPI_Comm comm);
#endif
   ~DeflationSpaces();
   DeflationSpaces(const DeflationSpaces &) = delete;
   DeflationSpaces &operator=(const DeflationSpaces &) = delete;

   /// Register the borrowed original square operator.
   void SetOperator(const Operator &A);
   /// Return the borrowed original operator, if registered.
   const Operator *GetOriginalOperator() const;
   /// Register one basis for both exact null spaces.
   void SetNullSpace(const Operator &N);
   /// Register distinct complete right and left exact null bases.
   void SetNullSpaces(const Operator &N_R, const Operator &N_L);
   /// Clear only null-space registration.
   void ClearNullSpace();
   /// Register shared right and left acceleration candidates (dense path).
   void SetCoarseSpace(const Operator &Z);
   /// Register distinct trial and test acceleration candidates (dense path).
   void SetCoarseSpaces(const Operator &Z_R, const Operator &Z_L);
   /** Register a shared operator coarse space with a borrowed coarse solver
       @a S for Z^T A Z. Setup calls S.SetOperator(). Pass
       @a exact_coarse_solve = true only if S applies (Z^T A Z)^{-1}, e.g. a
       direct solver; PROJECTED correction requires it. */
   void SetCoarseSpace(const Operator &Z, Solver &S,
                       bool exact_coarse_solve = false);
   /// Register paired operator coarse spaces with a solver for Z_L^T A Z_R.
   void SetCoarseSpaces(const Operator &Z_R, const Operator &Z_L, Solver &S,
                        bool exact_coarse_solve = false);
   /// Clear only coarse-space registration.
   void ClearCoarseSpace();
   /// Set rank, conditioning, and coarse solve options, invalidating setup.
   void SetSetupOptions(const DeflationSetupOptions &options);
   /// Return raw configured thresholds; -1 denotes automatic selection.
   const DeflationSetupOptions &GetSetupOptions() const;
   /// Whether null bases are registered.
   bool HasNullSpace() const;
   /// Whether coarse candidates are registered.
   bool HasCoarseSpace() const;
   /// Whether the null registration used the shared setter.
   bool UsesSharedNullBasis() const;
   /// Whether the coarse registration used the shared setter.
   bool UsesSharedCoarseBasis() const;
   /// Whether the coarse space uses the operator path with a user solver.
   bool UsesCoarseSolver() const;
   /// Whether the coarse solve is exact: always on the dense path.
   bool HasExactCoarseSolve() const;
   /// Whether all derived algebra is valid.
   bool IsSetup() const;
   /** Build orthonormal bases and the selected coarse solve. Passing true
       asserts that A is symmetric and positive definite on the null-space
       complement, and that the null and coarse bases are shared. Setup then
       uses V = U and Cholesky, and ApplySolutionProjector() uses the cached
       A U in place of V^T A. For nonsymmetric A this gives a wrong P_R, so
       pass false. */
   void Setup(bool symmetric_positive_definite = false);
   /// Force rebuilding after in-place input changes; see Setup().
   void Update(bool symmetric_positive_definite = false);
   /// Return the registered null basis dimension after setup.
   int GetNullSpaceDimension() const;
   /** Return the global acceleration coarse dimension after setup. On the
       operator path this is the sum of the local widths of Z_R, so the call
       is collective there. */
   int GetCoarseSpaceDimension() const;
   /// Return the replicated V^T A U of the dense path; empty without one.
   const DenseMatrix &GetCoarseMatrix() const;
   /** Return the coarse operator after setup: the dense V^T A U, or the
       operator path's Z_L^T A Z_R. Null without a coarse space. */
   const Operator *GetCoarseOperator() const;
   /// Apply Pi_R, allowing x and y to alias.
   void ProjectSolution(const Vector &x, Vector &y) const;
   /// Apply Pi_L, allowing b and y to alias.
   void ProjectRHS(const Vector &b, Vector &y) const;
   /// Apply Q = U (V^T A U)^{-1} V^T, or Pi_R Z_R S Z_L^T Pi_L.
   void ApplyCoarseCorrection(const Vector &r, Vector &y) const;
   /// Apply P_L = I - A Q.
   void ApplyResidualProjector(const Vector &r, Vector &y) const;
   /** Apply P_L and Q with one coarse restriction. The input may alias
       either output. Overlapping output storage, including distinct Vector
       views, is rejected. */
   void ApplyResidualProjectorAndCoarseCorrection(
      const Vector &r, Vector &projected, Vector &coarse) const;
   /// Apply P_R = I - Q A; see Setup() and cache_transpose_images.
   void ApplySolutionProjector(const Vector &x, Vector &y) const;
   /// Return the norm of the RHS component removed by Pi_L.
   real_t GetIncompatibleRHSNorm(const Vector &b) const;
   /// Test the configured Euclidean RHS compatibility inequality.
   bool IsConsistent(const Vector &b, real_t atol, real_t rtol) const;
   /// Optionally test A N_R and A^T N_L; completeness is not established.
   bool ValidateNullSpaces(real_t absolute_tolerance) const;
};

/** Shared solver lifecycle. CG requires a symmetric positive semidefinite A,
    shared spaces, and a fixed SPD fine preconditioner on the complement.
    GMRES requires a fixed linear fine preconditioner; FGMRES permits a
    varying one. Paired null spaces require an explicit preconditioner which
    maps the left complement invertibly to the right complement. These are
    caller preconditions, not properties that Setup can prove.

    PROJECTED solves P_L A y = P_L Pi_L b and returns
    Q Pi_L b + Pi_R P_R y. BALANCED applies
    Pi_R (P_R M^{-1} P_L + Q) Pi_L and returns Pi_R x_internal.
    Both b and the original operator are preserved. b and x must not alias;
    projector methods do support aliasing. Inputs and the fine preconditioner
    are borrowed and must outlive registration. Update rebuilds after in-place
    input changes. The solver is non-reentrant.

    Iterations run in an owned native Krylov solver. Native convergence is
    verified on the physical residual, with bounded continuation within the
    original iteration budget. A controller or iteration output enables
    physical verification at every accepted iteration; otherwise transient
    physical convergence within a native call can be missed. User callbacks
    receive physical iterates and total iteration counts, with one reset per
    outer solve. Deflated GMRES/FGMRES always use two orthogonalization passes.
    Explicitly qualified calls to the base Krylov Mult are unsupported. */
/// Only CGSolver, GMRESSolver, and FGMRESSolver are explicitly instantiated.
template <typename KrylovSolver>
class DeflatedSolverBase : public KrylovSolver
{
private:
   /// Own shared solver algebra, adapters, and solve diagnostics.
   class Impl;
   mutable std::unique_ptr<Impl> impl;
   /// Invalidate setup binding and previous solve statistics.
   void Invalidate();
   /// Clear solve diagnostics without rebuilding algebra or preconditioner.
   void ResetSolveInfo();
   /// Report wrapper-detected failures on all ranks before the next stage.
   void CheckCollective(bool good, const char *message) const;
   /// Stop the collective solve when any rank's controller requests it.
   bool CollectiveStop(bool local_stop) const;
   /// Whether @a local holds on any rank; collective on the MPI constructor.
   bool AnyRank(bool local) const;

protected:
   /// Construct with a local Krylov inner product and stable internal adapters.
   DeflatedSolverBase();
#ifdef MFEM_USE_MPI
   /// Construct with global Krylov products over the borrowed communicator.
   explicit DeflatedSolverBase(MPI_Comm comm);
#endif
   /// Return the inherited restart dimension, or zero for CG.
   int RestartDimension() const;

public:
   ~DeflatedSolverBase() override;
   DeflatedSolverBase(const DeflatedSolverBase &) = delete;
   DeflatedSolverBase &operator=(const DeflatedSolverBase &) = delete;
   /// Register the original operator, never an internal projection.
   void SetOperator(const Operator &A) override;
   /// Register a borrowed fine preconditioner for the original operator.
   void SetPreconditioner(Solver &M_inverse) override;
   /// Restore the identity fine preconditioner where admissible.
   void ClearPreconditioner();
   /// Return the borrowed original operator, if registered.
   const Operator *GetOriginalOperator() const;
   /// Return the borrowed fine preconditioner, if registered.
   Solver *GetUserPreconditioner() const;
   /// Register one complete basis for both exact null spaces.
   void SetNullSpace(const Operator &N);
   /// Register distinct complete null bases; rejected by CG.
   void SetNullSpaces(const Operator &N_R, const Operator &N_L);
   /// Clear only the null-space registration.
   void ClearNullSpace();
   /// Register shared acceleration candidates.
   void SetCoarseSpace(const Operator &Z);
   /// Register paired acceleration candidates; rejected by CG.
   void SetCoarseSpaces(const Operator &Z_R, const Operator &Z_L);
   /** Register a shared operator coarse space with a coarse solver; see
       DeflationSpaces. CG requires a fixed SPD S; PROJECTED correction
       requires @a exact_coarse_solve. */
   void SetCoarseSpace(const Operator &Z, Solver &S,
                       bool exact_coarse_solve = false);
   /// Register paired operator coarse spaces; rejected by CG.
   void SetCoarseSpaces(const Operator &Z_R, const Operator &Z_L, Solver &S,
                        bool exact_coarse_solve = false);
   /// Clear only the coarse-space registration.
   void ClearCoarseSpace();
   /// Choose projected or balanced correction; balanced is the default.
   void SetCoarseCorrectionType(CoarseCorrectionType type);
   /// Return the configured correction type.
   CoarseCorrectionType GetCoarseCorrectionType() const;
   /// Configure rank, conditioning, and coarse solve options.
   void SetSetupOptions(const DeflationSetupOptions &options);
   /// Return the configured setup thresholds.
   const DeflationSetupOptions &GetSetupOptions() const;
   /// Configure incompatible RHS handling.
   void SetRHSOptions(const DeflationRHSOptions &options);
   /// Return the configured incompatible RHS policy.
   const DeflationRHSOptions &GetRHSOptions() const;
   /// Build deflation algebra and bind the fine preconditioner to original A.
   void Setup();
   /// Force rebuilding after in-place changes to registered inputs.
   void Update();
   /// Whether setup and internal adapters are current.
   bool IsSetup() const;
   /// Return the common projection and coarse algebra.
   const DeflationSpaces &GetDeflationSpaces() const;
   /// Return statistics for the most recent completed solve.
   const DeflationSolveInfo &GetSolveInfo() const;
   /// Solve and return a physical, right-null-free solution.
   void Mult(const Vector &b, Vector &x) const override;
};

/// Deflated PCG for symmetric operators positive on the null complement.
class DeflatedCGSolver final : public DeflatedSolverBase<CGSolver>
{
public:
   /// Construct a local solver.
   DeflatedCGSolver();
#ifdef MFEM_USE_MPI
   /// Construct a solver with global dot products over the communicator.
   explicit DeflatedCGSolver(MPI_Comm comm);
#endif
};

/// Deflated restarted GMRES with fixed linear fine preconditioning.
class DeflatedGMRESSolver final : public DeflatedSolverBase<GMRESSolver>
{
public:
   /// Construct a local solver.
   DeflatedGMRESSolver();
#ifdef MFEM_USE_MPI
   /// Construct a solver with global dot products over the communicator.
   explicit DeflatedGMRESSolver(MPI_Comm comm);
#endif
};

/// Deflated FGMRES allowing variable fine preconditioning within a solve.
class DeflatedFGMRESSolver final : public DeflatedSolverBase<FGMRESSolver>
{
public:
   /// Construct a local solver.
   DeflatedFGMRESSolver();
#ifdef MFEM_USE_MPI
   /// Construct a solver with global dot products over the communicator.
   explicit DeflatedFGMRESSolver(MPI_Comm comm);
#endif
};

} // namespace mfem

#endif // MFEM_DEFLATION_HPP
