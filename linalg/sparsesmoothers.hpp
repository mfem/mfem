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

#ifndef MFEM_SPARSEMATSMOOTHERS
#define MFEM_SPARSEMATSMOOTHERS

#include "../config/config.hpp"
#include "sparsemat.hpp"

#include <memory>

namespace mfem
{

/// Abstract base class for smoothers created from a SparseMatrix.
class SparseSmoother : public MatrixInverse
{
protected:
   const SparseMatrix *oper = nullptr; ///< The underlying matrix.

   /// Pointer to the transpose of the underlying matrix. If the matrix is
   /// symmetric, this will be the same as @a oper. If the matrix is not
   /// symmetric, the transpose will be formed and stored in @a At. The
   /// transpose will only be formed if MultTranspose() is called.
   mutable const SparseMatrix *oper_T = nullptr;

   mutable std::unique_ptr<SparseMatrix> At; ///< Transpose of A, if needed.

   void EnsureTranspose() const; ///< Ensure that the transpose is set.

public:
   SparseSmoother() = default;

   SparseSmoother(const SparseMatrix &a) { SetOperator(a); }

   /// Sets the underlying matrix. @a a must be a SparseMatrix.
   void SetOperator(const Operator &a) override;
};

/// Gauss-Seidel smoother of a sparse matrix.
class GSSmoother : public SparseSmoother
{
public:
   enum GSType
   {
      SYMMETRIC, ///< Forward Gauss-Seidel, then backward.
      FORWARD, ///< Forward Gauss-Seidel ($L^{-1}$).
      BACKWARD ///< Backward Gauss-Seidel ($U^{-1}$).
   };
protected:
   GSType type; ///< Type of Gauss-Seidel, see GSSmoother::GSType.
   int iterations; ///< Number of stationary iterations.

public:
   /// @brief Create a Gauss-Seidel smoother. SetOperator() will need to be
   /// called with a SparseMatrix before first use.
   ///
   /// @param[in]  t        Type of GS smoother (see GSSmoother::GSType)
   /// @param[in]  it       Number of stationary iterations to perform
   GSSmoother(GSType t = SYMMETRIC, int it = 1) { type = t; iterations = it; }

   /// @brief Create a Jacobi smoother using the SparseMatrix @a a.
   ///
   /// @param[in]  a        The underlying SparseMatrix
   /// @param[in]  t        Type of GS smoother (see GSSmoother::GSType)
   /// @param[in]  it       Number of stationary iterations to perform
   GSSmoother(const SparseMatrix &a, GSType t = SYMMETRIC, int it = 1)
      : GSSmoother(t, it) { SetOperator(a); }

   /// Same as GSSmoother(GSType,int), for backwards compatibility.
   GSSmoother(int t, int it = 1) : GSSmoother(GSType(t), it) { }

   /// @brief Same as GSSmoother(const SparseMatrix&,GSType,int), for
   /// backwards compatibility.
   GSSmoother(const SparseMatrix &a, int t, int it = 1)
      : GSSmoother(a, GSType(t), it) { }

   /// @brief Application of the Gauss-Seidel smoother.
   ///
   /// Applies a stationary Gauss-Seidel iteration. If Solver::iterative_mode is
   /// true, then @a y is used as the initial guess, and Gauss-Seidel is applied
   /// to the residual $x - Ay$.
   void Mult(const Vector &x, Vector &y) const override;

   /// Application of the transpose of the Gauss-Seidel smoother.
   void MultTranspose(const Vector &x, Vector &y) const override;
};

/// Jacobi-type diagonal smoother of a sparse matrix.
class DSmoother : public SparseSmoother
{
public:
   enum JacobiType
   {
      JACOBI,       ///< Scale by the diagonal of the matrix.
      L1_JACOBI,    ///< Scale by the l1-norm of the rows.
      LUMPED_JACOBI ///< Scale by the sum of the rows.
   };
protected:
   JacobiType type; ///< Type of diagonal scaling, see DSmoother::JacobiType.
   real_t scale; ///< Scaling (damping) factor.
   int iterations; ///< Number of stationary iterations to perform.

   /// @brief Uses abs values of the diagonal entries. Relevant only with type
   /// JacobiType::JACOBI.
   bool use_abs_diag = false;

   mutable Vector z; ///< Temporary work vector.

   /// Apply the Jacobi smoother (used internally by Mult() and MultTranspose())
   void Mult_(const SparseMatrix &A, const Vector &x, Vector &y) const;

public:
   /// @brief Create a Jacobi smoother. SetOperator() will need to be called
   /// with a SparseMatrix before first use.
   ///
   /// @param[in]  t        Type of Jacobi smoother (see DSmoother::JacobiType)
   /// @param[in]  s        Scaling factor
   /// @param[in]  it       Number of stationary iterations to perform
   DSmoother(JacobiType t = JACOBI, real_t s = 1., int it = 1)
   { type = t; scale = s; iterations = it; }

   /// @brief Create a Jacobi smoother using the SparseMatrix @a a.
   ///
   /// @param[in]  a        The underlying SparseMatrix
   /// @param[in]  t        Type of Jacobi smoother (see DSmoother::JacobiType)
   /// @param[in]  s        Scaling factor
   /// @param[in]  it       Number of stationary iterations to perform
   DSmoother(const SparseMatrix &a, JacobiType t = JACOBI, real_t s = 1.,
             int it = 1) : DSmoother(t, s, it) { SetOperator(a); }

   /// @brief Same as DSmoother(JacobiType,real_t,int), for backwards compatbility.
   DSmoother(int t, real_t s = 1., int it = 1)
      : DSmoother(JacobiType(t), s, it) { }

   /// @brief Same as DSmoother(const SparseMatrix&,JacobiType,real_t,int), for
   /// backwards compatbility.
   DSmoother(const SparseMatrix &a, int t, real_t s = 1., int it = 1)
      : DSmoother(a, JacobiType(t), s, it) { }

   /// @brief Replace diagonal entries with their absolute values. Relevant only
   /// with JacobiType::JACOBI.
   void SetPositiveDiagonal(bool pos_diag = true) { use_abs_diag = pos_diag; }

   /// @brief Apply the Jacobi smoother.
   ///
   /// Applies a stationary iteration with diagonal scaling. If
   /// Solver::iterative_mode is true, then @a y is used as the initial guess
   /// (and the diagonal scaling is applied to the residual $x - Ay$, giving
   /// $D^{-1}(x - Ay)$).
   ///
   /// By default, Solver::iterative_mode is false and only one iteration is
   /// performed, corresponding to $y = D^{-1}x$.
   void Mult(const Vector &x, Vector &y) const override;

   /// @brief Apply the transpose of the Jacobi smoother.
   ///
   /// If the underlying matrix is symmetric, or if only one iteration is
   /// performed with zero initial guess (Solver::iterative_mode is false), then
   /// this is the same as Mult(). For non-symmetric matrices with iteration
   /// count greater than one, only JacobiType::JACOBI is supported.
   void MultTranspose(const Vector &x, Vector &y) const override;
};

}

#endif
