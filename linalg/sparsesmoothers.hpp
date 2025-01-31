// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

template <class T>
class SparseSmootherMP : public MatrixInverseMP<T>
{
protected:
   const SparseMatrixMP<T> *oper;

public:
   SparseSmootherMP() { oper = NULL; }

   SparseSmootherMP(const SparseMatrixMP<T> &a)
      : MatrixInverseMP<T>(a) { oper = &a; }

   void SetOperator(const OperatorMP<T> &a) override;
};

using SparseSmoother = SparseSmootherMP<real_t>;

/// Data type for Gauss-Seidel smoother of sparse matrix
template <class T>
class GSSmootherMP : public SparseSmootherMP<T>
{
protected:
   int type; // 0, 1, 2 - symmetric, forward, backward
   int iterations;

public:
   /// Create GSSmoother.
   GSSmootherMP(int t = 0, int it = 1) { type = t; iterations = it; }

   /// Create GSSmoother.
   GSSmootherMP(const SparseMatrixMP<T> &a, int t = 0, int it = 1)
      : SparseSmootherMP<T>(a) { type = t; iterations = it; }

   /// Matrix vector multiplication with GS Smoother.
   void Mult(const VectorMP<T> &x, VectorMP<T> &y) const override;
};

using GSSmoother = GSSmootherMP<real_t>;


/// Data type for scaled Jacobi-type smoother of sparse matrix
class DSmoother : public SparseSmoother
{
protected:
   int type; // 0, 1, 2 - scaled Jacobi, scaled l1-Jacobi, scaled lumped-Jacobi
   real_t scale;
   int iterations;
   /// Uses abs values of the diagonal entries. Relevant only when type = 0.
   bool use_abs_diag = false;

   mutable Vector z;

public:
   /// Create Jacobi smoother.
   DSmoother(int t = 0, real_t s = 1., int it = 1)
   { type = t; scale = s; iterations = it; }

   /// Create Jacobi smoother.
   DSmoother(const SparseMatrix &a, int t = 0, real_t s = 1., int it = 1);

   /// Replace diag entries with their abs values. Relevant only when type = 0.
   void SetPositiveDiagonal(bool pos_diag = true) { use_abs_diag = pos_diag; }

   /// Matrix vector multiplication with Jacobi smoother.
   void Mult(const Vector &x, Vector &y) const override;
};

}

#endif
