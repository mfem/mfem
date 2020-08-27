// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

class SparseSmoother : public MatrixInverse
{
protected:
   const SparseMatrix *oper;

public:
   SparseSmoother() { oper = NULL; }

   SparseSmoother(const SparseMatrix &a)
      : MatrixInverse(a) { oper = &a; }

   virtual void SetOperator(const Operator &a);
};

/// Data type for Gauss-Seidel smoother of sparse matrix
class GSSmoother : public SparseSmoother
{
protected:
   int type; // 0, 1, 2 - symmetric, forward, backward
   int iterations;

public:
   /// Create GSSmoother.
   GSSmoother(int t = 0, int it = 1) { type = t; iterations = it; }

   /// Create GSSmoother.
   GSSmoother(const SparseMatrix &a, int t = 0, int it = 1)
      : SparseSmoother(a) { type = t; iterations = it; }

   /// Matrix vector multiplication with GS Smoother.
   virtual void Mult(const Vector &x, Vector &y) const;
};

/// Data type for scaled Jacobi-type smoother of sparse matrix
class DSmoother : public SparseSmoother
{
protected:
   int type; // 0, 1, 2 - scaled Jacobi, scaled l1-Jacobi, scaled lumped-Jacobi
   double scale;
   int iterations;

   mutable Vector z;

public:
   /// Create Jacobi smoother.
   DSmoother(int t = 0, double s = 1., int it = 1)
   { type = t; scale = s; iterations = it; }

   /// Create Jacobi smoother.
   DSmoother(const SparseMatrix &a, int t = 0, double s = 1., int it = 1);

   /// Matrix vector multiplication with Jacobi smoother.
   virtual void Mult(const Vector &x, Vector &y) const;
};

}

#endif
