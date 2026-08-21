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

#ifndef MFEM_TRIPACK
#define MFEM_TRIPACK

#include "../config/config.hpp"
#include "vector.hpp"

namespace mfem
{

/// Packed storage for a batch of symmetric matrices of fixed size.
/// Storage is packed lower-triangular in LAPACK/MAGMA column-major convention.
class TriPackLowerMatrix
{
private:
   Vector data;
   int nrows = 0;
   int nmats = 0;

public:
   TriPackLowerMatrix() = default;

   TriPackLowerMatrix(int n, int batch_size)
   {
      SetSize(n, batch_size);
   }

   MFEM_HOST_DEVICE static int PackedSize(const int n)
   {
      return n*(n + 1)/2;
   }

   /// Packed index for (i,j) in the lower triangle (requires i >= j).
   MFEM_HOST_DEVICE static int LowerIndex(const int i, const int j, const int n)
   {
      return j*(2*n + 1 - j)/2 + (i - j);
   }

   /// Packed index for (i,j) in symmetric storage (maps to lower triangle).
   MFEM_HOST_DEVICE static int Index(const int i, const int j, const int n)
   {
      return (i >= j) ? LowerIndex(i, j, n) : LowerIndex(j, i, n);
   }

   void SetSize(const int n, const int batch_size)
   {
      nrows = n;
      nmats = batch_size;
      data.SetSize(batch_size*PackedSize(n));
   }

   int GetNumRows() const { return nrows; }

   int GetNumMatrices() const { return nmats; }

   int GetPackedSize() const { return PackedSize(nrows); }

   int Size() const { return data.Size(); }

   void UseDevice(bool use_dev) { data.UseDevice(use_dev); }

   TriPackLowerMatrix &operator=(real_t value)
   {
      data = value;
      return *this;
   }

   Vector &Data() { return data; }
   const Vector &Data() const { return data; }
};

namespace tripack
{

bool CompareWithFull(const TriPackLowerMatrix &packed, const Vector &full,
                     real_t tol = 0.0);

void Mult(const TriPackLowerMatrix &packed, const Vector &x, Vector &y);

void Lump(const TriPackLowerMatrix &packed, Vector &lump);

void ComputeCholeskyLower(const TriPackLowerMatrix &packed_lower,
                          TriPackLowerMatrix &lower_factor);

void SolveLower(const TriPackLowerMatrix &lower_factor,
                const Vector &rhs,
                Vector &sol);

void SolveLowerTranspose(const TriPackLowerMatrix &lower_factor,
                         const Vector &rhs,
                         Vector &sol);

void SolveCholeskyLower(const TriPackLowerMatrix &lower_factor,
                        const Vector &rhs,
                        Vector &sol);

/// Compute the inverse of the Cholesky lower factor for a batch of SPD matrices.
///
/// Given packed lower-triangular matrices A (SPD), this routine computes the
/// packed lower-triangular matrices L^{-1}, where A = L L^T.
///
/// This is intended for fast inverse applications using (L^{-1})^T (L^{-1}).
void ComputeCholeskyLowerInverse(const TriPackLowerMatrix &packed_lower,
                                 TriPackLowerMatrix &lower_inverse);

} // namespace tripack

} // namespace mfem

#endif
