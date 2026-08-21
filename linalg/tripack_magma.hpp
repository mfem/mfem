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

#ifndef MFEM_TRIPACK_MAGMA
#define MFEM_TRIPACK_MAGMA

#include "../config/config.hpp"
#include "tripack.hpp"

#ifdef MFEM_USE_MAGMA

#include "batched/magma.hpp"

namespace mfem
{

/// Workspace + operations for MAGMA packed-lower batched Cholesky and solve.
///
/// This class factors a batch of SPD matrices stored in packed lower-triangular
/// format (LAPACK/MAGMA column-major packed storage) and applies the inverse via
/// MAGMA batched triangular solves.
class MagmaPackedLowerCholesky
{
private:
   int n = 0;
   int batch_size = 0;
   int packed_size = 0;

   mutable Array<real_t *> factor_ptrs;
   mutable Array<real_t *> rhs_ptrs;
   Array<magma_int_t> info;

   magma_queue_t queue = nullptr;

public:
   MagmaPackedLowerCholesky();

   void SetQueue(magma_queue_t q) { queue = q; }

   int GetNumRows() const { return n; }
   int GetNumMatrices() const { return batch_size; }
   int GetPackedSize() const { return packed_size; }

   /// Factor packed-lower matrices A into L (in-place copy then factor).
   void Factor(const TriPackLowerMatrix &A,
               TriPackLowerMatrix &L);

   /// Solve A x = b using L from Factor(), overwriting rhs_sol with x.
   void SolveInPlace(const TriPackLowerMatrix &L,
                     Vector &rhs_sol) const;
};

/// Workspace + operations for MAGMA packed-lower batched inverse and apply.
///
/// This class computes the inverse of a batch of SPD matrices stored in packed
/// lower-triangular format (LAPACK/MAGMA column-major packed storage) using
/// MAGMA's `ppinv_batched`. The resulting packed inverse can be applied to a
/// batch of vectors using MAGMA's packed-symmetric batched matvec when
/// available, falling back to an MFEM device kernel for larger sizes.
class MagmaPackedLowerInverse
{
private:
   int n = 0;
   int batch_size = 0;
   int packed_size = 0;

   mutable Array<real_t *> inv_ptrs;
   mutable Array<real_t *> rhs_ptrs;
   mutable Vector work;
   Array<magma_int_t> info;

   magma_queue_t queue = nullptr;

public:
   MagmaPackedLowerInverse();

   void SetQueue(magma_queue_t q) { queue = q; }

   int GetNumRows() const { return n; }
   int GetNumMatrices() const { return batch_size; }
   int GetPackedSize() const { return packed_size; }

   /// Compute packed inverse of A into A_inv (in-place copy then invert).
   void Compute(const TriPackLowerMatrix &A,
                TriPackLowerMatrix &A_inv);

   /// Apply packed inverse to rhs_sol, overwriting rhs_sol with the result.
   void ApplyInPlace(const TriPackLowerMatrix &A_inv,
                     Vector &rhs_sol) const;
};

namespace tripack
{
namespace magma
{

inline void ComputeCholeskyLower(
   const TriPackLowerMatrix &packed_lower,
   TriPackLowerMatrix &lower_factor,
   MagmaPackedLowerCholesky &ws)
{
   ws.Factor(packed_lower, lower_factor);
}

inline void SolveCholeskyLowerInPlace(
   const TriPackLowerMatrix &lower_factor,
   Vector &rhs_sol,
   MagmaPackedLowerCholesky &ws)
{
   ws.SolveInPlace(lower_factor, rhs_sol);
}

inline void ComputeInverseLower(
   const TriPackLowerMatrix &packed_lower,
   TriPackLowerMatrix &lower_inverse,
   MagmaPackedLowerInverse &ws)
{
   ws.Compute(packed_lower, lower_inverse);
}

inline void ApplyInverseLowerInPlace(
   const TriPackLowerMatrix &lower_inverse,
   Vector &rhs_sol,
   MagmaPackedLowerInverse &ws)
{
   ws.ApplyInPlace(lower_inverse, rhs_sol);
}

} // namespace magma
} // namespace tripack

} // namespace mfem

#endif // MFEM_USE_MAGMA

#endif // MFEM_TRIPACK_MAGMA
