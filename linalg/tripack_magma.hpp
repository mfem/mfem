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
   void Factor(const TriPackMatrix<TriangularPart::LOWER> &A,
               TriPackMatrix<TriangularPart::LOWER> &L);

   /// Solve A x = b using L from Factor(), overwriting rhs_sol with x.
   void SolveInPlace(const TriPackMatrix<TriangularPart::LOWER> &L,
                     Vector &rhs_sol) const;
};

namespace tripack
{
namespace magma
{

inline void ComputeCholeskyLower(
   const TriPackMatrix<TriangularPart::LOWER> &packed_lower,
   TriPackMatrix<TriangularPart::LOWER> &lower_factor,
   MagmaPackedLowerCholesky &ws)
{
   ws.Factor(packed_lower, lower_factor);
}

inline void SolveCholeskyLowerInPlace(
   const TriPackMatrix<TriangularPart::LOWER> &lower_factor,
   Vector &rhs_sol,
   MagmaPackedLowerCholesky &ws)
{
   ws.SolveInPlace(lower_factor, rhs_sol);
}

} // namespace magma
} // namespace tripack

} // namespace mfem

#endif // MFEM_USE_MAGMA

#endif // MFEM_TRIPACK_MAGMA
