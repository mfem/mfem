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

#ifndef MFEM_SPARSITYPATTERN
#define MFEM_SPARSITYPATTERN

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "sparsemat.hpp"

#include <algorithm>

namespace mfem
{

/** @brief The sparsity pattern of a matrix in CSR form, copied so that a matrix
    seen later can be tested against it.

    The direct solver wrappers use this to decide whether a symbolic
    factorization computed earlier still describes the matrix in hand, and may
    therefore be reused rather than recomputed; see
    UMFPackSolver::SetReuseSymbolic().

    The test is exact: the size and the number of nonzeros first, which rejects
    most changes at O(1), and then the pattern itself, entry by entry, at O(nnz)
    integer compares. That is a fraction of a percent of the factorization it
    guards -- some 1 ms against the 100 ms symbolic analysis of a matrix with a
    million nonzeros -- and it is the whole safety of reusing an analysis, so it
    is not worth shortening.

    In particular it is not enough to ask whether this is the same matrix object
    with the same I and J arrays, tempting though that O(1) test is for a matrix
    reassembled in place. A matrix destroyed and rebuilt with a different
    structure, but the same size and the same number of nonzeros, asks the
    allocator for blocks of exactly the sizes just freed and readily lands on
    the same addresses; a matrix built in a loop can even be the same object
    every time. Identity would then hold for a pattern that had changed, which
    is the one failure this check exists to prevent, and it would be silent.

    The pattern is copied rather than pointed at so that the comparison stays
    valid, and stays safe, after the matrix it was taken from is destroyed. */
class RetainedSparsityPattern
{
private:
   int size, nnz;
   Array<int> I_copy, J_copy;
   bool set;

public:
   RetainedSparsityPattern() : size(0), nnz(0), set(false) { }

   /// Whether a pattern has been recorded and not since cleared.
   bool IsSet() const { return set; }

   /// Forget the pattern. Matches() returns false until the next Set().
   void Clear()
   {
      size = nnz = 0;
      I_copy.DeleteAll();
      J_copy.DeleteAll();
      set = false;
   }

   /** @brief Record the pattern of a CSR matrix with @a s rows and @a nz
       nonzeros, row pointers @a I and column indices @a J. */
   void Set(int s, int nz, const int *I, const int *J)
   {
      MFEM_ASSERT(I && J, "no pattern to record");
      MFEM_ASSERT(I[s] == nz, "the row pointers do not end at nnz");
      size = s;
      nnz = nz;
      I_copy.SetSize(size+1);
      J_copy.SetSize(nnz);
      std::copy(I, I + size + 1, I_copy.begin());
      std::copy(J, J + nnz, J_copy.begin());
      set = true;
   }

   /// Record the sparsity pattern of the finalized matrix @a A.
   void Set(const SparseMatrix &A)
   {
      Set(A.Height(), A.NumNonZeroElems(), A.HostReadI(), A.HostReadJ());
   }

   /** @brief Whether the CSR matrix with @a s rows, @a nz nonzeros, row
       pointers @a I and column indices @a J has the recorded pattern. */
   bool Matches(int s, int nz, const int *I, const int *J) const
   {
      if (!set || s != size || nz != nnz || !I || !J) { return false; }
      return std::equal(I_copy.begin(), I_copy.end(), I) &&
             std::equal(J_copy.begin(), J_copy.end(), J);
   }

   /// Whether the finalized matrix @a A has the recorded pattern.
   bool Matches(const SparseMatrix &A) const
   {
      const int *I = A.HostReadI();
      if (!I) { return false; }
      return Matches(A.Height(), I[A.Height()], I, A.HostReadJ());
   }
};

} // namespace mfem

#endif
