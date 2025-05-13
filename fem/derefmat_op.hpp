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

#ifndef MFEM_DEREFMAT_OP
#define MFEM_DEREFMAT_OP

#include "fespace.hpp"

#include "kernel_dispatch.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

struct DerefineMatrixOp : public Operator
{
   FiniteElementSpace *fespace;
   /// offsets into block_storage
   Array<int> block_offsets;
   /// offsets into row_idcs
   Array<int> block_row_idcs_offsets;
   /// offsets into col_idcs
   Array<int> block_col_idcs_offsets;
   /// mapping for row dofs, INT_MAX indicates the block row should be ignored.
   /// negative means the row data should be negated.
   Array<int> row_idcs;
   /// mapping for col dofs, negative means the col data should be negated.
   Array<int> col_idcs;
   /// dense block matrices which can be reused to construct the full matrix
   /// operation. These are stored contiguously and blocks have no restrictions
   /// on shape (can be rectangle and differ from block to block).
   Vector block_storage;
   /// maximum height of any block in block_storage for GPU
   /// parallelization, or 1 for CPU runs.
   int max_rows;

   using MultKernelType = void (*)(const DerefineMatrixOp &, const Vector &,
                                   Vector &);
   /// template args: ordering, atomic
   MFEM_REGISTER_KERNELS(MultKernel, MultKernelType, (Ordering::Type, bool));

   struct Kernels
   {
      Kernels();
   };

   void Mult(const Vector &x, Vector &y) const;

   DerefineMatrixOp(FiniteElementSpace &fespace_, int old_ndofs,
                    const Table *old_elem_dof, const Table *old_elem_fos);
};

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
#endif
