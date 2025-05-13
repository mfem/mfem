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

#ifndef MFEM_PDEREFMAT_OP
#define MFEM_PDEREFMAT_OP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pfespace.hpp"

#include "kernel_dispatch.hpp"

#include <vector>

namespace mfem
{

/// \cond DO_NOT_DOCUMENT
struct ParDerefineMatrixOp : public Operator
{
   ParFiniteElementSpace *fespace;
   /// offsets into block_storage for diagonal
   Array<int> block_offsets;
   /// offsets into row_idcs for diagonal
   Array<int> block_row_idcs_offsets;
   /// offsets into col_idcs for diagonal
   Array<int> block_col_idcs_offsets;

   /// offsets into block_storage for off-diagonal
   Array<int> off_diag_block_offsets;
   /// offsets into row_idcs for off-diagonal
   Array<int> block_off_diag_row_idcs_offsets;
   Array<int> block_off_diag_col_offsets;
   Array<int> block_off_diag_widths;
   /// mapping for row dofs, INT_MAX indicates the block row should be ignored.
   /// negative means the row data should be negated.
   /// only for diagonal blocks
   Array<int> row_idcs;
   /// mapping for col dofs, negative means the col data should be negated.
   /// only for diagonal blocks
   Array<int> col_idcs;

   Array<int> pack_col_idcs;

   /// mapping for row dofs, INT_MAX indicates the block row should be ignored.
   /// negative means the row data should be negated.
   /// only for off-diagonal blocks
   Array<int> row_off_diag_idcs;
   /// dense block matrices which can be reused to construct the full matrix
   /// operation. These are stored contiguously and blocks have no restrictions
   /// on shape (can be rectangle and differ from block to block).
   /// This is only for the diagonal block.
   Vector block_storage;
   /// maximum height of any block in block_storage for GPU
   /// parallelization, or 1 for CPU runs.
   int max_rows;

   /// quasi Ordering::byNODES, broken into sections by ranks we need to send
   /// the data to
   mutable Vector xghost_send;
   /// quasi Ordering::byNODES, broken into sections by ranks we received
   /// the data from
   mutable Vector xghost_recv;
   /// maps off-diagonal k to segment
   Array<int> recv_segment_idcs;
   /// cumulative count of dofs which will be received from other ranks
   Array<int> recv_segments;
   /// Source rank of each recv segment
   Array<int> recv_ranks;
   /// What send segment each entry in send_permutations corresponds to
   Array<int> send_segment_idcs;
   /// cumulative count of dofs which will be sent to other ranks
   Array<int> send_segments;
   /// Destination rank of each send segment
   Array<int> send_ranks;
   /// how to permute/sign change values from our local x to send to other ranks
   Array<int> send_permutations;
   /// internal buffer for MPI requests
   mutable std::vector<MPI_Request> requests;

   using MultKernelType = void (*)(const ParDerefineMatrixOp &, const Vector &,
                                   Vector &);
   /// template args: ordering, atomic
   MFEM_REGISTER_KERNELS(MultKernel, MultKernelType, (Ordering::Type, bool));

   struct Kernels
   {
      Kernels();
   };

   void Mult(const Vector &x, Vector &y) const;

   ParDerefineMatrixOp(ParFiniteElementSpace &fespace_, int old_ndofs,
                       const Table *old_elem_dof, const Table *old_elem_fos);
};
/// \endcond DO_NOT_DOCUMENT
} // namespace mfem

#endif

#endif
