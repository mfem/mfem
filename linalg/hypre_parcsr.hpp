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

#ifndef MFEM_HYPRE_PARCSR_HPP
#define MFEM_HYPRE_PARCSR_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

// Enable internal hypre timing routines
#define HYPRE_TIMING

#include "_hypre_parcsr_mv.h"

// Define macro wrappers for hypre_TAlloc, hypre_CTAlloc and hypre_TFree:
// mfem_hypre_TAlloc, mfem_hypre_CTAlloc, and mfem_hypre_TFree, respectively.
// Note: these macros are used in hypre.cpp, hypre_parcsr.cpp, and perhaps
// other locations in the future.
#if MFEM_HYPRE_VERSION < 21400

#define mfem_hypre_TAlloc(type, size) hypre_TAlloc(type, size)
#define mfem_hypre_CTAlloc(type, size) hypre_CTAlloc(type, size)
#define mfem_hypre_TFree(ptr) hypre_TFree(ptr)

#else // MFEM_HYPRE_VERSION >= 21400

#define mfem_hypre_TAlloc(type, size) \
   hypre_TAlloc(type, size, HYPRE_MEMORY_HOST)
#define mfem_hypre_CTAlloc(type, size) \
   hypre_CTAlloc(type, size, HYPRE_MEMORY_HOST)
#define mfem_hypre_TFree(ptr) hypre_TFree(ptr, HYPRE_MEMORY_HOST)

// Notes regarding allocation and deallocation of hypre objects in 2.14.0
//-----------------------------------------------------------------------
//
// 1. hypre_CSRMatrix: i, j, data, and rownnz use HYPRE_MEMORY_SHARED while the
//    hypre_CSRMatrix structure uses HYPRE_MEMORY_HOST.
//
//    Note: the function HYPRE_CSRMatrixCreate creates the i array using
//          HYPRE_MEMORY_HOST!
//    Note: the functions hypre_CSRMatrixAdd and hypre_CSRMatrixMultiply create
//          C_i using HYPRE_MEMORY_HOST!
//
// 2. hypre_Vector: data uses HYPRE_MEMORY_SHARED while the hypre_Vector
//    structure uses HYPRE_MEMORY_HOST.
//
// 3. hypre_ParVector: the structure hypre_ParVector uses HYPRE_MEMORY_HOST;
//    partitioning uses HYPRE_MEMORY_HOST.
//
// 4. hypre_ParCSRMatrix: the structure hypre_ParCSRMatrix uses
//    HYPRE_MEMORY_HOST; col_map_offd, row_starts, col_starts, rowindices,
//    rowvalues also use HYPRE_MEMORY_HOST.
//
//    Note: the function hypre_ParCSRMatrixToCSRMatrixAll allocates matrix_i
//          using HYPRE_MEMORY_HOST!
//
// 5. The goal for the MFEM wrappers of hypre objects is to support only the
//    standard hypre build case, i.e. when hypre is build without device support
//    and all memory types correspond to host memory. In this case memory
//    allocated with operator new can be used by hypre but (as usual) it must
//    not be owned by hypre.

#endif // #if MFEM_HYPRE_VERSION < 21400

namespace mfem
{

// This module contains functions that are logically part of HYPRE, and might
// become part of HYPRE at some point. In the meantime the module can be
// thought of as an extension of HYPRE.

namespace internal
{

/** Parallel essential BC elimination from the system A*X = B.
    (Adapted from hypre_ParCSRMatrixEliminateRowsCols.) */
void hypre_ParCSRMatrixEliminateAXB(hypre_ParCSRMatrix *A,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
                                    hypre_ParVector *X,
                                    hypre_ParVector *B);

/** Parallel essential BC elimination from matrix A only. The eliminated
    elements are stored in a new matrix Ae, so that (A + Ae) equals the original
    matrix A. */
void hypre_ParCSRMatrixEliminateAAe(hypre_ParCSRMatrix *A,
                                    hypre_ParCSRMatrix **Ae,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
                                    int ignore_rows = 0);

/** Eliminate rows from a hypre ParCSRMatrix, setting all entries in the listed
    rows of the matrix to zero. */
void hypre_ParCSRMatrixEliminateRows(hypre_ParCSRMatrix *A,
                                     HYPRE_Int num_rows_to_elim,
                                     const HYPRE_Int *rows_to_elim);

/** Split matrix 'A' into nr x nc blocks, return nr x nc pointers to
    new parallel matrices. The array 'blocks' needs to be preallocated to hold
    nr x nc pointers. If 'interleaved' == 0 the matrix is split into contiguous
    blocks (AAABBBCCC) otherwise the blocks are interleaved (ABCABCABC).
    The local number of rows of A must be divisible by nr. The local number of
    columns of A must be divisible by nc. */
void hypre_ParCSRMatrixSplit(hypre_ParCSRMatrix *A,
                             HYPRE_Int nr, HYPRE_Int nc,
                             hypre_ParCSRMatrix **blocks,
                             int interleaved_rows, int interleaved_cols);

typedef int HYPRE_Bool;
#define HYPRE_MPI_BOOL MPI_INT

/// Computes y = alpha * |A| * x + beta * y, using entry-wise absolute values of matrix A
void hypre_CSRMatrixAbsMatvec(hypre_CSRMatrix *A,
                              HYPRE_Real alpha,
                              HYPRE_Real *x,
                              HYPRE_Real beta,
                              HYPRE_Real *y);

/// Computes y = alpha * |At| * x + beta * y, using entry-wise absolute values of the transpose of matrix A
void hypre_CSRMatrixAbsMatvecT(hypre_CSRMatrix *A,
                               HYPRE_Real alpha,
                               HYPRE_Real *x,
                               HYPRE_Real beta,
                               HYPRE_Real *y);

/// Computes y = alpha * |A| * x + beta * y, using entry-wise absolute values of matrix A
void hypre_ParCSRMatrixAbsMatvec(hypre_ParCSRMatrix *A,
                                 HYPRE_Real alpha,
                                 HYPRE_Real *x,
                                 HYPRE_Real beta,
                                 HYPRE_Real *y);

/// Computes y = alpha * |At| * x + beta * y, using entry-wise absolute values of the transpose of matrix A
void hypre_ParCSRMatrixAbsMatvecT(hypre_ParCSRMatrix *A,
                                  HYPRE_Real alpha,
                                  HYPRE_Real *x,
                                  HYPRE_Real beta,
                                  HYPRE_Real *y);

/** The "Boolean" analog of y = alpha * A * x + beta * y, where elements in the
    sparsity pattern of the CSR matrix A are treated as "true". */
void hypre_CSRMatrixBooleanMatvec(hypre_CSRMatrix *A,
                                  HYPRE_Bool alpha,
                                  HYPRE_Bool *x,
                                  HYPRE_Bool beta,
                                  HYPRE_Bool *y);

/** The "Boolean" analog of y = alpha * A^T * x + beta * y, where elements in
    the sparsity pattern of the CSR matrix A are treated as "true". */
void hypre_CSRMatrixBooleanMatvecT(hypre_CSRMatrix *A,
                                   HYPRE_Bool alpha,
                                   HYPRE_Bool *x,
                                   HYPRE_Bool beta,
                                   HYPRE_Bool *y);

hypre_ParCSRCommHandle *
hypre_ParCSRCommHandleCreate_bool(HYPRE_Int            job,
                                  hypre_ParCSRCommPkg *comm_pkg,
                                  HYPRE_Bool          *send_data,
                                  HYPRE_Bool          *recv_data);

/** The "Boolean" analog of y = alpha * A * x + beta * y, where elements in the
    sparsity pattern of the ParCSR matrix A are treated as "true". */
void hypre_ParCSRMatrixBooleanMatvec(hypre_ParCSRMatrix *A,
                                     HYPRE_Bool alpha,
                                     HYPRE_Bool *x,
                                     HYPRE_Bool beta,
                                     HYPRE_Bool *y);

/** The "Boolean" analog of y = alpha * A^T * x + beta * y, where elements in
    the sparsity pattern of the ParCSR matrix A are treated as "true". */
void hypre_ParCSRMatrixBooleanMatvecT(hypre_ParCSRMatrix *A,
                                      HYPRE_Bool alpha,
                                      HYPRE_Bool *x,
                                      HYPRE_Bool beta,
                                      HYPRE_Bool *y);

/** Perform the operation A += beta*B, assuming that the sparsity pattern of A
    contains that of B. */
HYPRE_Int
hypre_CSRMatrixSum(hypre_CSRMatrix *A,
                   HYPRE_Complex    beta,
                   hypre_CSRMatrix *B);

/** Return a new matrix containing the sum of A and B, assuming that both
    matrices use the same row and column partitions. The col_map_offd do not
    need to be the same, but a more efficient algorithm is used if that's the
    case. */
hypre_ParCSRMatrix *
hypre_ParCSRMatrixAdd(hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix *B);

/** Perform the operation A += beta*B, assuming that both matrices use the same
    row and column partitions and the same col_map_offd arrays, or B has an empty
    off-diagonal block. We also assume that the sparsity pattern of A contains
    that of B. */
HYPRE_Int
hypre_ParCSRMatrixSum(hypre_ParCSRMatrix *A,
                      HYPRE_Complex       beta,
                      hypre_ParCSRMatrix *B);

/** Initialize all entries of A with value. */
HYPRE_Int
hypre_CSRMatrixSetConstantValues(hypre_CSRMatrix *A,
                                 HYPRE_Complex    value);

/** Initialize all entries of A with value. */
HYPRE_Int
hypre_ParCSRMatrixSetConstantValues(hypre_ParCSRMatrix *A,
                                    HYPRE_Complex       value);

} // namespace mfem::internal

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
