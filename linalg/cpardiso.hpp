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

#ifndef MFEM_CPARDISO
#define MFEM_CPARDISO

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_MKL_CPARDISO

#include "mkl_cluster_sparse_solver.h"
#include "operator.hpp"
#include "sparsitypattern.hpp"

namespace mfem
{
/**
 * @brief MKL Parallel Direct Sparse Solver for Clusters
 *
 * Interface to MKL CPardiso: the MPI-enabled Intel MKL version of Pardiso
 */
class CPardisoSolver : public Solver
{
public:
   enum MatType
   {
      REAL_STRUCTURE_SYMMETRIC = 1,
      REAL_NONSYMMETRIC = 11
   };

   /**
    * @brief Construct a new CPardisoSolver object
    *
    * @param comm MPI Communicator
    */
   CPardisoSolver(MPI_Comm comm);

   /**
    * @brief Set the Operator object and perform factorization
    *
    * @a op needs to be of type HypreParMatrix. The contents are copied and
    * reordered in an internal CSR structure.
    *
    * @param op Operator to use in factorization and solve
    */
   void SetOperator(const Operator &op) override;

   /**
    * @brief Solve
    *
    * @param b RHS vector
    * @param x Solution vector
    */
   void Mult(const Vector &b, Vector &x) const override;

   /**
    * @brief Set the print level for MKL CPardiso
    *
    * Prints statistics after the factorization and after each solve.
    *
    * @param print_lvl Print level
    */
   void SetPrintLevel(int print_lvl);

   /**
    * @brief Set the matrix type
    *
    * The matrix type supported is either real and symmetric or real and
    * non-symmetric.
    *
    * @param mat_type Matrix type
    */
   void SetMatrixType(MatType mat_type);

   /**
    * @brief Retain the analysis phase across SetOperator() calls and reuse it
    * whenever the sparsity pattern is unchanged. Off by default.
    *
    * This is PardisoSolver::SetReuseSymbolic() for the cluster solver, and the
    * saving is the same: phase 11, the reordering and symbolic factorization,
    * depends only on the pattern, so a caller refactorizing a matrix of fixed
    * structure otherwise repeats it on every SetOperator() call.
    *
    * The pattern compared is that of the local CSR matrix this rank builds
    * from the HypreParMatrix -- the merged diagonal and off-diagonal blocks,
    * with global column indices -- together with the index of its first row.
    *
    * @note The decision is reduced across the communicator: the analysis is
    * reused only if every rank finds its own pattern unchanged. It has to be,
    * because cluster_sparse_solver() is collective, and ranks that disagreed
    * about whether to call phase 11 would deadlock.
    *
    * @param reuse Whether to reuse the analysis
    */
   void SetReuseSymbolic(bool reuse = true);

   /// Whether analysis reuse is enabled; see SetReuseSymbolic().
   bool GetReuseSymbolic() const { return reuse_symbolic; }

   /**
    * @brief The number of analyses actually performed, that is, of phase 11
    * calls. Without reuse this is the number of SetOperator() calls; with
    * reuse it is the number of times the pattern changed. The same on every
    * rank, since the decision is collective.
    */
   long GetNumSymbolicFactorizations() const { return num_symbolic; }

   /**
    * @brief The number of numeric factorizations actually performed, that is,
    * of phase 22 calls. Reuse never skips one of these.
    */
   long GetNumNumericFactorizations() const { return num_numeric; }

   ~CPardisoSolver();

private:
   MPI_Fint comm_;

   // Global number of rows
   int m;

   // First row index of the global matrix on the local MPI rank
   int first_row;

   // Local number of nonzero entries
   int nnz_loc;

   // Local number of rows, obtained from a ParCSR matrix
   int m_loc;

   // CSR data structure for the copy data of the local CSR matrix
   int *csr_rowptr = nullptr;
   real_t *reordered_csr_nzval = nullptr;
   int *reordered_csr_colind = nullptr;

   // Internal solver memory pointer pt,
   // 32-bit: int pt[64]
   // 64-bit: long int pt[64] or void *pt[64] should be OK on both architectures
   mutable void *pt[64] = {0};

   // Solver control parameters, detailed description can be found in the
   // constructor.
   mutable int iparm[64] = {0};
   mutable int maxfct, mnum, msglvl, phase, error;
   int mtype;
   int nrhs;

   // Dummy variables
   mutable int idum;
   mutable real_t ddum;

   // Whether the analysis may be retained and reused; see SetReuseSymbolic()
   bool reuse_symbolic = false;

   // The pattern of the local CSR matrix the retained analysis was made for,
   // and the first row index that went with it; empty unless reusing
   RetainedSparsityPattern pattern;
   int pattern_first_row = -1;

   // Whether pt holds a factorization, and so has memory to release
   bool factored = false;

   // The number of phase 11 and phase 22 calls actually made
   long num_symbolic = 0;
   long num_numeric = 0;

   // Release the memory held by pt, if there is any
   void ReleaseFactorization();
};
} // namespace mfem

#endif
#endif // MFEM_USE_MKL_CPARDISO
#endif // MFEM_USE_MPI
