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

#ifndef MFEM_PARDISO
#define MFEM_PARDISO

#include "../config/config.hpp"

#ifdef MFEM_USE_MKL_PARDISO

#include "mkl_pardiso.h"
#include "operator.hpp"
#include "sparsitypattern.hpp"

namespace mfem
{
/**
 * @brief MKL Parallel Direct Sparse Solver PARDISO
 *
 * Interface to MKL PARDISO: the direct sparse solver based on PARDISO
 */
class PardisoSolver : public mfem::Solver
{
public:
   enum MatType
   {
      REAL_STRUCTURE_SYMMETRIC = 1,
      REAL_SYMMETRIC_POSITIVE_DEFINITE = 2,
      REAL_SYMMETRIC_INDEFINITE = -2,
      REAL_NONSYMMETRIC = 11
   };

   /**
    * @brief Construct a new PardisoSolver object
    *
    */
   PardisoSolver();

   /**
    * @brief Set the Operator object and perform factorization
    *
    * @a op needs to be of type SparseMatrix.
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
    * @brief Set the print level for MKL Pardiso
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
    * PARDISO's phase 11, the reordering and symbolic factorization, depends
    * only on the sparsity pattern. A caller that refactorizes a matrix whose
    * pattern does not change -- a Newton iteration, an implicit time step, a
    * continuation loop, a parameter sweep -- otherwise repeats it on every
    * SetOperator() call and uses it once. With reuse, such a call is phase 22
    * alone.
    *
    * The pattern is checked, not assumed, by the exact comparison described in
    * UMFPackSolver::SetReuseSymbolic(): when it has changed, the factorization
    * is released and phase 11 is run again.
    *
    * @param reuse Whether to reuse the analysis
    */
   void SetReuseSymbolic(bool reuse = true);

   /// Whether analysis reuse is enabled; see SetReuseSymbolic().
   bool GetReuseSymbolic() const { return reuse_symbolic; }

   /**
    * @brief The number of analyses actually performed, that is, of PARDISO
    * phase 11 calls. Without reuse this is the number of SetOperator() calls;
    * with reuse it is the number of times the pattern changed.
    */
   long GetNumSymbolicFactorizations() const { return num_symbolic; }

   /**
    * @brief The number of numeric factorizations actually performed, that is,
    * of PARDISO phase 22 calls. Reuse never skips one of these.
    */
   long GetNumNumericFactorizations() const { return num_numeric; }

   ~PardisoSolver();

private:
   // Global number of rows
   int m;

   // Number of nonzero entries
   int nnz;

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

   // The pattern the retained analysis was made for; empty unless reusing
   RetainedSparsityPattern pattern;

   // Whether pt holds a factorization, and so has memory to release
   bool factored = false;

   // The number of phase 11 and phase 22 calls actually made
   long num_symbolic = 0;
   long num_numeric = 0;

   // Release the memory held by pt, if there is any
   void ReleaseFactorization();
};
} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO

#endif
