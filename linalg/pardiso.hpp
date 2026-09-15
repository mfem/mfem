// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
    * @brief Solve for several right-hand sides in one pass
    *
    * The factors are walked once rather than once per column: PARDISO's
    * solve phase is called a single time with @a nrhs set to `X.Size()`,
    * which is one triangular sweep over the factors instead of `X.Size()`
    * of them. The arithmetic performed on any one column is unchanged, so
    * the saving is in memory traffic and call overhead, not in flops.
    *
    * `X.Size() == 1` forwards to Mult() and allocates nothing, following
    * STRUMPACKSolver::ArrayMult().
    *
    * @param X RHS vectors, all of length Width()
    * @param Y Solution vectors, all of length Height()
    */
   void ArrayMult(const Array<const Vector *> &X,
                  Array<Vector *> &Y) const override;

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
   // Number of right-hand sides of the next solve phase. Mutable because
   // ArrayMult() is const and sets it from the number of columns it is
   // given; CuDSSSolver does the same. PARDISO reads it only in the solve
   // phase, so changing it after the factorization is legal.
   mutable int nrhs;

   // Column-major packing of the right-hand sides and the solutions, sized
   // nrhs*m and reused across calls, as in STRUMPACKSolver. Empty until a
   // multi-column ArrayMult() asks for them.
   mutable Vector rhs_buf, sol_buf;

   // Dummy variables
   mutable int idum;
   mutable real_t ddum;
};
} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO

#endif
