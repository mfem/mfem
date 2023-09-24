// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PARDISO_COMP
#define MFEM_PARDISO_COMP

#include "../config/config.hpp"
#include <complex>
#include <vector>

#ifdef MFEM_USE_MKL_PARDISO

#define MKL_Complex16 std::complex<double>

#include "mkl_pardiso.h"
#include "operator.hpp"
#include "sparsemat.hpp"

namespace mfem
{
struct ComplexCSRMatrix
{
    std::vector<std::complex<double>> values;
    std::vector<int> row_offsets;
    std::vector<int> cols;
    int numNonZeros;
};

/**
 * @brief MKL Parallel Direct Sparse Solver PARDISO
 *
 * Interface to MKL PARDISO: the direct sparse complex solver based on PARDISO
 */
class PardisoCompSolver : public mfem::Solver
{
public:
    enum MatType
    {
        COMPLEX_STRUCTURE_SYMMETRIC = 3,
        COMPLEX_HERMITIAN_POS_INDEFINITE = 4,
        COMPLEX_HERMITIAN_INDEFINITE = -4,
        COMPLEX_SYMMETRIC = 6,
        COMPLEX_UNSYMMETRIC = 13
   };

   /**
    * @brief Construct a new PardisoCompSolver object
    *
    */
   PardisoCompSolver(MatType mat_type = MatType::COMPLEX_UNSYMMETRIC);

   /**
    * @brief Set the Operator object and perform factorization
    *
    * @a op needs to be of type ComplexSparseMatrix.
    *
    * @param op Operator to use in factorization and solve
    */
   void SetOperator(const Operator &op) override;

   /**
    * @ brief convert the data to a new complex array
    * @x_r
    * @x_i
    */
   void CSRRealToComplex(const SparseMatrix* x_r, const SparseMatrix* x_i);

   /**
    * @brief Solve
    *
    * @param b RHS vector
    * @param x Solution vector
    */
   void Mult(const Vector &b, Vector &x) const override;

   void GetResidual(const Vector& b, const Vector& x) const;

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

   ~PardisoCompSolver();

private:
   // Global number of rows
   int m;

   // Number of nonzero entries
   int nnz;

   // CSR data structure for the copy data of the local CSR matrix
   ComplexCSRMatrix *complexCSR = nullptr;
   int *csr_rowptr = nullptr;
   std::complex<double> *reordered_csr_nzval = nullptr;
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
   mutable std::complex<double> ddum;
};
} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO

#endif
