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

#ifndef MFEM_CPARDISO
#define MFEM_CPARDISO

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_MKL_CPARDISO

#include "mkl_cluster_sparse_solver.h"
#include "operator.hpp"

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

   ~CPardisoSolver();

private:
   MPI_Comm comm_;

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
   double *reordered_csr_nzval = nullptr;
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
   mutable double ddum;
};
} // namespace mfem

#endif
#endif // MFEM_USE_MKL_CPARDISO
#endif // MFEM_USE_MPI
