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

#ifndef MFEM_SUPERLU
#define MFEM_SUPERLU

#include "../config/config.hpp"

#ifdef MFEM_USE_SUPERLU
#ifdef MFEM_USE_MPI

#include "operator.hpp"
#include "hypre.hpp"
#include <mpi.h>

namespace mfem
{

namespace superlu
{

// Copy selected enumerations from SuperLU (from superlu_enum_consts.h)
#ifdef MFEM_USE_SUPERLU5
typedef enum
{
   NOROWPERM,
   LargeDiag,
   MY_PERMR
} RowPerm;
#else
/// Define the type of row permutation
typedef enum
{
   /// No row permutation
   NOROWPERM,
   /** @brief Duff/Koster algorithm to make the diagonals large compared to the
       off-diagonals. Use LargeDiag for SuperLU version 5 and below. */
   LargeDiag_MC64,
   /** @brief Parallel approximate weight perfect matching to make the diagonals
       large compared to the off-diagonals.  Option doesn't exist in SuperLU
       version 5 and below. */
   LargeDiag_HWPM,
   /// User defined row permutation
   MY_PERMR
} RowPerm;
#endif

/// Define the type of column permutation
typedef enum
{
   /// Natural ordering
   NATURAL,
   /// Minimum degree ordering on structure of $ A^T*A $
   MMD_ATA,
   /// Minimum degree ordering on structure of $ A^T+A $
   MMD_AT_PLUS_A,
   /// Approximate minimum degree column ordering
   COLAMD,
   /// Sequential ordering on structure of $ A^T+A $ using the METIS package
   METIS_AT_PLUS_A,
   /** @brief Sequential ordering on structure of $ A^T+A $ using the
       PARMETIS package */
   PARMETIS,
   /// Use the Zoltan library from Sandia to define the column ordering
   ZOLTAN,
   /// User defined column permutation
   MY_PERMC
} ColPerm;

/// Define how to do iterative refinement
typedef enum
{
   /// No interative refinement
   NOREFINE,
   /// Iterative refinement accumulating residuals in a float.
   SLU_SINGLE=1,
   /// Iterative refinement accumulating residuals in a double.
   SLU_DOUBLE,
   /// Iterative refinement accumulating residuals in a higher precision variable.
   SLU_EXTRA
} IterRefine;

/** @brief Define the information that is provided about the matrix
    factorization ahead of time. */
typedef enum
{
   /// No information is provided, do the full factorization.
   DOFACT,
   /** @brief Matrix A will be factored assuming the sparsity is the same as a
       previous factorization.  Column permutations will be reused. */
   SamePattern,
   /** @brief Matrix A will be factored assuming the sparsity is the same and
       the matrix as a previous are similar as a previous factorization.  Column
       permutations and row permutations will be reused. */
   SamePattern_SameRowPerm,
   /** @brief The matrix A was provided in fully factored form and no
       factorization is needed. */
   FACTORED
} Fact;

} // namespace superlu

class SuperLURowLocMatrix : public Operator
{
public:
   /** @brief Creates a general parallel matrix from a local CSR matrix on each
       processor described by the I, J and data arrays. The local matrix should
       be of size (local) nrows by (global) glob_ncols. The new parallel matrix
       contains copies of all input arrays (so they can be deleted). */
   SuperLURowLocMatrix(MPI_Comm comm,
                       int num_loc_rows, HYPRE_BigInt first_loc_row,
                       HYPRE_BigInt glob_nrows, HYPRE_BigInt glob_ncols,
                       int *I, HYPRE_BigInt *J, double *data);

   /** @brief Creates a copy of the parallel matrix hypParMat in SuperLU's
       RowLoc format. All data is copied so the original matrix may be
       deleted. */
   SuperLURowLocMatrix(const Operator &op);

   ~SuperLURowLocMatrix();

   /// Matrix Vector products are not supported for this type of matrix
   void Mult(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("SuperLURowLocMatrix::Mult: Matrix vector products are not "
                 "supported!");
   }

   void *InternalData() const { return rowLocPtr_; }

   /// Get the MPI communicator for this matrix
   MPI_Comm GetComm() const { return comm_; }

   /// Get the number of global rows in this matrix
   HYPRE_BigInt GetGlobalNumRows() const { return num_global_rows_; }

   /// Get the number of global columns in this matrix
   HYPRE_BigInt GetGlobalNumColumns() const { return num_global_cols_; }

private:
   MPI_Comm     comm_;
   void        *rowLocPtr_;
   HYPRE_BigInt num_global_rows_, num_global_cols_;
};

/** The MFEM wrapper around the SuperLU Direct Solver class.

    The mfem::SuperLUSolver class uses the SuperLU_DIST library to perform LU
    factorization of a parallel sparse matrix. The solver is capable of handling
    double precision types. It is currently maintained by Xiaoye Sherry Li at
    NERSC, see http://crd-legacy.lbl.gov/~xiaoye/SuperLU/.
*/
class SuperLUSolver : public Solver
{
public:
   /** @brief Constructor with MPI_Comm parameter.

       @a npdep is the replication factor for the matrix
       data and must be a power of 2 and divide evenly
       into the number of processors. */
   SuperLUSolver(MPI_Comm comm, int npdep = 1);

   /** @brief Constructor with SuperLU matrix object.

       @a npdep is the replication factor for the matrix
       data and must be a power of 2 and divide evenly
       into the number of processors. */
   SuperLUSolver(SuperLURowLocMatrix &A, int npdep = 1);

   /// Default destructor.
   ~SuperLUSolver();

   /** @brief Set the operator/matrix.
       @note  @a A must be a SuperLURowLocMatrix. */
   void SetOperator(const Operator &op);

   /** @brief Factor and solve the linear system $ y = Op^{-1} x $
       @note Factorization modifies the operator matrix. */
   void Mult(const Vector &x, Vector &y) const;

   /** @brief Factor and solve the linear systems $ y_i = Op^{-1} x_i $
       for all i in the @a X and @a Y arrays.
       @note Factorization modifies the operator matrix. */
   void ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const;

   /** @brief Factor and solve the transposed linear system
       $ y = Op^{-T} x $
       @note Factorization modifies the operator matrix. */
   void MultTranspose(const Vector &x, Vector &y) const;

   /** @brief Factor and solve the transposed linear systems
       $ y_i = Op^{-T} x_i $ for all i in the @a X and @a Y arrays.
       @note Factorization modifies the operator matrix. */
   void ArrayMultTranspose(const Array<const Vector *> &X,
                           Array<Vector *> &Y) const;

   /// Specify whether to print the solver statistics (default true)
   void SetPrintStatistics(bool print_stat);

   /** @brief Specify whether to equibrate the system scaling to make
      the rows and columns have unit norms.  (default true) */
   void SetEquilibriate(bool equil);

   /** @brief Specify how to permute the columns of the matrix.

      Supported options are:
      superlu::NATURAL, superlu::MMD_ATA, superlu::MMD_AT_PLUS_A,
      superlu::COLAMD, superlu::METIS_AT_PLUS_A (default),
      superlu::PARMETIS, superlu::ZOLTAN, superlu::MY_PERMC */
   void SetColumnPermutation(superlu::ColPerm col_perm);

   /** @brief Specify how to permute the rows of the matrix.

      Supported options are:
      superlu::NOROWPERM, superlu::LargeDiag (default), superlu::MY_PERMR for
      SuperLU version 5.  For later versions the supported options are:
      superlu::NOROWPERM, superlu::LargeDiag_MC64 (default),
      superlu::LargeDiag_HWPM, superlu::MY_PERMR */
   void SetRowPermutation(superlu::RowPerm row_perm);

   /** @brief Specify how to handle iterative refinement

      Supported options are:
      superlu::NOREFINE, superlu::SLU_SINGLE,
      superlu::SLU_DOUBLE (default), superlu::SLU_EXTRA */
   void SetIterativeRefine(superlu::IterRefine iter_ref);

   /** @brief Specify whether to replace tiny diagonals encountered
       during pivot with $ \sqrt{\epsilon} \lVert A \rVert $
       (default false) */
   void SetReplaceTinyPivot(bool rtp);

   /// Specify the number of levels in the look-ahead factorization (default 10)
   void SetNumLookAheads(int num_lookaheads);

   /** @brief Specifies whether to use the elimination tree computed from the
       serial symbolic factorization to perform static scheduling
       (default false) */
   void SetLookAheadElimTree(bool etree);

   /** @brief Specify whether the matrix has a symmetric pattern to avoid extra
       work (default false) */
   void SetSymmetricPattern(bool sym);

   /** @brief Specify whether to perform parallel symbolic factorization.
       @note If true SuperLU will use superlu::PARMETIS for the Column
       Permutation regardless of the setting */
   void SetParSymbFact(bool par);

   /** @brief Specify what information has been provided ahead of time about the
       factorization of A.

       Supported options are:
       superlu::DOFACT, superlu::SamePattern, superlu::SamePattern_SameRowPerm,
       superlu::FACTORED*/
   void SetFact(superlu::Fact fact);

   // Processor grid for SuperLU_DIST.
   const int nprow_, npcol_, npdep_;

private:
   // Initialize the solver.
   void Init(MPI_Comm comm);

   // Handle error message from call to SuperLU solver.
   void HandleError(int info) const;

protected:
   const SuperLURowLocMatrix *APtr_;
   mutable Vector             sol_;
   mutable int                nrhs_;

   /** The actual types of the following pointers are hidden to avoid exposing
       the SuperLU header files to the entire library. Their types are given in
       the trailing comments. The reason that this is necessary is that SuperLU
       defines these structs differently for use with its real and complex
       solvers. If we want to add support for SuperLU's complex solvers one day
       we will need to hide these types to avoid name conflicts. */
   void *optionsPtr_;          // superlu_options_t *
   void *ScalePermstructPtr_;  //  ScalePermsruct_t *
   void *LUstructPtr_;         //        LUstruct_t *
   void *SOLVEstructPtr_;      //     SOLVEstruct_t *
   void *gridPtr_;             //        gridinfo_t * or gridinfo3d_t *
};

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
#endif // MFEM_SUPERLU
