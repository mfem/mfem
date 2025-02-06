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

#ifndef MFEM_STRUMPACK
#define MFEM_STRUMPACK

#include "../config/config.hpp"

#ifdef MFEM_USE_STRUMPACK
#ifdef MFEM_USE_MPI

#include "operator.hpp"
#include "hypre.hpp"
#include <mpi.h>

// STRUMPACK headers
#include "StrumpackSparseSolverMPIDist.hpp"
#include "StrumpackSparseSolverMixedPrecisionMPIDist.hpp"

namespace mfem
{

class STRUMPACKRowLocMatrix : public Operator
{
public:
   /** @brief Creates a general parallel matrix from a local CSR matrix on each
       processor.

       The CSR matrix is described by the I, J and data arrays. The local matrix
       should be of size (local) nrows by (global) glob_ncols. The new parallel
       matrix contains copies of all input arrays (so they can be deleted). */
   STRUMPACKRowLocMatrix(MPI_Comm comm,
                         int num_loc_rows, HYPRE_BigInt first_loc_row,
                         HYPRE_BigInt glob_nrows, HYPRE_BigInt glob_ncols,
                         int *I, HYPRE_BigInt *J, double *data,
                         bool sym_sparse = false);

   /** @brief Creates a copy of the parallel matrix hypParMat in STRUMPACK's
       RowLoc format.

       All data is copied so the original matrix may be deleted. */
   STRUMPACKRowLocMatrix(const Operator &op, bool sym_sparse = false);

   ~STRUMPACKRowLocMatrix();

   /// Matrix vector products are not supported for this type of matrix.
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ABORT("STRUMPACKRowLocMatrix::Mult: Matrix vector products are not "
                 "supported!");
   }

   /// Get the MPI Comm being used by the parallel matrix.
   MPI_Comm GetComm() const { return A_->comm(); }

   /// Get access to the internal CSR matrix.
   strumpack::CSRMatrixMPI<double, HYPRE_BigInt> *GetA() const { return A_; }

private:
   strumpack::CSRMatrixMPI<double, HYPRE_BigInt> *A_;
};

/** The MFEM STRUMPACK Direct Solver class.

    The mfem::STRUMPACKSolver class uses the STRUMPACK library to perform LU
    factorization of a parallel sparse matrix. The solver is capable of handling
    double precision types. See
    http://portal.nersc.gov/project/sparse/strumpack/. */
template <typename STRUMPACKSolverType>
class STRUMPACKSolverBase : public Solver
{
protected:
   /** @brief Constructor with MPI_Comm parameter and command line arguments.

       STRUMPACKSolverBase::SetFromCommandLine must be called for the command
       line arguments to be used. */
   STRUMPACKSolverBase(MPI_Comm comm, int argc, char *argv[]);

   /** @brief Constructor with STRUMPACK matrix object and command line
       arguments.

       STRUMPACKSolverBase::SetFromCommandLine must be called for the command
       line arguments to be used. */
   STRUMPACKSolverBase(STRUMPACKRowLocMatrix &A, int argc, char *argv[]);

public:
   /// Default destructor.
   virtual ~STRUMPACKSolverBase();

   /// Factor and solve the linear system $y = Op^{-1} x $.
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Factor and solve the linear systems $ Y_i = Op^{-1} X_i $
       across the array of vectors. */
   void ArrayMult(const Array<const Vector *> &X,
                  Array<Vector *> &Y) const override;

   /** @brief Set the operator/matrix.
       \note  @a A must be a STRUMPACKRowLocMatrix. */
   void SetOperator(const Operator &op) override;

   /** @brief Set options that were captured from the command line.

       These were captured in the constructer STRUMPACKSolverBase. Refer
       to the STRUMPACK documentation for details. */
   void SetFromCommandLine();

   /// Set up verbose printing during the factor step
   void SetPrintFactorStatistics(bool print_stat);

   /// Set up verbose printing during the solve step
   void SetPrintSolveStatistics(bool print_stat);

   /// Set the relative tolerance for interative solvers
   void SetRelTol(double rtol);

   /// Set the absolute tolerance for iterative solvers
   void SetAbsTol(double atol);

   /// Set the maximum number of iterations for iterative solvers
   void SetMaxIter(int max_it);

   /** @brief Set the flag controlling reuse of the symbolic factorization for
       multiple operators.

       This method must be called before repeated calls to SetOperator. */
   void SetReorderingReuse(bool reuse);

   /** @brief Enable GPU off-loading available if STRUMPACK was compiled with
       CUDA.
       @note Input/Output from MFEM to STRUMPACK is all still through host
       memory. */
   void EnableGPU();

   /** @brief Disable GPU off-loading available if STRUMPACK was compiled with
       CUDA.
       @note Input/Output from MFEM to STRUMPACK is all still through host
       memory. */
   void DisableGPU();

   /** @brief Set the Krylov solver method to use
    *
    * STRUMPACK is an (approximate) direct solver. It can be used as a direct
    * solver or as a preconditioner. To use STRUMPACK as only a preconditioner,
    * set the Krylov solver to DIRECT. STRUMPACK also provides iterative solvers
    * which can use the preconditioner, and these iterative solvers can also be
    * used without preconditioner.
    *
    * Supported values are:
    *  - AUTO:           Use iterative refinement if no HSS compression is
    *                    used, otherwise use GMRES
    *  - DIRECT:         No outer iterative solver, just a single application
    *                    of the multifrontal solver
    *  - REFINE:         Iterative refinement
    *  - PREC_GMRES:     Preconditioned GMRES
    *                    The preconditioner is the (approx) multifrontal solver
    *  - GMRES:          UN-preconditioned GMRES (for testing mainly)
    *  - PREC_BICGSTAB:  Preconditioned BiCGStab
    *                    The preconditioner is the (approx) multifrontal solver
    *  - BICGSTAB:       UN-preconditioned BiCGStab. (for testing mainly)
    */
   void SetKrylovSolver(strumpack::KrylovSolver method);

   /** @brief Set matrix reordering strategy
    *
    * Supported reorderings are:
    *  - NATURAL:    Do not reorder the system
    *  - METIS:      Use Metis nested-dissection reordering (default)
    *  - PARMETIS:   Use ParMetis nested-dissection reordering
    *  - SCOTCH:     Use Scotch nested-dissection reordering
    *  - PTSCOTCH:   Use PT-Scotch nested-dissection reordering
    *  - RCM:        Use RCM reordering
    *  - GEOMETRIC:  A simple geometric nested dissection code that
    *                only works for regular meshes
    *  - AMD:        Approximate minimum degree
    *  - MMD:        Multiple minimum degree
    *  - AND:        Nested dissection
    *  - MLF:        Minimum local fill
    *  - SPECTRAL:   Spectral nested dissection
    */
   void SetReorderingStrategy(strumpack::ReorderingStrategy method);

   /** @brief Configure static pivoting for stability.
    *
    * The static pivoting in STRUMPACK
    * permutes the sparse input matrix in order to get large (nonzero) elements
    * on the diagonal. If the input matrix is already diagonally dominant, this
    * reordering can be disabled.
    *
    * Supported matching algorithms are:
    *  - NONE:                          Don't do anything
    *  - MAX_CARDINALITY:               Maximum cardinality
    *  - MAX_SMALLEST_DIAGONAL:         Maximum smallest diagonal value
    *  - MAX_SMALLEST_DIAGONAL_2:       Same as MAX_SMALLEST_DIAGONAL
    *                                   but different algorithm
    *  - MAX_DIAGONAL_SUM:              Maximum sum of diagonal values
    *  - MAX_DIAGONAL_PRODUCT_SCALING:  Maximum product of diagonal values
    *                                   and row and column scaling (default)
    *  - COMBBLAS:                      Use AWPM from CombBLAS (only with
    *                                   version >= 3)
    */
   void SetMatching(strumpack::MatchingJob job);

   /** @brief Select compression for sparse data types
    *
    * Enable support for rank-structured data formats, which can be used
    * for compression within the sparse solver.
    *
    * Supported compression types are:
    *  - NONE:           No compression, purely direct solver (default)
    *  - HSS:            HSS compression of frontal matrices
    *  - BLR:            Block low-rank compression of fronts
    *  - HODLR:          Hierarchically Off-diagonal Low-Rank
    *                    compression of frontal matrices
    *  - BLR_HODLR:      Block low-rank compression of medium
    *                    fronts and Hierarchically Off-diagonal
    *                    Low-Rank compression of large fronts
    *  - ZFP_BLR_HODLR:  ZFP compression for small fronts,
    *                    Block low-rank compression of medium
    *                    fronts and Hierarchically Off-diagonal
    *                    Low-Rank compression of large fronts
    *  - LOSSLESS:       Lossless compression
    *  - LOSSY:          Lossy compression
    *
    * For versions of STRUMPACK < 5, we support only NONE, HSS, and BLR.
    * BLR_HODLR and ZPR_BLR_HODLR are supported in STRUMPACK >= 6.
    */
   void SetCompression(strumpack::CompressionType type);

   /** @brief Set the relative tolerance for low rank compression methods
    *
    * This currently affects BLR, HSS, and HODLR.  Use
    * STRUMPACKSolverBase::SetCompression to set the proper compression type.
    */
   void SetCompressionRelTol(double rtol);

   /** @brief Set the absolute tolerance for low rank compression methods
    *
    * This currently affects BLR, HSS, and HODLR.  Use
    * STRUMPACKSolverBase::SetCompression to set the proper compression type.
    */
   void SetCompressionAbsTol(double atol);

#if STRUMPACK_VERSION_MAJOR >= 5
   /** @brief Set the precision for the lossy compression option
    *
    *  Use STRUMPACKSolverBase::SetCompression to set the proper compression
    * type. */
   void SetCompressionLossyPrecision(int precision);

   /** @brief Set the number of butterfly levels for the HODLR compression
    *  option.
    *
    *  Use STRUMPACKSolverBase::SetCompression to set the proper compression
    *  type. */
   void SetCompressionButterflyLevels(int levels);
#endif

private:
   /// Helper method for calling the STRUMPACK factoriation routine.
   void FactorInternal() const;

protected:
   const STRUMPACKRowLocMatrix *APtr_;
   STRUMPACKSolverType         *solver_;

   bool factor_verbose_;
   bool solve_verbose_;
   bool reorder_reuse_;

   mutable Vector rhs_, sol_;
   mutable int    nrhs_;
};

class STRUMPACKSolver :
   public STRUMPACKSolverBase<strumpack::
   SparseSolverMPIDist<double, HYPRE_BigInt>>
{
public:
   /// Constructor with the MPI Comm parameter
   STRUMPACKSolver(MPI_Comm comm);

   /// Constructor with STRUMPACK matrix object.
   STRUMPACKSolver(STRUMPACKRowLocMatrix &A);

   /** @brief Constructor with MPI_Comm parameter and command line arguments.

       SetFromCommandLine must be called for the command line arguments
       to be used.
   */
   STRUMPACKSolver(MPI_Comm comm, int argc, char *argv[]);
   MFEM_DEPRECATED STRUMPACKSolver(int argc, char *argv[], MPI_Comm comm)
      : STRUMPACKSolver(comm, argc, argv) {}

   /** @brief Constructor with STRUMPACK matrix object and command line
       arguments.

       SetFromCommandLine must be called for the command line arguments
       to be used.
   */
   STRUMPACKSolver(STRUMPACKRowLocMatrix &A, int argc, char *argv[]);

   /// Destructor.
   ~STRUMPACKSolver() {}
};

#if STRUMPACK_VERSION_MAJOR >= 7
class STRUMPACKMixedPrecisionSolver :
   public STRUMPACKSolverBase<strumpack::
   SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>
{
public:
   /// Constructor with MPI_Comm parameter.
   STRUMPACKMixedPrecisionSolver(MPI_Comm comm);

   /// Constructor with STRUMPACK matrix object.
   STRUMPACKMixedPrecisionSolver(STRUMPACKRowLocMatrix &A);

   /** @brief Constructor with MPI_Comm parameter and command line arguments.

       SetFromCommandLine must be called for the command line arguments
       to be used.
   */
   STRUMPACKMixedPrecisionSolver(MPI_Comm comm, int argc, char *argv[]);

   /** @brief Constructor with STRUMPACK matrix object and command line
       arguments.

       SetFromCommandLine must be called for the command line arguments
       to be used.
   */
   STRUMPACKMixedPrecisionSolver(STRUMPACKRowLocMatrix &A,
                                 int argc, char *argv[]);

   /// Destructor.
   ~STRUMPACKMixedPrecisionSolver() {}
};
#endif

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
#endif // MFEM_STRUMPACK
