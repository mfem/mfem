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

#ifndef MFEM_HYPRE
#define MFEM_HYPRE

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>

// Enable internal hypre timing routines
#define HYPRE_TIMING

// hypre header files
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "temp_multivector.h"
#include "../general/globals.hpp"

#ifdef HYPRE_COMPLEX
#error "MFEM does not work with HYPRE's complex numbers support"
#endif

#include "sparsemat.hpp"
#include "hypre_parcsr.hpp"
#ifdef MFEM_USE_SUNDIALS
#include <nvector/nvector_parhyp.h>
#endif

namespace mfem
{

class ParFiniteElementSpace;
class HypreParMatrix;

namespace internal
{

template <typename int_type>
inline int to_int(int_type i)
{
   MFEM_ASSERT(int_type(int(i)) == i, "overflow converting int_type to int");
   return int(i);
}

// Specialization for to_int(int)
template <> inline int to_int(int i) { return i; }

// Convert a HYPRE_Int to int
#ifdef HYPRE_BIGINT
template <>
inline int to_int(HYPRE_Int i)
{
   MFEM_ASSERT(HYPRE_Int(int(i)) == i, "overflow converting HYPRE_Int to int");
   return int(i);
}
#endif

}

/// Wrapper for hypre's parallel vector class
class HypreParVector : public Vector
{
private:
   int own_ParVector;

   /// The actual object
   hypre_ParVector *x;

   friend class HypreParMatrix;

   // Set Vector::data and Vector::size from *x
   inline void _SetDataAndSize_();

public:

   HypreParVector() {}

   /** @brief Creates vector with given global size and parallel partitioning of
       the rows/columns given by @a col. */
   /** @anchor hypre_partitioning_descr
       The partitioning is defined in one of two ways depending on the
       configuration of HYPRE:
       1. If HYPRE_AssumedPartitionCheck() returns true (the default),
          then col is of length 2 and the local processor owns columns
          [col[0],col[1]).
       2. If HYPRE_AssumedPartitionCheck() returns false, then col is of
          length (number of processors + 1) and processor P owns columns
          [col[P],col[P+1]) i.e. each processor has a copy of the same col
          array. */
   HypreParVector(MPI_Comm comm, HYPRE_Int glob_size, HYPRE_Int *col);
   /** @brief Creates vector with given global size, partitioning of the
       columns, and data. */
   /** The data must be allocated and destroyed outside. If @a _data is NULL, a
       dummy vector without a valid data array will be created. See @ref
       hypre_partitioning_descr "here" for a description of the @a col array. */
   HypreParVector(MPI_Comm comm, HYPRE_Int glob_size, double *_data,
                  HYPRE_Int *col);
   /// Creates vector compatible with y
   HypreParVector(const HypreParVector &y);
   /// Creates vector compatible with (i.e. in the domain of) A or A^T
   explicit HypreParVector(const HypreParMatrix &A, int transpose = 0);
   /// Creates vector wrapping y
   explicit HypreParVector(HYPRE_ParVector y);
   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   explicit HypreParVector(ParFiniteElementSpace *pfes);

   /// MPI communicator
   MPI_Comm GetComm() { return x->comm; }

   void WrapHypreParVector(hypre_ParVector *y);

   /// Returns the row partitioning
   inline HYPRE_Int *Partitioning() { return x->partitioning; }

   /// Returns the global number of rows
   inline HYPRE_Int GlobalSize() { return x->global_size; }

   /// Typecasting to hypre's hypre_ParVector*
   operator hypre_ParVector*() const { return x; }
#ifndef HYPRE_PAR_VECTOR_STRUCT
   /// Typecasting to hypre's HYPRE_ParVector, a.k.a. void *
   operator HYPRE_ParVector() const { return (HYPRE_ParVector) x; }
#endif
   /// Changes the ownership of the the vector
   hypre_ParVector *StealParVector() { own_ParVector = 0; return x; }

   /// Sets ownership of the internal hypre_ParVector
   void SetOwnership(int own) { own_ParVector = own; }

   /// Gets ownership of the internal hypre_ParVector
   int GetOwnership() const { return own_ParVector; }

   /// Returns the global vector in each processor
   Vector* GlobalVector() const;

   /// Set constant values
   HypreParVector& operator= (double d);
   /// Define '=' for hypre vectors.
   HypreParVector& operator= (const HypreParVector &y);

   /// Sets the data of the Vector and the hypre_ParVector to @a _data.
   /** Must be used only for HypreParVector%s that do not own the data,
       e.g. created with the constructor:
       HypreParVector(MPI_Comm, HYPRE_Int, double *, HYPRE_Int *). */
   void SetData(double *_data);

   /// Set random values
   HYPRE_Int Randomize(HYPRE_Int seed);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname) const;

   /// Calls hypre's destroy function
   ~HypreParVector();

#ifdef MFEM_USE_SUNDIALS
   /// Return a new wrapper SUNDIALS N_Vector of type SUNDIALS_NVEC_PARHYP.
   /** The returned N_Vector must be destroyed by the caller. */
   virtual N_Vector ToNVector() { return N_VMake_ParHyp(x); }

   /** @brief Update an existing wrapper SUNDIALS N_Vector of type
       SUNDIALS_NVEC_PARHYP to point to this Vector. */
   virtual void ToNVector(N_Vector &nv);
#endif
};

/// Returns the inner product of x and y
double InnerProduct(HypreParVector &x, HypreParVector &y);
double InnerProduct(HypreParVector *x, HypreParVector *y);


/** @brief Compute the l_p norm of the Vector which is split without overlap
    across the given communicator. */
double ParNormlp(const Vector &vec, double p, MPI_Comm comm);


/// Wrapper for hypre's ParCSR matrix class
class HypreParMatrix : public Operator
{
private:
   /// The actual object
   hypre_ParCSRMatrix *A;

   /// Auxiliary vectors for typecasting
   mutable HypreParVector *X, *Y;

   // Flags indicating ownership of A->diag->{i,j,data}, A->offd->{i,j,data},
   // and A->col_map_offd.
   // The possible values for diagOwner are:
   //  -1: no special treatment of A->diag (default)
   //   0: prevent hypre from destroying A->diag->{i,j,data}
   //   1: same as 0, plus take ownership of A->diag->{i,j}
   //   2: same as 0, plus take ownership of A->diag->data
   //   3: same as 0, plus take ownership of A->diag->{i,j,data}
   // The same values and rules apply to offdOwner and A->offd.
   // The possible values for colMapOwner are:
   //  -1: no special treatment of A->col_map_offd (default)
   //   0: prevent hypre from destroying A->col_map_offd
   //   1: same as 0, plus take ownership of A->col_map_offd
   // All owned arrays are destroyed with 'delete []'.
   signed char diagOwner, offdOwner, colMapOwner;

   // Does the object own the pointer A?
   signed char ParCSROwner;

   // Initialize with defaults. Does not initialize inherited members.
   void Init();

   // Delete all owned data. Does not perform re-initialization with defaults.
   void Destroy();

   // Copy (shallow/deep, based on HYPRE_BIGINT) the I and J arrays from csr to
   // hypre_csr. Shallow copy the data. Return the appropriate ownership flag.
   static char CopyCSR(SparseMatrix *csr, hypre_CSRMatrix *hypre_csr);
   // Copy (shallow or deep, based on HYPRE_BIGINT) the I and J arrays from
   // bool_csr to hypre_csr. Allocate the data array and set it to all ones.
   // Return the appropriate ownership flag.
   static char CopyBoolCSR(Table *bool_csr, hypre_CSRMatrix *hypre_csr);

   // Copy the j array of a hypre_CSRMatrix to the given J array, converting
   // the indices from HYPRE_Int to int.
   static void CopyCSR_J(hypre_CSRMatrix *hypre_csr, int *J);

public:
   /// An empty matrix to be used as a reference to an existing matrix
   HypreParMatrix();

   void WrapHypreParCSRMatrix(hypre_ParCSRMatrix *a, bool owner = true)
   {
      A = a;
      if (!owner) { ParCSROwner = 0; }
      height = GetNumRows();
      width = GetNumCols();
   }

   /// Converts hypre's format to HypreParMatrix
   /** If @a owner is false, ownership of @a a is not transferred */
   explicit HypreParMatrix(hypre_ParCSRMatrix *a, bool owner = true)
   {
      Init();
      WrapHypreParCSRMatrix(a, owner);
   }

   /** Creates block-diagonal square parallel matrix. Diagonal is given by diag
       which must be in CSR format (finalized). The new HypreParMatrix does not
       take ownership of any of the input arrays.

       @warning The ordering of the columns in each row in @a *diag may be
       changed by this constructor to ensure that the first entry in each row is
       the diagonal one. This is expected by most hypre functions. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int glob_size, HYPRE_Int *row_starts,
                  SparseMatrix *diag); // constructor with 4 arguments, v1

   /// Creates block-diagonal rectangular parallel matrix.
   /** Diagonal is given by @a diag which must be in CSR format (finalized). The
       new HypreParMatrix does not take ownership of any of the input arrays.
       See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts,
                  SparseMatrix *diag); // constructor with 6 arguments, v1

   /// Creates general (rectangular) parallel matrix.
   /** The new HypreParMatrix does not take ownership of any of the input
       arrays. See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts, SparseMatrix *diag, SparseMatrix *offd,
                  HYPRE_Int *cmap); // constructor with 8 arguments

   /// Creates general (rectangular) parallel matrix.
   /** The new HypreParMatrix takes ownership of all input arrays, except
       @a col_starts and @a row_starts. See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a row_starts and @a
       col_starts. */
   HypreParMatrix(MPI_Comm comm,
                  HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
                  HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                  HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
                  HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
                  HYPRE_Int offd_num_cols,
                  HYPRE_Int *offd_col_map); // constructor with 13 arguments

   /// Creates a parallel matrix from SparseMatrix on processor 0.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                  SparseMatrix *a); // constructor with 4 arguments, v2

   /// Creates boolean block-diagonal rectangular parallel matrix.
   /** The new HypreParMatrix does not take ownership of any of the input
       arrays. See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_Int global_num_rows,
                  HYPRE_Int global_num_cols, HYPRE_Int *row_starts,
                  HYPRE_Int *col_starts,
                  Table *diag); // constructor with 6 arguments, v2

   /// Creates boolean rectangular parallel matrix.
   /** The new HypreParMatrix takes ownership of the arrays @a i_diag,
       @a j_diag, @a i_offd, @a j_offd, and @a cmap; does not take ownership of
       the arrays @a row and @a col. See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a row and @a col. */
   HypreParMatrix(MPI_Comm comm, int id, int np, HYPRE_Int *row, HYPRE_Int *col,
                  HYPRE_Int *i_diag, HYPRE_Int *j_diag, HYPRE_Int *i_offd,
                  HYPRE_Int *j_offd, HYPRE_Int *cmap,
                  HYPRE_Int cmap_size); // constructor with 11 arguments

   /** @brief Creates a general parallel matrix from a local CSR matrix on each
       processor described by the @a I, @a J and @a data arrays. */
   /** The local matrix should be of size (local) @a nrows by (global)
       @a glob_ncols. The new parallel matrix contains copies of all input
       arrays (so they can be deleted). See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a rows and @a cols. */
   HypreParMatrix(MPI_Comm comm, int nrows, HYPRE_Int glob_nrows,
                  HYPRE_Int glob_ncols, int *I, HYPRE_Int *J,
                  double *data, HYPRE_Int *rows,
                  HYPRE_Int *cols); // constructor with 9 arguments

   /** @brief Copy constructor for a ParCSR matrix which creates a deep copy of
       structure and data from @a P. */
   HypreParMatrix(const HypreParMatrix &P);

   /// Make this HypreParMatrix a reference to 'master'
   void MakeRef(const HypreParMatrix &master);

   /// MPI communicator
   MPI_Comm GetComm() const { return A->comm; }
   void BuildComm() const { hypre_MatvecCommPkgCreate(A); }

   /// Typecasting to hypre's hypre_ParCSRMatrix*
   operator hypre_ParCSRMatrix*() const { return A; }
#ifndef HYPRE_PAR_CSR_MATRIX_STRUCT
   /// Typecasting to hypre's HYPRE_ParCSRMatrix, a.k.a. void *
   operator HYPRE_ParCSRMatrix() { return (HYPRE_ParCSRMatrix) A; }
#endif
   /// Changes the ownership of the the matrix
   hypre_ParCSRMatrix* StealData();

   /// Explicitly set the three ownership flags, see docs for diagOwner etc.
   void SetOwnerFlags(signed char diag, signed char offd, signed char colmap)
   { diagOwner = diag, offdOwner = offd, colMapOwner = colmap; }

   /// Get diag ownership flag
   signed char OwnsDiag() const { return diagOwner; }
   /// Get offd ownership flag
   signed char OwnsOffd() const { return offdOwner; }
   /// Get colmap ownership flag
   signed char OwnsColMap() const { return colMapOwner; }

   /** If the HypreParMatrix does not own the row-starts array, make a copy of
       it that the HypreParMatrix will own. If the col-starts array is the same
       as the row-starts array, col-starts is also replaced. */
   void CopyRowStarts();
   /** If the HypreParMatrix does not own the col-starts array, make a copy of
       it that the HypreParMatrix will own. If the row-starts array is the same
       as the col-starts array, row-starts is also replaced. */
   void CopyColStarts();

   /// Returns the global number of nonzeros
   inline HYPRE_Int NNZ() const { return A->num_nonzeros; }
   /// Returns the row partitioning
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline HYPRE_Int *RowPart() { return A->row_starts; }
   /// Returns the column partitioning
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline HYPRE_Int *ColPart() { return A->col_starts; }
   /// Returns the row partitioning (const version)
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline const HYPRE_Int *RowPart() const { return A->row_starts; }
   /// Returns the column partitioning (const version)
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline const HYPRE_Int *ColPart() const { return A->col_starts; }
   /// Returns the global number of rows
   inline HYPRE_Int M() const { return A->global_num_rows; }
   /// Returns the global number of columns
   inline HYPRE_Int N() const { return A->global_num_cols; }

   /// Get the local diagonal of the matrix.
   void GetDiag(Vector &diag) const;
   /// Get the local diagonal block. NOTE: 'diag' will not own any data.
   void GetDiag(SparseMatrix &diag) const;
   /// Get the local off-diagonal block. NOTE: 'offd' will not own any data.
   void GetOffd(SparseMatrix &offd, HYPRE_Int* &cmap) const;
   /// Get on-processor rows as CSR matrix.
   void GetProcRows(SparseMatrix &colCSRMat);

   /** Split the matrix into M x N equally sized blocks of parallel matrices.
       The size of 'blocks' must already be set to M x N. */
   void GetBlocks(Array2D<HypreParMatrix*> &blocks,
                  bool interleaved_rows = false,
                  bool interleaved_cols = false) const;

   /// Returns the transpose of *this
   HypreParMatrix * Transpose() const;

   /** Returns principle submatrix given by array of indices of connections
       with relative size > \@ threshold in *this. */
   HypreParMatrix * ExtractSubmatrix(Array<int> &indices,
                                     double threshhold=0.0) const;

   /// Returns the number of rows in the diagonal block of the ParCSRMatrix
   int GetNumRows() const
   {
      return internal::to_int(
                hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)));
   }

   /// Returns the number of columns in the diagonal block of the ParCSRMatrix
   int GetNumCols() const
   {
      return internal::to_int(
                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A)));
   }

   /// Return the global number of rows
   HYPRE_Int GetGlobalNumRows() const
   { return hypre_ParCSRMatrixGlobalNumRows(A); }

   /// Return the global number of columns
   HYPRE_Int GetGlobalNumCols() const
   { return hypre_ParCSRMatrixGlobalNumCols(A); }

   /// Return the parallel row partitioning array.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   HYPRE_Int *GetRowStarts() const { return hypre_ParCSRMatrixRowStarts(A); }

   /// Return the parallel column partitioning array.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   HYPRE_Int *GetColStarts() const { return hypre_ParCSRMatrixColStarts(A); }

   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HypreParVector &x, HypreParVector &y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A^t * x + beta * y
   HYPRE_Int MultTranspose(HypreParVector &x, HypreParVector &y,
                           double alpha = 1.0, double beta = 0.0);

   void Mult(double a, const Vector &x, double b, Vector &y) const;
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   virtual void Mult(const Vector &x, Vector &y) const
   { Mult(1.0, x, 0.0, y); }
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { MultTranspose(1.0, x, 0.0, y); }

   /** The "Boolean" analog of y = alpha * A * x + beta * y, where elements in
       the sparsity pattern of the matrix are treated as "true". */
   void BooleanMult(int alpha, const int *x, int beta, int *y)
   {
      internal::hypre_ParCSRMatrixBooleanMatvec(A, alpha, const_cast<int*>(x),
                                                beta, y);
   }

   /** The "Boolean" analog of y = alpha * A^T * x + beta * y, where elements in
       the sparsity pattern of the matrix are treated as "true". */
   void BooleanMultTranspose(int alpha, const int *x, int beta, int *y)
   {
      internal::hypre_ParCSRMatrixBooleanMatvecT(A, alpha, const_cast<int*>(x),
                                                 beta, y);
   }

   /// Initialize all entries with value.
   HypreParMatrix &operator=(double value)
   { internal::hypre_ParCSRMatrixSetConstantValues(A, value); return *this; }

   /** Perform the operation `*this += B`, assuming that both matrices use the
       same row and column partitions and the same col_map_offd arrays, or B has
       an empty off-diagonal block. We also assume that the sparsity pattern of
       `*this` contains that of `B`. */
   HypreParMatrix &operator+=(const HypreParMatrix &B) { return Add(1.0, B); }

   /** Perform the operation `*this += beta*B`, assuming that both matrices use
       the same row and column partitions and the same col_map_offd arrays, or
       B has an empty off-diagonal block. We also assume that the sparsity
       pattern of `*this` contains that of `B`. For a more general case consider
       the stand-alone function ParAdd described below. */
   HypreParMatrix &Add(const double beta, const HypreParMatrix &B)
   {
      MFEM_VERIFY(internal::hypre_ParCSRMatrixSum(A, beta, B.A) == 0,
                  "error in hypre_ParCSRMatrixSum");
      return *this;
   }

   /** @brief Multiply the HypreParMatrix on the left by a block-diagonal
       parallel matrix @a D and return the result as a new HypreParMatrix. */
   /** If @a D has a different number of rows than @a A (this matrix), @a D's
       row starts array needs to be given (as returned by the methods
       GetDofOffsets/GetTrueDofOffsets of ParFiniteElementSpace). The new
       matrix @a D*A uses copies of the row- and column-starts arrays, so "this"
       matrix and @a row_starts can be deleted.
       @note This operation is local and does not require communication. */
   HypreParMatrix* LeftDiagMult(const SparseMatrix &D,
                                HYPRE_Int* row_starts = NULL) const;

   /// Scale the local row i by s(i).
   void ScaleRows(const Vector & s);
   /// Scale the local row i by 1./s(i)
   void InvScaleRows(const Vector & s);
   /// Scale all entries by s: A_scaled = s*A.
   void operator*=(double s);

   /// Remove values smaller in absolute value than some threshold
   void Threshold(double threshold = 0.0);

   /// If a row contains only zeros, set its diagonal to 1.
   void EliminateZeroRows() { hypre_ParCSRMatrixFixZeroRows(A); }

   /** Eliminate rows and columns from the matrix, and rows from the vector B.
       Modify B with the BC values in X. */
   void EliminateRowsCols(const Array<int> &rows_cols, const HypreParVector &X,
                          HypreParVector &B);

   /** Eliminate rows and columns from the matrix and store the eliminated
       elements in a new matrix Ae (returned), so that the modified matrix and
       Ae sum to the original matrix. */
   HypreParMatrix* EliminateRowsCols(const Array<int> &rows_cols);

   /** Eliminate columns from the matrix and store the eliminated elements in a
       new matrix Ae (returned) so that the modified matrix and Ae sum to the
       original matrix. */
   HypreParMatrix* EliminateCols(const Array<int> &cols);

   /// Eliminate rows from the diagonal and off-diagonal blocks of the matrix.
   void EliminateRows(const Array<int> &rows);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname, HYPRE_Int offi = 0, HYPRE_Int offj = 0);
   /// Reads the matrix from a file
   void Read(MPI_Comm comm, const char *fname);
   /// Read a matrix saved as a HYPRE_IJMatrix
   void Read_IJMatrix(MPI_Comm comm, const char *fname);

   /// Print information about the hypre_ParCSRCommPkg of the HypreParMatrix.
   void PrintCommPkg(std::ostream &out = mfem::out) const;

   /// Calls hypre's destroy function
   virtual ~HypreParMatrix() { Destroy(); }

   Type GetType() const { return Hypre_ParCSR; }
};

int BlockInvScal(const HypreParMatrix *A, HypreParMatrix *C,
                 const Vector *b, HypreParVector *d, int block, int job);

/** @brief Return a new matrix `C = alpha*A + beta*B`, assuming that both `A`
    and `B` use the same row and column partitions and the same `col_map_offd`
    arrays. */
HypreParMatrix *Add(double alpha, const HypreParMatrix &A,
                    double beta,  const HypreParMatrix &B);
HypreParMatrix *HypreParMatrixAdd(double alpha, const HypreParMatrix &A,
                                  double beta,  const HypreParMatrix &B);

/** Returns the matrix @a A * @a B. Returned matrix does not necessarily own
    row or column starts unless the bool @a own_matrix is set to true. */
HypreParMatrix * ParMult(const HypreParMatrix *A, const HypreParMatrix *B,
                         bool own_matrix = false);

/// Returns the matrix A + B
/** It is assumed that both matrices use the same row and column partitions and
    the same col_map_offd arrays. */
HypreParMatrix * ParAdd(const HypreParMatrix *A, const HypreParMatrix *B);

/// Returns the matrix P^t * A * P
HypreParMatrix * RAP(const HypreParMatrix *A, const HypreParMatrix *P);
/// Returns the matrix Rt^t * A * P
HypreParMatrix * RAP(const HypreParMatrix * Rt, const HypreParMatrix *A,
                     const HypreParMatrix *P);

/// Returns a merged hypre matrix constructed from hypre matrix blocks.
/** It is assumed that all block matrices use the same communicator, and the
    block sizes are consistent in rows and columns. Rows and columns are
    renumbered but not redistributed in parallel, e.g. the block rows owned by
    each process remain on that process in the resulting matrix. Some blocks can
    be NULL. Each block and the entire system can be rectangular. Scalability to
    extremely large processor counts is limited by global MPI communication, see
    GatherBlockOffsetData in hypre.cpp. */
HypreParMatrix * HypreParMatrixFromBlocks(Array2D<HypreParMatrix*> &blocks,
                                          Array2D<double> *blockCoeff=NULL);

/** Eliminate essential BC specified by 'ess_dof_list' from the solution X to
    the r.h.s. B. Here A is a matrix with eliminated BC, while Ae is such that
    (A+Ae) is the original (Neumann) matrix before elimination. */
void EliminateBC(HypreParMatrix &A, HypreParMatrix &Ae,
                 const Array<int> &ess_dof_list, const Vector &X, Vector &B);


/// Parallel smoothers in hypre
class HypreSmoother : public Solver
{
protected:
   /// The linear system matrix
   HypreParMatrix *A;
   /// Right-hand side and solution vectors
   mutable HypreParVector *B, *X;
   /// Temporary vectors
   mutable HypreParVector *V, *Z;
   /// FIR Filter Temporary Vectors
   mutable HypreParVector *X0, *X1;

   /** Smoother type from hypre_ParCSRRelax() in ams.c plus extensions, see the
       enumeration Type below. */
   int type;
   /// Number of relaxation sweeps
   int relax_times;
   /// Damping coefficient (usually <= 1)
   double relax_weight;
   /// SOR parameter (usually in (0,2))
   double omega;
   /// Order of the smoothing polynomial
   int poly_order;
   /// Fraction of spectrum to smooth for polynomial relaxation
   double poly_fraction;
   /// Apply the polynomial smoother to A or D^{-1/2} A D^{-1/2}
   int poly_scale;

   /// Taubin's lambda-mu method parameters
   double lambda;
   double mu;
   int taubin_iter;

   /// l1 norms of the rows of A
   double *l1_norms;
   /// If set, take absolute values of the computed l1_norms
   bool pos_l1_norms;
   /// Number of CG iterations to determine eigenvalue estimates
   int eig_est_cg_iter;
   /// Maximal eigenvalue estimate for polynomial smoothing
   double max_eig_est;
   /// Minimal eigenvalue estimate for polynomial smoothing
   double min_eig_est;
   /// Parameters for windowing function of FIR filter
   double window_params[3];

   /// Combined coefficients for windowing and Chebyshev polynomials.
   double* fir_coeffs;

public:
   /** Hypre smoother types:
       0    = Jacobi
       1    = l1-scaled Jacobi
       2    = l1-scaled block Gauss-Seidel/SSOR
       4    = truncated l1-scaled block Gauss-Seidel/SSOR
       5    = lumped Jacobi
       6    = Gauss-Seidel
       16   = Chebyshev
       1001 = Taubin polynomial smoother
       1002 = FIR polynomial smoother. */
   enum Type { Jacobi = 0, l1Jacobi = 1, l1GS = 2, l1GStr = 4, lumpedJacobi = 5,
               GS = 6, TS = 10, Chebyshev = 16, Taubin = 1001, FIR = 1002
             };

   HypreSmoother();

   HypreSmoother(HypreParMatrix &_A, int type = l1GS,
                 int relax_times = 1, double relax_weight = 1.0,
                 double omega = 1.0, int poly_order = 2,
                 double poly_fraction = .3, int eig_est_cg_iter = 10);

   /// Set the relaxation type and number of sweeps
   void SetType(int type, int relax_times = 1);
   /// Set SOR-related parameters
   void SetSOROptions(double relax_weight, double omega);
   /// Set parameters for polynomial smoothing
   /** By default, 10 iterations of CG are used to estimate the eigenvalues.
       Setting eig_est_cg_iter = 0 uses hypre's hypre_ParCSRMaxEigEstimate() instead. */
   void SetPolyOptions(int poly_order, double poly_fraction,
                       int eig_est_cg_iter = 10);
   /// Set parameters for Taubin's lambda-mu method
   void SetTaubinOptions(double lambda, double mu, int iter);

   /// Convenience function for setting canonical windowing parameters
   void SetWindowByName(const char* window_name);
   /// Set parameters for windowing function for FIR smoother.
   void SetWindowParameters(double a, double b, double c);
   /// Compute window and Chebyshev coefficients for given polynomial order.
   void SetFIRCoefficients(double max_eig);

   /// After computing l1-norms, replace them with their absolute values.
   /** By default, the l1-norms take their sign from the corresponding diagonal
       entries in the associated matrix. */
   void SetPositiveDiagonal(bool pos = true) { pos_l1_norms = pos; }

   /** Set/update the associated operator. Must be called after setting the
       HypreSmoother type and options. */
   virtual void SetOperator(const Operator &op);

   /// Relax the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual ~HypreSmoother();
};


/// Abstract class for hypre's solvers and preconditioners
class HypreSolver : public Solver
{
public:
   /// How to treat errors returned by hypre function calls.
   enum ErrorMode
   {
      IGNORE_HYPRE_ERRORS, ///< Ignore hypre errors (see e.g. HypreADS)
      WARN_HYPRE_ERRORS,   ///< Issue warnings on hypre errors
      ABORT_HYPRE_ERRORS   ///< Abort on hypre errors (default in base class)
   };

protected:
   /// The linear system matrix
   HypreParMatrix *A;

   /// Right-hand side and solution vector
   mutable HypreParVector *B, *X;

   /// Was hypre's Setup function called already?
   mutable int setup_called;
   mutable HYPRE_Int num_iterations;
   mutable double final_res_norm;

   /// How to treat hypre errors.
   mutable ErrorMode error_mode;

public:
   HypreSolver();

   HypreSolver(HypreParMatrix *_A);

   int GetNumIterations() const { return num_iterations; }
   double GetFinalNorm() const { return final_res_norm; }

   /// Typecast to HYPRE_Solver -- return the solver
   virtual operator HYPRE_Solver() const = 0;

   /// hypre's internal Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const = 0;
   /// hypre's internal Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const = 0;

   virtual void SetOperator(const Operator &op)
   { mfem_error("HypreSolvers do not support SetOperator!"); }

   /// Solve the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   /** @brief Set the behavior for treating hypre errors, see the ErrorMode
       enum. The default mode in the base class is ABORT_HYPRE_ERRORS. */
   /** Currently, there are three cases in derived classes where the error flag
       is set to IGNORE_HYPRE_ERRORS:
       * in the method HypreBoomerAMG::SetElasticityOptions(), and
       * in the constructor of classes HypreAMS and HypreADS.
       The reason for this is that a nonzero hypre error is returned) when
       hypre_ParCSRComputeL1Norms() encounters zero row in a matrix, which is
       expected in some cases with the above solvers. */
   void SetErrorMode(ErrorMode err_mode) const { error_mode = err_mode; }

   virtual ~HypreSolver();
};


/// Abstract class for hypre's solvers and preconditioners
class HypreTriSolve : public HypreSolver
{
public:
   HypreTriSolve() : HypreSolver() { }
   explicit HypreTriSolve(HypreParMatrix &A) : HypreSolver(&A) { }
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSolve; }

   HypreParMatrix* GetData() { return A; }
   virtual ~HypreTriSolve() { }
};


/// PCG solver in hypre
class HyprePCG : public HypreSolver
{
private:
   HYPRE_Solver pcg_solver;

   HypreSolver * precond;

public:
   HyprePCG(MPI_Comm comm);

   HyprePCG(HypreParMatrix &_A);

   virtual void SetOperator(const Operator &op);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /** Use the L2 norm of the residual for measuring PCG convergence, plus
       (optionally) 1) periodically recompute true residuals from scratch; and
       2) enable residual-based stopping criteria. */
   void SetResidualConvergenceOptions(int res_frequency=-1, double rtol=0.0);

   /// non-hypre setting
   void SetZeroInintialIterate() { iterative_mode = false; }

   void GetNumIterations(int &num_iterations)
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   /// The typecast to HYPRE_Solver returns the internal pcg_solver
   virtual operator HYPRE_Solver() const { return pcg_solver; }

   /// PCG Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSetup; }
   /// PCG Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSolve; }

   /// Solve Ax=b with hypre's PCG
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HyprePCG();
};

/// GMRES solver in hypre
class HypreGMRES : public HypreSolver
{
private:
   HYPRE_Solver gmres_solver;

   HypreSolver * precond;

   /// Default, generally robust, GMRES options
   void SetDefaultOptions();

public:
   HypreGMRES(MPI_Comm comm);

   HypreGMRES(HypreParMatrix &_A);

   virtual void SetOperator(const Operator &op);

   void SetTol(double tol);
   void SetAbsTol(double tol);
   void SetMaxIter(int max_iter);
   void SetKDim(int dim);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// non-hypre setting
   void SetZeroInintialIterate() { iterative_mode = false; }

   /// The typecast to HYPRE_Solver returns the internal gmres_solver
   virtual operator HYPRE_Solver() const  { return gmres_solver; }

   /// GMRES Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSetup; }
   /// GMRES Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSolve; }

   /// Solve Ax=b with hypre's GMRES
   virtual void Mult (const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HypreGMRES();
};

/// The identity operator as a hypre solver
class HypreIdentity : public HypreSolver
{
public:
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentitySetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentity; }

   virtual ~HypreIdentity() { }
};

/// Jacobi preconditioner in hypre
class HypreDiagScale : public HypreSolver
{
public:
   HypreDiagScale() : HypreSolver() { }
   explicit HypreDiagScale(HypreParMatrix &A) : HypreSolver(&A) { }
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual void SetOperator(const Operator &op);

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale; }

   HypreParMatrix* GetData() { return A; }
   virtual ~HypreDiagScale() { }
};

/// The ParaSails preconditioner in hypre
class HypreParaSails : public HypreSolver
{
private:
   HYPRE_Solver sai_precond;

   /// Default, generally robust, ParaSails options
   void SetDefaultOptions();

   // If sai_precond is NULL, this method allocates it and sets default options.
   // Otherwise the method saves the options from sai_precond, destroys it,
   // allocates a new object, and sets its options to the saved values.
   void ResetSAIPrecond(MPI_Comm comm);

public:
   HypreParaSails(MPI_Comm comm);

   HypreParaSails(HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   void SetSymmetry(int sym);

   /// The typecast to HYPRE_Solver returns the internal sai_precond
   virtual operator HYPRE_Solver() const { return sai_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSolve; }

   virtual ~HypreParaSails();
};

/** The Euclid preconditioner in Hypre

    Euclid implements the Parallel Incomplete LU factorization technique. For
    more information see:

    "A Scalable Parallel Algorithm for Incomplete Factor Preconditioning" by
    David Hysom and Alex Pothen, https://doi.org/10.1137/S1064827500376193
*/
class HypreEuclid : public HypreSolver
{
private:
   HYPRE_Solver euc_precond;

   /// Default, generally robust, Euclid options
   void SetDefaultOptions();

   // If euc_precond is NULL, this method allocates it and sets default options.
   // Otherwise the method saves the options from euc_precond, destroys it,
   // allocates a new object, and sets its options to the saved values.
   void ResetEuclidPrecond(MPI_Comm comm);

public:
   HypreEuclid(MPI_Comm comm);

   HypreEuclid(HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   /// The typecast to HYPRE_Solver returns the internal euc_precond
   virtual operator HYPRE_Solver() const { return euc_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSolve; }

   virtual ~HypreEuclid();
};

/// The BoomerAMG solver in hypre
class HypreBoomerAMG : public HypreSolver
{
private:
   HYPRE_Solver amg_precond;

   /// Rigid body modes
   Array<HYPRE_ParVector> rbms;

   /// Finite element space for elasticity problems, see SetElasticityOptions()
   ParFiniteElementSpace *fespace;

   /// Recompute the rigid-body modes vectors (in the rbms array)
   void RecomputeRBMs();

   /// Default, generally robust, BoomerAMG options
   void SetDefaultOptions();

   // If amg_precond is NULL, allocates it and sets default options.
   // Otherwise saves the options from amg_precond, destroys it, allocates a new
   // one, and sets its options to the saved values.
   void ResetAMGPrecond();

public:
   HypreBoomerAMG();

   HypreBoomerAMG(HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   /** More robust options for systems, such as elasticity. Note that BoomerAMG
       assumes Ordering::byVDIM in the finite element space used to generate the
       matrix A. */
   void SetSystemsOptions(int dim);

   /** A special elasticity version of BoomerAMG that takes advantage of
       geometric rigid body modes and could perform better on some problems, see
       "Improving algebraic multigrid interpolation operators for linear
       elasticity problems", Baker, Kolev, Yang, NLAA 2009, DOI:10.1002/nla.688.
       As with SetSystemsOptions(), this solver assumes Ordering::byVDIM. */
   void SetElasticityOptions(ParFiniteElementSpace *fespace);

   /* distance parameter takes on values {1,2,15} for lAIR, meaning R is built using
   distance 1 neighbors, distance two neighbors, or distance two on processor and
   distance 1 off processor (i.e., distance 1.5 --> 15).        */
   void SetLAIROptions(int distance=15,  std::string prerelax="",
                       std::string postrelax="FFC", double strength_tol=0.1,
                       double strength_tolR=0.01, double filter_tolR=0.0,
                       int interp_type=100, int relax_type=3, double filterA_tol=0.0,
                       int splitting=6, int blksize=0, int Sabs=0);

   void SetNAIROptions(int neumann_degree=2,  std::string prerelax="A",
                       std::string postrelax="F", double strength_tol=0.1,
                       double strength_tolR=0.01, double filter_tolR=0.0,
                       int interp_type=100, int relax_type=10, double filterA_tol=0.0,
                       int splitting=6, int blksize=0, int Sabs=0);

   void SetCoord(int dim, float *coord);

   void SetPrintLevel(int print_level)
   { HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level); }

   void SetMaxIter(int max_iter)
   { HYPRE_BoomerAMGSetMaxIter(amg_precond, max_iter); }

   void SetMaxLevels(int max_levels)
   { HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels); }

   void SetTol(double tol)
   { HYPRE_BoomerAMGSetTol(amg_precond, tol); }

   void SetStrengthThresh(double strength)
   { HYPRE_BoomerAMGSetStrongThreshold(amg_precond, strength); }

   void SetStrengthThreshR(double strengthR)
   { HYPRE_BoomerAMGSetStrongThresholdR(amg_precond, strengthR); }

   void SetFilterThreshR(double filterR)
   { HYPRE_BoomerAMGSetFilterThresholdR(amg_precond, filterR); }

   void SetInterpolation(int interp_type)
   { HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type); }

   void SetRestriction(int restrict_type)
   { HYPRE_BoomerAMGSetRestriction(amg_precond, restrict_type); }

   void SetCoarsening(int coarsen_type)
   { HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type); }

   void SetRelaxType(int relax_type)
   { HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type); }

   void SetRelaxCycle(int prerelax, int postrelax)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, prerelax,  1);
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, postrelax, 2);
   }

   void GetNumIterations(int &num_it)
   { HYPRE_BoomerAMGGetNumIterations(amg_precond, &num_it); }

   void SetCycleType(int cycle_type)
   { HYPRE_BoomerAMGSetCycleType(amg_precond, cycle_type); }

   void SetNodal(int blocksize)
   {
      HYPRE_BoomerAMGSetNumFunctions(amg_precond, blocksize);
      HYPRE_BoomerAMGSetNodal(amg_precond, 1);
   }

   void SetAggressiveCoarsening(int num_levels)
   { HYPRE_BoomerAMGSetAggNumLevels(amg_precond, num_levels); }

   void SetTriangular()
   { HYPRE_BoomerAMGSetIsTriangular(amg_precond, 1); }

   void SetGMRESSwitchR(int gmres_switch)
   { HYPRE_BoomerAMGSetGMRESSwitchR(amg_precond, gmres_switch); }

   /// The typecast to HYPRE_Solver returns the internal amg_precond
   virtual operator HYPRE_Solver() const { return amg_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; }

   virtual void Mult (const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HypreBoomerAMG();
};

/// Compute the discrete gradient matrix between the nodal linear and ND1 spaces
HypreParMatrix* DiscreteGrad(ParFiniteElementSpace *edge_fespace,
                             ParFiniteElementSpace *vert_fespace);
/// Compute the discrete curl matrix between the ND1 and RT0 spaces
HypreParMatrix* DiscreteCurl(ParFiniteElementSpace *face_fespace,
                             ParFiniteElementSpace *edge_fespace);

/// The Auxiliary-space Maxwell Solver in hypre
class HypreAMS : public HypreSolver
{
private:
   /// Constuct AMS solver from finite element space
   void Init(ParFiniteElementSpace *edge_space);

   HYPRE_Solver ams;

   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *Pi, *Pix, *Piy, *Piz;

public:
   HypreAMS(ParFiniteElementSpace *edge_fespace);

   HypreAMS(HypreParMatrix &A, ParFiniteElementSpace *edge_fespace);

   virtual void SetOperator(const Operator &op);

   void SetPrintLevel(int print_lvl);

   /// Set this option when solving a curl-curl problem with zero mass term
   void SetSingularProblem() { HYPRE_AMSSetBetaPoissonMatrix(ams, NULL); }

   /// The typecast to HYPRE_Solver returns the internal ams object
   virtual operator HYPRE_Solver() const { return ams; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSolve; }

   virtual ~HypreAMS();
};

/// The Auxiliary-space Divergence Solver in hypre
class HypreADS : public HypreSolver
{
private:
   /// Constuct ADS solver from finite element space
   void Init(ParFiniteElementSpace *face_fespace);

   HYPRE_Solver ads;

   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Discrete curl matrix
   HypreParMatrix *C;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;
   /// Raviart-Thomas interpolation matrix and its components
   HypreParMatrix *RT_Pi, *RT_Pix, *RT_Piy, *RT_Piz;

public:
   HypreADS(ParFiniteElementSpace *face_fespace);

   HypreADS(HypreParMatrix &A, ParFiniteElementSpace *face_fespace);

   virtual void SetOperator(const Operator &op);

   void SetPrintLevel(int print_lvl);

   /// The typecast to HYPRE_Solver returns the internal ads object
   virtual operator HYPRE_Solver() const { return ads; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSolve; }

   virtual ~HypreADS();
};

/** LOBPCG eigenvalue solver in hypre

    The Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG)
    eigenvalue solver is designed to find the lowest eigenmodes of the
    generalized eigenvalue problem:
       A x = lambda M x
    where A is symmetric, potentially indefinite and M is symmetric positive
    definite. The eigenvectors are M-orthonormal, meaning that
       x^T M x = 1 and x^T M y = 0,
    if x and y are distinct eigenvectors. The matrix M is optional and is
    assumed to be the identity if left unset.

    The efficiency of LOBPCG relies on the availability of a suitable
    preconditioner for the matrix A. The preconditioner is supplied through the
    SetPreconditioner() method. It should be noted that the operator used with
    the preconditioner need not be A itself.

    For more information regarding LOBPCG see "Block Locally Optimal
    Preconditioned Eigenvalue Xolvers (BLOPEX) in Hypre and PETSc" by
    A. Knyazev, M. Argentati, I. Lashuk, and E. Ovtchinnikov, SISC, 29(5),
    2224-2239, 2007.
*/
class HypreLOBPCG
{
private:
   MPI_Comm comm;
   int myid;
   int numProcs;
   int nev;   // Number of desired eigenmodes
   int seed;  // Random seed used for initial vectors

   HYPRE_Int glbSize; // Global number of DoFs in the linear system
   HYPRE_Int * part;  // Row partitioning of the linear system

   // Pointer to HYPRE's solver struct
   HYPRE_Solver lobpcg_solver;

   // Interface for matrix storage type
   mv_InterfaceInterpreter interpreter;

   // Interface for setting up and performing matrix-vector products
   HYPRE_MatvecFunctions matvec_fn;

   // Eigenvalues
   Array<double> eigenvalues;

   // Forward declaration
   class HypreMultiVector;

   // MultiVector to store eigenvectors
   HypreMultiVector * multi_vec;

   // Empty vectors used to setup the matrices and preconditioner
   HypreParVector * x;

   // An optional operator which projects vectors into a desired subspace
   Operator * subSpaceProj;

   /// Internal class to represent a set of eigenvectors
   class HypreMultiVector
   {
   private:
      // Pointer to hypre's multi-vector object
      mv_MultiVectorPtr mv_ptr;

      // Wrappers for each member of the multivector
      HypreParVector ** hpv;

      // Number of vectors in the multivector
      int nv;

   public:
      HypreMultiVector(int n, HypreParVector & v,
                       mv_InterfaceInterpreter & interpreter);
      ~HypreMultiVector();

      /// Set random values
      void Randomize(HYPRE_Int seed);

      /// Extract a single HypreParVector object
      HypreParVector & GetVector(unsigned int i);

      /// Transfers ownership of data to returned array of vectors
      HypreParVector ** StealVectors();

      operator mv_MultiVectorPtr() const { return mv_ptr; }

      mv_MultiVectorPtr & GetMultiVector() { return mv_ptr; }
   };

   static void    * OperatorMatvecCreate( void *A, void *x );
   static HYPRE_Int OperatorMatvec( void *matvec_data,
                                    HYPRE_Complex alpha,
                                    void *A,
                                    void *x,
                                    HYPRE_Complex beta,
                                    void *y );
   static HYPRE_Int OperatorMatvecDestroy( void *matvec_data );

   static HYPRE_Int PrecondSolve(void *solver,
                                 void *A,
                                 void *b,
                                 void *x);
   static HYPRE_Int PrecondSetup(void *solver,
                                 void *A,
                                 void *b,
                                 void *x);

public:
   HypreLOBPCG(MPI_Comm comm);
   ~HypreLOBPCG();

   void SetTol(double tol);
   void SetRelTol(double rel_tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int logging);
   void SetNumModes(int num_eigs) { nev = num_eigs; }
   void SetPrecondUsageMode(int pcg_mode);
   void SetRandomSeed(int s) { seed = s; }
   void SetInitialVectors(int num_vecs, HypreParVector ** vecs);

   // The following four methods support general operators
   void SetPreconditioner(Solver & precond);
   void SetOperator(Operator & A);
   void SetMassMatrix(Operator & M);
   void SetSubSpaceProjector(Operator & proj) { subSpaceProj = &proj; }

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(Array<double> & eigenvalues);

   /// Extract a single eigenvector
   HypreParVector & GetEigenvector(unsigned int i);

   /// Transfer ownership of the converged eigenvectors
   HypreParVector ** StealEigenvectors() { return multi_vec->StealVectors(); }
};

/** AME eigenvalue solver in hypre

    The Auxiliary space Maxwell Eigensolver (AME) is designed to find
    the lowest eigenmodes of the generalized eigenvalue problem:
       Curl Curl x = lambda M x
    where the Curl Curl operator is discretized using Nedelec finite element
    basis functions. Properties of this discretization are essential to
    eliminating the large null space of the Curl Curl operator.

    This eigensolver relies upon the LOBPCG eigensolver internally. It is also
    expected that the preconditioner supplied to this method will be the
    HypreAMS preconditioner defined above.

    As with LOBPCG, the operator set in the preconditioner need not be the same
    as A. This flexibility may be useful in solving eigenproblems which bare a
    strong resemblance to the Curl Curl problems for which AME is designed.

    Unlike LOBPCG, this eigensolver requires that the mass matrix be set.
    It is possible to circumvent this by passing an identity operator as the
    mass matrix but it seems unlikely that this would be useful so it is not the
    default behavior.
*/
class HypreAME
{
private:
   int myid;
   int numProcs;
   int nev;   // Number of desired eigenmodes
   bool setT;

   // Pointer to HYPRE's AME solver struct
   HYPRE_Solver ame_solver;

   // Pointer to HYPRE's AMS solver struct
   HypreSolver * ams_precond;

   // Eigenvalues
   HYPRE_Real * eigenvalues;

   // MultiVector to store eigenvectors
   HYPRE_ParVector * multi_vec;

   HypreParVector ** eigenvectors;

   void createDummyVectors();

public:
   HypreAME(MPI_Comm comm);
   ~HypreAME();

   void SetTol(double tol);
   void SetRelTol(double rel_tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int logging);
   void SetNumModes(int num_eigs);

   // The following four methods support operators of type HypreParMatrix.
   void SetPreconditioner(HypreSolver & precond);
   void SetOperator(HypreParMatrix & A);
   void SetMassMatrix(HypreParMatrix & M);

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(Array<double> & eigenvalues);

   /// Extract a single eigenvector
   HypreParVector & GetEigenvector(unsigned int i);

   /// Transfer ownership of the converged eigenvectors
   HypreParVector ** StealEigenvectors();
};

}

#endif // MFEM_USE_MPI

#endif
