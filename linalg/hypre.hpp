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

#if defined(HYPRE_USING_GPU) && \
    !(defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
#error "Unsupported GPU build of HYPRE! Only CUDA and HIP builds are supported."
#endif
#if defined(HYPRE_USING_CUDA) && !defined(MFEM_USE_CUDA)
#error "MFEM_USE_CUDA=YES is required when HYPRE is built with CUDA!"
#endif
#if defined(HYPRE_USING_HIP) && !defined(MFEM_USE_HIP)
#error "MFEM_USE_HIP=YES is required when HYPRE is built with HIP!"
#endif

// MFEM_HYPRE_FORALL is a macro similar to mfem::forall, but it executes on the
// device that hypre was configured with (no matter what device was selected
// in MFEM's runtime configuration).
#if defined(HYPRE_USING_CUDA)
#define MFEM_HYPRE_FORALL(i, N,...) CuWrap1D(N, [=] MFEM_DEVICE      \
                                       (int i) {__VA_ARGS__})
#elif defined(HYPRE_USING_HIP)
#define MFEM_HYPRE_FORALL(i, N,...) HipWrap1D(N, [=] MFEM_DEVICE     \
                                        (int i) {__VA_ARGS__})
#else
#define MFEM_HYPRE_FORALL(i, N,...) for (int i = 0; i < N; i++) { __VA_ARGS__ }
#endif

#include "sparsemat.hpp"
#include "hypre_parcsr.hpp"

namespace mfem
{

class ParFiniteElementSpace;
class HypreParMatrix;


/// @brief A simple singleton class for hypre's global settings, that 1) calls
/// HYPRE_Init() and sets some GPU-relevant options at construction and 2) calls
/// HYPRE_Finalize() at destruction.
class Hypre
{
public:
   /// @brief Initialize hypre by calling HYPRE_Init() and set default options.
   /// After calling Hypre::Init(), hypre will be finalized automatically at
   /// program exit.
   ///
   /// Calling HYPRE_Finalize() manually is not compatible with this class.
   static void Init() { Instance(); }

   /// @brief Finalize hypre (called automatically at program exit if
   /// Hypre::Init() has been called).
   ///
   /// Multiple calls to Hypre::Finalize() have no effect. This function can be
   /// called manually to more precisely control when hypre is finalized.
   static void Finalize();

private:
   /// Calls HYPRE_Init() when the singleton is constructed.
   Hypre();

   /// The singleton destructor (called at program exit) finalizes hypre.
   ~Hypre() { Finalize(); }

   /// Set the default hypre global options (mostly GPU-relevant).
   void SetDefaultOptions();

   /// Create and return the Hypre singleton object.
   static Hypre &Instance()
   {
      static Hypre hypre;
      return hypre;
   }

   bool finalized = false; ///< Has Hypre::Finalize() been called already?
};


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

} // namespace internal


/// The MemoryClass used by Hypre objects.
inline constexpr MemoryClass GetHypreMemoryClass()
{
#if !defined(HYPRE_USING_GPU)
   return MemoryClass::HOST;
#elif defined(HYPRE_USING_UNIFIED_MEMORY)
   return MemoryClass::MANAGED;
#else
   return MemoryClass::DEVICE;
#endif
}

/// The MemoryType used by MFEM when allocating arrays for Hypre objects.
inline MemoryType GetHypreMemoryType()
{
#if !defined(HYPRE_USING_GPU)
   return Device::GetHostMemoryType();
#elif defined(HYPRE_USING_UNIFIED_MEMORY)
   return MemoryType::MANAGED;
#else
   return MemoryType::DEVICE;
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

   /// Default constructor, no underlying @a hypre_ParVector is created.
   HypreParVector()
   {
      own_ParVector = false;
      x = NULL;
   }

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
   HypreParVector(MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt *col);
   /** @brief Creates vector with given global size, partitioning of the
       columns, and data. */
   /** The data must be allocated and destroyed outside. If @a data_ is NULL, a
       dummy vector without a valid data array will be created. See @ref
       hypre_partitioning_descr "here" for a description of the @a col array.

       If @a is_device_ptr is true, the pointer @a data_ is assumed to be
       allocated in the memory location HYPRE_MEMORY_DEVICE. */
   HypreParVector(MPI_Comm comm, HYPRE_BigInt glob_size, double *data_,
                  HYPRE_BigInt *col, bool is_device_ptr = false);
   /// Creates a deep copy of @a y
   HypreParVector(const HypreParVector &y);
   /// Move constructor for HypreParVector. "Steals" data from its argument.
   HypreParVector(HypreParVector&& other);
   /// Creates vector compatible with (i.e. in the domain of) A or A^T
   explicit HypreParVector(const HypreParMatrix &A, int transpose = 0);
   /// Creates vector wrapping y
   explicit HypreParVector(HYPRE_ParVector y);
   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   explicit HypreParVector(ParFiniteElementSpace *pfes);

   /// \brief Constructs a  @p HypreParVector *compatible* with the calling vector
   /// - meaning that it will be the same size and have the same partitioning.
   HypreParVector CreateCompatibleVector() const;

   /// MPI communicator
   MPI_Comm GetComm() const { return x->comm; }

   /// Converts hypre's format to HypreParVector
   void WrapHypreParVector(hypre_ParVector *y, bool owner=true);

   /// Returns the parallel row/column partitioning
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline const HYPRE_BigInt *Partitioning() const { return x->partitioning; }

   /// @brief Returns a non-const pointer to the parallel row/column
   /// partitioning.
   /// Deprecated in favor of HypreParVector::Partitioning() const.
   MFEM_DEPRECATED
   inline HYPRE_BigInt *Partitioning() { return x->partitioning; }

   /// Returns the global number of rows
   inline HYPRE_BigInt GlobalSize() const { return x->global_size; }

   /// Typecasting to hypre's hypre_ParVector*
   operator hypre_ParVector*() const { return x; }
#ifndef HYPRE_PAR_VECTOR_STRUCT
   /// Typecasting to hypre's HYPRE_ParVector, a.k.a. void *
   operator HYPRE_ParVector() const { return (HYPRE_ParVector) x; }
#endif
   /// Changes the ownership of the vector
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
   /// Move assignment
   HypreParVector& operator= (HypreParVector &&y);

   using Vector::Read;

   /// Sets the data of the Vector and the hypre_ParVector to @a data_.
   /** Must be used only for HypreParVector%s that do not own the data,
       e.g. created with the constructor:
       HypreParVector(MPI_Comm, HYPRE_BigInt, double *, HYPRE_BigInt *). */
   void SetData(double *data_);

   /** @brief Prepare the HypreParVector for read access in hypre's device
       memory space, HYPRE_MEMORY_DEVICE. */
   void HypreRead() const;

   /** @brief Prepare the HypreParVector for read and write access in hypre's
       device memory space, HYPRE_MEMORY_DEVICE. */
   void HypreReadWrite();

   /** @brief Prepare the HypreParVector for write access in hypre's device
       memory space, HYPRE_MEMORY_DEVICE. */
   void HypreWrite();

   /** @brief Replace the HypreParVector's data with the given Memory, @a mem,
       and prepare the vector for read access in hypre's device memory space,
       HYPRE_MEMORY_DEVICE. */
   /** This method must be used with HypreParVector%s that do not own the data,
       e.g. created with the constructor:
       HypreParVector(MPI_Comm, HYPRE_BigInt, double *, HYPRE_BigInt *).

       The Memory @a mem must be accessible with the hypre MemoryClass defined
       by GetHypreMemoryClass(). */
   void WrapMemoryRead(const Memory<double> &mem);

   /** @brief Replace the HypreParVector's data with the given Memory, @a mem,
       and prepare the vector for read and write access in hypre's device memory
       space, HYPRE_MEMORY_DEVICE. */
   /** This method must be used with HypreParVector%s that do not own the data,
       e.g. created with the constructor:
       HypreParVector(MPI_Comm, HYPRE_BigInt, double *, HYPRE_BigInt *).

       The Memory @a mem must be accessible with the hypre MemoryClass defined
       by GetHypreMemoryClass(). */
   void WrapMemoryReadWrite(Memory<double> &mem);

   /** @brief Replace the HypreParVector's data with the given Memory, @a mem,
       and prepare the vector for write access in hypre's device memory space,
       HYPRE_MEMORY_DEVICE. */
   /** This method must be used with HypreParVector%s that do not own the data,
       e.g. created with the constructor:
       HypreParVector(MPI_Comm, HYPRE_BigInt, double *, HYPRE_BigInt *).

       The Memory @a mem must be accessible with the hypre MemoryClass defined
       by GetHypreMemoryClass(). */
   void WrapMemoryWrite(Memory<double> &mem);

   /// Set random values
   HYPRE_Int Randomize(HYPRE_Int seed);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname) const;

   /// Reads a HypreParVector from files saved with HypreParVector::Print
   void Read(MPI_Comm comm, const char *fname);

   /// Calls hypre's destroy function
   ~HypreParVector();
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
   /** @brief Auxiliary buffers for the case when the input or output arrays in
       methods like Mult(double, const Vector &, double, Vector &) need to be
       deep copied in order to be used by hypre. */
   mutable Memory<double> auxX, auxY;

   // Flags indicating ownership of A->diag->{i,j,data}, A->offd->{i,j,data},
   // and A->col_map_offd.
   // The possible values for diagOwner are:
   //  -1: no special treatment of A->diag (default)
   //      when hypre is built with CUDA support, A->diag owns the "host"
   //      pointers (according to A->diag->owns_data)
   //  -2: used when hypre is built with CUDA support, A->diag owns the "hypre"
   //      pointers (according to A->diag->owns_data)
   //   0: prevent hypre from destroying A->diag->{i,j,data}
   //   1: same as 0, plus own the "host" A->diag->{i,j}
   //   2: same as 0, plus own the "host" A->diag->data
   //   3: same as 0, plus own the "host" A->diag->{i,j,data}
   // The same values and rules apply to offdOwner and A->offd.
   // The possible values for colMapOwner are:
   //  -1: no special treatment of A->col_map_offd (default)
   //   0: prevent hypre from destroying A->col_map_offd
   //   1: same as 0, plus take ownership of A->col_map_offd
   // All owned arrays are destroyed with 'delete []'.
   signed char diagOwner, offdOwner, colMapOwner;

   // Does the object own the pointer A?
   signed char ParCSROwner;

   MemoryIJData mem_diag, mem_offd;

   // Initialize with defaults. Does not initialize inherited members.
   void Init();

   // Delete all owned data. Does not perform re-initialization with defaults.
   void Destroy();

   void Read(MemoryClass mc) const;
   void ReadWrite(MemoryClass mc);
   // The Boolean flags are used in Destroy().
   void Write(MemoryClass mc, bool set_diag = true, bool set_offd = true);

   // Copy (shallow/deep, based on HYPRE_BIGINT) the I and J arrays from csr to
   // hypre_csr. Shallow copy the data. Return the appropriate ownership flag.
   // The CSR arrays are wrapped in the mem_csr struct which is used to move
   // these arrays to device, if necessary.
   static signed char CopyCSR(SparseMatrix *csr,
                              MemoryIJData &mem_csr,
                              hypre_CSRMatrix *hypre_csr,
                              bool mem_owner);
   // Copy (shallow or deep, based on HYPRE_BIGINT) the I and J arrays from
   // bool_csr to hypre_csr. Allocate the data array and set it to all ones.
   // Return the appropriate ownership flag. The CSR arrays are wrapped in the
   // mem_csr struct which is used to move these arrays to device, if necessary.
   static signed char CopyBoolCSR(Table *bool_csr,
                                  MemoryIJData &mem_csr,
                                  hypre_CSRMatrix *hypre_csr);

   // Wrap the data from h_mat into mem with the given ownership flag.
   // If the new Memory arrays in mem are not suitable to be accessed via
   // GetHypreMemoryClass(), then mem will be re-allocated using the memory type
   // returned by GetHypreMemoryType(), the data will be deep copied, and h_mat
   // will be updated with the new pointers.
   static signed char HypreCsrToMem(hypre_CSRMatrix *h_mat, MemoryType h_mat_mt,
                                    bool own_ija, MemoryIJData &mem);

public:
   /// An empty matrix to be used as a reference to an existing matrix
   HypreParMatrix();

   /// Converts hypre's format to HypreParMatrix
   /** If @a owner is false, ownership of @a a is not transferred */
   void WrapHypreParCSRMatrix(hypre_ParCSRMatrix *a, bool owner = true);

   /// Converts hypre's format to HypreParMatrix
   /** If @a owner is false, ownership of @a a is not transferred */
   explicit HypreParMatrix(hypre_ParCSRMatrix *a, bool owner = true)
   {
      Init();
      WrapHypreParCSRMatrix(a, owner);
   }

   /// Creates block-diagonal square parallel matrix.
   /** Diagonal is given by @a diag which must be in CSR format (finalized). The
       new HypreParMatrix does not take ownership of any of the input arrays.
       See @ref hypre_partitioning_descr "here" for a description of the row
       partitioning array @a row_starts.

       @warning The ordering of the columns in each row in @a *diag may be
       changed by this constructor to ensure that the first entry in each row is
       the diagonal one. This is expected by most hypre functions. */
   HypreParMatrix(MPI_Comm comm, HYPRE_BigInt glob_size,
                  HYPRE_BigInt *row_starts,
                  SparseMatrix *diag); // constructor with 4 arguments, v1

   /// Creates block-diagonal rectangular parallel matrix.
   /** Diagonal is given by @a diag which must be in CSR format (finalized). The
       new HypreParMatrix does not take ownership of any of the input arrays.
       See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_BigInt global_num_rows,
                  HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                  HYPRE_BigInt *col_starts,
                  SparseMatrix *diag); // constructor with 6 arguments, v1

   /// Creates general (rectangular) parallel matrix.
   /** The new HypreParMatrix does not take ownership of any of the input
       arrays, if @a own_diag_offd is false (default). If @a own_diag_offd is
       true, ownership of @a diag and @a offd is transferred to the
       HypreParMatrix.

       See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_BigInt global_num_rows,
                  HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                  HYPRE_BigInt *col_starts, SparseMatrix *diag,
                  SparseMatrix *offd, HYPRE_BigInt *cmap,
                  bool own_diag_offd = false); // constructor with 8+1 arguments

   /// Creates general (rectangular) parallel matrix.
   /** The new HypreParMatrix takes ownership of all input arrays, except
       @a col_starts and @a row_starts. See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a row_starts and @a
       col_starts.

       If @a hypre_arrays is false, all arrays (except @a row_starts and
       @a col_starts) are assumed to be allocated according to the MemoryType
       returned by Device::GetHostMemoryType(). If @a hypre_arrays is true, then
       the same arrays are assumed to be allocated by hypre as host arrays. */
   HypreParMatrix(MPI_Comm comm,
                  HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols,
                  HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                  HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
                  HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
                  HYPRE_Int offd_num_cols,
                  HYPRE_BigInt *offd_col_map,
                  bool hypre_arrays = false); // constructor with 13+1 arguments

   /// Creates a parallel matrix from SparseMatrix on processor 0.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_BigInt *row_starts,
                  HYPRE_BigInt *col_starts,
                  SparseMatrix *a); // constructor with 4 arguments, v2

   /// Creates boolean block-diagonal rectangular parallel matrix.
   /** The new HypreParMatrix does not take ownership of any of the input
       arrays. See @ref hypre_partitioning_descr "here" for a description of the
       partitioning arrays @a row_starts and @a col_starts. */
   HypreParMatrix(MPI_Comm comm, HYPRE_BigInt global_num_rows,
                  HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                  HYPRE_BigInt *col_starts,
                  Table *diag); // constructor with 6 arguments, v2

   /// Creates boolean rectangular parallel matrix.
   /** The new HypreParMatrix takes ownership of the arrays @a i_diag,
       @a j_diag, @a i_offd, @a j_offd, and @a cmap; does not take ownership of
       the arrays @a row and @a col. See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a row and @a col. */
   HypreParMatrix(MPI_Comm comm, int id, int np, HYPRE_BigInt *row,
                  HYPRE_BigInt *col,
                  HYPRE_Int *i_diag, HYPRE_Int *j_diag, HYPRE_Int *i_offd,
                  HYPRE_Int *j_offd, HYPRE_BigInt *cmap,
                  HYPRE_Int cmap_size); // constructor with 11 arguments

   /** @brief Creates a general parallel matrix from a local CSR matrix on each
       processor described by the @a I, @a J and @a data arrays. */
   /** The local matrix should be of size (local) @a nrows by (global)
       @a glob_ncols. The new parallel matrix contains copies of all input
       arrays (so they can be deleted). See @ref hypre_partitioning_descr "here"
       for a description of the partitioning arrays @a rows and @a cols. */
   HypreParMatrix(MPI_Comm comm, int nrows, HYPRE_BigInt glob_nrows,
                  HYPRE_BigInt glob_ncols, int *I, HYPRE_BigInt *J,
                  double *data, HYPRE_BigInt *rows,
                  HYPRE_BigInt *cols); // constructor with 9 arguments

   /** @brief Copy constructor for a ParCSR matrix which creates a deep copy of
       structure and data from @a P. */
   HypreParMatrix(const HypreParMatrix &P);

   /// Make this HypreParMatrix a reference to 'master'
   void MakeRef(const HypreParMatrix &master);

   /// MPI communicator
   MPI_Comm GetComm() const { return A->comm; }

   /// Typecasting to hypre's hypre_ParCSRMatrix*
   operator hypre_ParCSRMatrix*() const { return A; }
#ifndef HYPRE_PAR_CSR_MATRIX_STRUCT
   /// Typecasting to hypre's HYPRE_ParCSRMatrix, a.k.a. void *
   operator HYPRE_ParCSRMatrix() { return (HYPRE_ParCSRMatrix) A; }
#endif
   /// Changes the ownership of the matrix
   hypre_ParCSRMatrix* StealData();

   /// Explicitly set the three ownership flags, see docs for diagOwner etc.
   void SetOwnerFlags(signed char diag, signed char offd, signed char colmap);

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
   inline HYPRE_BigInt NNZ() const { return A->num_nonzeros; }
   /// Returns the row partitioning
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline HYPRE_BigInt *RowPart() { return A->row_starts; }
   /// Returns the column partitioning
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline HYPRE_BigInt *ColPart() { return A->col_starts; }
   /// Returns the row partitioning (const version)
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline const HYPRE_BigInt *RowPart() const { return A->row_starts; }
   /// Returns the column partitioning (const version)
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   inline const HYPRE_BigInt *ColPart() const { return A->col_starts; }
   /// Returns the global number of rows
   inline HYPRE_BigInt M() const { return A->global_num_rows; }
   /// Returns the global number of columns
   inline HYPRE_BigInt N() const { return A->global_num_cols; }

   /// Get the local diagonal of the matrix.
   void GetDiag(Vector &diag) const;
   /// Get the local diagonal block. NOTE: 'diag' will not own any data.
   void GetDiag(SparseMatrix &diag) const;
   /// Get the local off-diagonal block. NOTE: 'offd' will not own any data.
   void GetOffd(SparseMatrix &offd, HYPRE_BigInt* &cmap) const;
   /** @brief Get a single SparseMatrix containing all rows from this processor,
       merged from the diagonal and off-diagonal blocks stored by the
       HypreParMatrix. */
   /** @note The number of columns in the SparseMatrix will be the global number
       of columns in the parallel matrix, so using this method may result in an
       integer overflow in the column indices. */
   void MergeDiagAndOffd(SparseMatrix &merged);

   /// Return the diagonal of the matrix (Operator interface).
   void AssembleDiagonal(Vector &diag) const override { GetDiag(diag); }

   /** Split the matrix into M x N equally sized blocks of parallel matrices.
       The size of 'blocks' must already be set to M x N. */
   void GetBlocks(Array2D<HypreParMatrix*> &blocks,
                  bool interleaved_rows = false,
                  bool interleaved_cols = false) const;

   /// Returns the transpose of *this
   HypreParMatrix * Transpose() const;

   /** Returns principle submatrix given by array of indices of connections
       with relative size > @a threshold in *this. */
#if MFEM_HYPRE_VERSION >= 21800
   HypreParMatrix *ExtractSubmatrix(const Array<int> &indices,
                                    double threshold=0.0) const;
#endif

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
   HYPRE_BigInt GetGlobalNumRows() const
   { return hypre_ParCSRMatrixGlobalNumRows(A); }

   /// Return the global number of columns
   HYPRE_BigInt GetGlobalNumCols() const
   { return hypre_ParCSRMatrixGlobalNumCols(A); }

   /// Return the parallel row partitioning array.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   HYPRE_BigInt *GetRowStarts() const { return hypre_ParCSRMatrixRowStarts(A); }

   /// Return the parallel column partitioning array.
   /** See @ref hypre_partitioning_descr "here" for a description of the
       partitioning array. */
   HYPRE_BigInt *GetColStarts() const { return hypre_ParCSRMatrixColStarts(A); }

   MemoryClass GetMemoryClass() const override { return GetHypreMemoryClass(); }

   /// Ensure the action of the transpose is performed fast.
   /** When HYPRE is built for GPUs, this method will construct and store the
       transposes of the 'diag' and 'offd' CSR matrices. When HYPRE is not built
       for GPUs, this method is a no-op.

       This method is automatically called by MultTranspose().

       If the matrix is modified the old transpose blocks can be deleted by
       calling ResetTranspose(). */
   void EnsureMultTranspose() const;

   /** @brief Reset (destroy) the internal transpose matrix that is created by
       EnsureMultTranspose() and MultTranspose().

       If the matrix is modified, this method should be called to delete the
       out-of-date transpose that is stored internally. */
   void ResetTranspose() const;

   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HypreParVector &x, HypreParVector &y,
                  double alpha = 1.0, double beta = 0.0) const;
   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                  double alpha = 1.0, double beta = 0.0) const;

   /// Computes y = alpha * A^t * x + beta * y
   /** If the matrix is modified, call ResetTranspose() and optionally
       EnsureMultTranspose() to make sure this method uses the correct updated
       transpose. */
   HYPRE_Int MultTranspose(HypreParVector &x, HypreParVector &y,
                           double alpha = 1.0, double beta = 0.0) const;

   void Mult(double a, const Vector &x, double b, Vector &y) const;

   /// Computes y = alpha * A^t * x + beta * y
   /** If the matrix is modified, call ResetTranspose() and optionally
       EnsureMultTranspose() to make sure this method uses the correct updated
       transpose. */
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   void Mult(const Vector &x, Vector &y) const override
   { Mult(1.0, x, 0.0, y); }

   /// Computes y = A^t * x
   /** If the matrix is modified, call ResetTranspose() and optionally
       EnsureMultTranspose() to make sure this method uses the correct updated
       transpose. */
   void MultTranspose(const Vector &x, Vector &y) const override
   { MultTranspose(1.0, x, 0.0, y); }

   void AddMult(const Vector &x, Vector &y, const double a = 1.0) const override
   { Mult(a, x, 1.0, y); }
   void AddMultTranspose(const Vector &x, Vector &y,
                         const double a = 1.0) const override
   { MultTranspose(a, x, 1.0, y); }

   using Operator::Mult;
   using Operator::MultTranspose;

   /** @brief Computes y = a * |A| * x + b * y, using entry-wise absolute values
       of the matrix A. */
   void AbsMult(double a, const Vector &x, double b, Vector &y) const;

   /** @brief Computes y = a * |At| * x + b * y, using entry-wise absolute
       values of the transpose of the matrix A. */
   void AbsMultTranspose(double a, const Vector &x, double b, Vector &y) const;

   /** @brief The "Boolean" analog of y = alpha * A * x + beta * y, where
       elements in the sparsity pattern of the matrix are treated as "true". */
   void BooleanMult(int alpha, const int *x, int beta, int *y)
   {
      HostRead();
      internal::hypre_ParCSRMatrixBooleanMatvec(A, alpha, const_cast<int*>(x),
                                                beta, y);
      HypreRead();
   }

   /** @brief The "Boolean" analog of y = alpha * A^T * x + beta * y, where
       elements in the sparsity pattern of the matrix are treated as "true". */
   void BooleanMultTranspose(int alpha, const int *x, int beta, int *y)
   {
      HostRead();
      internal::hypre_ParCSRMatrixBooleanMatvecT(A, alpha, const_cast<int*>(x),
                                                 beta, y);
      HypreRead();
   }

   /// Initialize all entries with value.
   HypreParMatrix &operator=(double value)
   {
#if MFEM_HYPRE_VERSION < 22200
      internal::hypre_ParCSRMatrixSetConstantValues(A, value);
#else
      hypre_ParCSRMatrixSetConstantValues(A, value);
#endif
      return *this;
   }

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
                                HYPRE_BigInt* row_starts = NULL) const;

   /// Scale the local row i by s(i).
   void ScaleRows(const Vector & s);
   /// Scale the local row i by 1./s(i)
   void InvScaleRows(const Vector & s);
   /// Scale all entries by s: A_scaled = s*A.
   void operator*=(double s);

   /// Remove values smaller in absolute value than some threshold
   void Threshold(double threshold = 0.0);

   /** @brief Wrapper for hypre_ParCSRMatrixDropSmallEntries in different
       versions of hypre. Drop off-diagonal entries that are smaller than
       tol * l2 norm of its row */
   /** For HYPRE versions < 2.14, this method just calls Threshold() with
       threshold = tol * max(l2 row norm). */
   void DropSmallEntries(double tol);

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

   /** @brief Eliminate essential BC specified by @a ess_dof_list from the
       solution @a X to the r.h.s. @a B. */
   /** This matrix is the matrix with eliminated BC, while @a Ae is such that
       (A+Ae) is the original (Neumann) matrix before elimination. */
   void EliminateBC(const HypreParMatrix &Ae, const Array<int> &ess_dof_list,
                    const Vector &X, Vector &B) const;

   /** @brief Eliminate essential (Dirichlet) boundary conditions.

       @param[in] ess_dofs indices of the degrees of freedom belonging to the
                           essential boundary conditions.
       @param[in] diag_policy policy for diagonal entries. */
   void EliminateBC(const Array<int> &ess_dofs,
                    DiagonalPolicy diag_policy);

   /// Update the internal hypre_ParCSRMatrix object, A, to be on host.
   /** After this call A's diagonal and off-diagonal should not be modified
       until after a suitable call to {Host,Hypre}{Write,ReadWrite}. */
   void HostRead() const { Read(Device::GetHostMemoryClass()); }

   /// Update the internal hypre_ParCSRMatrix object, A, to be on host.
   /** After this call A's diagonal and off-diagonal can be modified on host
       and subsequent calls to Hypre{Read,Write,ReadWrite} will require a deep
       copy of the data if hypre is built with device support. */
   void HostReadWrite() { ReadWrite(Device::GetHostMemoryClass()); }

   /// Update the internal hypre_ParCSRMatrix object, A, to be on host.
   /** Similar to HostReadWrite(), except that the data will never be copied
       from device to host to ensure host contains the correct current data. */
   void HostWrite() { Write(Device::GetHostMemoryClass()); }

   /** @brief Update the internal hypre_ParCSRMatrix object, A, to be in hypre
       memory space. */
   /** After this call A's diagonal and off-diagonal should not be modified
       until after a suitable call to {Host,Hypre}{Write,ReadWrite}. */
   void HypreRead() const { Read(GetHypreMemoryClass()); }

   /** @brief Update the internal hypre_ParCSRMatrix object, A, to be in hypre
       memory space. */
   /** After this call A's diagonal and off-diagonal can be modified in hypre
       memory space and subsequent calls to Host{Read,Write,ReadWrite} will
       require a deep copy of the data if hypre is built with device support. */
   void HypreReadWrite() { ReadWrite(GetHypreMemoryClass()); }

   /** @brief Update the internal hypre_ParCSRMatrix object, A, to be in hypre
       memory space. */
   /** Similar to HostReadWrite(), except that the data will never be copied
       from host to hypre memory space to ensure the latter contains the correct
       current data. */
   void HypreWrite() { Write(GetHypreMemoryClass()); }

   Memory<HYPRE_Int> &GetDiagMemoryI() { return mem_diag.I; }
   Memory<HYPRE_Int> &GetDiagMemoryJ() { return mem_diag.J; }
   Memory<double> &GetDiagMemoryData() { return mem_diag.data; }

   const Memory<HYPRE_Int> &GetDiagMemoryI() const { return mem_diag.I; }
   const Memory<HYPRE_Int> &GetDiagMemoryJ() const { return mem_diag.J; }
   const Memory<double> &GetDiagMemoryData() const { return mem_diag.data; }

   /// Prints the locally owned rows in parallel
   void Print(const char *fname, HYPRE_Int offi = 0, HYPRE_Int offj = 0) const;
   /// Reads the matrix from a file
   void Read(MPI_Comm comm, const char *fname);
   /// Read a matrix saved as a HYPRE_IJMatrix
   void Read_IJMatrix(MPI_Comm comm, const char *fname);

   /// Print information about the hypre_ParCSRCommPkg of the HypreParMatrix.
   void PrintCommPkg(std::ostream &out = mfem::out) const;

   /** @brief Print sizes and hashes for all data arrays of the HypreParMatrix
       from the local MPI rank. */
   /** This is a compact text representation of the local data of the
       HypreParMatrix that can be used to compare matrices from different runs
       without the need to save the whole matrix. */
   void PrintHash(std::ostream &out) const;

   /// Calls hypre's destroy function
   virtual ~HypreParMatrix() { Destroy(); }

   Type GetType() const { return Hypre_ParCSR; }
};

/// @brief Make @a A_hyp steal ownership of its diagonal part @a A_diag.
///
/// If @a A_hyp does not own I and J, then they are aliases pointing to the I
/// and J arrays in @a A_diag. In that case, this function swaps the memory
/// objects. Similarly for the data array.
///
/// After this function is called, @a A_hyp will own all of the arrays of its
/// diagonal part.
///
/// @note I and J can only be aliases when HYPRE_BIGINT is disabled.
void HypreStealOwnership(HypreParMatrix &A_hyp, SparseMatrix &A_diag);

#if MFEM_HYPRE_VERSION >= 21800

enum class BlockInverseScaleJob
{
   MATRIX_ONLY,
   RHS_ONLY,
   MATRIX_AND_RHS
};

/** Constructs and applies block diagonal inverse of HypreParMatrix.
    The enum @a job specifies whether the matrix or the RHS should be
    scaled (or both). */
void BlockInverseScale(const HypreParMatrix *A, HypreParMatrix *C,
                       const Vector *b, HypreParVector *d,
                       int blocksize, BlockInverseScaleJob job);
#endif

/** @brief Return a new matrix `C = alpha*A + beta*B`, assuming that both `A`
    and `B` use the same row and column partitions and the same `col_map_offd`
    arrays. */
HypreParMatrix *Add(double alpha, const HypreParMatrix &A,
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
    GatherBlockOffsetData() in hypre.cpp. */
HypreParMatrix * HypreParMatrixFromBlocks(Array2D<HypreParMatrix*> &blocks,
                                          Array2D<double> *blockCoeff=NULL);

/** @brief Eliminate essential BC specified by @a ess_dof_list from the solution
    @a X to the r.h.s. @a B. */
/** Here @a A is a matrix with eliminated BC, while @a Ae is such that (A+Ae) is
    the original (Neumann) matrix before elimination. */
void EliminateBC(const HypreParMatrix &A, const HypreParMatrix &Ae,
                 const Array<int> &ess_dof_list, const Vector &X, Vector &B);


/// Parallel smoothers in hypre
class HypreSmoother : public Solver
{
protected:
   /// The linear system matrix
   HypreParMatrix *A;
   /// Right-hand side and solution vectors
   mutable HypreParVector *B, *X;
   /** @brief Auxiliary buffers for the case when the input or output arrays in
       methods like Mult(const Vector &, Vector &) need to be deep copied in
       order to be used by hypre. */
   mutable Memory<double> auxB, auxX;
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

   /// A flag that indicates whether the linear system matrix A is symmetric
   bool A_is_symmetric;

public:
   /** Hypre smoother types:
       0    = Jacobi
       1    = l1-scaled Jacobi
       2    = l1-scaled block Gauss-Seidel/SSOR
       4    = truncated l1-scaled block Gauss-Seidel/SSOR
       5    = lumped Jacobi
       6    = Gauss-Seidel
       10   = On-processor forward solve for matrix w/ triangular structure
       16   = Chebyshev
       1001 = Taubin polynomial smoother
       1002 = FIR polynomial smoother. */
   enum Type { Jacobi = 0, l1Jacobi = 1, l1GS = 2, l1GStr = 4, lumpedJacobi = 5,
               GS = 6, OPFS = 10, Chebyshev = 16, Taubin = 1001, FIR = 1002
             };
#if !defined(HYPRE_USING_GPU)
   static constexpr Type default_type = l1GS;
#else
   static constexpr Type default_type = l1Jacobi;
#endif

   HypreSmoother();

   HypreSmoother(const HypreParMatrix &A_, int type = default_type,
                 int relax_times = 1, double relax_weight = 1.0,
                 double omega = 1.0, int poly_order = 2,
                 double poly_fraction = .3, int eig_est_cg_iter = 10);

   /// Set the relaxation type and number of sweeps
   void SetType(HypreSmoother::Type type, int relax_times = 1);
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

   /** Explicitly indicate whether the linear system matrix A is symmetric. If A
       is symmetric, the smoother will also be symmetric. In this case, calling
       MultTranspose will be redirected to Mult. (This is also done if the
       smoother is diagonal.) By default, A is assumed to be nonsymmetric. */
   void SetOperatorSymmetry(bool is_sym) { A_is_symmetric = is_sym; }

   /** Set/update the associated operator. Must be called after setting the
       HypreSmoother type and options. */
   virtual void SetOperator(const Operator &op);

   /// Relax the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;
   using Operator::Mult;

   /// Apply transpose of the smoother to relax the linear system Ax=b
   virtual void MultTranspose(const Vector &b, Vector &x) const;

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
   const HypreParMatrix *A;

   /// Right-hand side and solution vector
   mutable HypreParVector *B, *X;

   mutable Memory<double> auxB, auxX;

   /// Was hypre's Setup function called already?
   mutable int setup_called;

   /// How to treat hypre errors.
   mutable ErrorMode error_mode;

   /// @brief Makes the internal HypreParVector%s @a B and @a X wrap the input
   /// vectors @a b and @a x.
   ///
   /// Returns true if @a x can be shallow-copied, false otherwise.
   bool WrapVectors(const Vector &b, Vector &x) const;

public:
   HypreSolver();

   HypreSolver(const HypreParMatrix *A_);

   /// Typecast to HYPRE_Solver -- return the solver
   virtual operator HYPRE_Solver() const = 0;

   /// hypre's internal Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const = 0;
   /// hypre's internal Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const = 0;

   ///@{

   /// @brief Set up the solver (if not set up already, also called
   /// automatically by HypreSolver::Mult).
   virtual void Setup(const HypreParVector &b, HypreParVector &x) const;
   /// @brief Set up the solver (if not set up already, also called
   /// automatically by HypreSolver::Mult).
   virtual void Setup(const Vector &b, Vector &x) const;

   ///@}

   virtual void SetOperator(const Operator &op)
   { mfem_error("HypreSolvers do not support SetOperator!"); }

   virtual MemoryClass GetMemoryClass() const { return GetHypreMemoryClass(); }

   ///@{

   /// Solve the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   /// Solve the linear system Ax=b
   virtual void Mult(const Vector &b, Vector &x) const;
   using Operator::Mult;

   ///@}

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


#if MFEM_HYPRE_VERSION >= 21800
/** Preconditioner for HypreParMatrices that are triangular in some ordering.
   Finds correct ordering and performs forward substitution on processor
   as approximate inverse. Exact on one processor. */
class HypreTriSolve : public HypreSolver
{
public:
   HypreTriSolve() : HypreSolver() { }
   explicit HypreTriSolve(const HypreParMatrix &A) : HypreSolver(&A) { }
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSolve; }

   const HypreParMatrix* GetData() const { return A; }

   /// Deprecated. Use HypreTriSolve::GetData() const instead.
   MFEM_DEPRECATED HypreParMatrix* GetData()
   { return const_cast<HypreParMatrix*>(A); }

   virtual ~HypreTriSolve() { }
};
#endif

/// PCG solver in hypre
class HyprePCG : public HypreSolver
{
private:
   HYPRE_Solver pcg_solver;

   HypreSolver * precond;

public:
   HyprePCG(MPI_Comm comm);

   HyprePCG(const HypreParMatrix &A_);

   virtual void SetOperator(const Operator &op);

   void SetTol(double tol);
   void SetAbsTol(double atol);
   void SetMaxIter(int max_iter);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /** Use the L2 norm of the residual for measuring PCG convergence, plus
       (optionally) 1) periodically recompute true residuals from scratch; and
       2) enable residual-based stopping criteria. */
   void SetResidualConvergenceOptions(int res_frequency=-1, double rtol=0.0);

   /// deprecated: use SetZeroInitialIterate()
   MFEM_DEPRECATED void SetZeroInintialIterate() { iterative_mode = false; }

   /// non-hypre setting
   void SetZeroInitialIterate() { iterative_mode = false; }

   void GetNumIterations(int &num_iterations) const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   void GetFinalResidualNorm(double &final_res_norm) const
   {
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver,
                                                  &final_res_norm);
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

   HypreGMRES(const HypreParMatrix &A_);

   virtual void SetOperator(const Operator &op);

   void SetTol(double tol);
   void SetAbsTol(double tol);
   void SetMaxIter(int max_iter);
   void SetKDim(int dim);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// deprecated: use SetZeroInitialIterate()
   MFEM_DEPRECATED void SetZeroInintialIterate() { iterative_mode = false; }

   /// non-hypre setting
   void SetZeroInitialIterate() { iterative_mode = false; }

   void GetNumIterations(int &num_iterations) const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRGMRESGetNumIterations(gmres_solver, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   void GetFinalResidualNorm(double &final_res_norm) const
   {
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(gmres_solver,
                                                    &final_res_norm);
   }

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

/// Flexible GMRES solver in hypre
class HypreFGMRES : public HypreSolver
{
private:
   HYPRE_Solver fgmres_solver;

   HypreSolver * precond;

   /// Default, generally robust, FGMRES options
   void SetDefaultOptions();

public:
   HypreFGMRES(MPI_Comm comm);

   HypreFGMRES(const HypreParMatrix &A_);

   virtual void SetOperator(const Operator &op);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetKDim(int dim);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// deprecated: use SetZeroInitialIterate()
   MFEM_DEPRECATED void SetZeroInintialIterate() { iterative_mode = false; }

   /// non-hypre setting
   void SetZeroInitialIterate() { iterative_mode = false; }

   void GetNumIterations(int &num_iterations) const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRFlexGMRESGetNumIterations(fgmres_solver, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   void GetFinalResidualNorm(double &final_res_norm) const
   {
      HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(fgmres_solver,
                                                        &final_res_norm);
   }

   /// The typecast to HYPRE_Solver returns the internal fgmres_solver
   virtual operator HYPRE_Solver() const  { return fgmres_solver; }

   /// FGMRES Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRFlexGMRESSetup; }
   /// FGMRES Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRFlexGMRESSolve; }

   /// Solve Ax=b with hypre's FGMRES
   virtual void Mult (const HypreParVector &b, HypreParVector &x) const;
   using HypreSolver::Mult;

   virtual ~HypreFGMRES();
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
   explicit HypreDiagScale(const HypreParMatrix &A) : HypreSolver(&A) { }
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual void SetOperator(const Operator &op);

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale; }

   const HypreParMatrix* GetData() const { return A; }

   /// Deprecated. Use HypreDiagScale::GetData() const instead.
   MFEM_DEPRECATED HypreParMatrix* GetData()
   { return const_cast<HypreParMatrix*>(A); }

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

   HypreParaSails(const HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   void SetParams(double threshold, int max_levels);
   void SetFilter(double filter);
   void SetLoadBal(double loadbal);
   void SetReuse(int reuse);
   void SetLogging(int logging);
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

   HypreEuclid(const HypreParMatrix &A);

   void SetLevel(int level);
   void SetStats(int stats);
   void SetMemory(int mem);
   void SetBJ(int bj);
   void SetRowScale(int row_scale);

   virtual void SetOperator(const Operator &op);

   /// The typecast to HYPRE_Solver returns the internal euc_precond
   virtual operator HYPRE_Solver() const { return euc_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSolve; }

   virtual ~HypreEuclid();
};

#if MFEM_HYPRE_VERSION >= 21900
/**
@brief Wrapper for Hypre's native parallel ILU preconditioner.

The default ILU factorization type is ILU(k).  If you need to change this, or
any other option, you can use the HYPRE_Solver method to cast the object for use
with Hypre's native functions. For example, if want to use natural ordering
rather than RCM reordering, you can use the following approach:

@code
mfem::HypreILU ilu();
int reorder_type = 0;
HYPRE_ILUSetLocalReordering(ilu, reorder_type);
@endcode
*/
class HypreILU : public HypreSolver
{
private:
   HYPRE_Solver ilu_precond;

   /// Set the ILU default options
   void SetDefaultOptions();

   /** Reset the ILU preconditioner.
   @note If ilu_precond is NULL, this method allocates; otherwise it destroys
   ilu_precond and allocates a new object.  In both cases the default options
   are set. */
   void ResetILUPrecond();

public:
   /// Constructor; sets the default options
   HypreILU();

   virtual ~HypreILU();

   /// Set the fill level for ILU(k); the default is k=1.
   void SetLevelOfFill(HYPRE_Int lev_fill);

   void SetType(HYPRE_Int ilu_type);
   void SetMaxIter(HYPRE_Int max_iter);
   void SetTol(HYPRE_Real tol);
   void SetLocalReordering(HYPRE_Int reorder_type);

   /// Set the print level: 0 = none, 1 = setup, 2 = solve, 3 = setup+solve
   void SetPrintLevel(HYPRE_Int print_level);

   /// The typecast to HYPRE_Solver returns the internal ilu_precond
   virtual operator HYPRE_Solver() const { return ilu_precond; }

   virtual void SetOperator(const Operator &op);

   /// ILU Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSetup; }

   /// ILU Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSolve; }
};
#endif

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

   HypreBoomerAMG(const HypreParMatrix &A);

   virtual void SetOperator(const Operator &op);

   /** More robust options for systems, such as elasticity. */
   void SetSystemsOptions(int dim, bool order_bynodes=false);

   /** A special elasticity version of BoomerAMG that takes advantage of
       geometric rigid body modes and could perform better on some problems, see
       "Improving algebraic multigrid interpolation operators for linear
       elasticity problems", Baker, Kolev, Yang, NLAA 2009, DOI:10.1002/nla.688.
       This solver assumes Ordering::byVDIM in the FiniteElementSpace used to
       construct A. */
   void SetElasticityOptions(ParFiniteElementSpace *fespace);

#if MFEM_HYPRE_VERSION >= 21800
   /** Hypre parameters to use AIR AMG solve for advection-dominated problems.
       See "Nonsymmetric Algebraic Multigrid Based on Local Approximate Ideal
       Restriction (AIR)," Manteuffel, Ruge, Southworth, SISC (2018),
       DOI:/10.1137/17M1144350. Options: "distanceR" -> distance of neighbor
       DOFs for the restriction operator; options include 1, 2, and 15 (1.5).
       Strings "prerelax" and "postrelax" indicate points to relax on:
       F = F-points, C = C-points, A = all points. E.g., FFC -> relax on
       F-points, relax again on F-points, then relax on C-points. */
   void SetAdvectiveOptions(int distance=15,  const std::string &prerelax="",
                            const std::string &postrelax="FFC");

   /// Expert option - consult hypre documentation/team
   void SetStrongThresholdR(double strengthR)
   { HYPRE_BoomerAMGSetStrongThresholdR(amg_precond, strengthR); }

   /// Expert option - consult hypre documentation/team
   void SetFilterThresholdR(double filterR)
   { HYPRE_BoomerAMGSetFilterThresholdR(amg_precond, filterR); }

   /// Expert option - consult hypre documentation/team
   void SetRestriction(int restrict_type)
   { HYPRE_BoomerAMGSetRestriction(amg_precond, restrict_type); }

   /// Expert option - consult hypre documentation/team
   void SetIsTriangular()
   { HYPRE_BoomerAMGSetIsTriangular(amg_precond, 1); }

   /// Expert option - consult hypre documentation/team
   void SetGMRESSwitchR(int gmres_switch)
   { HYPRE_BoomerAMGSetGMRESSwitchR(amg_precond, gmres_switch); }

   /// Expert option - consult hypre documentation/team
   void SetCycleNumSweeps(int prerelax, int postrelax)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, prerelax,  1);
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, postrelax, 2);
   }
#endif

   void SetPrintLevel(int print_level)
   { HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level); }

   void SetMaxIter(int max_iter)
   { HYPRE_BoomerAMGSetMaxIter(amg_precond, max_iter); }

   /// Expert option - consult hypre documentation/team
   void SetMaxLevels(int max_levels)
   { HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels); }

   /// Expert option - consult hypre documentation/team
   void SetTol(double tol)
   { HYPRE_BoomerAMGSetTol(amg_precond, tol); }

   /// Expert option - consult hypre documentation/team
   void SetStrengthThresh(double strength)
   { HYPRE_BoomerAMGSetStrongThreshold(amg_precond, strength); }

   /// Expert option - consult hypre documentation/team
   void SetInterpolation(int interp_type)
   { HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type); }

   /// Expert option - consult hypre documentation/team
   void SetCoarsening(int coarsen_type)
   { HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type); }

   /// Expert option - consult hypre documentation/team
   void SetRelaxType(int relax_type)
   { HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type); }

   /// Expert option - consult hypre documentation/team
   void SetCycleType(int cycle_type)
   { HYPRE_BoomerAMGSetCycleType(amg_precond, cycle_type); }

   void GetNumIterations(int &num_iterations) const
   {
      HYPRE_Int num_it;
      HYPRE_BoomerAMGGetNumIterations(amg_precond, &num_it);
      num_iterations = internal::to_int(num_it);
   }

   /// Expert option - consult hypre documentation/team
   void SetNodal(int blocksize)
   {
      HYPRE_BoomerAMGSetNumFunctions(amg_precond, blocksize);
      HYPRE_BoomerAMGSetNodal(amg_precond, 1);
   }

   /// Expert option - consult hypre documentation/team
   void SetAggressiveCoarsening(int num_levels)
   { HYPRE_BoomerAMGSetAggNumLevels(amg_precond, num_levels); }

   /// The typecast to HYPRE_Solver returns the internal amg_precond
   virtual operator HYPRE_Solver() const { return amg_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; }

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
   /// Construct AMS solver from finite element space
   void Init(ParFiniteElementSpace *edge_space);

   /// Create the hypre solver object and set the default options, given the
   /// space dimension @a sdim and cycle type @a cycle_type.
   void MakeSolver(int sdim, int cycle_type);

   /// Construct the gradient and interpolation matrices associated with
   /// @a edge_fespace, and add them to the solver.
   void MakeGradientAndInterpolation(ParFiniteElementSpace *edge_fespace,
                                     int cycle_type);

   // Recreates another AMS solver with the same options when SetOperator is
   // called multiple times.
   void ResetAMSPrecond();

   /// The underlying hypre solver object
   HYPRE_Solver ams;
   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *Pi, *Pix, *Piy, *Piz;

   /// AMS cycle type
   int ams_cycle_type = 0;
   /// Spatial dimension of the underlying mesh
   int space_dim = 0;
   /// Flag set if `SetSingularProblem` is called, needed in `ResetAMSPrecond`
   bool singular = false;
   /// Flag set if `SetPrintLevel` is called, needed in `ResetAMSPrecond`
   int print_level = 1;

public:
   /// @brief Construct the AMS solver on the given edge finite element space.
   ///
   /// HypreAMS::SetOperator must be called to set the system matrix.
   HypreAMS(ParFiniteElementSpace *edge_fespace);

   /// Construct the AMS solver using the given matrix and finite element space.
   HypreAMS(const HypreParMatrix &A, ParFiniteElementSpace *edge_fespace);

   /// @brief Construct the AMS solver using the provided discrete gradient
   /// matrix @a G_ and the vertex coordinate vectors @a x_, @a y_, and @a z_.
   ///
   /// For 2D problems, @a z_ may be NULL. All other parameters must be
   /// non-NULL. The solver assumes ownership of G_, x_, y_, and z_.
   HypreAMS(const HypreParMatrix &A, HypreParMatrix *G_, HypreParVector *x_,
            HypreParVector *y_, HypreParVector *z_=NULL);

   virtual void SetOperator(const Operator &op);

   void SetPrintLevel(int print_lvl);

   /// Set this option when solving a curl-curl problem with zero mass term
   void SetSingularProblem()
   {
      HYPRE_AMSSetBetaPoissonMatrix(ams, NULL);
      singular = true;
   }

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
   /// Construct ADS solver from finite element space
   void Init(ParFiniteElementSpace *face_fespace);

   /// Create the hypre solver object and set the default options, using the
   /// cycle type cycle_type and AMS cycle type ams_cycle_type data members.
   void MakeSolver();

   /// Construct the discrete curl, gradient and interpolation matrices
   /// associated with @a face_fespace, and add them to the solver.
   void MakeDiscreteMatrices(ParFiniteElementSpace *face_fespace);

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

   /// ADS cycle type
   const int cycle_type = 11;
   /// AMS cycle type
   const int ams_cycle_type = 14;
   /// ADS print level
   int print_level = 1;

   // Recreates another ADS solver with the same options when SetOperator is
   // called multiple times.
   void ResetADSPrecond();
public:
   HypreADS(ParFiniteElementSpace *face_fespace);

   HypreADS(const HypreParMatrix &A, ParFiniteElementSpace *face_fespace);

   /// @brief Construct the ADS solver using the provided discrete curl matrix
   /// @a C, discrete gradient matrix @a G_ and vertex coordinate vectors @a x_,
   /// @a y_, and @a z_.
   ///
   /// None of the inputs may be NULL. The solver assumes ownership of C_, G_,
   /// x_, y_, and z_.
   HypreADS(const HypreParMatrix &A, HypreParMatrix *C_, HypreParMatrix *G_,
            HypreParVector *x_, HypreParVector *y_, HypreParVector *z_);

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

   HYPRE_BigInt glbSize; // Global number of DoFs in the linear system
   HYPRE_BigInt * part;  // Row partitioning of the linear system

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
   void GetEigenvalues(Array<double> & eigenvalues) const;

   /// Extract a single eigenvector
   const HypreParVector & GetEigenvector(unsigned int i) const;

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

   // HypreParVector wrappers to contain eigenvectors
   mutable HypreParVector ** eigenvectors;

   void createDummyVectors() const;

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
   void SetOperator(const HypreParMatrix & A);
   void SetMassMatrix(const HypreParMatrix & M);

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(Array<double> & eigenvalues) const;

   /// Extract a single eigenvector
   const HypreParVector & GetEigenvector(unsigned int i) const;

   /// Transfer ownership of the converged eigenvectors
   HypreParVector ** StealEigenvectors();
};

}

#endif // MFEM_USE_MPI

#endif
