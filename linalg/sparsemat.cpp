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

// Implementation of sparse matrix

#include "linalg.hpp"
#include "../general/forall.hpp"
#include "../general/table.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/backends.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

#if defined(MFEM_USE_CUDA)
#define MFEM_cu_or_hip(stub) cu##stub
#define MFEM_Cu_or_Hip(stub) Cu##stub
#define MFEM_CU_or_HIP(stub) CU##stub
#define MFEM_CUDA_or_HIP(stub) CUDA##stub

#if CUSPARSE_VERSION >=  11400
#define MFEM_GPUSPARSE_ALG CUSPARSE_SPMV_CSR_ALG1
#else // CUSPARSE_VERSION >= 11400
#define MFEM_GPUSPARSE_ALG CUSPARSE_CSRMV_ALG1
#endif // CUSPARSE_VERSION >= 11400

#elif defined(MFEM_USE_HIP)
#define MFEM_cu_or_hip(stub) hip##stub
#define MFEM_Cu_or_Hip(stub) Hip##stub
#define MFEM_CU_or_HIP(stub) HIP##stub
#define MFEM_CUDA_or_HIP(stub) HIP##stub

// https://hipsparse.readthedocs.io/en/latest/usermanual.html#hipsparsespmvalg-t
#define MFEM_GPUSPARSE_ALG HIPSPARSE_CSRMV_ALG1
#endif // defined(MFEM_USE_CUDA)

#if defined(MFEM_USE_SINGLE)
#define MFEM_CUDA_or_HIP_REAL_T MFEM_CUDA_or_HIP(_R_32F)
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_CUDA_or_HIP_REAL_T MFEM_CUDA_or_HIP(_R_64F)
#endif

namespace mfem
{

using namespace std;

#ifdef MFEM_USE_CUDA_OR_HIP
int SparseMatrix::SparseMatrixCount = 0;
// doxygen doesn't like the macro-assisted typename so let's skip parsing it:
/// @cond Suppress_Doxygen_warnings
MFEM_cu_or_hip(sparseHandle_t) SparseMatrix::handle = nullptr;
/// @endcond
#ifndef MFEM_CUDA_1897_WORKAROUND
size_t SparseMatrix::bufferSize = 0;
void * SparseMatrix::dBuffer = nullptr;
#endif
#endif // MFEM_USE_CUDA_OR_HIP

void SparseMatrix::InitGPUSparse()
{
   // Initialize cuSPARSE/hipSPARSE library
#ifdef MFEM_USE_CUDA_OR_HIP
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      if (!handle) { MFEM_cu_or_hip(sparseCreate)(&handle); }
      useGPUSparse=true;
      SparseMatrixCount++;
   }
   else
   {
      useGPUSparse=false;
   }
#endif // MFEM_USE_CUDA_OR_HIP
}

void SparseMatrix::ClearGPUSparse()
{
#ifdef MFEM_USE_CUDA_OR_HIP
   if (initBuffers)
   {
#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
      MFEM_cu_or_hip(sparseDestroySpMat)(matA_descr);
      MFEM_cu_or_hip(sparseDestroyDnVec)(vecX_descr);
      MFEM_cu_or_hip(sparseDestroyDnVec)(vecY_descr);
#else
      cusparseDestroyMatDescr(matA_descr);
#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
      initBuffers = false;
   }
#endif // MFEM_USE_CUDA_OR_HIP
}

SparseMatrix::SparseMatrix(int nrows, int ncols)
   : AbstractSparseMatrix(nrows, (ncols >= 0) ? ncols : nrows),
     Rows(new RowNode *[nrows]),
     current_row(-1),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     At(NULL),
     isSorted(false)
{
   // We probably do not need to set the ownership flags here.
   I.SetHostPtrOwner(true);
   J.SetHostPtrOwner(true);
   A.SetHostPtrOwner(true);

   for (int i = 0; i < nrows; i++)
   {
      Rows[i] = NULL;
   }

#ifdef MFEM_USE_MEMALLOC
   NodesMem = new RowNodeAlloc;
#endif

   InitGPUSparse();
}

SparseMatrix::SparseMatrix(int *i, int *j, real_t *data, int m, int n)
   : AbstractSparseMatrix(m, n),
     Rows(NULL),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     At(NULL),
     isSorted(false)
{
   I.Wrap(i, height+1, true);
   J.Wrap(j, I[height], true);
   A.Wrap(data, I[height], true);

#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif

   InitGPUSparse();
}

SparseMatrix::SparseMatrix(int *i, int *j, real_t *data, int m, int n,
                           bool ownij, bool owna, bool issorted)
   : AbstractSparseMatrix(m, n),
     Rows(NULL),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     At(NULL),
     isSorted(issorted)
{
   I.Wrap(i, height+1, ownij);
   J.Wrap(j, I[height], ownij);

#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif

   if (data)
   {
      A.Wrap(data, I[height], owna);
   }
   else
   {
      const int nnz = I[height];
      A.New(nnz);
      for (int ii=0; ii<nnz; ++ii)
      {
         A[ii] = 0.0;
      }
   }

   InitGPUSparse();
}

SparseMatrix::SparseMatrix(int nrows, int ncols, int rowsize)
   : AbstractSparseMatrix(nrows, ncols)
   , Rows(NULL)
   , ColPtrJ(NULL)
   , ColPtrNode(NULL)
   , At(NULL)
   , isSorted(false)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   I.New(nrows + 1);
   J.New(nrows * rowsize);
   A.New(nrows * rowsize);

   for (int i = 0; i <= nrows; i++)
   {
      I[i] = i * rowsize;
   }

   InitGPUSparse();
}

SparseMatrix::SparseMatrix(const SparseMatrix &mat, bool copy_graph,
                           MemoryType mt)
   : AbstractSparseMatrix(mat.Height(), mat.Width())
{
   if (mat.Finalized())
   {
      mat.HostReadI();
      const int nnz = mat.I[height];
      if (copy_graph)
      {
         I.New(height+1, mt == MemoryType::PRESERVE ? mat.I.GetMemoryType() : mt);
         J.New(nnz, mt == MemoryType::PRESERVE ? mat.J.GetMemoryType() : mt);
         I.CopyFrom(mat.I, height+1);
         J.CopyFrom(mat.J, nnz);
      }
      else
      {
         I = mat.I;
         J = mat.J;
         I.ClearOwnerFlags();
         J.ClearOwnerFlags();
      }
      A.New(nnz, mt == MemoryType::PRESERVE ? mat.A.GetMemoryType() : mt);
      A.CopyFrom(mat.A, nnz);

      Rows = NULL;
#ifdef MFEM_USE_MEMALLOC
      NodesMem = NULL;
#endif
   }
   else
   {
#ifdef MFEM_USE_MEMALLOC
      NodesMem = new RowNodeAlloc;
#endif
      Rows = new RowNode *[height];
      for (int i = 0; i < height; i++)
      {
         RowNode **node_pp = &Rows[i];
         for (RowNode *node_p = mat.Rows[i]; node_p; node_p = node_p->Prev)
         {
#ifdef MFEM_USE_MEMALLOC
            RowNode *new_node_p = NodesMem->Alloc();
#else
            RowNode *new_node_p = new RowNode;
#endif
            new_node_p->Value = node_p->Value;
            new_node_p->Column = node_p->Column;
            *node_pp = new_node_p;
            node_pp = &new_node_p->Prev;
         }
         *node_pp = NULL;
      }

      // We probably do not need to set the ownership flags here.
      I.SetHostPtrOwner(true);
      J.SetHostPtrOwner(true);
      A.SetHostPtrOwner(true);
   }

   current_row = -1;
   ColPtrJ = NULL;
   ColPtrNode = NULL;
   At = NULL;
   isSorted = mat.isSorted;

   InitGPUSparse();
}

SparseMatrix::SparseMatrix(const Vector &v)
   : AbstractSparseMatrix(v.Size(), v.Size())
   , Rows(NULL)
   , ColPtrJ(NULL)
   , ColPtrNode(NULL)
   , At(NULL)
   , isSorted(true)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   I.New(height + 1);
   J.New(height);
   A.New(height);

   for (int i = 0; i <= height; i++)
   {
      I[i] = i;
   }

   for (int r=0; r<height; r++)
   {
      J[r] = r;
      A[r] = v[r];
   }

   InitGPUSparse();
}

void SparseMatrix::OverrideSize(int height_, int width_)
{
   height = height_;
   width = width_;
}

SparseMatrix& SparseMatrix::operator=(const SparseMatrix &rhs)
{
   Clear();

   SparseMatrix copy(rhs);
   Swap(copy);

   return *this;
}

void SparseMatrix::MakeRef(const SparseMatrix &master)
{
   MFEM_ASSERT(master.Finalized(), "'master' must be finalized");
   Clear();
   height = master.Height();
   width = master.Width();
   I = master.I; I.ClearOwnerFlags();
   J = master.J; J.ClearOwnerFlags();
   A = master.A; A.ClearOwnerFlags();
   isSorted = master.isSorted;
}

void SparseMatrix::SetEmpty()
{
   height = width = 0;
   I.Reset();
   J.Reset();
   A.Reset();
   Rows = NULL;
   current_row = -1;
   ColPtrJ = NULL;
   ColPtrNode = NULL;
   At = NULL;
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   isSorted = false;

   ClearGPUSparse();
}

int SparseMatrix::RowSize(const int i) const
{
   int gi = i;
   if (gi < 0)
   {
      gi = -1-gi;
   }

   if (I)
   {
      return I[gi+1]-I[gi];
   }

   int s = 0;
   RowNode *row = Rows[gi];
   for ( ; row != NULL; row = row->Prev)
      if (row->Value != 0.0)
      {
         s++;
      }
   return s;
}

int SparseMatrix::MaxRowSize() const
{
   int max_row_size=0;
   int rowSize=0;
   if (I)
   {
      for (int i=0; i < height; ++i)
      {
         rowSize = I[i+1]-I[i];
         max_row_size = (max_row_size > rowSize) ? max_row_size : rowSize;
      }
   }
   else
   {
      for (int i=0; i < height; ++i)
      {
         rowSize = RowSize(i);
         max_row_size = (max_row_size > rowSize) ? max_row_size : rowSize;
      }
   }

   return max_row_size;
}

int *SparseMatrix::GetRowColumns(const int row)
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   return J + I[row];
}

const int *SparseMatrix::GetRowColumns(const int row) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   return J + I[row];
}

real_t *SparseMatrix::GetRowEntries(const int row)
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   return A + I[row];
}

const real_t *SparseMatrix::GetRowEntries(const int row) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   return A + I[row];
}

void SparseMatrix::SetWidth(int newWidth)
{
   if (newWidth == width)
   {
      // Nothing to be done here
      return;
   }
   else if (newWidth == -1)
   {
      // Compute the actual width
      width = ActualWidth();
      // No need to reset the ColPtr, since the new ColPtr will be shorter.
   }
   else if (newWidth > width)
   {
      // We need to reset ColPtr, since now we may have additional columns.
      if (Rows != NULL)
      {
         delete [] ColPtrNode;
         ColPtrNode = static_cast<RowNode **>(NULL);
      }
      else
      {
         delete [] ColPtrJ;
         ColPtrJ = static_cast<int *>(NULL);
      }
      width = newWidth;
   }
   else
   {
      // Check that the new width is bigger or equal to the actual width.
      MFEM_ASSERT(newWidth >= ActualWidth(),
                  "The new width needs to be bigger or equal to the actual width");
      width = newWidth;
   }
}


void SparseMatrix::SortColumnIndices()
{
   MFEM_VERIFY(Finalized(), "Matrix is not Finalized!");

   if (isSorted)
   {
      return;
   }

#ifdef MFEM_USE_CUDA_OR_HIP
   if (Device::Allows(Backend::CUDA_MASK) || Device::Allows(Backend::HIP_MASK))
   {
      const int m = Height();
      const int n = Width();
      const int nnzA = J.Capacity();
      const int *d_ia = ReadI();
      int *d_ja = ReadWriteJ();

      // Get size of temporary buffer needed to sort the column indices,
      // allocate the temporary buffer.
      size_t pBufferSizeInBytes;
      MFEM_cu_or_hip(sparseXcsrsort_bufferSizeExt)(handle, m, n, nnzA, d_ia,
                                                   d_ja, &pBufferSizeInBytes);
      void *pBuffer = MFEM_Cu_or_Hip(MemAlloc)(&pBuffer, pBufferSizeInBytes);

      // Create matrix descriptor, will have default values
      // CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_MATRIX_TYPE_GENERAL.
      MFEM_cu_or_hip(sparseMatDescr_t) matA_descr;
      MFEM_cu_or_hip(sparseCreateMatDescr)(&matA_descr);

      // Initialize permutation to identity
      Array<int> P(nnzA);
      int *d_P = P.Write();
      mfem::forall(nnzA, [=] MFEM_HOST_DEVICE (int i) { d_P[i] = i; });

      // Sort the column indices. The array d_ja will now be sorted. The
      // permutation required to sort the values will be returned in d_P.
      MFEM_cu_or_hip(sparseXcsrsort)(handle, m, n, nnzA, matA_descr, d_ia, d_ja,
                                     d_P, pBuffer);

      // Create a copy of the unsorted matrix values.
      real_t *d_a = ReadWriteData();
      void *d_a_unsorted = MFEM_Cu_or_Hip(MemAlloc)(&d_a_unsorted,
                                                    nnzA * sizeof(real_t));
      MFEM_Cu_or_Hip(MemcpyDtoD)(d_a_unsorted, d_a, nnzA * sizeof(real_t));

      // Create the (input) dense vector with the unsorted values.
      MFEM_cu_or_hip(sparseDnVecDescr_t) d_a_dense;
      MFEM_cu_or_hip(sparseCreateDnVec)(&d_a_dense, nnzA, d_a_unsorted,
                                        MFEM_CUDA_or_HIP_REAL_T);

      // Create the (output) sparse vector that will have the sorted values.
      MFEM_cu_or_hip(sparseSpVecDescr_t) d_a_sparse;
      MFEM_cu_or_hip(sparseCreateSpVec)(&d_a_sparse, nnzA, nnzA, d_P, d_a,
                                        MFEM_CU_or_HIP(SPARSE_INDEX_32I),
                                        MFEM_CU_or_HIP(SPARSE_INDEX_BASE_ZERO),
                                        MFEM_CUDA_or_HIP_REAL_T);

      // Sort the matrix values using the permutation vector.
      MFEM_cu_or_hip(sparseGather)(handle, d_a_dense, d_a_sparse);

      // The above calls may be asynchronous, so we need to wait for them to
      // finish before we can free memory.
      MFEM_STREAM_SYNC;

      MFEM_cu_or_hip(sparseDestroyDnVec)(d_a_dense);
      MFEM_cu_or_hip(sparseDestroySpVec)(d_a_sparse);
      MFEM_cu_or_hip(sparseDestroyMatDescr)(matA_descr);

      MFEM_Cu_or_Hip(MemFree)(d_a_unsorted);
      MFEM_Cu_or_Hip(MemFree)(pBuffer);
   }
   else
#endif // MFEM_USE_CUDA_OR_HIP
   {
      const int * Ip=HostReadI();
      HostReadWriteJ();
      HostReadWriteData();

      Array<Pair<int,real_t> > row;
      for (int j = 0, i = 0; i < height; i++)
      {
         int end = Ip[i+1];
         row.SetSize(end - j);
         for (int k = 0; k < row.Size(); k++)
         {
            row[k].one = J[j+k];
            row[k].two = A[j+k];
         }
         row.Sort();
         for (int k = 0; k < row.Size(); k++, j++)
         {
            J[j] = row[k].one;
            A[j] = row[k].two;
         }
      }
   }
   isSorted = true;
}

void SparseMatrix::MoveDiagonalFirst()
{
   MFEM_VERIFY(Finalized(), "Matrix is not Finalized!");

   for (int row = 0, end = 0; row < height; row++)
   {
      int start = end, j;
      end = I[row+1];
      for (j = start; true; j++)
      {
         MFEM_VERIFY(j < end, "diagonal entry not found in row = " << row);
         if (J[j] == row) { break; }
      }
      const real_t diag = A[j];
      for ( ; j > start; j--)
      {
         J[j] = J[j-1];
         A[j] = A[j-1];
      }
      J[start] = row;
      A[start] = diag;
   }
}

real_t &SparseMatrix::Elem(int i, int j)
{
   return operator()(i,j);
}

const real_t &SparseMatrix::Elem(int i, int j) const
{
   return operator()(i,j);
}

real_t &SparseMatrix::operator()(int i, int j)
{
   MFEM_ASSERT(i < height && i >= 0 && j < width && j >= 0,
               "Trying to access element outside of the matrix.  "
               << "height = " << height << ", "
               << "width = " << width << ", "
               << "i = " << i << ", "
               << "j = " << j);

   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int k = I[i], end = I[i+1]; k < end; k++)
   {
      if (J[k] == j)
      {
         return A[k];
      }
   }

   MFEM_ABORT("Did not find i = " << i << ", j = " << j << " in matrix.");
   return A[0];
}

const real_t &SparseMatrix::operator()(int i, int j) const
{
   static const real_t zero = 0.0;

   MFEM_ASSERT(i < height && i >= 0 && j < width && j >= 0,
               "Trying to access element outside of the matrix.  "
               << "height = " << height << ", "
               << "width = " << width << ", "
               << "i = " << i << ", "
               << "j = " << j);

   if (Finalized())
   {
      HostReadI();
      HostReadJ();
      HostReadData();
      for (int k = I[i], end = I[i+1]; k < end; k++)
      {
         if (J[k] == j)
         {
            return A[k];
         }
      }
   }
   else
   {
      for (RowNode *node_p = Rows[i]; node_p != NULL; node_p = node_p->Prev)
      {
         if (node_p->Column == j)
         {
            return node_p->Value;
         }
      }
   }

   return zero;
}

void SparseMatrix::GetDiag(Vector & d) const
{
   MFEM_VERIFY(height == width, "Matrix must be square, not height = "
               << height << ", width = " << width);
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   d.SetSize(height);

   const auto II = this->ReadI();
   const auto JJ = this->ReadJ();
   const auto AA = this->ReadData();
   auto dd = d.Write();

   mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
   {
      const int begin = II[i];
      const int end = II[i+1];
      int j;
      for (j = begin; j < end; j++)
      {
         if (JJ[j] == i)
         {
            dd[i] = AA[j];
            break;
         }
      }
      if (j == end)
      {
         dd[i] = 0.;
      }
   });
}

/// Produces a DenseMatrix from a SparseMatrix
DenseMatrix *SparseMatrix::ToDenseMatrix() const
{
   int num_rows = this->Height();
   int num_cols = this->Width();

   DenseMatrix * B = new DenseMatrix(num_rows, num_cols);

   this->ToDenseMatrix(*B);

   return B;
}

/// Produces a DenseMatrix from a SparseMatrix
void SparseMatrix::ToDenseMatrix(DenseMatrix & B) const
{
   B.SetSize(height, width);
   B = 0.0;

   for (int r=0; r<height; r++)
   {
      const int    * col = this->GetRowColumns(r);
      const real_t * val = this->GetRowEntries(r);

      for (int cj=0; cj<this->RowSize(r); cj++)
      {
         B(r, col[cj]) = val[cj];
      }
   }
}

void SparseMatrix::Mult(const Vector &x, Vector &y) const
{
   if (Finalized()) { y.UseDevice(true); }
   y = 0.0;
   AddMult(x, y);
}

void SparseMatrix::AddMult(const Vector &x, Vector &y, const real_t a) const
{
   MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
               << ") must match matrix width (" << width << ")");
   MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
               << ") must match matrix height (" << height << ")");

   if (!Finalized())
   {
      const real_t *xp = x.HostRead();
      real_t *yp = y.HostReadWrite();

      // The matrix is not finalized, but multiplication is still possible
      for (int i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         real_t b = 0.0;
         for ( ; row != NULL; row = row->Prev)
         {
            b += row->Value * xp[row->Column];
         }
         *yp += a * b;
         yp++;
      }
      return;
   }

#ifndef MFEM_USE_LEGACY_OPENMP
   const int height = this->height;
   const int nnz = J.Capacity();
   auto d_I = Read(I, height+1);
   auto d_J = Read(J, nnz);
   auto d_A = Read(A, nnz);
   auto d_x = x.Read();
   auto d_y = y.ReadWrite();

   // Skip if matrix has no non-zeros
   if (nnz == 0) {return;}
   if ((Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK)) && useGPUSparse)
   {
#ifdef MFEM_USE_CUDA_OR_HIP
      const real_t alpha = a;
      const real_t beta  = 1.0;

      // Setup descriptors
      if (!initBuffers)
      {
#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
         // Setup matrix descriptor
         MFEM_cu_or_hip(sparseCreateCsr)(
            &matA_descr,Height(),
            Width(),
            J.Capacity(),
            const_cast<int *>(d_I),
            const_cast<int *>(d_J),
            const_cast<real_t *>(d_A),
            MFEM_CU_or_HIP(SPARSE_INDEX_32I),
            MFEM_CU_or_HIP(SPARSE_INDEX_32I),
            MFEM_CU_or_HIP(SPARSE_INDEX_BASE_ZERO),
            MFEM_CUDA_or_HIP_REAL_T);

         // Create handles for input/output vectors
         MFEM_cu_or_hip(sparseCreateDnVec)(&vecX_descr,
                                           x.Size(),
                                           const_cast<real_t *>(d_x),
                                           MFEM_CUDA_or_HIP_REAL_T);
         MFEM_cu_or_hip(sparseCreateDnVec)(&vecY_descr, y.Size(), d_y,
                                           MFEM_CUDA_or_HIP_REAL_T);
#else
         cusparseCreateMatDescr(&matA_descr);
         cusparseSetMatIndexBase(matA_descr, CUSPARSE_INDEX_BASE_ZERO);
         cusparseSetMatType(matA_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
         initBuffers = true;
      }
      // Allocate kernel space. Buffer is shared between different sparsemats
      size_t newBufferSize = 0;

      MFEM_cu_or_hip(sparseSpMV_bufferSize)(
         handle,
         MFEM_CU_or_HIP(SPARSE_OPERATION_NON_TRANSPOSE),
         &alpha,
         matA_descr,
         vecX_descr,
         &beta,
         vecY_descr,
         MFEM_CUDA_or_HIP_REAL_T,
         MFEM_GPUSPARSE_ALG,
         &newBufferSize);

      // Check if we need to resize
      if (newBufferSize > bufferSize)
      {
         bufferSize = newBufferSize;
         if (dBuffer != nullptr) { MFEM_Cu_or_Hip(MemFree)(dBuffer); }
         MFEM_Cu_or_Hip(MemAlloc)(&dBuffer, bufferSize);
      }

#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
      // Update input/output vectors
      MFEM_cu_or_hip(sparseDnVecSetValues)(vecX_descr,
                                           const_cast<real_t *>(d_x));
      MFEM_cu_or_hip(sparseDnVecSetValues)(vecY_descr, d_y);

      // Y = alpha A * X + beta * Y
      MFEM_cu_or_hip(sparseSpMV)(
         handle,
         MFEM_CU_or_HIP(SPARSE_OPERATION_NON_TRANSPOSE),
         &alpha,
         matA_descr,
         vecX_descr,
         &beta,
         vecY_descr,
         MFEM_CUDA_or_HIP_REAL_T,
         MFEM_GPUSPARSE_ALG,
         dBuffer);
#else
#ifdef MFEM_USE_SINGLE
      cusparseScsrmv(handle,
#else
      cusparseDcsrmv(handle,
#endif
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     Height(),
                     Width(),
                     J.Capacity(),
                     &alpha,
                     matA_descr,
                     const_cast<real_t *>(d_A),
                     const_cast<int *>(d_I),
                     const_cast<int *>(d_J),
                     const_cast<real_t *>(d_x),
                     &beta,
                     d_y);
#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
#endif // MFEM_USE_CUDA_OR_HIP
   }
   else
   {
      // Native version
      mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
      {
         real_t d = 0.0;
         const int end = d_I[i+1];
         for (int j = d_I[i]; j < end; j++)
         {
            d += d_A[j] * d_x[d_J[j]];
         }
         d_y[i] += a * d;
      });

   }

#else // MFEM_USE_LEGACY_OPENMP
   const real_t *Ap = A, *xp = x.GetData();
   real_t *yp = y.GetData();
   const int *Jp = J, *Ip = I;

   #pragma omp parallel for
   for (int i = 0; i < height; i++)
   {
      real_t d = 0.0;
      const int end = Ip[i+1];
      for (int j = Ip[i]; j < end; j++)
      {
         d += Ap[j] * xp[Jp[j]];
      }
      yp[i] += a * d;
   }
#endif // MFEM_USE_LEGACY_OPENMP
}

void SparseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   if (Finalized()) { y.UseDevice(true); }
   y = 0.0;
   AddMultTranspose(x, y);
}

void SparseMatrix::AddMultTranspose(const Vector &x, Vector &y,
                                    const real_t a) const
{
   MFEM_ASSERT(height == x.Size(), "Input vector size (" << x.Size()
               << ") must match matrix height (" << height << ")");
   MFEM_ASSERT(width == y.Size(), "Output vector size (" << y.Size()
               << ") must match matrix width (" << width << ")");

   if (!Finalized())
   {
      real_t *yp = y.HostReadWrite();
      const real_t *xp = x.HostRead();
      // The matrix is not finalized, but multiplication is still possible
      for (int i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         real_t b = a * xp[i];
         for ( ; row != NULL; row = row->Prev)
         {
            yp[row->Column] += row->Value * b;
         }
      }
      return;
   }

   EnsureMultTranspose();
   if (At)
   {
      At->AddMult(x, y, a);
   }
   else
   {
      real_t *yp = y.HostReadWrite();
      const real_t *xp = x.HostRead();

      const int *Ip = HostRead(I, height+1);
      const int nnz = Ip[height];
      const int *Jp = HostRead(J, nnz);
      const real_t *Ap = HostRead(A, nnz);

      for (int i = 0; i < height; i++)
      {
         const real_t xi = a * xp[i];
         const int end = Ip[i+1];
         for (int j = Ip[i]; j < end; j++)
         {
            const int Jj = Jp[j];
            yp[Jj] += Ap[j] * xi;
         }
      }
   }
}

void SparseMatrix::BuildTranspose() const
{
   if (At == NULL)
   {
      At = Transpose(*this);
   }
}

void SparseMatrix::ResetTranspose() const
{
   delete At;
   At = NULL;
}

void SparseMatrix::EnsureMultTranspose() const
{
   if (Device::Allows(~Backend::CPU_MASK))
   {
      BuildTranspose();
   }
}

void SparseMatrix::PartMult(
   const Array<int> &rows, const Vector &x, Vector &y) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   const int n = rows.Size();
   const int nnz = J.Capacity();
   auto d_rows = rows.Read();
   auto d_I = Read(I, height+1);
   auto d_J = Read(J, nnz);
   auto d_A = Read(A, nnz);
   auto d_x = x.Read();
   auto d_y = y.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int r = d_rows[i];
      const int end = d_I[r + 1];
      real_t a = 0.0;
      for (int j = d_I[r]; j < end; j++)
      {
         a += d_A[j] * d_x[d_J[j]];
      }
      d_y[r] = a;
   });
}

void SparseMatrix::PartAddMult(
   const Array<int> &rows, const Vector &x, Vector &y, const real_t a) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < rows.Size(); i++)
   {
      int r = rows[i];
      int end = I[r + 1];
      real_t val = 0.0;
      for (int j = I[r]; j < end; j++)
      {
         val += A[j] * x(J[j]);
      }
      y(r) += a * val;
   }
}

void SparseMatrix::BooleanMult(const Array<int> &x, Array<int> &y) const
{
   MFEM_ASSERT(Finalized(), "Matrix must be finalized.");
   MFEM_ASSERT(x.Size() == Width(), "Input vector size (" << x.Size()
               << ") must match matrix width (" << Width() << ")");

   y.SetSize(Height(), Device::GetDeviceMemoryType());

   const int height = Height();
   const int nnz = J.Capacity();
   auto d_I = Read(I, height+1);
   auto d_J = Read(J, nnz);
   auto d_x = Read(x.GetMemory(), x.Size());
   auto d_y = Write(y.GetMemory(), y.Size());
   mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
   {
      bool d_yi = false;
      const int end = d_I[i+1];
      for (int j = d_I[i]; j < end; j++)
      {
         if (d_x[d_J[j]])
         {
            d_yi = true;
            break;
         }
      }
      d_y[i] = d_yi;
   });
}

void SparseMatrix::BooleanMultTranspose(const Array<int> &x,
                                        Array<int> &y) const
{
   MFEM_ASSERT(Finalized(), "Matrix must be finalized.");
   MFEM_ASSERT(x.Size() == Height(), "Input vector size (" << x.Size()
               << ") must match matrix height (" << Height() << ")");

   y.SetSize(Width());
   y = 0;

   for (int i = 0; i < Height(); i++)
   {
      if (x[i])
      {
         int end = I[i+1];
         for (int j = I[i]; j < end; j++)
         {
            y[J[j]] = x[i];
         }
      }
   }
}

void SparseMatrix::AbsMult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
               << ") must match matrix width (" << width << ")");
   MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
               << ") must match matrix height (" << height << ")");

   if (Finalized()) { y.UseDevice(true); }
   y = 0.0;

   if (!Finalized())
   {
      const real_t *xp = x.HostRead();
      real_t *yp = y.HostReadWrite();

      // The matrix is not finalized, but multiplication is still possible
      for (int i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         real_t b = 0.0;
         for ( ; row != NULL; row = row->Prev)
         {
            b += std::abs(row->Value) * xp[row->Column];
         }
         *yp += b;
         yp++;
      }
      return;
   }

   const int height = this->height;
   const int nnz = J.Capacity();
   auto d_I = Read(I, height+1);
   auto d_J = Read(J, nnz);
   auto d_A = Read(A, nnz);
   auto d_x = x.Read();
   auto d_y = y.ReadWrite();
   mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
   {
      real_t d = 0.0;
      const int end = d_I[i+1];
      for (int j = d_I[i]; j < end; j++)
      {
         d += std::abs(d_A[j]) * d_x[d_J[j]];
      }
      d_y[i] += d;
   });
}

void SparseMatrix::AbsMultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == x.Size(), "Input vector size (" << x.Size()
               << ") must match matrix height (" << height << ")");
   MFEM_ASSERT(width == y.Size(), "Output vector size (" << y.Size()
               << ") must match matrix width (" << width << ")");

   y = 0.0;

   if (!Finalized())
   {
      real_t *yp = y.GetData();
      // The matrix is not finalized, but multiplication is still possible
      for (int i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         real_t b = x(i);
         for ( ; row != NULL; row = row->Prev)
         {
            yp[row->Column] += fabs(row->Value) * b;
         }
      }
      return;
   }

   EnsureMultTranspose();
   if (At)
   {
      At->AbsMult(x, y);
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         const real_t xi = x[i];
         const int end = I[i+1];
         for (int j = I[i]; j < end; j++)
         {
            const int Jj = J[j];
            y[Jj] += std::abs(A[j]) * xi;
         }
      }
   }
}

real_t SparseMatrix::InnerProduct(const Vector &x, const Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "x.Size() = " << x.Size()
               << " must be equal to Width() = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "y.Size() = " << y.Size()
               << " must be equal to Height() = " << Height());

   x.HostRead();
   y.HostRead();
   if (Finalized())
   {
      const int nnz = J.Capacity();
      HostRead(I, height+1);
      HostRead(J, nnz);
      HostRead(A, nnz);
   }

   real_t prod = 0.0;
   for (int i = 0; i < height; i++)
   {
      real_t a = 0.0;
      if (A)
      {
         for (int j = I[i], end = I[i+1]; j < end; j++)
         {
            a += A[j] * x(J[j]);
         }
      }
      else
      {
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
         {
            a += np->Value * x(np->Column);
         }
      }
      prod += a * y(i);
   }

   return prod;
}

void SparseMatrix::GetRowSums(Vector &x) const
{
   if (Finalized())
   {
      auto d_I = ReadI();
      auto d_A = ReadData();
      auto d_x = x.Write();
      mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
      {
         real_t sum = 0.0;
         for (int j = d_I[i], end = d_I[i+1]; j < end; j++)
         {
            sum += d_A[j];
         }
         d_x[i] = sum;
      });
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         real_t a = 0.0;
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
         {
            a += np->Value;
         }
         x(i) = a;
      }
   }
}

real_t SparseMatrix::GetRowNorml1(int irow) const
{
   MFEM_VERIFY(irow < height,
               "row " << irow << " not in matrix with height " << height);

   real_t a = 0.0;
   if (A)
   {
      for (int j = I[irow], end = I[irow+1]; j < end; j++)
      {
         a += fabs(A[j]);
      }
   }
   else
   {
      for (RowNode *np = Rows[irow]; np != NULL; np = np->Prev)
      {
         a += fabs(np->Value);
      }
   }

   return a;
}

void SparseMatrix::Threshold(real_t tol, bool fix_empty_rows)
{
   MFEM_ASSERT(Finalized(), "Matrix must be finalized.");
   real_t atol;
   atol = std::abs(tol);

   fix_empty_rows = height == width ? fix_empty_rows : false;

   real_t *newA;
   int *newI, *newJ;
   int i, j, nz;

   newI = Memory<int>(height+1);
   newI[0] = 0;
   for (i = 0, nz = 0; i < height; i++)
   {
      bool found = false;
      for (j = I[i]; j < I[i+1]; j++)
         if (std::abs(A[j]) > atol)
         {
            found = true;
            nz++;
         }
      if (fix_empty_rows && !found) { nz++; }
      newI[i+1] = nz;
   }

   newJ = Memory<int>(nz);
   newA = Memory<real_t>(nz);
   // Assume we're sorted until we find out otherwise
   isSorted = true;
   for (i = 0, nz = 0; i < height; i++)
   {
      bool found = false;
      int lastCol = -1;
      for (j = I[i]; j < I[i+1]; j++)
         if (std::abs(A[j]) > atol)
         {
            found = true;
            newJ[nz] = J[j];
            newA[nz] = A[j];
            if ( lastCol > newJ[nz] )
            {
               isSorted = false;
            }
            lastCol = newJ[nz];
            nz++;
         }
      if (fix_empty_rows && !found)
      {
         newJ[nz] = i;
         newA[nz] = 0.0;
         nz++;
      }
   }
   Destroy();
   I.Wrap(newI, height+1, true);
   J.Wrap(newJ, I[height], true);
   A.Wrap(newA, I[height], true);
}

void SparseMatrix::Finalize(int skip_zeros, bool fix_empty_rows)
{
   int i, j, nr, nz;
   RowNode *aux;

   if (Finalized())
   {
      return;
   }

   delete [] ColPtrNode;
   ColPtrNode = NULL;

   I.New(height+1);
   I[0] = 0;
   for (i = 1; i <= height; i++)
   {
      nr = 0;
      for (aux = Rows[i-1]; aux != NULL; aux = aux->Prev)
      {
         if (skip_zeros && aux->Value == 0.0)
         {
            if (skip_zeros == 2) { continue; }
            if ((i-1) != aux->Column) { continue; }

            bool found = false;
            real_t found_val = 0.0; // init to suppress gcc warning
            for (RowNode *other = Rows[aux->Column]; other != NULL; other = other->Prev)
            {
               if (other->Column == (i-1))
               {
                  found = true;
                  found_val = other->Value;
                  break;
               }
            }
            if (found && found_val == 0.0) { continue; }

         }
         nr++;
      }
      if (fix_empty_rows && !nr) { nr = 1; }
      I[i] = I[i-1] + nr;
   }

   nz = I[height];
   J.New(nz);
   A.New(nz);
   // Assume we're sorted until we find out otherwise
   isSorted = true;
   for (j = i = 0; i < height; i++)
   {
      int lastCol = -1;
      nr = 0;
      for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
      {
         if (skip_zeros && aux->Value == 0.0)
         {
            if (skip_zeros == 2) { continue; }
            if (i != aux->Column) { continue; }

            bool found = false;
            real_t found_val = 0.0; // init to suppress gcc warning
            for (RowNode *other = Rows[aux->Column]; other != NULL; other = other->Prev)
            {
               if (other->Column == i)
               {
                  found = true;
                  found_val = other->Value;
                  break;
               }
            }
            if (found && found_val == 0.0) { continue; }
         }

         J[j] = aux->Column;
         A[j] = aux->Value;

         if ( lastCol > J[j] )
         {
            isSorted = false;
         }
         lastCol = J[j];

         j++;
         nr++;
      }
      if (fix_empty_rows && !nr)
      {
         J[j] = i;
         A[j] = 1.0;
         j++;
      }
   }

#ifdef MFEM_USE_MEMALLOC
   delete NodesMem;
   NodesMem = NULL;
#else
   for (i = 0; i < height; i++)
   {
      RowNode *node_p = Rows[i];
      while (node_p != NULL)
      {
         aux = node_p;
         node_p = node_p->Prev;
         delete aux;
      }
   }
#endif

   delete [] Rows;
   Rows = NULL;
}

void SparseMatrix::GetBlocks(Array2D<SparseMatrix *> &blocks) const
{
   int br = blocks.NumRows(), bc = blocks.NumCols();
   int nr = (height + br - 1)/br, nc = (width + bc - 1)/bc;

   for (int j = 0; j < bc; j++)
   {
      for (int i = 0; i < br; i++)
      {
         int *bI = Memory<int>(nr + 1);
         for (int k = 0; k <= nr; k++)
         {
            bI[k] = 0;
         }
         blocks(i,j) = new SparseMatrix(bI, NULL, NULL, nr, nc);
      }
   }

   for (int gr = 0; gr < height; gr++)
   {
      int bi = gr/nr, i = gr%nr + 1;
      if (Finalized())
      {
         for (int j = I[gr]; j < I[gr+1]; j++)
         {
            if (A[j] != 0.0)
            {
               blocks(bi, J[j]/nc)->I[i]++;
            }
         }
      }
      else
      {
         for (RowNode *n_p = Rows[gr]; n_p != NULL; n_p = n_p->Prev)
         {
            if (n_p->Value != 0.0)
            {
               blocks(bi, n_p->Column/nc)->I[i]++;
            }
         }
      }
   }

   for (int j = 0; j < bc; j++)
   {
      for (int i = 0; i < br; i++)
      {
         SparseMatrix &b = *blocks(i,j);
         int nnz = 0, rs;
         for (int k = 1; k <= nr; k++)
         {
            rs = b.I[k], b.I[k] = nnz, nnz += rs;
         }
         b.J.New(nnz);
         b.A.New(nnz);
      }
   }

   for (int gr = 0; gr < height; gr++)
   {
      int bi = gr/nr, i = gr%nr + 1;
      if (Finalized())
      {
         for (int j = I[gr]; j < I[gr+1]; j++)
         {
            if (A[j] != 0.0)
            {
               SparseMatrix &b = *blocks(bi, J[j]/nc);
               b.J[b.I[i]] = J[j] % nc;
               b.A[b.I[i]] = A[j];
               b.I[i]++;
            }
         }
      }
      else
      {
         for (RowNode *n_p = Rows[gr]; n_p != NULL; n_p = n_p->Prev)
         {
            if (n_p->Value != 0.0)
            {
               SparseMatrix &b = *blocks(bi, n_p->Column/nc);
               b.J[b.I[i]] = n_p->Column % nc;
               b.A[b.I[i]] = n_p->Value;
               b.I[i]++;
            }
         }
      }
   }
}

real_t SparseMatrix::IsSymmetric() const
{
   if (height != width)
   {
      return infinity();
   }

   real_t symm = 0.0;
   if (Empty())
   {
      // return 0.0;
   }
   else if (Finalized())
   {
      for (int i = 1; i < height; i++)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            if (J[j] < i)
            {
               symm = std::max(symm, std::abs(A[j]-(*this)(J[j],i)));
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *node_p = Rows[i]; node_p != NULL; node_p = node_p->Prev)
         {
            int col = node_p->Column;
            if (col < i)
            {
               symm = std::max(symm, std::abs(node_p->Value-(*this)(col,i)));
            }
         }
      }
   }
   return symm;
}

void SparseMatrix::Symmetrize()
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   int i, j;
   for (i = 1; i < height; i++)
   {
      for (j = I[i]; j < I[i+1]; j++)
      {
         if (J[j] < i)
         {
            A[j] += (*this)(J[j],i);
            A[j] *= 0.5;
            (*this)(J[j],i) = A[j];
         }
      }
   }
}

int SparseMatrix::NumNonZeroElems() const
{
   if (Finalized())
   {
      HostReadI();
      return I[height];
   }
   else
   {
      int nnz = 0;

      for (int i = 0; i < height; i++)
      {
         for (RowNode *node_p = Rows[i]; node_p != NULL; node_p = node_p->Prev)
         {
            nnz++;
         }
      }

      return nnz;
   }
}

real_t SparseMatrix::MaxNorm() const
{
   real_t m = 0.0;

   if (A)
   {
      int nnz = I[height];
      for (int j = 0; j < nnz; j++)
      {
         m = std::max(m, std::abs(A[j]));
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *n_p = Rows[i]; n_p != NULL; n_p = n_p->Prev)
         {
            m = std::max(m, std::abs(n_p->Value));
         }
      }
   }
   return m;
}

int SparseMatrix::CountSmallElems(real_t tol) const
{
   int counter = 0;

   if (A)
   {
      const int nz = I[height];
      const real_t *Ap = A;

      for (int i = 0; i < nz; i++)
      {
         counter += (std::abs(Ap[i]) <= tol);
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            counter += (std::abs(aux->Value) <= tol);
         }
      }
   }

   return counter;
}

int SparseMatrix::CheckFinite() const
{
   if (Empty())
   {
      return 0;
   }
   else if (Finalized())
   {
      return mfem::CheckFinite(A, I[height]);
   }
   else
   {
      int counter = 0;
      for (int i = 0; i < height; i++)
      {
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            counter += !IsFinite(aux->Value);
         }
      }
      return counter;
   }
}

MatrixInverse *SparseMatrix::Inverse() const
{
   return NULL;
}

void SparseMatrix::EliminateRow(int row, const real_t sol, Vector &rhs)
{
   RowNode *aux;

   MFEM_ASSERT(row < height && row >= 0,
               "Row " << row << " not in matrix of height " << height);

   MFEM_VERIFY(!Finalized(), "Matrix must NOT be finalized.");

   for (aux = Rows[row]; aux != NULL; aux = aux->Prev)
   {
      rhs(aux->Column) -= sol * aux->Value;
      aux->Value = 0.0;
   }
}

void SparseMatrix::EliminateRow(int row, DiagonalPolicy dpolicy)
{
   RowNode *aux;

   MFEM_ASSERT(row < height && row >= 0,
               "Row " << row << " not in matrix of height " << height);
   MFEM_ASSERT(dpolicy != DIAG_KEEP, "Diagonal policy must not be DIAG_KEEP");
   MFEM_ASSERT(dpolicy != DIAG_ONE || height == width,
               "if dpolicy == DIAG_ONE, matrix must be square, not height = "
               << height << ",  width = " << width);

   if (Rows == NULL)
   {
      for (int i=I[row]; i < I[row+1]; ++i)
      {
         A[i]=0.0;
      }
   }
   else
   {
      for (aux = Rows[row]; aux != NULL; aux = aux->Prev)
      {
         aux->Value = 0.0;
      }
   }

   if (dpolicy == DIAG_ONE)
   {
      SearchRow(row, row) = 1.;
   }
}

void SparseMatrix::EliminateCol(int col, DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(col < width && col >= 0,
               "Col " << col << " not in matrix of width " << width);
   MFEM_ASSERT(dpolicy != DIAG_KEEP, "Diagonal policy must not be DIAG_KEEP");
   MFEM_ASSERT(dpolicy != DIAG_ONE || height == width,
               "if dpolicy == DIAG_ONE, matrix must be square, not height = "
               << height << ",  width = " << width);

   if (Rows == NULL)
   {
      const int nnz = I[height];
      for (int jpos = 0; jpos != nnz; ++jpos)
      {
         if (J[jpos] == col)
         {
            A[jpos] = 0.0;
         }
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            if (aux->Column == col)
            {
               aux->Value = 0.0;
               break;
            }
         }
      }
   }

   if (dpolicy == DIAG_ONE)
   {
      SearchRow(col, col) = 1.0;
   }
}

void SparseMatrix::EliminateCols(const Array<int> &cols, const Vector *x,
                                 Vector *b)
{
   if (Rows == NULL)
   {
      for (int i = 0; i < height; i++)
      {
         for (int jpos = I[i]; jpos != I[i+1]; ++jpos)
         {
            if (cols[ J[jpos]])
            {
               if (x && b)
               {
                  (*b)(i) -= A[jpos] * (*x)( J[jpos] );
               }
               A[jpos] = 0.0;
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            if (cols[aux -> Column])
            {
               if (x && b)
               {
                  (*b)(i) -= aux -> Value * (*x)(aux -> Column);
               }
               aux->Value = 0.0;
            }
         }
      }
   }
}

void SparseMatrix::EliminateCols(const Array<int> &col_marker, SparseMatrix &Ae)
{
   if (Rows)
   {
      RowNode *nd;
      for (int row = 0; row < height; row++)
      {
         for (nd = Rows[row]; nd != NULL; nd = nd->Prev)
         {
            if (col_marker[nd->Column])
            {
               Ae.Add(row, nd->Column, nd->Value);
               nd->Value = 0.0;
            }
         }
      }
   }
   else
   {
      for (int row = 0; row < height; row++)
      {
         for (int j = I[row]; j < I[row+1]; j++)
         {
            if (col_marker[J[j]])
            {
               Ae.Add(row, J[j], A[j]);
               A[j] = 0.0;
            }
         }
      }
   }
}


void SparseMatrix::EliminateRowCol(int rc, const real_t sol, Vector &rhs,
                                   DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);
   HostReadWriteI();
   HostReadWriteJ();
   HostReadWriteData();

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         const int col = J[j];
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_KEEP:
                  rhs(rc) = A[j] * sol;
                  break;
               case DIAG_ONE:
                  A[j] = 1.0;
                  rhs(rc) = sol;
                  break;
               case DIAG_ZERO:
                  A[j] = 0.;
                  rhs(rc) = 0.;
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowCol () #2");
                  break;
            }
         }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
            {
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol () #3");
               }
               else if (J[k] == rc)
               {
                  rhs(col) -= sol * A[k];
                  A[k] = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      for (RowNode *aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         const int col = aux->Column;
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_KEEP:
                  rhs(rc) = aux->Value * sol;
                  break;
               case DIAG_ONE:
                  aux->Value = 1.0;
                  rhs(rc) = sol;
                  break;
               case DIAG_ZERO:
                  aux->Value = 0.;
                  rhs(rc) = 0.;
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowCol () #4");
                  break;
            }
         }
         else
         {
            aux->Value = 0.0;
            for (RowNode *node = Rows[col]; 1; node = node->Prev)
            {
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol () #5");
               }
               else if (node->Column == rc)
               {
                  rhs(col) -= sol * node->Value;
                  node->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
}

void SparseMatrix::EliminateRowColMultipleRHS(int rc, const Vector &sol,
                                              DenseMatrix &rhs,
                                              DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);
   MFEM_ASSERT(sol.Size() == rhs.Width(), "solution size (" << sol.Size()
               << ") must match rhs width (" << rhs.Width() << ")");

   const int num_rhs = rhs.Width();
   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         const int col = J[j];
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_KEEP:
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = A[j] * sol(r);
                  }
                  break;
               case DIAG_ONE:
                  A[j] = 1.0;
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = sol(r);
                  }
                  break;
               case DIAG_ZERO:
                  A[j] = 0.;
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = 0.;
                  }
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #3");
                  break;
            }
         }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
            {
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #4");
               }
               else if (J[k] == rc)
               {
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(col,r) -= sol(r) * A[k];
                  }
                  A[k] = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      for (RowNode *aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         const int col = aux->Column;
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_KEEP:
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = aux->Value * sol(r);
                  }
                  break;
               case DIAG_ONE:
                  aux->Value = 1.0;
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = sol(r);
                  }
                  break;
               case DIAG_ZERO:
                  aux->Value = 0.;
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(rc,r) = 0.;
                  }
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #5");
                  break;
            }
         }
         else
         {
            aux->Value = 0.0;
            for (RowNode *node = Rows[col]; 1; node = node->Prev)
            {
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #6");
               }
               else if (node->Column == rc)
               {
                  for (int r = 0; r < num_rhs; r++)
                  {
                     rhs(col,r) -= sol(r) * node->Value;
                  }
                  node->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
}

void SparseMatrix::EliminateRowCol(int rc, DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);

   if (Rows == NULL)
   {
      const auto &II = this->I; // only use const access for I
      const auto &JJ = this->J; // only use const access for J
      for (int j = II[rc]; j < II[rc+1]; j++)
      {
         const int col = JJ[j];
         if (col == rc)
         {
            if (dpolicy == DIAG_ONE)
            {
               A[j] = 1.0;
            }
            else if (dpolicy == DIAG_ZERO)
            {
               A[j] = 0.0;
            }
         }
         else
         {
            A[j] = 0.0;
            for (int k = II[col]; 1; k++)
            {
               if (k == II[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #2");
               }
               else if (JJ[k] == rc)
               {
                  A[k] = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      RowNode *aux, *node;

      for (aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         const int col = aux->Column;
         if (col == rc)
         {
            if (dpolicy == DIAG_ONE)
            {
               aux->Value = 1.0;
            }
            else if (dpolicy == DIAG_ZERO)
            {
               aux->Value = 0.;
            }
         }
         else
         {
            aux->Value = 0.0;
            for (node = Rows[col]; 1; node = node->Prev)
            {
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #3");
               }
               else if (node->Column == rc)
               {
                  node->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
}

// This is almost identical to EliminateRowCol(int, int), except for
// the A[j] = value; and aux->Value = value; lines.
void SparseMatrix::EliminateRowColDiag(int rc, real_t value)
{
   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         const int col = J[j];
         if (col == rc)
         {
            A[j] = value;
         }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
            {
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #2");
               }
               else if (J[k] == rc)
               {
                  A[k] = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      RowNode *aux, *node;

      for (aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         const int col = aux->Column;
         if (col == rc)
         {
            aux->Value = value;
         }
         else
         {
            aux->Value = 0.0;
            for (node = Rows[col]; 1; node = node->Prev)
            {
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #3");
               }
               else if (node->Column == rc)
               {
                  node->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
}

void SparseMatrix::EliminateRowCol(int rc, SparseMatrix &Ae,
                                   DiagonalPolicy dpolicy)
{
   if (Rows)
   {
      RowNode *nd, *nd2;
      for (nd = Rows[rc]; nd != NULL; nd = nd->Prev)
      {
         const int col = nd->Column;
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_ONE:
                  Ae.Add(rc, rc, nd->Value - 1.0);
                  nd->Value = 1.0;
                  break;
               case DIAG_ZERO:
                  Ae.Add(rc, rc, nd->Value);
                  nd->Value = 0.;
                  break;
               case DIAG_KEEP:
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowCol #1");
                  break;
            }
         }
         else
         {
            Ae.Add(rc, col, nd->Value);
            nd->Value = 0.0;
            for (nd2 = Rows[col]; 1; nd2 = nd2->Prev)
            {
               if (nd2 == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol #2");
               }
               else if (nd2->Column == rc)
               {
                  Ae.Add(col, rc, nd2->Value);
                  nd2->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         const int col = J[j];
         if (col == rc)
         {
            switch (dpolicy)
            {
               case DIAG_ONE:
                  Ae.Add(rc, rc, A[j] - 1.0);
                  A[j] = 1.0;
                  break;
               case DIAG_ZERO:
                  Ae.Add(rc, rc, A[j]);
                  A[j] = 0.;
                  break;
               case DIAG_KEEP:
                  break;
               default:
                  mfem_error("SparseMatrix::EliminateRowCol #3");
                  break;
            }
         }
         else
         {
            Ae.Add(rc, col, A[j]);
            A[j] = 0.0;
            for (int k = I[col]; true; k++)
            {
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol #4");
               }
               else if (J[k] == rc)
               {
                  Ae.Add(col, rc, A[k]);
                  A[k] = 0.0;
                  break;
               }
            }
         }
      }
   }
}

void SparseMatrix::EliminateBC(const Array<int> &ess_dofs,
                               DiagonalPolicy diag_policy)
{
   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.Read();
   const auto dI = ReadI();
   const auto dJ = ReadJ();
   auto dA = ReadWriteData();

   mfem::forall(n_ess_dofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int idof = ess_dofs_d[i];
      for (int j=dI[idof]; j<dI[idof+1]; ++j)
      {
         const int jdof = dJ[j];
         if (jdof != idof)
         {
            dA[j] = 0.0;
            for (int k=dI[jdof]; k<dI[jdof+1]; ++k)
            {
               if (dJ[k] == idof)
               {
                  dA[k] = 0.0;
                  break;
               }
            }
         }
         else
         {
            if (diag_policy == DiagonalPolicy::DIAG_ONE)
            {
               dA[j] = 1.0;
            }
            else if (diag_policy == DiagonalPolicy::DIAG_ZERO)
            {
               dA[j] = 0.0;
            }
            // else (diag_policy == DiagonalPolicy::DIAG_KEEP)
         }
      }
   });
}

void SparseMatrix::SetDiagIdentity()
{
   for (int i = 0; i < height; i++)
   {
      if (I[i+1] == I[i]+1 && fabs(A[I[i]]) < 1e-16)
      {
         A[I[i]] = 1.0;
      }
   }
}

void SparseMatrix::EliminateZeroRows(const real_t threshold)
{
   for (int i = 0; i < height; i++)
   {
      real_t zero = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         zero += fabs(A[j]);
      }
      if (zero <= threshold)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            A[j] = (J[j] == i) ? 1.0 : 0.0;
         }
      }
   }
}

void SparseMatrix::Gauss_Seidel_forw(const Vector &x, Vector &y) const
{
   if (!Finalized())
   {
      real_t *yp = y.GetData();
      const real_t *xp = x.GetData();
      RowNode *diag_p, *n_p, **R = Rows;

      const int s = height;
      for (int i = 0; i < s; i++)
      {
         real_t sum = 0.0;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
         {
            const int c = n_p->Column;
            if (c == i)
            {
               diag_p = n_p;
            }
            else
            {
               sum += n_p->Value * yp[c];
            }
         }

         if (diag_p != NULL && diag_p->Value != 0.0)
         {
            yp[i] = (xp[i] - sum) / diag_p->Value;
         }
         else if (xp[i] == sum)
         {
            yp[i] = sum;
         }
         else
         {
            mfem_error("SparseMatrix::Gauss_Seidel_forw()");
         }
      }
   }
   else
   {
      const int s = height;
      const int nnz = J.Capacity();
      const int *Ip = HostRead(I, s+1);
      const int *Jp = HostRead(J, nnz);
      const real_t *Ap = HostRead(A, nnz);
      real_t *yp = y.HostReadWrite();
      const real_t *xp = x.HostRead();

      for (int i = 0, j = Ip[0]; i < s; i++)
      {
         const int end = Ip[i+1];
         real_t sum = 0.0;
         int d = -1;
         for ( ; j < end; j++)
         {
            const int c = Jp[j];
            if (c == i)
            {
               d = j;
            }
            else
            {
               sum += Ap[j] * yp[c];
            }
         }

         if (d >= 0 && Ap[d] != 0.0)
         {
            yp[i] = (xp[i] - sum) / Ap[d];
         }
         else if (xp[i] == sum)
         {
            yp[i] = sum;
         }
         else
         {
            mfem_error("SparseMatrix::Gauss_Seidel_forw(...) #2");
         }
      }
   }
}

void SparseMatrix::Gauss_Seidel_back(const Vector &x, Vector &y) const
{
   if (!Finalized())
   {
      real_t *yp = y.GetData();
      const real_t *xp = x.GetData();
      RowNode *diag_p, *n_p, **R = Rows;

      for (int i = height-1; i >= 0; i--)
      {
         real_t sum = 0.;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
         {
            const int c = n_p->Column;
            if (c == i)
            {
               diag_p = n_p;
            }
            else
            {
               sum += n_p->Value * yp[c];
            }
         }

         if (diag_p != NULL && diag_p->Value != 0.0)
         {
            yp[i] = (xp[i] - sum) / diag_p->Value;
         }
         else if (xp[i] == sum)
         {
            yp[i] = sum;
         }
         else
         {
            mfem_error("SparseMatrix::Gauss_Seidel_back()");
         }
      }
   }
   else
   {
      const int s = height;
      const int nnz = J.Capacity();
      const int *Ip = HostRead(I, s+1);
      const int *Jp = HostRead(J, nnz);
      const real_t *Ap = HostRead(A, nnz);
      real_t *yp = y.HostReadWrite();
      const real_t *xp = x.HostRead();

      for (int i = s-1, j = Ip[s]-1; i >= 0; i--)
      {
         const int beg = Ip[i];
         real_t sum = 0.;
         int d = -1;
         for ( ; j >= beg; j--)
         {
            const int c = Jp[j];
            if (c == i)
            {
               d = j;
            }
            else
            {
               sum += Ap[j] * yp[c];
            }
         }

         if (d >= 0 && Ap[d] != 0.0)
         {
            yp[i] = (xp[i] - sum) / Ap[d];
         }
         else if (xp[i] == sum)
         {
            yp[i] = sum;
         }
         else
         {
            mfem_error("SparseMatrix::Gauss_Seidel_back(...) #2");
         }
      }
   }
}

real_t SparseMatrix::GetJacobiScaling() const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   real_t sc = 1.0;
   for (int i = 0; i < height; i++)
   {
      int d = -1;
      real_t norm = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         if (J[j] == i)
         {
            d = j;
         }
         norm += fabs(A[j]);
      }
      if (d >= 0 && A[d] != 0.0)
      {
         real_t a = 1.8 * fabs(A[d]) / norm;
         if (a < sc)
         {
            sc = a;
         }
      }
      else
      {
         mfem_error("SparseMatrix::GetJacobiScaling() #2");
      }
   }
   return sc;
}

void SparseMatrix::Jacobi(const Vector &b, const Vector &x0, Vector &x1,
                          real_t sc, bool use_abs_diag) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < height; i++)
   {
      int d = -1;
      real_t sum = b(i);
      for (int j = I[i]; j < I[i+1]; j++)
      {
         if (J[j] == i)
         {
            d = j;
         }
         else
         {
            sum -= A[j] * x0(J[j]);
         }
      }
      if (d >= 0 && A[d] != 0.0)
      {
         const real_t diag = (use_abs_diag) ? fabs(A[d]) : A[d];
         x1(i) = sc * (sum / diag) + (1.0 - sc) * x0(i);
      }
      else
      {
         mfem_error("SparseMatrix::Jacobi(...) #2");
      }
   }
}

void SparseMatrix::DiagScale(const Vector &b, Vector &x,
                             real_t sc, bool use_abs_diag) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   const int H = height;
   const int nnz = J.Capacity();
   const bool use_dev = b.UseDevice() || x.UseDevice();

   const auto Ap = Read(A, nnz, use_dev);
   const auto Ip = Read(I, height+1, use_dev);
   const auto Jp = Read(J, nnz, use_dev);

   const auto bp = b.Read(use_dev);
   auto xp = x.Write(use_dev);

   mfem::forall_switch(use_dev, H, [=] MFEM_HOST_DEVICE (int i)
   {
      const int end = Ip[i+1];
      for (int j = Ip[i]; true; j++)
      {
         if (j == end)
         {
            MFEM_ABORT_KERNEL("Diagonal not found in SparseMatrix::DiagScale");
         }
         if (Jp[j] == i)
         {
            const real_t diag = (use_abs_diag) ? fabs(Ap[j]) : Ap[j];
            if (diag == 0.0)
            {
               MFEM_ABORT_KERNEL("Zero diagonal in SparseMatrix::DiagScale");
            }
            xp[i] = sc * bp[i] / diag;
            break;
         }
      }
   });
}

template <bool useFabs>
static void JacobiDispatch(const Vector &b, const Vector &x0, Vector &x1,
                           const Memory<int> &I, const Memory<int> &J,
                           const Memory<real_t> &A, const int height,
                           const real_t sc)
{
   const bool useDevice = b.UseDevice() || x0.UseDevice() || x1.UseDevice();

   const auto bp  = b.Read(useDevice);
   const auto x0p = x0.Read(useDevice);
   auto       x1p = x1.Write(useDevice);

   const auto Ip = Read(I, height+1, useDevice);
   const auto Jp = Read(J, J.Capacity(), useDevice);
   const auto Ap = Read(A, J.Capacity(), useDevice);

   mfem::forall_switch(useDevice, height, [=] MFEM_HOST_DEVICE (int i)
   {
      real_t resi = bp[i], norm = 0.0;
      for (int j = Ip[i]; j < Ip[i+1]; j++)
      {
         resi -= Ap[j] * x0p[Jp[j]];
         if (useFabs)
         {
            norm += fabs(Ap[j]);
         }
         else
         {
            norm += Ap[j];
         }
      }
      if (norm > 0.0)
      {
         x1p[i] = x0p[i] + sc * resi / norm;
      }
      else
      {
         if (useFabs)
         {
            MFEM_ABORT_KERNEL("L1 norm of row is zero.");
         }
         else
         {
            MFEM_ABORT_KERNEL("sum of row is zero.");
         }
      }
   });
}

void SparseMatrix::Jacobi2(const Vector &b, const Vector &x0, Vector &x1,
                           real_t sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");
   JacobiDispatch<true>(b,x0,x1,I,J,A,height,sc);
}

void SparseMatrix::Jacobi3(const Vector &b, const Vector &x0, Vector &x1,
                           real_t sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");
   JacobiDispatch<false>(b,x0,x1,I,J,A,height,sc);
}

void SparseMatrix::AddSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                const DenseMatrix &subm, int skip_zeros)
{
   int i, j, gi, gj, s, t;
   real_t a;

   if (Finalized())
   {
      HostReadI();
      HostReadJ();
      HostReadWriteData();
   }

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) { gi = -1-gi, s = -1; }
      else { s = 1; }
      MFEM_ASSERT(gi < height,
                  "Trying to insert a row " << gi << " outside the matrix height "
                  << height);
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to insert a column " << gj << " outside the matrix width "
                     << width);
         a = subm(i, j);
         if (skip_zeros && a == 0.0)
         {
            // Skip assembly of zero elements if either:
            // (i) user specified to skip zeros regardless of symmetry, or
            // (ii) symmetry is not broken.
            if (skip_zeros == 2 || &rows != &cols || subm(j, i) == 0.0)
            {
               continue;
            }
         }
         if (t < 0) { a = -a; }
         _Add_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::Set(const int i, const int j, const real_t val)
{
   real_t a = val;
   int gi, gj, s, t;

   if ((gi=i) < 0) { gi = -1-gi, s = -1; }
   else { s = 1; }
   MFEM_ASSERT(gi < height,
               "Trying to set a row " << gi << " outside the matrix height "
               << height);
   if ((gj=j) < 0) { gj = -1-gj, t = -s; }
   else { t = s; }
   MFEM_ASSERT(gj < width,
               "Trying to set a column " << gj << " outside the matrix width "
               << width);
   if (t < 0) { a = -a; }
   _Set_(gi, gj, a);
}

void SparseMatrix::Add(const int i, const int j, const real_t val)
{
   int gi, gj, s, t;
   real_t a = val;

   if ((gi=i) < 0) { gi = -1-gi, s = -1; }
   else { s = 1; }
   MFEM_ASSERT(gi < height,
               "Trying to insert a row " << gi << " outside the matrix height "
               << height);
   if ((gj=j) < 0) { gj = -1-gj, t = -s; }
   else { t = s; }
   MFEM_ASSERT(gj < width,
               "Trying to insert a column " << gj << " outside the matrix width "
               << width);
   if (t < 0) { a = -a; }
   _Add_(gi, gj, a);
}

void SparseMatrix::SetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                const DenseMatrix &subm, int skip_zeros)
{
   int i, j, gi, gj, s, t;
   real_t a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) { gi = -1-gi, s = -1; }
      else { s = 1; }
      MFEM_ASSERT(gi < height,
                  "Trying to set a row " << gi << " outside the matrix height "
                  << height);
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         a = subm(i, j);
         if (skip_zeros && a == 0.0)
         {
            // Skip assembly of zero elements if either:
            // (i) user specified to skip zeros regardless of symmetry, or
            // (ii) symmetry is not broken.
            if (skip_zeros == 2 || &rows != &cols || subm(j, i) == 0.0)
            {
               continue;
            }
         }
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to set a column " << gj << " outside the matrix width "
                     << width);
         if (t < 0) { a = -a; }
         _Set_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::SetSubMatrixTranspose(const Array<int> &rows,
                                         const Array<int> &cols,
                                         const DenseMatrix &subm,
                                         int skip_zeros)
{
   int i, j, gi, gj, s, t;
   real_t a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) { gi = -1-gi, s = -1; }
      else { s = 1; }
      MFEM_ASSERT(gi < height,
                  "Trying to set a row " << gi << " outside the matrix height "
                  << height);
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         a = subm(j, i);
         if (skip_zeros && a == 0.0)
         {
            // Skip assembly of zero elements if either:
            // (i) user specified to skip zeros regardless of symmetry, or
            // (ii) symmetry is not broken.
            if (skip_zeros == 2 || &rows != &cols || subm(j, i) == 0.0)
            {
               continue;
            }
         }
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to set a column " << gj << " outside the matrix width "
                     << width);
         if (t < 0) { a = -a; }
         _Set_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::GetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                DenseMatrix &subm) const
{
   int i, j, gi, gj, s, t;
   real_t a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) { gi = -1-gi, s = -1; }
      else { s = 1; }
      MFEM_ASSERT(gi < height,
                  "Trying to read a row " << gi << " outside the matrix height "
                  << height);
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to read a column " << gj << " outside the matrix width "
                     << width);
         a = _Get_(gj);
         subm(i, j) = (t < 0) ? (-a) : (a);
      }
      ClearColPtr();
   }
}

bool SparseMatrix::RowIsEmpty(const int row) const
{
   int gi;

   if ((gi=row) < 0)
   {
      gi = -1-gi;
   }
   MFEM_ASSERT(gi < height,
               "Trying to query a row " << gi << " outside the matrix height "
               << height);
   if (Rows)
   {
      return (Rows[gi] == NULL);
   }
   else
   {
      return (I[gi] == I[gi+1]);
   }
}

int SparseMatrix::GetRow(const int row, Array<int> &cols, Vector &srow) const
{
   RowNode *n;
   int j, gi;

   if ((gi=row) < 0) { gi = -1-gi; }
   MFEM_ASSERT(gi < height,
               "Trying to read a row " << gi << " outside the matrix height "
               << height);
   if (Rows)
   {
      for (n = Rows[gi], j = 0; n; n = n->Prev)
      {
         j++;
      }
      cols.SetSize(j);
      srow.SetSize(j);
      for (n = Rows[gi], j = 0; n; n = n->Prev, j++)
      {
         cols[j] = n->Column;
         srow(j) = n->Value;
      }
      if (row < 0)
      {
         srow.Neg();
      }

      return 0;
   }
   else
   {
      j = I[gi];
      cols.MakeRef(const_cast<int*>((const int*)J) + j, I[gi+1]-j);
      srow.NewDataAndSize(
         const_cast<real_t*>((const real_t*)A) + j, cols.Size());
      MFEM_ASSERT(row >= 0, "Row not valid: " << row << ", height: " << height);
      return 1;
   }
}

void SparseMatrix::SetRow(const int row, const Array<int> &cols,
                          const Vector &srow)
{
   int gi, gj, s, t;
   real_t a;

   if ((gi=row) < 0) { gi = -1-gi, s = -1; }
   else { s = 1; }
   MFEM_ASSERT(gi < height,
               "Trying to set a row " << gi << " outside the matrix height "
               << height);

   if (!Finalized())
   {
      SetColPtr(gi);
      for (int j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to set a column " << gj << " outside the matrix"
                     " width " << width);
         a = srow(j);
         if (t < 0) { a = -a; }
         _Set_(gj, a);
      }
      ClearColPtr();
   }
   else
   {
      MFEM_ASSERT(cols.Size() == RowSize(gi), "");
      MFEM_ASSERT(cols.Size() == srow.Size(), "");

      for (int i = I[gi], j = 0; j < cols.Size(); j++, i++)
      {
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to set a column " << gj << " outside the matrix"
                     " width " << width);

         J[i] = gj;
         A[i] = srow[j] * t;
      }
   }
}

void SparseMatrix::AddRow(const int row, const Array<int> &cols,
                          const Vector &srow)
{
   int j, gi, gj, s, t;
   real_t a;

   MFEM_VERIFY(!Finalized(), "Matrix must NOT be finalized.");

   if ((gi=row) < 0) { gi = -1-gi, s = -1; }
   else { s = 1; }
   MFEM_ASSERT(gi < height,
               "Trying to insert a row " << gi << " outside the matrix height "
               << height);
   SetColPtr(gi);
   for (j = 0; j < cols.Size(); j++)
   {
      if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
      else { t = s; }
      MFEM_ASSERT(gj < width,
                  "Trying to insert a column " << gj << " outside the matrix width "
                  << width);
      a = srow(j);
      if (a == 0.0)
      {
         continue;
      }
      if (t < 0) { a = -a; }
      _Add_(gj, a);
   }
   ClearColPtr();
}

void SparseMatrix::ScaleRow(const int row, const real_t scale)
{
   int i;

   if ((i=row) < 0)
   {
      i = -1-i;
   }
   if (Rows != NULL)
   {
      RowNode *aux;

      for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
      {
         aux -> Value *= scale;
      }
   }
   else
   {
      int j, end = I[i+1];

      for (j = I[i]; j < end; j++)
      {
         A[j] *= scale;
      }
   }
}

void SparseMatrix::ScaleRows(const Vector & sl)
{
   real_t scale;
   if (Rows != NULL)
   {
      RowNode *aux;
      for (int i=0; i < height; ++i)
      {
         scale = sl(i);
         for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
         {
            aux -> Value *= scale;
         }
      }
   }
   else
   {
      int j, end;

      for (int i=0; i < height; ++i)
      {
         end = I[i+1];
         scale = sl(i);
         for (j = I[i]; j < end; j++)
         {
            A[j] *= scale;
         }
      }
   }
}

void SparseMatrix::ScaleColumns(const Vector & sr)
{
   if (Rows != NULL)
   {
      RowNode *aux;
      for (int i=0; i < height; ++i)
      {
         for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
         {
            aux -> Value *= sr(aux->Column);
         }
      }
   }
   else
   {
      int j, end;

      for (int i=0; i < height; ++i)
      {
         end = I[i+1];
         for (j = I[i]; j < end; j++)
         {
            A[j] *= sr(J[j]);
         }
      }
   }
}

SparseMatrix &SparseMatrix::operator+=(const SparseMatrix &B)
{
   MFEM_ASSERT(height == B.height && width == B.width,
               "Mismatch of this matrix size and rhs.  This height = "
               << height << ", width = " << width << ", B.height = "
               << B.height << ", B.width = " << B.width);

   for (int i = 0; i < height; i++)
   {
      SetColPtr(i);
      if (B.Rows)
      {
         for (RowNode *aux = B.Rows[i]; aux != NULL; aux = aux->Prev)
         {
            _Add_(aux->Column, aux->Value);
         }
      }
      else
      {
         for (int j = B.I[i]; j < B.I[i+1]; j++)
         {
            _Add_(B.J[j], B.A[j]);
         }
      }
      ClearColPtr();
   }

   return (*this);
}

void SparseMatrix::Add(const real_t a, const SparseMatrix &B)
{
   for (int i = 0; i < height; i++)
   {
      B.SetColPtr(i);
      if (Rows)
      {
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
         {
            np->Value += a * B._Get_(np->Column);
         }
      }
      else
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            A[j] += a * B._Get_(J[j]);
         }
      }
      B.ClearColPtr();
   }
}

SparseMatrix &SparseMatrix::operator=(real_t a)
{
   if (Rows == NULL)
   {
      const int nnz = J.Capacity();
      real_t *h_A = HostWrite(A, nnz);
      for (int i = 0; i < nnz; i++)
      {
         h_A[i] = a;
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
         {
            node_p -> Value = a;
         }
      }
   }

   return (*this);
}

SparseMatrix &SparseMatrix::operator*=(real_t a)
{
   if (Rows == NULL)
   {
      for (int i = 0, nnz = I[height]; i < nnz; i++)
      {
         A[i] *= a;
      }
   }
   else
   {
      for (int i = 0; i < height; i++)
      {
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
         {
            node_p -> Value *= a;
         }
      }
   }

   return (*this);
}

void SparseMatrix::Print(std::ostream & os, int width_) const
{
   int i, j;

   if (A.Empty())
   {
      RowNode *nd;
      for (i = 0; i < height; i++)
      {
         os << "[row " << i << "]\n";
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++)
         {
            os << " (" << nd->Column << "," << nd->Value << ")";
            if ( !((j+1) % width_) )
            {
               os << '\n';
            }
         }
         if (j % width_)
         {
            os << '\n';
         }
      }
      return;
   }

   // HostRead forces synchronization
   HostReadI();
   HostReadJ();
   HostReadData();
   for (i = 0; i < height; i++)
   {
      os << "[row " << i << "]\n";
      for (j = I[i]; j < I[i+1]; j++)
      {
         os << " (" << J[j] << "," << A[j] << ")";
         if ( !((j+1-I[i]) % width_) )
         {
            os << '\n';
         }
      }
      if ((j-I[i]) % width_)
      {
         os << '\n';
      }
   }
}

void SparseMatrix::PrintMatlab(std::ostream & os) const
{
   os << "% size " << height << " " << width << "\n";
   os << "% Non Zeros " << NumNonZeroElems() << "\n";

   int i, j;
   ios::fmtflags old_fmt = os.flags();
   os.setf(ios::scientific);
   std::streamsize old_prec = os.precision(14);

   if (A.Empty())
   {
      RowNode *nd;
      for (i = 0; i < height; i++)
      {
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++)
         {
            os << i+1 << " " << nd->Column+1 << " " << nd->Value << '\n';
         }
      }
   }
   else
   {
      // HostRead forces synchronization
      HostReadI();
      HostReadJ();
      HostReadData();
      for (i = 0; i < height; i++)
      {
         for (j = I[i]; j < I[i+1]; j++)
         {
            os << i+1 << " " << J[j]+1 << " " << A[j] << '\n';
         }
      }
   }
   // Write a zero entry at (m,n) to make sure MATLAB doesn't shrink the matrix
   os << height << " " << width << " 0.0\n";
   os.precision(old_prec);
   os.flags(old_fmt);
}

void SparseMatrix::PrintMathematica(std::ostream & os) const
{
   int i, j;
   ios::fmtflags old_fmt = os.flags();
   os.setf(ios::scientific);
   std::streamsize old_prec = os.precision(14);

   os << "(* Read file into Mathematica using: "
      << "myMat = Get[\"this_file_name\"] *)\n";
   os << "SparseArray[";

   if (A == NULL)
   {
      RowNode *nd;
      int c = 0;
      os << "{\n";
      for (i = 0; i < height; i++)
      {
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++, c++)
         {
            os << "{"<< i+1 << ", " << nd->Column+1
               << "} -> Internal`StringToMReal[\"" << nd->Value << "\"]";
            if (c < NumNonZeroElems() - 1) { os << ","; }
            os << '\n';
         }
      }
      os << "}\n";
   }
   else
   {
      // HostRead forces synchronization
      HostReadI();
      HostReadJ();
      HostReadData();
      int c = 0;
      os << "{\n";
      for (i = 0; i < height; i++)
      {
         for (j = I[i]; j < I[i+1]; j++, c++)
         {
            os << "{" << i+1 << ", " << J[j]+1
               << "} -> Internal`StringToMReal[\"" << A[j] << "\"]";
            if (c < NumNonZeroElems() - 1) { os << ","; }
            os << '\n';
         }
      }
      os << "}";
   }

   os << ",{" << height << "," << width << "}]\n";

   os.precision(old_prec);
   os.flags(old_fmt);
}

void SparseMatrix::PrintMM(std::ostream & os) const
{
   int i, j;
   ios::fmtflags old_fmt = os.flags();
   os.setf(ios::scientific);
   std::streamsize old_prec = os.precision(14);

   os << "%%MatrixMarket matrix coordinate real general" << '\n'
      << "% Generated by MFEM" << '\n';

   os << height << " " << width << " " << NumNonZeroElems() << '\n';

   if (A.Empty())
   {
      RowNode *nd;
      for (i = 0; i < height; i++)
      {
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++)
         {
            os << i+1 << " " << nd->Column+1 << " " << nd->Value << '\n';
         }
      }
   }
   else
   {
      // HostRead forces synchronization
      HostReadI();
      HostReadJ();
      HostReadData();
      for (i = 0; i < height; i++)
      {
         for (j = I[i]; j < I[i+1]; j++)
         {
            os << i+1 << " " << J[j]+1 << " " << A[j] << '\n';
         }
      }
   }
   os.precision(old_prec);
   os.flags(old_fmt);
}

void SparseMatrix::PrintCSR(std::ostream & os) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   int i;

   os << height << '\n';  // number of rows

   // HostRead forces synchronization
   HostReadI();
   HostReadJ();
   HostReadData();
   for (i = 0; i <= height; i++)
   {
      os << I[i]+1 << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      os << J[i]+1 << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      os << A[i] << '\n';
   }
}

void SparseMatrix::PrintCSR2(std::ostream & os) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   int i;

   os << height << '\n'; // number of rows
   os << width << '\n';  // number of columns

   // HostRead forces synchronization
   HostReadI();
   HostReadJ();
   HostReadData();
   for (i = 0; i <= height; i++)
   {
      os << I[i] << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      os << J[i] << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      os << A[i] << '\n';
   }
}

void SparseMatrix::PrintInfo(std::ostream &os) const
{
   const real_t MiB = 1024.*1024;
   int nnz = NumNonZeroElems();
   real_t pz = 100./nnz;
   int nz = CountSmallElems(0.0);
   real_t max_norm = MaxNorm();
   real_t symm = IsSymmetric();
   int nnf = CheckFinite();
   int ns12 = CountSmallElems(1e-12*max_norm);
   int ns15 = CountSmallElems(1e-15*max_norm);
   int ns18 = CountSmallElems(1e-18*max_norm);

   os <<
      "SparseMatrix statistics:\n"
      "  Format                      : " <<
      (Empty() ? "(empty)" : (Finalized() ? "CSR" : "LIL")) << "\n"
      "  Dimensions                  : " << height << " x " << width << "\n"
      "  Number of entries (total)   : " << nnz << "\n"
      "  Number of entries (per row) : " << 1.*nnz/Height() << "\n"
      "  Number of stored zeros      : " << nz*pz << "% (" << nz << ")\n"
      "  Number of Inf/Nan entries   : " << nnf*pz << "% ("<< nnf << ")\n"
      "  Norm, max |a_ij|            : " << max_norm << "\n"
      "  Symmetry, max |a_ij-a_ji|   : " << symm << "\n"
      "  Number of small entries:\n"
      "    |a_ij| <= 1e-12*Norm      : " << ns12*pz << "% (" << ns12 << ")\n"
      "    |a_ij| <= 1e-15*Norm      : " << ns15*pz << "% (" << ns15 << ")\n"
      "    |a_ij| <= 1e-18*Norm      : " << ns18*pz << "% (" << ns18 << ")\n";
   if (Finalized())
   {
      os << "  Memory used by CSR          : " <<
         (sizeof(int)*(height+1+nnz)+sizeof(real_t)*nnz)/MiB << " MiB\n";
   }
   if (Rows != NULL)
   {
      size_t used_mem = sizeof(RowNode*)*height;
#ifdef MFEM_USE_MEMALLOC
      used_mem += NodesMem->MemoryUsage();
#else
      for (int i = 0; i < height; i++)
      {
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            used_mem += sizeof(RowNode);
         }
      }
#endif
      os << "  Memory used by LIL          : " << used_mem/MiB << " MiB\n";
   }
}

void SparseMatrix::Destroy()
{
   I.Delete();
   J.Delete();
   A.Delete();

   if (Rows != NULL)
   {
#if !defined(MFEM_USE_MEMALLOC)
      for (int i = 0; i < height; i++)
      {
         RowNode *aux, *node_p = Rows[i];
         while (node_p != NULL)
         {
            aux = node_p;
            node_p = node_p->Prev;
            delete aux;
         }
      }
#endif
      delete [] Rows;
   }

   delete [] ColPtrJ;
   delete [] ColPtrNode;
#ifdef MFEM_USE_MEMALLOC
   delete NodesMem;
#endif
   delete At;

   ClearGPUSparse();
}

int SparseMatrix::ActualWidth() const
{
   int awidth = 0;
   if (A)
   {
      const int *start_j = J;
      const int *end_j = J + I[height];
      for (const int *jptr = start_j; jptr != end_j; ++jptr)
      {
         awidth = std::max(awidth, *jptr + 1);
      }
   }
   else
   {
      RowNode *aux;
      for (int i = 0; i < height; i++)
      {
         for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
         {
            awidth = std::max(awidth, aux->Column + 1);
         }
      }
   }
   return awidth;
}

void SparseMatrixFunction (SparseMatrix & S, real_t (*f)(real_t))
{
   int n = S.NumNonZeroElems();
   real_t * s = S.GetData();

   for (int i = 0; i < n; i++)
   {
      s[i] = f(s[i]);
   }
}

SparseMatrix *Transpose (const SparseMatrix &A)
{
   MFEM_VERIFY(
      A.Finalized(),
      "Finalize must be called before Transpose. Use TransposeRowMatrix instead");

   int i, j, end;
   const int *A_i, *A_j;
   int m, n, nnz, *At_i, *At_j;
   const real_t *A_data;
   real_t *At_data;

   m      = A.Height(); // number of rows of A
   n      = A.Width();  // number of columns of A
   nnz    = A.NumNonZeroElems();
   A_i    = A.HostReadI();
   A_j    = A.HostReadJ();
   A_data = A.HostReadData();

   At_i = Memory<int>(n+1);
   At_j = Memory<int>(nnz);
   At_data = Memory<real_t>(nnz);

   for (i = 0; i <= n; i++)
   {
      At_i[i] = 0;
   }
   for (i = 0; i < nnz; i++)
   {
      At_i[A_j[i]+1]++;
   }
   for (i = 1; i < n; i++)
   {
      At_i[i+1] += At_i[i];
   }

   for (i = j = 0; i < m; i++)
   {
      end = A_i[i+1];
      for ( ; j < end; j++)
      {
         At_j[At_i[A_j[j]]] = i;
         At_data[At_i[A_j[j]]] = A_data[j];
         At_i[A_j[j]]++;
      }
   }

   for (i = n; i > 0; i--)
   {
      At_i[i] = At_i[i-1];
   }
   At_i[0] = 0;

   return  new SparseMatrix(At_i, At_j, At_data, n, m);
}

SparseMatrix *TransposeAbstractSparseMatrix (const AbstractSparseMatrix &A,
                                             int useActualWidth)
{
   int i, j;
   int m, n, nnz, *At_i, *At_j;
   real_t *At_data;
   Array<int> Acols;
   Vector Avals;

   m = A.Height(); // number of rows of A
   if (useActualWidth)
   {
      n = 0;
      int tmp;
      for (i = 0; i < m; i++)
      {
         A.GetRow(i, Acols, Avals);
         if (Acols.Size())
         {
            tmp = Acols.Max();
            if (tmp > n)
            {
               n = tmp;
            }
         }
      }
      ++n;
   }
   else
   {
      n = A.Width(); // number of columns of A
   }
   nnz = A.NumNonZeroElems();

   At_i = Memory<int>(n+1);
   At_j = Memory<int>(nnz);
   At_data = Memory<real_t>(nnz);

   for (i = 0; i <= n; i++)
   {
      At_i[i] = 0;
   }

   for (i = 0; i < m; i++)
   {
      A.GetRow(i, Acols, Avals);
      for (j = 0; j<Acols.Size(); ++j)
      {
         At_i[Acols[j]+1]++;
      }
   }
   for (i = 1; i < n; i++)
   {
      At_i[i+1] += At_i[i];
   }

   for (i = 0; i < m; i++)
   {
      A.GetRow(i, Acols, Avals);
      for (j = 0; j<Acols.Size(); ++j)
      {
         At_j[At_i[Acols[j]]] = i;
         At_data[At_i[Acols[j]]] = Avals[j];
         At_i[Acols[j]]++;
      }
   }

   for (i = n; i > 0; i--)
   {
      At_i[i] = At_i[i-1];
   }
   At_i[0] = 0;

   return new SparseMatrix(At_i, At_j, At_data, n, m);
}


SparseMatrix *Mult (const SparseMatrix &A, const SparseMatrix &B,
                    SparseMatrix *OAB)
{
   int nrowsA, ncolsA, nrowsB, ncolsB;
   const int *A_i, *A_j, *B_i, *B_j;
   int *C_i, *C_j, *B_marker;
   const real_t *A_data, *B_data;
   real_t *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   real_t a_entry, b_entry;
   SparseMatrix *C;

   nrowsA = A.Height();
   ncolsA = A.Width();
   nrowsB = B.Height();
   ncolsB = B.Width();

   MFEM_VERIFY(ncolsA == nrowsB,
               "number of columns of A (" << ncolsA
               << ") must equal number of rows of B (" << nrowsB << ")");

   A_i    = A.HostReadI();
   A_j    = A.HostReadJ();
   A_data = A.HostReadData();
   B_i    = B.HostReadI();
   B_j    = B.HostReadJ();
   B_data = B.HostReadData();

   B_marker = new int[ncolsB];

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   if (OAB == NULL)
   {
      C_i = Memory<int>(nrowsA+1);

      C_i[0] = num_nonzeros = 0;
      for (ic = 0; ic < nrowsA; ic++)
      {
         for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
         {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
            {
               jb = B_j[ib];
               if (B_marker[jb] != ic)
               {
                  B_marker[jb] = ic;
                  num_nonzeros++;
               }
            }
         }
         C_i[ic+1] = num_nonzeros;
      }

      C_j    = Memory<int>(num_nonzeros);
      C_data = Memory<real_t>(num_nonzeros);

      C = new SparseMatrix(C_i, C_j, C_data, nrowsA, ncolsB);

      for (ib = 0; ib < ncolsB; ib++)
      {
         B_marker[ib] = -1;
      }
   }
   else
   {
      C = OAB;

      MFEM_VERIFY(nrowsA == C->Height() && ncolsB == C->Width(),
                  "Input matrix sizes do not match output sizes"
                  << " nrowsA = " << nrowsA
                  << ", C->Height() = " << C->Height()
                  << " ncolsB = " << ncolsB
                  << ", C->Width() = " << C->Width());

      // C_i    = C->HostReadI(); // not used
      C_j    = C->HostWriteJ();
      C_data = C->HostWriteData();
   }

   counter = 0;
   for (ic = 0; ic < nrowsA; ic++)
   {
      // row_start = C_i[ic];
      row_start = counter;
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         ja = A_j[ia];
         a_entry = A_data[ia];
         for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
         {
            jb = B_j[ib];
            b_entry = B_data[ib];
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               if (OAB == NULL)
               {
                  C_j[counter] = jb;
               }
               C_data[counter] = a_entry*b_entry;
               counter++;
            }
            else
            {
               C_data[B_marker[jb]] += a_entry*b_entry;
            }
         }
      }
   }

   MFEM_VERIFY(
      OAB == NULL || counter == OAB->NumNonZeroElems(),
      "With pre-allocated output matrix, number of non-zeros ("
      << OAB->NumNonZeroElems()
      << ") did not match number of entries changed from matrix-matrix multiply, "
      << counter);

   delete [] B_marker;

   return C;
}

SparseMatrix * TransposeMult(const SparseMatrix &A, const SparseMatrix &B)
{
   SparseMatrix *At  = Transpose(A);
   SparseMatrix *AtB = Mult(*At, B);
   delete At;
   return AtB;
}

SparseMatrix *MultAbstractSparseMatrix (const AbstractSparseMatrix &A,
                                        const AbstractSparseMatrix &B)
{
   int nrowsA, ncolsA, nrowsB, ncolsB;
   int *C_i, *C_j, *B_marker;
   real_t *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   real_t a_entry, b_entry;
   SparseMatrix *C;

   nrowsA = A.Height();
   ncolsA = A.Width();
   nrowsB = B.Height();
   ncolsB = B.Width();

   MFEM_VERIFY(ncolsA == nrowsB,
               "number of columns of A (" << ncolsA
               << ") must equal number of rows of B (" << nrowsB << ")");

   B_marker = new int[ncolsB];

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   C_i = Memory<int>(nrowsA+1);

   C_i[0] = num_nonzeros = 0;

   Array<int> colsA, colsB;
   Vector dataA, dataB;
   for (ic = 0; ic < nrowsA; ic++)
   {
      A.GetRow(ic, colsA, dataA);
      for (ia = 0; ia < colsA.Size(); ia++)
      {
         ja = colsA[ia];
         B.GetRow(ja, colsB, dataB);
         for (ib = 0; ib < colsB.Size(); ib++)
         {
            jb = colsB[ib];
            if (B_marker[jb] != ic)
            {
               B_marker[jb] = ic;
               num_nonzeros++;
            }
         }
      }
      C_i[ic+1] = num_nonzeros;
   }

   C_j    = Memory<int>(num_nonzeros);
   C_data = Memory<real_t>(num_nonzeros);

   C = new SparseMatrix(C_i, C_j, C_data, nrowsA, ncolsB);

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   counter = 0;
   for (ic = 0; ic < nrowsA; ic++)
   {
      row_start = counter;
      A.GetRow(ic, colsA, dataA);
      for (ia = 0; ia < colsA.Size(); ia++)
      {
         ja = colsA[ia];
         a_entry = dataA[ia];
         B.GetRow(ja, colsB, dataB);
         for (ib = 0; ib < colsB.Size(); ib++)
         {
            jb = colsB[ib];
            b_entry = dataB[ib];
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[counter] = jb;
               C_data[counter] = a_entry*b_entry;
               counter++;
            }
            else
            {
               C_data[B_marker[jb]] += a_entry*b_entry;
            }
         }
      }
   }

   delete [] B_marker;

   return C;
}

DenseMatrix *Mult (const SparseMatrix &A, DenseMatrix &B)
{
   DenseMatrix *C = new DenseMatrix(A.Height(), B.Width());
   Vector columnB, columnC;
   for (int j = 0; j < B.Width(); ++j)
   {
      B.GetColumnReference(j, columnB);
      C->GetColumnReference(j, columnC);
      A.Mult(columnB, columnC);
   }
   return C;
}

DenseMatrix *RAP (const SparseMatrix &A, DenseMatrix &P)
{
   DenseMatrix R (P, 't'); // R = P^T
   DenseMatrix *AP   = Mult (A, P);
   DenseMatrix *RAP_ = new DenseMatrix(R.Height(), AP->Width());
   Mult (R, *AP, *RAP_);
   delete AP;
   return RAP_;
}

DenseMatrix *RAP(DenseMatrix &A, const SparseMatrix &P)
{
   SparseMatrix *R  = Transpose(P);
   DenseMatrix  *RA = Mult(*R, A);
   DenseMatrix   AtP(*RA, 't');
   delete RA;
   DenseMatrix  *RAtP = Mult(*R, AtP);
   delete R;
   DenseMatrix * RAP_ = new DenseMatrix(*RAtP, 't');
   delete RAtP;
   return RAP_;
}

SparseMatrix *RAP (const SparseMatrix &A, const SparseMatrix &R,
                   SparseMatrix *ORAP)
{
   SparseMatrix *P  = Transpose (R);
   SparseMatrix *AP = Mult (A, *P);
   delete P;
   SparseMatrix *RAP_ = Mult (R, *AP, ORAP);
   delete AP;
   return RAP_;
}

SparseMatrix *RAP(const SparseMatrix &Rt, const SparseMatrix &A,
                  const SparseMatrix &P)
{
   SparseMatrix * R = Transpose(Rt);
   SparseMatrix * RA = Mult(*R,A);
   delete R;
   SparseMatrix * RAP_ = Mult(*RA, P);
   delete RA;
   return RAP_;
}

SparseMatrix *Mult_AtDA (const SparseMatrix &A, const Vector &D,
                         SparseMatrix *OAtDA)
{
   int i, At_nnz, *At_j;
   real_t *At_data;

   SparseMatrix *At = Transpose (A);
   At_nnz  = At -> NumNonZeroElems();
   At_j    = At -> GetJ();
   At_data = At -> GetData();
   for (i = 0; i < At_nnz; i++)
   {
      At_data[i] *= D(At_j[i]);
   }
   SparseMatrix *AtDA = Mult (*At, A, OAtDA);
   delete At;
   return AtDA;
}

SparseMatrix * Add(real_t a, const SparseMatrix & A, real_t b,
                   const SparseMatrix & B)
{
   int nrows = A.Height();
   int ncols = A.Width();

   int * C_i = Memory<int>(nrows+1);
   int * C_j;
   real_t * C_data;

   const int *A_i = A.HostReadI();
   const int *A_j = A.HostReadJ();
   const real_t *A_data = A.HostReadData();

   const int *B_i = B.HostReadI();
   const int *B_j = B.HostReadJ();
   const real_t *B_data = B.HostReadData();

   int * marker = new int[ncols];
   std::fill(marker, marker+ncols, -1);

   int num_nonzeros = 0, jcol;
   C_i[0] = 0;
   for (int ic = 0; ic < nrows; ic++)
   {
      for (int ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      for (int ib = B_i[ic]; ib < B_i[ic+1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] != ic)
         {
            marker[jcol] = ic;
            num_nonzeros++;
         }
      }
      C_i[ic+1] = num_nonzeros;
   }

   C_j = Memory<int>(num_nonzeros);
   C_data = Memory<real_t>(num_nonzeros);

   for (int ia = 0; ia < ncols; ia++)
   {
      marker[ia] = -1;
   }

   int pos = 0;
   for (int ic = 0; ic < nrows; ic++)
   {
      for (int ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         C_data[pos] = a*A_data[ia];
         marker[jcol] = pos;
         pos++;
      }
      for (int ib = B_i[ic]; ib < B_i[ic+1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] < C_i[ic])
         {
            C_j[pos] = jcol;
            C_data[pos] = b*B_data[ib];
            marker[jcol] = pos;
            pos++;
         }
         else
         {
            C_data[marker[jcol]] += b*B_data[ib];
         }
      }
   }

   delete[] marker;
   return new SparseMatrix(C_i, C_j, C_data, nrows, ncols);
}

SparseMatrix * Add(const SparseMatrix & A, const SparseMatrix & B)
{
   return Add(1.,A,1.,B);
}

SparseMatrix * Add(Array<SparseMatrix *> & Ai)
{
   MFEM_ASSERT(Ai.Size() > 0, "invalid size Ai.Size() = " << Ai.Size());

   SparseMatrix * accumulate = Ai[0];
   SparseMatrix * result = accumulate;

   for (int i=1; i < Ai.Size(); ++i)
   {
      result = Add(*accumulate, *Ai[i]);
      if (i != 1)
      {
         delete accumulate;
      }

      accumulate = result;
   }

   return result;
}

/// B += alpha * A
void Add(const SparseMatrix &A,
         real_t alpha, DenseMatrix &B)
{
   for (int r = 0; r < B.Height(); r++)
   {
      const int    * colA = A.GetRowColumns(r);
      const real_t * valA = A.GetRowEntries(r);
      for (int i=0; i<A.RowSize(r); i++)
      {
         B(r, colA[i]) += alpha * valA[i];
      }
   }
}

/// Produces a block matrix with blocks A_{ij}*B
DenseMatrix *OuterProduct(const DenseMatrix &A, const DenseMatrix &B)
{
   int mA = A.Height(), nA = A.Width();
   int mB = B.Height(), nB = B.Width();

   DenseMatrix *C = new DenseMatrix(mA * mB, nA * nB);
   *C = 0.0;
   for (int i=0; i<mA; i++)
   {
      for (int j=0; j<nA; j++)
      {
         C->AddMatrix(A(i,j), B, i * mB, j * nB);
      }
   }
   return C;
}

/// Produces a block matrix with blocks A_{ij}*B
SparseMatrix *OuterProduct(const DenseMatrix &A, const SparseMatrix &B)
{
   int mA = A.Height(), nA = A.Width();
   int mB = B.Height(), nB = B.Width();

   SparseMatrix *C = new SparseMatrix(mA * mB, nA * nB);

   for (int i=0; i<mA; i++)
   {
      for (int j=0; j<nA; j++)
      {
         for (int r=0; r<mB; r++)
         {
            const int    * colB = B.GetRowColumns(r);
            const real_t * valB = B.GetRowEntries(r);

            for (int cj=0; cj<B.RowSize(r); cj++)
            {
               C->Set(i * mB + r, j * nB + colB[cj], A(i,j) * valB[cj]);
            }
         }
      }
   }
   C->Finalize();

   return C;
}

/// Produces a block matrix with blocks A_{ij}*B
SparseMatrix *OuterProduct(const SparseMatrix &A, const DenseMatrix &B)
{
   int mA = A.Height(), nA = A.Width();
   int mB = B.Height(), nB = B.Width();

   SparseMatrix *C = new SparseMatrix(mA * mB, nA * nB);

   for (int r=0; r<mA; r++)
   {
      const int    * colA = A.GetRowColumns(r);
      const real_t * valA = A.GetRowEntries(r);

      for (int aj=0; aj<A.RowSize(r); aj++)
      {
         for (int i=0; i<mB; i++)
         {
            for (int j=0; j<nB; j++)
            {
               C->Set(r * mB + i, colA[aj] * nB + j, valA[aj] * B(i, j));
            }
         }
      }
   }
   C->Finalize();

   return C;
}

/// Produces a block matrix with blocks A_{ij}*B
SparseMatrix *OuterProduct(const SparseMatrix &A, const SparseMatrix &B)
{
   int mA = A.Height(), nA = A.Width();
   int mB = B.Height(), nB = B.Width();

   SparseMatrix *C = new SparseMatrix(mA * mB, nA * nB);

   for (int ar=0; ar<mA; ar++)
   {
      const int    * colA = A.GetRowColumns(ar);
      const real_t * valA = A.GetRowEntries(ar);

      for (int aj=0; aj<A.RowSize(ar); aj++)
      {
         for (int br=0; br<mB; br++)
         {
            const int    * colB = B.GetRowColumns(br);
            const real_t * valB = B.GetRowEntries(br);

            for (int bj=0; bj<B.RowSize(br); bj++)
            {
               C->Set(ar * mB + br, colA[aj] * nB + colB[bj],
                      valA[aj] * valB[bj]);
            }
         }
      }
   }
   C->Finalize();

   return C;
}

void SparseMatrix::Swap(SparseMatrix &other)
{
   mfem::Swap(width, other.width);
   mfem::Swap(height, other.height);
   mfem::Swap(I, other.I);
   mfem::Swap(J, other.J);
   mfem::Swap(A, other.A);
   mfem::Swap(Rows, other.Rows);
   mfem::Swap(current_row, other.current_row);
   mfem::Swap(ColPtrJ, other.ColPtrJ);
   mfem::Swap(ColPtrNode, other.ColPtrNode);
   mfem::Swap(At, other.At);

#ifdef MFEM_USE_MEMALLOC
   mfem::Swap(NodesMem, other.NodesMem);
#endif

   mfem::Swap(isSorted, other.isSorted);
}

SparseMatrix::~SparseMatrix()
{
   Destroy();
#ifdef MFEM_USE_CUDA_OR_HIP
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
#ifdef MFEM_CUDA_1897_WORKAROUND
      if (dBuffer)
      {
         MFEM_Cu_or_Hip(MemFree)(dBuffer);
         dBuffer = nullptr;
         bufferSize = 0;
      }
#endif
      if (SparseMatrixCount==1)
      {
         if (handle)
         {
            MFEM_cu_or_hip(sparseDestroy)(handle);
            handle = nullptr;
         }
#ifndef MFEM_CUDA_1897_WORKAROUND
         if (dBuffer)
         {
            MFEM_Cu_or_Hip(MemFree)(dBuffer);
            dBuffer = nullptr;
            bufferSize = 0;
         }
#endif
      }
      SparseMatrixCount--;
   }
#endif // MFEM_USE_CUDA_OR_HIP
}

}
