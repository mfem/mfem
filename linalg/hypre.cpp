// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "linalg.hpp"
#include "../fem/fem.hpp"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std;

namespace mfem
{

template<typename TargetT, typename SourceT>
static TargetT *DuplicateAs(const SourceT *array, int size,
                            bool cplusplus = true)
{
   TargetT *target_array = cplusplus ? new TargetT[size]
                           /*     */ : hypre_TAlloc(TargetT, size);
   for (int i = 0; i < size; i++)
   {
      target_array[i] = array[i];
   }
   return target_array;
}

inline void HypreParVector::_SetDataAndSize_()
{
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  internal::to_int(
                     hypre_VectorSize(hypre_ParVectorLocalVector(x))));
}

HypreParVector::HypreParVector(MPI_Comm comm, HYPRE_Int glob_size,
                               HYPRE_Int *col) : Vector()
{
   x = hypre_ParVectorCreate(comm,glob_size,col);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(MPI_Comm comm, HYPRE_Int glob_size,
                               double *_data, HYPRE_Int *col) : Vector()
{
   x = hypre_ParVectorCreate(comm,glob_size,col);
   hypre_ParVectorSetDataOwner(x,1); // owns the seq vector
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),0);
   hypre_ParVectorSetPartitioningOwner(x,0);
   double tmp = 0.0;
   hypre_VectorData(hypre_ParVectorLocalVector(x)) = &tmp;
   // If hypre_ParVectorLocalVector(x) and &tmp are non-NULL,
   // hypre_ParVectorInitialize(x) does not allocate memory!
   hypre_ParVectorInitialize(x);
   // Set the internal data array to the one passed in
   hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParVector &y) : Vector()
{
   x = hypre_ParVectorCreate(y.x -> comm, y.x -> global_size,
                             y.x -> partitioning);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParMatrix &A,
                               int transpose) : Vector()
{
   if (!transpose)
   {
      x = hypre_ParVectorInDomainOf(const_cast<HypreParMatrix&>(A));
   }
   else
   {
      x = hypre_ParVectorInRangeOf(const_cast<HypreParMatrix&>(A));
   }
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(HYPRE_ParVector y) : Vector()
{
   x = (hypre_ParVector *) y;
   _SetDataAndSize_();
   own_ParVector = 0;
}

HypreParVector::HypreParVector(ParFiniteElementSpace *pfes)
{
   x = hypre_ParVectorCreate(pfes->GetComm(), pfes->GlobalTrueVSize(),
                             pfes->GetTrueDofOffsets());
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

Vector * HypreParVector::GlobalVector() const
{
   hypre_Vector *hv = hypre_ParVectorToVectorAll(*this);
   Vector *v = new Vector(hv->data, internal::to_int(hv->size));
   v->MakeDataOwner();
   hypre_SeqVectorSetDataOwner(hv,0);
   hypre_SeqVectorDestroy(hv);
   return v;
}

HypreParVector& HypreParVector::operator=(double d)
{
   hypre_ParVectorSetConstantValues(x,d);
   return *this;
}

HypreParVector& HypreParVector::operator=(const HypreParVector &y)
{
#ifdef MFEM_DEBUG
   if (size != y.Size())
   {
      mfem_error("HypreParVector::operator=");
   }
#endif

   for (int i = 0; i < size; i++)
   {
      data[i] = y.data[i];
   }
   return *this;
}

void HypreParVector::SetData(double *_data)
{
   Vector::data = hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
}

HYPRE_Int HypreParVector::Randomize(HYPRE_Int seed)
{
   return hypre_ParVectorSetRandomValues(x,seed);
}

void HypreParVector::Print(const char *fname) const
{
   hypre_ParVectorPrint(x,fname);
}

HypreParVector::~HypreParVector()
{
   if (own_ParVector)
   {
      hypre_ParVectorDestroy(x);
   }
}

#ifdef MFEM_USE_SUNDIALS

void HypreParVector::ToNVector(N_Vector &nv)
{
   MFEM_ASSERT(nv && N_VGetVectorID(nv) == SUNDIALS_NVEC_PARHYP,
               "invalid N_Vector");
   N_VectorContent_ParHyp nv_c = (N_VectorContent_ParHyp)(nv->content);
   MFEM_ASSERT(nv_c->own_parvector == FALSE, "invalid N_Vector");
   nv_c->local_length = x->local_vector->size;
   nv_c->global_length = x->global_size;
   nv_c->comm = x->comm;
   nv_c->x = x;
}

#endif // MFEM_USE_SUNDIALS


double InnerProduct(HypreParVector *x, HypreParVector *y)
{
   return hypre_ParVectorInnerProd(*x, *y);
}

double InnerProduct(HypreParVector &x, HypreParVector &y)
{
   return hypre_ParVectorInnerProd(x, y);
}


double ParNormlp(const Vector &vec, double p, MPI_Comm comm)
{
   double norm = 0.0;
   if (p == 1.0)
   {
      double loc_norm = vec.Norml1();
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
   if (p == 2.0)
   {
      double loc_norm = vec*vec;
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
      norm = sqrt(norm);
   }
   if (p < std::numeric_limits<double>::infinity())
   {
      double sum = 0.0;
      for (int i = 0; i < vec.Size(); i++)
      {
         sum += pow(fabs(vec(i)), p);
      }
      MPI_Allreduce(&sum, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
      norm = pow(norm, 1.0/p);
   }
   else
   {
      double loc_norm = vec.Normlinf();
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_MAX, comm);
   }
   return norm;
}


void HypreParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   diagOwner = offdOwner = colMapOwner = -1;
   ParCSROwner = 1;
}

HypreParMatrix::HypreParMatrix()
{
   Init();
   height = width = 0;
}

char HypreParMatrix::CopyCSR(SparseMatrix *csr, hypre_CSRMatrix *hypre_csr)
{
   hypre_CSRMatrixData(hypre_csr) = csr->GetData();
#ifndef HYPRE_BIGINT
   hypre_CSRMatrixI(hypre_csr) = csr->GetI();
   hypre_CSRMatrixJ(hypre_csr) = csr->GetJ();
   // Prevent hypre from destroying hypre_csr->{i,j,data}
   return 0;
#else
   hypre_CSRMatrixI(hypre_csr) =
      DuplicateAs<HYPRE_Int>(csr->GetI(), csr->Height()+1);
   hypre_CSRMatrixJ(hypre_csr) =
      DuplicateAs<HYPRE_Int>(csr->GetJ(), csr->NumNonZeroElems());
   // Prevent hypre from destroying hypre_csr->{i,j,data}, own {i,j}
   return 1;
#endif
}

char HypreParMatrix::CopyBoolCSR(Table *bool_csr, hypre_CSRMatrix *hypre_csr)
{
   int nnz = bool_csr->Size_of_connections();
   double *data = new double[nnz];
   for (int i = 0; i < nnz; i++)
   {
      data[i] = 1.0;
   }
   hypre_CSRMatrixData(hypre_csr) = data;
#ifndef HYPRE_BIGINT
   hypre_CSRMatrixI(hypre_csr) = bool_csr->GetI();
   hypre_CSRMatrixJ(hypre_csr) = bool_csr->GetJ();
   // Prevent hypre from destroying hypre_csr->{i,j,data}, own {data}
   return 2;
#else
   hypre_CSRMatrixI(hypre_csr) =
      DuplicateAs<HYPRE_Int>(bool_csr->GetI(), bool_csr->Size()+1);
   hypre_CSRMatrixJ(hypre_csr) =
      DuplicateAs<HYPRE_Int>(bool_csr->GetJ(), nnz);
   // Prevent hypre from destroying hypre_csr->{i,j,data}, own {i,j,data}
   return 3;
#endif
}

void HypreParMatrix::CopyCSR_J(hypre_CSRMatrix *hypre_csr, int *J)
{
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(hypre_csr);
   for (HYPRE_Int j = 0; j < nnz; j++)
   {
      J[j] = int(hypre_CSRMatrixJ(hypre_csr)[j]);
   }
}

// Square block-diagonal constructor
HypreParMatrix::HypreParMatrix(MPI_Comm comm, HYPRE_Int glob_size,
                               HYPRE_Int *row_starts, SparseMatrix *diag)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, glob_size, glob_size, row_starts,
                                row_starts, 0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, A->diag);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = hypre_CTAlloc(HYPRE_Int, diag->Height()+1);

   /* Don't need to call these, since they allocate memory only
      if it was not already allocated */
   // hypre_CSRMatrixInitialize(A->diag);
   // hypre_ParCSRMatrixInitialize(A);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
#ifdef HYPRE_BIGINT
   CopyCSR_J(A->diag, diag->GetJ());
#endif

   hypre_MatvecCommPkgCreate(A);
}

// Rectangular block-diagonal constructor
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_Int global_num_rows,
                               HYPRE_Int global_num_cols,
                               HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                               SparseMatrix *diag)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, A->diag);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = hypre_CTAlloc(HYPRE_Int, diag->Height()+1);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);
}

// General rectangular constructor with diagonal and off-diagonal
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_Int global_num_rows,
                               HYPRE_Int global_num_cols,
                               HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                               SparseMatrix *diag, SparseMatrix *offd,
                               HYPRE_Int *cmap)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts,
                                offd->Width(), diag->NumNonZeroElems(),
                                offd->NumNonZeroElems());
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, A->diag);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   offdOwner = CopyCSR(offd, A->offd);
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;
   // Prevent hypre from destroying A->col_map_offd
   colMapOwner = 0;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);
}

// General rectangular constructor with diagonal and off-diagonal
HypreParMatrix::HypreParMatrix(
   MPI_Comm comm,
   HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
   HYPRE_Int *row_starts, HYPRE_Int *col_starts,
   HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
   HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
   HYPRE_Int offd_num_cols, HYPRE_Int *offd_col_map)
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, offd_num_cols, 0, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   HYPRE_Int local_num_rows = hypre_CSRMatrixNumRows(A->diag);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag) = diag_i;
   hypre_CSRMatrixJ(A->diag) = diag_j;
   hypre_CSRMatrixData(A->diag) = diag_data;
   hypre_CSRMatrixNumNonzeros(A->diag) = diag_i[local_num_rows];
   hypre_CSRMatrixSetRownnz(A->diag);
   // Prevent hypre from destroying A->diag->{i,j,data}, own A->diag->{i,j,data}
   diagOwner = 3;

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   hypre_CSRMatrixI(A->offd) = offd_i;
   hypre_CSRMatrixJ(A->offd) = offd_j;
   hypre_CSRMatrixData(A->offd) = offd_data;
   hypre_CSRMatrixNumNonzeros(A->offd) = offd_i[local_num_rows];
   hypre_CSRMatrixSetRownnz(A->offd);
   // Prevent hypre from destroying A->offd->{i,j,data}, own A->offd->{i,j,data}
   offdOwner = 3;

   hypre_ParCSRMatrixColMapOffd(A) = offd_col_map;
   // Prevent hypre from destroying A->col_map_offd, own A->col_map_offd
   colMapOwner = 1;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

// Constructor from a CSR matrix on rank 0
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                               SparseMatrix *sm_a)
{
   MFEM_ASSERT(sm_a != NULL, "invalid input");
   MFEM_VERIFY(!HYPRE_AssumedPartitionCheck(),
               "this method can not be used with assumed partition");

   Init();

   hypre_CSRMatrix *csr_a;
   csr_a = hypre_CSRMatrixCreate(sm_a -> Height(), sm_a -> Width(),
                                 sm_a -> NumNonZeroElems());

   hypre_CSRMatrixSetDataOwner(csr_a,0);
   CopyCSR(sm_a, csr_a);
   hypre_CSRMatrixSetRownnz(csr_a);

   A = hypre_CSRMatrixToParCSRMatrix(comm, csr_a, row_starts, col_starts);

#ifdef HYPRE_BIGINT
   delete [] hypre_CSRMatrixI(csr_a);
   delete [] hypre_CSRMatrixJ(csr_a);
#endif
   hypre_CSRMatrixI(csr_a) = NULL;
   hypre_CSRMatrixDestroy(csr_a);

   height = GetNumRows();
   width = GetNumCols();

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }

   hypre_MatvecCommPkgCreate(A);
}

// Boolean, rectangular, block-diagonal constructor
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_Int global_num_rows,
                               HYPRE_Int global_num_cols,
                               HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                               Table *diag)
{
   Init();
   int nnz = diag->Size_of_connections();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, 0, nnz, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyBoolCSR(diag, A->diag);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = hypre_CTAlloc(HYPRE_Int, diag->Size()+1);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

// Boolean, general rectangular constructor with diagonal and off-diagonal
HypreParMatrix::HypreParMatrix(MPI_Comm comm, int id, int np,
                               HYPRE_Int *row, HYPRE_Int *col,
                               HYPRE_Int *i_diag, HYPRE_Int *j_diag,
                               HYPRE_Int *i_offd, HYPRE_Int *j_offd,
                               HYPRE_Int *cmap, HYPRE_Int cmap_size)
{
   HYPRE_Int diag_nnz, offd_nnz;

   Init();
   if (HYPRE_AssumedPartitionCheck())
   {
      diag_nnz = i_diag[row[1]-row[0]];
      offd_nnz = i_offd[row[1]-row[0]];

      A = hypre_ParCSRMatrixCreate(comm, row[2], col[2], row, col,
                                   cmap_size, diag_nnz, offd_nnz);
   }
   else
   {
      diag_nnz = i_diag[row[id+1]-row[id]];
      offd_nnz = i_offd[row[id+1]-row[id]];

      A = hypre_ParCSRMatrixCreate(comm, row[np], col[np], row, col,
                                   cmap_size, diag_nnz, offd_nnz);
   }

   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   HYPRE_Int i;

   double *a_diag = new double[diag_nnz];
   for (i = 0; i < diag_nnz; i++)
   {
      a_diag[i] = 1.0;
   }

   double *a_offd = new double[offd_nnz];
   for (i = 0; i < offd_nnz; i++)
   {
      a_offd[i] = 1.0;
   }

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = i_diag;
   hypre_CSRMatrixJ(A->diag)    = j_diag;
   hypre_CSRMatrixData(A->diag) = a_diag;
   hypre_CSRMatrixSetRownnz(A->diag);
   // Prevent hypre from destroying A->diag->{i,j,data}, own A->diag->{i,j,data}
   diagOwner = 3;

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   hypre_CSRMatrixI(A->offd)    = i_offd;
   hypre_CSRMatrixJ(A->offd)    = j_offd;
   hypre_CSRMatrixData(A->offd) = a_offd;
   hypre_CSRMatrixSetRownnz(A->offd);
   // Prevent hypre from destroying A->offd->{i,j,data}, own A->offd->{i,j,data}
   offdOwner = 3;

   hypre_ParCSRMatrixColMapOffd(A) = cmap;
   // Prevent hypre from destroying A->col_map_offd, own A->col_map_offd
   colMapOwner = 1;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row == col)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

// General rectangular constructor with diagonal and off-diagonal constructed
// from a CSR matrix that contains both diagonal and off-diagonal blocks
HypreParMatrix::HypreParMatrix(MPI_Comm comm, int nrows, HYPRE_Int glob_nrows,
                               HYPRE_Int glob_ncols, int *I, HYPRE_Int *J,
                               double *data, HYPRE_Int *rows, HYPRE_Int *cols)
{
   Init();

   // Determine partitioning size, and my column start and end
   int part_size;
   HYPRE_Int my_col_start, my_col_end; // my range: [my_col_start, my_col_end)
   if (HYPRE_AssumedPartitionCheck())
   {
      part_size = 2;
      my_col_start = cols[0];
      my_col_end = cols[1];
   }
   else
   {
      int myid;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &part_size);
      part_size++;
      my_col_start = cols[myid];
      my_col_end = cols[myid+1];
   }

   // Copy in the row and column partitionings
   HYPRE_Int *row_starts, *col_starts;
   if (rows == cols)
   {
      row_starts = col_starts = hypre_TAlloc(HYPRE_Int, part_size);
      for (int i = 0; i < part_size; i++)
      {
         row_starts[i] = rows[i];
      }
   }
   else
   {
      row_starts = hypre_TAlloc(HYPRE_Int, part_size);
      col_starts = hypre_TAlloc(HYPRE_Int, part_size);
      for (int i = 0; i < part_size; i++)
      {
         row_starts[i] = rows[i];
         col_starts[i] = cols[i];
      }
   }

   // Create a map for the off-diagonal indices - global to local. Count the
   // number of diagonal and off-diagonal entries.
   HYPRE_Int diag_nnz = 0, offd_nnz = 0, offd_num_cols = 0;
   map<HYPRE_Int, HYPRE_Int> offd_map;
   for (HYPRE_Int j = 0, loc_nnz = I[nrows]; j < loc_nnz; j++)
   {
      HYPRE_Int glob_col = J[j];
      if (my_col_start <= glob_col && glob_col < my_col_end)
      {
         diag_nnz++;
      }
      else
      {
         offd_map.insert(pair<const HYPRE_Int, HYPRE_Int>(glob_col, -1));
         offd_nnz++;
      }
   }
   // count the number of columns in the off-diagonal and set the local indices
   for (map<HYPRE_Int, HYPRE_Int>::iterator it = offd_map.begin();
        it != offd_map.end(); ++it)
   {
      it->second = offd_num_cols++;
   }

   // construct the global ParCSR matrix
   A = hypre_ParCSRMatrixCreate(comm, glob_nrows, glob_ncols,
                                row_starts, col_starts, offd_num_cols,
                                diag_nnz, offd_nnz);
   hypre_ParCSRMatrixInitialize(A);

   HYPRE_Int *diag_i, *diag_j, *offd_i, *offd_j, *offd_col_map;
   double *diag_data, *offd_data;
   diag_i = A->diag->i;
   diag_j = A->diag->j;
   diag_data = A->diag->data;
   offd_i = A->offd->i;
   offd_j = A->offd->j;
   offd_data = A->offd->data;
   offd_col_map = A->col_map_offd;

   diag_nnz = offd_nnz = 0;
   for (HYPRE_Int i = 0, j = 0; i < nrows; i++)
   {
      diag_i[i] = diag_nnz;
      offd_i[i] = offd_nnz;
      for (HYPRE_Int j_end = I[i+1]; j < j_end; j++)
      {
         HYPRE_Int glob_col = J[j];
         if (my_col_start <= glob_col && glob_col < my_col_end)
         {
            diag_j[diag_nnz] = glob_col - my_col_start;
            diag_data[diag_nnz] = data[j];
            diag_nnz++;
         }
         else
         {
            offd_j[offd_nnz] = offd_map[glob_col];
            offd_data[offd_nnz] = data[j];
            offd_nnz++;
         }
      }
   }
   diag_i[nrows] = diag_nnz;
   offd_i[nrows] = offd_nnz;
   for (map<HYPRE_Int, HYPRE_Int>::iterator it = offd_map.begin();
        it != offd_map.end(); ++it)
   {
      offd_col_map[it->second] = it->first;
   }

   hypre_ParCSRMatrixSetNumNonzeros(A);
   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }
   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::MakeRef(const HypreParMatrix &master)
{
   Destroy();
   Init();
   A = master.A;
   ParCSROwner = 0;
   height = master.GetNumRows();
   width = master.GetNumCols();
}

hypre_ParCSRMatrix* HypreParMatrix::StealData()
{
   // Only safe when (diagOwner == -1 && offdOwner == -1 && colMapOwner == -1)
   // Otherwise, there may be memory leaks or hypre may destroy arrays allocated
   // with operator new.
   MFEM_ASSERT(diagOwner == -1 && offdOwner == -1 && colMapOwner == -1, "");
   MFEM_ASSERT(ParCSROwner, "");
   hypre_ParCSRMatrix *R = A;
   A = NULL;
   Destroy();
   Init();
   return R;
}

void HypreParMatrix::CopyRowStarts()
{
   if (!A || hypre_ParCSRMatrixOwnsRowStarts(A) ||
       (hypre_ParCSRMatrixRowStarts(A) == hypre_ParCSRMatrixColStarts(A) &&
        hypre_ParCSRMatrixOwnsColStarts(A)))
   {
      return;
   }

   int row_starts_size;
   if (HYPRE_AssumedPartitionCheck())
   {
      row_starts_size = 2;
   }
   else
   {
      MPI_Comm_size(hypre_ParCSRMatrixComm(A), &row_starts_size);
      row_starts_size++; // num_proc + 1
   }

   HYPRE_Int *old_row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int *new_row_starts = hypre_CTAlloc(HYPRE_Int, row_starts_size);
   for (int i = 0; i < row_starts_size; i++)
   {
      new_row_starts[i] = old_row_starts[i];
   }

   hypre_ParCSRMatrixRowStarts(A) = new_row_starts;
   hypre_ParCSRMatrixOwnsRowStarts(A) = 1;

   if (hypre_ParCSRMatrixColStarts(A) == old_row_starts)
   {
      hypre_ParCSRMatrixColStarts(A) = new_row_starts;
      hypre_ParCSRMatrixOwnsColStarts(A) = 0;
   }
}

void HypreParMatrix::CopyColStarts()
{
   if (!A || hypre_ParCSRMatrixOwnsColStarts(A) ||
       (hypre_ParCSRMatrixRowStarts(A) == hypre_ParCSRMatrixColStarts(A) &&
        hypre_ParCSRMatrixOwnsRowStarts(A)))
   {
      return;
   }

   int col_starts_size;
   if (HYPRE_AssumedPartitionCheck())
   {
      col_starts_size = 2;
   }
   else
   {
      MPI_Comm_size(hypre_ParCSRMatrixComm(A), &col_starts_size);
      col_starts_size++; // num_proc + 1
   }

   HYPRE_Int *old_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int *new_col_starts = hypre_CTAlloc(HYPRE_Int, col_starts_size);
   for (int i = 0; i < col_starts_size; i++)
   {
      new_col_starts[i] = old_col_starts[i];
   }

   hypre_ParCSRMatrixColStarts(A) = new_col_starts;

   if (hypre_ParCSRMatrixRowStarts(A) == old_col_starts)
   {
      hypre_ParCSRMatrixRowStarts(A) = new_col_starts;
      hypre_ParCSRMatrixOwnsRowStarts(A) = 1;
      hypre_ParCSRMatrixOwnsColStarts(A) = 0;
   }
   else
   {
      hypre_ParCSRMatrixOwnsColStarts(A) = 1;
   }
}

void HypreParMatrix::GetDiag(Vector &diag) const
{
   int size = Height();
   diag.SetSize(size);
   for (int j = 0; j < size; j++)
   {
      diag(j) = A->diag->data[A->diag->i[j]];
      MFEM_ASSERT(A->diag->j[A->diag->i[j]] == j,
                  "the first entry in each row must be the diagonal one");
   }
}

static void MakeWrapper(const hypre_CSRMatrix *mat, SparseMatrix &wrapper)
{
   HYPRE_Int nr = hypre_CSRMatrixNumRows(mat);
   HYPRE_Int nc = hypre_CSRMatrixNumCols(mat);
#ifndef HYPRE_BIGINT
   SparseMatrix tmp(hypre_CSRMatrixI(mat),
                    hypre_CSRMatrixJ(mat),
                    hypre_CSRMatrixData(mat),
                    nr, nc, false, false, false);
#else
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(mat);
   SparseMatrix tmp(DuplicateAs<int>(hypre_CSRMatrixI(mat), nr+1),
                    DuplicateAs<int>(hypre_CSRMatrixJ(mat), nnz),
                    hypre_CSRMatrixData(mat),
                    nr, nc, true, false, false);
#endif
   wrapper.Swap(tmp);
}

void HypreParMatrix::GetDiag(SparseMatrix &diag) const
{
   MakeWrapper(A->diag, diag);
}

void HypreParMatrix::GetOffd(SparseMatrix &offd, HYPRE_Int* &cmap) const
{
   MakeWrapper(A->offd, offd);
   cmap = A->col_map_offd;
}

void HypreParMatrix::GetBlocks(Array2D<HypreParMatrix*> &blocks,
                               bool interleaved_rows,
                               bool interleaved_cols) const
{
   int nr = blocks.NumRows();
   int nc = blocks.NumCols();

   hypre_ParCSRMatrix **hypre_blocks = new hypre_ParCSRMatrix*[nr * nc];
   internal::hypre_ParCSRMatrixSplit(A, nr, nc, hypre_blocks,
                                     interleaved_rows, interleaved_cols);

   for (int i = 0; i < nr; i++)
   {
      for (int j = 0; j < nc; j++)
      {
         blocks[i][j] = new HypreParMatrix(hypre_blocks[i*nc + j]);
      }
   }

   delete [] hypre_blocks;
}

HypreParMatrix * HypreParMatrix::Transpose()
{
   hypre_ParCSRMatrix * At;
   hypre_ParCSRMatrixTranspose(A, &At, 1);
   hypre_ParCSRMatrixSetNumNonzeros(At);

   hypre_MatvecCommPkgCreate(At);

   return new HypreParMatrix(At);
}

HYPRE_Int HypreParMatrix::Mult(HypreParVector &x, HypreParVector &y,
                               double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a, A, x, b, y);
}

void HypreParMatrix::Mult(double a, const Vector &x, double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   if (X == NULL)
   {
      X = new HypreParVector(A->comm,
                             GetGlobalNumCols(),
                             x.GetData(),
                             GetColStarts());
      Y = new HypreParVector(A->comm,
                             GetGlobalNumRows(),
                             y.GetData(),
                             GetRowStarts());
   }
   else
   {
      X->SetData(x.GetData());
      Y->SetData(y.GetData());
   }

   hypre_ParCSRMatrixMatvec(a, A, *X, b, *Y);
}

void HypreParMatrix::MultTranspose(double a, const Vector &x,
                                   double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   // Note: x has the dimensions of Y (height), and
   //       y has the dimensions of X (width)
   if (X == NULL)
   {
      X = new HypreParVector(A->comm,
                             GetGlobalNumCols(),
                             y.GetData(),
                             GetColStarts());
      Y = new HypreParVector(A->comm,
                             GetGlobalNumRows(),
                             x.GetData(),
                             GetRowStarts());
   }
   else
   {
      X->SetData(y.GetData());
      Y->SetData(x.GetData());
   }

   hypre_ParCSRMatrixMatvecT(a, A, *Y, b, *X);
}

HYPRE_Int HypreParMatrix::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                               double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a, A, (hypre_ParVector *) x, b,
                                   (hypre_ParVector *) y);
}

HYPRE_Int HypreParMatrix::MultTranspose(HypreParVector & x, HypreParVector & y,
                                        double a, double b)
{
   return hypre_ParCSRMatrixMatvecT(a, A, x, b, y);
}

HypreParMatrix* HypreParMatrix::LeftDiagMult(const SparseMatrix &D,
                                             HYPRE_Int* row_starts) const
{
   const bool assumed_partition = HYPRE_AssumedPartitionCheck();
   if (row_starts == NULL)
   {
      row_starts = hypre_ParCSRMatrixRowStarts(A);
      MFEM_VERIFY(D.Height() == hypre_CSRMatrixNumRows(A->diag),
                  "the matrix D is NOT compatible with the row starts of"
                  " this HypreParMatrix, row_starts must be given.");
   }
   else
   {
      int offset;
      if (assumed_partition)
      {
         offset = 0;
      }
      else
      {
         MPI_Comm_rank(GetComm(), &offset);
      }
      int local_num_rows = row_starts[offset+1]-row_starts[offset];
      MFEM_VERIFY(local_num_rows == D.Height(), "the number of rows in D is "
                  " not compatible with the given row_starts");
   }
   // D.Width() will be checked for compatibility by the SparseMatrix
   // multiplication function, mfem::Mult(), called below.

   int part_size;
   HYPRE_Int global_num_rows;
   if (assumed_partition)
   {
      part_size = 2;
      global_num_rows = row_starts[2];
      // Here, we use row_starts[2], so row_starts must come from the methods
      // GetDofOffsets/GetTrueDofOffsets of ParFiniteElementSpace (HYPRE's
      // partitions have only 2 entries).
   }
   else
   {
      MPI_Comm_size(GetComm(), &part_size);
      global_num_rows = row_starts[part_size];
      part_size++;
   }

   HYPRE_Int *col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int *col_map_offd;

   // get the diag and offd blocks as SparseMatrix wrappers
   SparseMatrix A_diag, A_offd;
   GetDiag(A_diag);
   GetOffd(A_offd, col_map_offd);

   // multiply the blocks with D and create a new HypreParMatrix
   SparseMatrix* DA_diag = mfem::Mult(D, A_diag);
   SparseMatrix* DA_offd = mfem::Mult(D, A_offd);

   HypreParMatrix* DA =
      new HypreParMatrix(GetComm(),
                         global_num_rows, hypre_ParCSRMatrixGlobalNumCols(A),
                         DuplicateAs<HYPRE_Int>(row_starts, part_size, false),
                         DuplicateAs<HYPRE_Int>(col_starts, part_size, false),
                         DA_diag, DA_offd,
                         DuplicateAs<HYPRE_Int>(col_map_offd, A_offd.Width()));

   // When HYPRE_BIGINT is defined, we want DA_{diag,offd} to delete their I and
   // J arrays but not their data arrays; when HYPRE_BIGINT is not defined, we
   // don't want DA_{diag,offd} to delete anything.
#ifndef HYPRE_BIGINT
   DA_diag->LoseData();
   DA_offd->LoseData();
#else
   DA_diag->SetDataOwner(false);
   DA_offd->SetDataOwner(false);
#endif

   delete DA_diag;
   delete DA_offd;

   hypre_ParCSRMatrixSetRowStartsOwner(DA->A, 1);
   hypre_ParCSRMatrixSetColStartsOwner(DA->A, 1);

   DA->diagOwner = DA->offdOwner = 3;
   DA->colMapOwner = 1;

   return DA;
}

void HypreParMatrix::ScaleRows(const Vector &diag)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   if (hypre_CSRMatrixNumRows(A->diag) != diag.Size())
   {
      mfem_error("Note the Vector diag is not of compatible dimensions with A\n");
   }

   int size = Height();
   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);


   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   double val;
   HYPRE_Int jj;
   for (int i(0); i < size; ++i)
   {
      val = diag[i];
      for (jj = Adiag_i[i]; jj < Adiag_i[i+1]; ++jj)
      {
         Adiag_data[jj] *= val;
      }
      for (jj = Aoffd_i[i]; jj < Aoffd_i[i+1]; ++jj)
      {
         Aoffd_data[jj] *= val;
      }
   }
}

void HypreParMatrix::InvScaleRows(const Vector &diag)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   if (hypre_CSRMatrixNumRows(A->diag) != diag.Size())
   {
      mfem_error("Note the Vector diag is not of compatible dimensions with A\n");
   }

   int size = Height();
   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);


   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   double val;
   HYPRE_Int jj;
   for (int i(0); i < size; ++i)
   {
#ifdef MFEM_DEBUG
      if (0.0 == diag(i))
      {
         mfem_error("HypreParMatrix::InvDiagScale : Division by 0");
      }
#endif
      val = 1./diag(i);
      for (jj = Adiag_i[i]; jj < Adiag_i[i+1]; ++jj)
      {
         Adiag_data[jj] *= val;
      }
      for (jj = Aoffd_i[i]; jj < Aoffd_i[i+1]; ++jj)
      {
         Aoffd_data[jj] *= val;
      }
   }
}

void HypreParMatrix::operator*=(double s)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   HYPRE_Int size=hypre_CSRMatrixNumRows(A->diag);
   HYPRE_Int jj;

   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);
   for (jj = 0; jj < Adiag_i[size]; ++jj)
   {
      Adiag_data[jj] *= s;
   }

   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   for (jj = 0; jj < Aoffd_i[size]; ++jj)
   {
      Aoffd_data[jj] *= s;
   }
}

static void get_sorted_rows_cols(const Array<int> &rows_cols,
                                 Array<HYPRE_Int> &hypre_sorted)
{
   hypre_sorted.SetSize(rows_cols.Size());
   bool sorted = true;
   for (int i = 0; i < rows_cols.Size(); i++)
   {
      hypre_sorted[i] = rows_cols[i];
      if (i && rows_cols[i-1] > rows_cols[i]) { sorted = false; }
   }
   if (!sorted) { hypre_sorted.Sort(); }
}

void HypreParMatrix::Threshold(double threshold)
{
   int  ierr = 0;

   MPI_Comm comm;
   int num_procs;
   hypre_CSRMatrix * csr_A;
   hypre_CSRMatrix * csr_A_wo_z;
   hypre_ParCSRMatrix * parcsr_A_ptr;
   HYPRE_Int * row_starts = NULL; HYPRE_Int * col_starts = NULL;
   HYPRE_Int row_start = -1;   HYPRE_Int row_end = -1;
   HYPRE_Int col_start = -1;   HYPRE_Int col_end = -1;

   comm = hypre_ParCSRMatrixComm(A);

   MPI_Comm_size(comm, &num_procs);

   ierr += hypre_ParCSRMatrixGetLocalRange(A,
                                           &row_start,&row_end,
                                           &col_start,&col_end );

   row_starts = hypre_ParCSRMatrixRowStarts(A);
   col_starts = hypre_ParCSRMatrixColStarts(A);

   parcsr_A_ptr = hypre_ParCSRMatrixCreate(comm,row_starts[num_procs],
                                           col_starts[num_procs],row_starts,
                                           col_starts,0,0,0);

   csr_A = hypre_MergeDiagAndOffd(A);

   csr_A_wo_z =  hypre_CSRMatrixDeleteZeros(csr_A,threshold);

   /* hypre_CSRMatrixDeleteZeros will return a NULL pointer rather than a usable
      CSR matrix if it finds no non-zeros */
   if (csr_A_wo_z == NULL)
   {
      csr_A_wo_z = csr_A;
   }
   else
   {
      ierr += hypre_CSRMatrixDestroy(csr_A);
   }

   ierr += GenerateDiagAndOffd(csr_A_wo_z,parcsr_A_ptr,
                               col_start,col_end);

   ierr += hypre_CSRMatrixDestroy(csr_A_wo_z);

   ierr += hypre_ParCSRMatrixDestroy(A);

   A = parcsr_A_ptr;
}

void HypreParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B)
{
   Array<HYPRE_Int> rc_sorted;
   get_sorted_rows_cols(rows_cols, rc_sorted);

   internal::hypre_ParCSRMatrixEliminateAXB(
      A, rc_sorted.Size(), rc_sorted.GetData(), X, B);
}

HypreParMatrix* HypreParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Array<HYPRE_Int> rc_sorted;
   get_sorted_rows_cols(rows_cols, rc_sorted);

   hypre_ParCSRMatrix* Ae;
   internal::hypre_ParCSRMatrixEliminateAAe(
      A, &Ae, rc_sorted.Size(), rc_sorted.GetData());

   return new HypreParMatrix(Ae);
}

void HypreParMatrix::Print(const char *fname, HYPRE_Int offi, HYPRE_Int offj)
{
   hypre_ParCSRMatrixPrintIJ(A,offi,offj,fname);
}

void HypreParMatrix::Read(MPI_Comm comm, const char *fname)
{
   Destroy();
   Init();

   HYPRE_Int base_i, base_j;
   hypre_ParCSRMatrixReadIJ(comm, fname, &base_i, &base_j, &A);
   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::Read_IJMatrix(MPI_Comm comm, const char *fname)
{
   Destroy();
   Init();

   HYPRE_IJMatrix A_ij;
   HYPRE_IJMatrixRead(fname, comm, 5555, &A_ij); // HYPRE_PARCSR = 5555

   HYPRE_ParCSRMatrix A_parcsr;
   HYPRE_IJMatrixGetObject(A_ij, (void**) &A_parcsr);

   A = (hypre_ParCSRMatrix*)A_parcsr;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::Destroy()
{
   if ( X != NULL ) { delete X; }
   if ( Y != NULL ) { delete Y; }

   if (A == NULL) { return; }

   if (diagOwner >= 0)
   {
      if (diagOwner & 1)
      {
         delete [] hypre_CSRMatrixI(A->diag);
         delete [] hypre_CSRMatrixJ(A->diag);
      }
      hypre_CSRMatrixI(A->diag) = NULL;
      hypre_CSRMatrixJ(A->diag) = NULL;
      if (diagOwner & 2)
      {
         delete [] hypre_CSRMatrixData(A->diag);
      }
      hypre_CSRMatrixData(A->diag) = NULL;
   }
   if (offdOwner >= 0)
   {
      if (offdOwner & 1)
      {
         delete [] hypre_CSRMatrixI(A->offd);
         delete [] hypre_CSRMatrixJ(A->offd);
      }
      hypre_CSRMatrixI(A->offd) = NULL;
      hypre_CSRMatrixJ(A->offd) = NULL;
      if (offdOwner & 2)
      {
         delete [] hypre_CSRMatrixData(A->offd);
      }
      hypre_CSRMatrixData(A->offd) = NULL;
   }
   if (colMapOwner >= 0)
   {
      if (colMapOwner & 1)
      {
         delete [] hypre_ParCSRMatrixColMapOffd(A);
      }
      hypre_ParCSRMatrixColMapOffd(A) = NULL;
   }

   if (ParCSROwner)
   {
      hypre_ParCSRMatrixDestroy(A);
   }
}

HypreParMatrix *Add(double alpha, const HypreParMatrix &A,
                    double beta,  const HypreParMatrix &B)
{
   hypre_ParCSRMatrix *C_hypre =
      internal::hypre_ParCSRMatrixAdd(const_cast<HypreParMatrix &>(A),
                                      const_cast<HypreParMatrix &>(B));
   MFEM_VERIFY(C_hypre, "error in hypre_ParCSRMatrixAdd");

   hypre_MatvecCommPkgCreate(C_hypre);
   HypreParMatrix *C = new HypreParMatrix(C_hypre);
   *C = 0.0;
   C->Add(alpha, A);
   C->Add(beta, B);

   return C;
}

HypreParMatrix * ParMult(HypreParMatrix *A, HypreParMatrix *B)
{
   hypre_ParCSRMatrix * ab;
   ab = hypre_ParMatmul(*A,*B);
   hypre_ParCSRMatrixSetNumNonzeros(ab);

   hypre_MatvecCommPkgCreate(ab);

   return new HypreParMatrix(ab);
}

HypreParMatrix * ParAdd(HypreParMatrix *A, HypreParMatrix *B)
{
   hypre_ParCSRMatrix * C = internal::hypre_ParCSRMatrixAdd(*A,*B);

   hypre_MatvecCommPkgCreate(C);

   return new HypreParMatrix(C);
}

HypreParMatrix * RAP(HypreParMatrix *A, HypreParMatrix *P)
{
   HYPRE_Int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));

   hypre_ParCSRMatrix * rap;
   hypre_BoomerAMGBuildCoarseOperator(*P,*A,*P,&rap);
   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);

   /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
      from P (even if it does not own them)! */
   hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
   hypre_ParCSRMatrixSetColStartsOwner(rap,0);

   if (P_owns_its_col_starts)
   {
      hypre_ParCSRMatrixSetColStartsOwner(*P, 1);
   }

   return new HypreParMatrix(rap);
}

HypreParMatrix * RAP(HypreParMatrix * Rt, HypreParMatrix *A, HypreParMatrix *P)
{
   HYPRE_Int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
   HYPRE_Int Rt_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*Rt));

   hypre_ParCSRMatrix * rap;
   hypre_BoomerAMGBuildCoarseOperator(*Rt,*A,*P,&rap);

   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);
   if (!P_owns_its_col_starts)
   {
      /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
         from P (even if it does not own them)! */
      hypre_ParCSRMatrixSetColStartsOwner(rap,0);
   }
   if (!Rt_owns_its_col_starts)
   {
      /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
         from P (even if it does not own them)! */
      hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
   }
   return new HypreParMatrix(rap);
}

void EliminateBC(HypreParMatrix &A, HypreParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A);
   double *data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *I = hypre_CSRMatrixI(A_diag);
#ifdef MFEM_DEBUG
   HYPRE_Int    *J   = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix *)A);
   HYPRE_Int *I_offd = hypre_CSRMatrixI(A_offd);
   double *data_offd = hypre_CSRMatrixData(A_offd);
#endif

   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = data[I[r]] * X(r);
#ifdef MFEM_DEBUG
      // Check that in the rows specified by the ess_dof_list, the matrix A has
      // only one entry -- the diagonal.
      // if (I[r+1] != I[r]+1 || J[I[r]] != r || I_offd[r] != I_offd[r+1])
      if (J[I[r]] != r)
      {
         MFEM_ABORT("the diagonal entry must be the first entry in the row!");
      }
      for (int j = I[r]+1; j < I[r+1]; j++)
      {
         if (data[j] != 0.0)
         {
            MFEM_ABORT("all off-diagonal entries must be zero!");
         }
      }
      for (int j = I_offd[r]; j < I_offd[r+1]; j++)
      {
         if (data_offd[j] != 0.0)
         {
            MFEM_ABORT("all off-diagonal entries must be zero!");
         }
      }
#endif
   }
}

// Taubin or "lambda-mu" scheme, which alternates between positive and
// negative step sizes to approximate low-pass filter effect.

int ParCSRRelax_Taubin(hypre_ParCSRMatrix *A, // matrix to relax with
                       hypre_ParVector *f,    // right-hand side
                       double lambda,
                       double mu,
                       int N,
                       double max_eig,
                       hypre_ParVector *u,    // initial/updated approximation
                       hypre_ParVector *r     // another temp vector
                      )
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   double *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   for (int i = 0; i < N; i++)
   {
      // get residual: r = f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      double coef;
      (0 == (i % 2)) ? coef = lambda : coef = mu;

      for (HYPRE_Int j = 0; j < num_rows; j++)
      {
         u_data[j] += coef*r_data[j] / max_eig;
      }
   }

   return 0;
}

// FIR scheme, which uses Chebyshev polynomials and a window function
// to approximate a low-pass step filter.

int ParCSRRelax_FIR(hypre_ParCSRMatrix *A, // matrix to relax with
                    hypre_ParVector *f,    // right-hand side
                    double max_eig,
                    int poly_order,
                    double* fir_coeffs,
                    hypre_ParVector *u,    // initial/updated approximation
                    hypre_ParVector *x0,   // temporaries
                    hypre_ParVector *x1,
                    hypre_ParVector *x2,
                    hypre_ParVector *x3)

{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

   double *x0_data = hypre_VectorData(hypre_ParVectorLocalVector(x0));
   double *x1_data = hypre_VectorData(hypre_ParVectorLocalVector(x1));
   double *x2_data = hypre_VectorData(hypre_ParVectorLocalVector(x2));
   double *x3_data = hypre_VectorData(hypre_ParVectorLocalVector(x3));

   hypre_ParVectorCopy(u, x0);

   // x1 = f -A*x0/max_eig
   hypre_ParVectorCopy(f, x1);
   hypre_ParCSRMatrixMatvec(-1.0, A, x0, 1.0, x1);

   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x1_data[i] /= -max_eig;
   }

   // x1 = x0 -x1
   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x1_data[i] = x0_data[i] -x1_data[i];
   }

   // x3 = f0*x0 +f1*x1
   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x3_data[i] = fir_coeffs[0]*x0_data[i] +fir_coeffs[1]*x1_data[i];
   }

   for (int n = 2; n <= poly_order; n++)
   {
      // x2 = f - A*x1/max_eig
      hypre_ParVectorCopy(f, x2);
      hypre_ParCSRMatrixMatvec(-1.0, A, x1, 1.0, x2);

      for (HYPRE_Int i = 0; i < num_rows; i++)
      {
         x2_data[i] /= -max_eig;
      }

      // x2 = (x1-x0) +(x1-2*x2)
      // x3 = x3 +f[n]*x2
      // x0 = x1
      // x1 = x2

      for (HYPRE_Int i = 0; i < num_rows; i++)
      {
         x2_data[i] = (x1_data[i]-x0_data[i]) +(x1_data[i]-2*x2_data[i]);
         x3_data[i] += fir_coeffs[n]*x2_data[i];
         x0_data[i] = x1_data[i];
         x1_data[i] = x2_data[i];
      }
   }

   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      u_data[i] = x3_data[i];
   }

   return 0;
}

HypreSmoother::HypreSmoother() : Solver()
{
   type = 2;
   relax_times = 1;
   relax_weight = 1.0;
   omega = 1.0;
   poly_order = 2;
   poly_fraction = .3;
   lambda = 0.5;
   mu = -0.5;
   taubin_iter = 40;

   l1_norms = NULL;
   pos_l1_norms = false;
   B = X = V = Z = NULL;
   X0 = X1 = NULL;
   fir_coeffs = NULL;
}

HypreSmoother::HypreSmoother(HypreParMatrix &_A, int _type,
                             int _relax_times, double _relax_weight, double _omega,
                             int _poly_order, double _poly_fraction)
{
   type = _type;
   relax_times = _relax_times;
   relax_weight = _relax_weight;
   omega = _omega;
   poly_order = _poly_order;
   poly_fraction = _poly_fraction;

   l1_norms = NULL;
   pos_l1_norms = false;
   B = X = V = Z = NULL;
   X0 = X1 = NULL;
   fir_coeffs = NULL;

   SetOperator(_A);
}

void HypreSmoother::SetType(HypreSmoother::Type _type, int _relax_times)
{
   type = static_cast<int>(_type);
   relax_times = _relax_times;
}

void HypreSmoother::SetSOROptions(double _relax_weight, double _omega)
{
   relax_weight = _relax_weight;
   omega = _omega;
}

void HypreSmoother::SetPolyOptions(int _poly_order, double _poly_fraction)
{
   poly_order = _poly_order;
   poly_fraction = _poly_fraction;
}

void HypreSmoother::SetTaubinOptions(double _lambda, double _mu,
                                     int _taubin_iter)
{
   lambda = _lambda;
   mu = _mu;
   taubin_iter = _taubin_iter;
}

void HypreSmoother::SetWindowByName(const char* name)
{
   double a = -1, b, c;
   if (!strcmp(name,"Rectangular")) { a = 1.0,  b = 0.0,  c = 0.0; }
   if (!strcmp(name,"Hanning")) { a = 0.5,  b = 0.5,  c = 0.0; }
   if (!strcmp(name,"Hamming")) { a = 0.54, b = 0.46, c = 0.0; }
   if (!strcmp(name,"Blackman")) { a = 0.42, b = 0.50, c = 0.08; }
   if (a < 0)
   {
      mfem_error("HypreSmoother::SetWindowByName : name not recognized!");
   }

   SetWindowParameters(a, b, c);
}

void HypreSmoother::SetWindowParameters(double a, double b, double c)
{
   window_params[0] = a;
   window_params[1] = b;
   window_params[2] = c;
}

void HypreSmoother::SetOperator(const Operator &op)
{
   A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
   if (A == NULL)
   {
      mfem_error("HypreSmoother::SetOperator : not HypreParMatrix!");
   }

   height = A->Height();
   width = A->Width();

   if (B) { delete B; }
   if (X) { delete X; }
   if (V) { delete V; }
   if (Z) { delete Z; }
   if (l1_norms)
   {
      hypre_TFree(l1_norms);
   }
   delete X0;
   delete X1;

   X1 = X0 = Z = V = B = X = NULL;

   if (type >= 1 && type <= 4)
   {
      hypre_ParCSRComputeL1Norms(*A, type, NULL, &l1_norms);
   }
   else if (type == 5)
   {
      l1_norms = hypre_CTAlloc(double, height);
      Vector ones(height), diag(l1_norms, height);
      ones = 1.0;
      A->Mult(ones, diag);
      type = 1;
   }
   else
   {
      l1_norms = NULL;
   }
   if (l1_norms && pos_l1_norms)
   {
      for (int i = 0; i < height; i++)
      {
         l1_norms[i] = std::abs(l1_norms[i]);
      }
   }

   if (type == 16)
   {
      poly_scale = 1;
      hypre_ParCSRMaxEigEstimateCG(*A, poly_scale, 10,
                                   &max_eig_est, &min_eig_est);
      Z = new HypreParVector(*A);
   }
   else if (type == 1001 || type == 1002)
   {
      poly_scale = 0;
      hypre_ParCSRMaxEigEstimateCG(*A, poly_scale, 10,
                                   &max_eig_est, &min_eig_est);

      // The Taubin and FIR polynomials are defined on [0, 2]
      max_eig_est /= 2;

      // Compute window function, Chebyshev coefficients, and allocate temps.
      if (type == 1002)
      {
         // Temporaries for Chebyshev recursive evaluation
         Z = new HypreParVector(*A);
         X0 = new HypreParVector(*A);
         X1 = new HypreParVector(*A);

         SetFIRCoefficients(max_eig_est);
      }
   }
}

void HypreSmoother::SetFIRCoefficients(double max_eig)
{
   if (fir_coeffs)
   {
      delete [] fir_coeffs;
   }

   fir_coeffs = new double[poly_order+1];

   double* window_coeffs = new double[poly_order+1];
   double* cheby_coeffs = new double[poly_order+1];

   double a = window_params[0];
   double b = window_params[1];
   double c = window_params[2];
   for (int i = 0; i <= poly_order; i++)
   {
      double t = (i*M_PI)/(poly_order+1);
      window_coeffs[i] = a + b*cos(t) +c*cos(2*t);
   }

   double k_pb = poly_fraction*max_eig;
   double theta_pb = acos(1.0 -0.5*k_pb);
   double sigma = 0.0;
   cheby_coeffs[0] = (theta_pb +sigma)/M_PI;
   for (int i = 1; i <= poly_order; i++)
   {
      double t = i*(theta_pb+sigma);
      cheby_coeffs[i] = 2.0*sin(t)/(i*M_PI);
   }

   for (int i = 0; i <= poly_order; i++)
   {
      fir_coeffs[i] = window_coeffs[i]*cheby_coeffs[i];
   }

   delete[] window_coeffs;
   delete[] cheby_coeffs;
}

void HypreSmoother::Mult(const HypreParVector &b, HypreParVector &x) const
{
   if (A == NULL)
   {
      mfem_error("HypreSmoother::Mult (...) : HypreParMatrix A is missing");
      return;
   }

   if (!iterative_mode)
   {
      if (type == 0 && relax_times == 1)
      {
         HYPRE_ParCSRDiagScale(NULL, *A, b, x);
         if (relax_weight != 1.0)
         {
            x *= relax_weight;
         }
         return;
      }
      x = 0.0;
   }

   if (V == NULL)
   {
      V = new HypreParVector(*A);
   }

   if (type == 1001)
   {
      for (int sweep = 0; sweep < relax_times; sweep++)
      {
         ParCSRRelax_Taubin(*A, b, lambda, mu, taubin_iter,
                            max_eig_est,
                            x, *V);
      }
   }
   else if (type == 1002)
   {
      for (int sweep = 0; sweep < relax_times; sweep++)
      {
         ParCSRRelax_FIR(*A, b,
                         max_eig_est,
                         poly_order,
                         fir_coeffs,
                         x,
                         *X0, *X1, *V, *Z);
      }
   }
   else
   {
      if (Z == NULL)
         hypre_ParCSRRelax(*A, b, type,
                           relax_times, l1_norms, relax_weight, omega,
                           max_eig_est, min_eig_est, poly_order, poly_fraction,
                           x, *V, NULL);
      else
         hypre_ParCSRRelax(*A, b, type,
                           relax_times, l1_norms, relax_weight, omega,
                           max_eig_est, min_eig_est, poly_order, poly_fraction,
                           x, *V, *Z);
   }
}

void HypreSmoother::Mult(const Vector &b, Vector &x) const
{
   if (A == NULL)
   {
      mfem_error("HypreSmoother::Mult (...) : HypreParMatrix A is missing");
      return;
   }
   if (B == NULL)
   {
      B = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumRows(),
                             b.GetData(),
                             A -> GetRowStarts());
      X = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumCols(),
                             x.GetData(),
                             A -> GetColStarts());
   }
   else
   {
      B -> SetData(b.GetData());
      X -> SetData(x.GetData());
   }

   Mult(*B, *X);
}

HypreSmoother::~HypreSmoother()
{
   if (B) { delete B; }
   if (X) { delete X; }
   if (V) { delete V; }
   if (Z) { delete Z; }
   if (l1_norms)
   {
      hypre_TFree(l1_norms);
   }
   if (fir_coeffs)
   {
      delete [] fir_coeffs;
   }
   if (X0) { delete X0; }
   if (X1) { delete X1; }
}


HypreSolver::HypreSolver()
{
   A = NULL;
   setup_called = 0;
   B = X = NULL;
}

HypreSolver::HypreSolver(HypreParMatrix *_A)
   : Solver(_A->Height(), _A->Width())
{
   A = _A;
   setup_called = 0;
   B = X = NULL;
}

void HypreSolver::Mult(const HypreParVector &b, HypreParVector &x) const
{
   if (A == NULL)
   {
      mfem_error("HypreSolver::Mult (...) : HypreParMatrix A is missing");
      return;
   }
   if (!setup_called)
   {
      SetupFcn()(*this, *A, b, x);
      setup_called = 1;
   }

   if (!iterative_mode)
   {
      x = 0.0;
   }
   SolveFcn()(*this, *A, b, x);
}

void HypreSolver::Mult(const Vector &b, Vector &x) const
{
   if (A == NULL)
   {
      mfem_error("HypreSolver::Mult (...) : HypreParMatrix A is missing");
      return;
   }
   if (B == NULL)
   {
      B = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumRows(),
                             b.GetData(),
                             A -> GetRowStarts());
      X = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumCols(),
                             x.GetData(),
                             A -> GetColStarts());
   }
   else
   {
      B -> SetData(b.GetData());
      X -> SetData(x.GetData());
   }

   Mult(*B, *X);
}

HypreSolver::~HypreSolver()
{
   if (B) { delete B; }
   if (X) { delete X; }
}


HyprePCG::HyprePCG(HypreParMatrix &_A) : HypreSolver(&_A)
{
   MPI_Comm comm;

   iterative_mode = true;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
}

void HyprePCG::SetTol(double tol)
{
   HYPRE_PCGSetTol(pcg_solver, tol);
}

void HyprePCG::SetMaxIter(int max_iter)
{
   HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
}

void HyprePCG::SetLogging(int logging)
{
   HYPRE_PCGSetLogging(pcg_solver, logging);
}

void HyprePCG::SetPrintLevel(int print_lvl)
{
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_lvl);
}

void HyprePCG::SetPreconditioner(HypreSolver &precond)
{
   HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                             precond.SolveFcn(),
                             precond.SetupFcn(),
                             precond);
}

void HyprePCG::SetResidualConvergenceOptions(int res_frequency, double rtol)
{
   HYPRE_PCGSetTwoNorm(pcg_solver, 1);
   if (res_frequency > 0)
   {
      HYPRE_PCGSetRecomputeResidualP(pcg_solver, res_frequency);
   }
   if (rtol > 0.0)
   {
      HYPRE_PCGSetResidualTol(pcg_solver, rtol);
   }
}

void HyprePCG::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   HYPRE_Int time_index = 0;
   HYPRE_Int num_iterations;
   double final_res_norm;
   MPI_Comm comm;
   HYPRE_Int print_level;

   HYPRE_PCGGetPrintLevel(pcg_solver, &print_level);
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level%3);

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!setup_called)
   {
      if (print_level > 0 && print_level < 3)
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRPCGSetup(pcg_solver, *A, b, x);
      setup_called = 1;

      if (print_level > 0 && print_level < 3)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
   }

   if (print_level > 0 && print_level < 3)
   {
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
   }

   if (!iterative_mode)
   {
      x = 0.0;
   }

   HYPRE_ParCSRPCGSolve(pcg_solver, *A, b, x);

   if (print_level > 0)
   {
      if (print_level < 3)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver,
                                                  &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         cout << "PCG Iterations = " << num_iterations << endl
              << "Final PCG Relative Residual Norm = " << final_res_norm
              << endl;
      }
   }
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level);
}

HyprePCG::~HyprePCG()
{
   HYPRE_ParCSRPCGDestroy(pcg_solver);
}


HypreGMRES::HypreGMRES(HypreParMatrix &_A) : HypreSolver(&_A)
{
   MPI_Comm comm;

   int k_dim    = 50;
   int max_iter = 100;
   double tol   = 1e-6;

   iterative_mode = true;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRGMRESCreate(comm, &gmres_solver);
   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetTol(double tol)
{
   HYPRE_GMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetMaxIter(int max_iter)
{
   HYPRE_GMRESSetMaxIter(gmres_solver, max_iter);
}

void HypreGMRES::SetKDim(int k_dim)
{
   HYPRE_GMRESSetKDim(gmres_solver, k_dim);
}

void HypreGMRES::SetLogging(int logging)
{
   HYPRE_GMRESSetLogging(gmres_solver, logging);
}

void HypreGMRES::SetPrintLevel(int print_lvl)
{
   HYPRE_GMRESSetPrintLevel(gmres_solver, print_lvl);
}

void HypreGMRES::SetPreconditioner(HypreSolver &precond)
{
   HYPRE_ParCSRGMRESSetPrecond(gmres_solver,
                               precond.SolveFcn(),
                               precond.SetupFcn(),
                               precond);
}

void HypreGMRES::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   HYPRE_Int time_index = 0;
   HYPRE_Int num_iterations;
   double final_res_norm;
   MPI_Comm comm;
   HYPRE_Int print_level;

   HYPRE_GMRESGetPrintLevel(gmres_solver, &print_level);

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!setup_called)
   {
      if (print_level > 0)
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRGMRESSetup(gmres_solver, *A, b, x);
      setup_called = 1;

      if (print_level > 0)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
   }

   if (print_level > 0)
   {
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
   }

   if (!iterative_mode)
   {
      x = 0.0;
   }

   HYPRE_ParCSRGMRESSolve(gmres_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRGMRESGetNumIterations(gmres_solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(gmres_solver,
                                                    &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         cout << "GMRES Iterations = " << num_iterations << endl
              << "Final GMRES Relative Residual Norm = " << final_res_norm
              << endl;
      }
   }
}

HypreGMRES::~HypreGMRES()
{
   HYPRE_ParCSRGMRESDestroy(gmres_solver);
}


HypreParaSails::HypreParaSails(HypreParMatrix &A) : HypreSolver(&A)
{
   MPI_Comm comm;

   int    sai_max_levels = 1;
   double sai_threshold  = 0.1;
   double sai_filter     = 0.1;
   int    sai_sym        = 0;
   double sai_loadbal    = 0.0;
   int    sai_reuse      = 0;
   int    sai_logging    = 1;

   HYPRE_ParCSRMatrixGetComm(A, &comm);

   HYPRE_ParaSailsCreate(comm, &sai_precond);
   HYPRE_ParaSailsSetParams(sai_precond, sai_threshold, sai_max_levels);
   HYPRE_ParaSailsSetFilter(sai_precond, sai_filter);
   HYPRE_ParaSailsSetSym(sai_precond, sai_sym);
   HYPRE_ParaSailsSetLoadbal(sai_precond, sai_loadbal);
   HYPRE_ParaSailsSetReuse(sai_precond, sai_reuse);
   HYPRE_ParaSailsSetLogging(sai_precond, sai_logging);
}

void HypreParaSails::SetSymmetry(int sym)
{
   HYPRE_ParaSailsSetSym(sai_precond, sym);
}

HypreParaSails::~HypreParaSails()
{
   HYPRE_ParaSailsDestroy(sai_precond);
}


HypreBoomerAMG::HypreBoomerAMG()
{
   HYPRE_BoomerAMGCreate(&amg_precond);
   SetDefaultOptions();
}

HypreBoomerAMG::HypreBoomerAMG(HypreParMatrix &A) : HypreSolver(&A)
{
   HYPRE_BoomerAMGCreate(&amg_precond);
   SetDefaultOptions();
}

void HypreBoomerAMG::SetDefaultOptions()
{
   // AMG coarsening options:
   int coarsen_type = 10;   // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
   int agg_levels   = 1;    // number of aggressive coarsening levels
   double theta     = 0.25; // strength threshold: 0.25, 0.5, 0.8

   // AMG interpolation options:
   int interp_type  = 6;    // 6 = extended+i, 0 = classical
   int Pmax         = 4;    // max number of elements per row in P

   // AMG relaxation options:
   int relax_type   = 8;    // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
   int relax_sweeps = 1;    // relaxation sweeps on each level

   // Additional options:
   int print_level  = 1;    // print AMG iterations? 1 = no, 2 = yes
   int max_levels   = 25;   // max number of levels in AMG hierarchy

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
   HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
   HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

   // Use as a preconditioner (one V-cycle, zero tolerance)
   HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
   HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
}

void HypreBoomerAMG::ResetAMGPrecond()
{
   HYPRE_Int coarsen_type;
   HYPRE_Int agg_levels;
   HYPRE_Int relax_type;
   HYPRE_Int relax_sweeps;
   HYPRE_Real theta;
   HYPRE_Int interp_type;
   HYPRE_Int Pmax;
   HYPRE_Int print_level;
   HYPRE_Int dim;
   HYPRE_Int nrbms = rbms.Size();
   HYPRE_Int nodal;
   HYPRE_Int nodal_diag;
   HYPRE_Int relax_coarse;
   HYPRE_Int interp_vec_variant;
   HYPRE_Int q_max;
   HYPRE_Int smooth_interp_vectors;
   HYPRE_Int interp_refine;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)amg_precond;

   // read options from amg_precond
   HYPRE_BoomerAMGGetCoarsenType(amg_precond, &coarsen_type);
   agg_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   relax_type = hypre_ParAMGDataUserRelaxType(amg_data);
   relax_sweeps = hypre_ParAMGDataUserNumSweeps(amg_data);
   HYPRE_BoomerAMGGetStrongThreshold(amg_precond, &theta);
   hypre_BoomerAMGGetInterpType(amg_precond, &interp_type);
   HYPRE_BoomerAMGGetPMaxElmts(amg_precond, &Pmax);
   HYPRE_BoomerAMGGetPrintLevel(amg_precond, &print_level);
   HYPRE_BoomerAMGGetNumFunctions(amg_precond, &dim);
   if (nrbms) // elasticity solver options
   {
      nodal = hypre_ParAMGDataNodal(amg_data);
      nodal_diag = hypre_ParAMGDataNodalDiag(amg_data);
      HYPRE_BoomerAMGGetCycleRelaxType(amg_precond, &relax_coarse, 3);
      interp_vec_variant = hypre_ParAMGInterpVecVariant(amg_data);
      q_max = hypre_ParAMGInterpVecQMax(amg_data);
      smooth_interp_vectors = hypre_ParAMGSmoothInterpVectors(amg_data);
      interp_refine = hypre_ParAMGInterpRefine(amg_data);
   }

   HYPRE_BoomerAMGDestroy(amg_precond);
   HYPRE_BoomerAMGCreate(&amg_precond);

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
   HYPRE_BoomerAMGSetMaxLevels(amg_precond, 25);
   HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
   HYPRE_BoomerAMGSetMaxIter(amg_precond, 1); // one V-cycle
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
   HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);
   if (nrbms)
   {
      HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
      HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
      HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
      HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
      HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
      HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
      HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);
      RecomputeRBMs();
      HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());
   }
}

void HypreBoomerAMG::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   if (A) { ResetAMGPrecond(); }

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
}

void HypreBoomerAMG::SetSystemsOptions(int dim)
{
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

   // More robust options with respect to convergence
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);
}

// Rotational rigid-body mode functions, used in SetElasticityOptions()
static void func_rxy(const Vector &x, Vector &y)
{
   y = 0.0; y(0) = x(1); y(1) = -x(0);
}
static void func_ryz(const Vector &x, Vector &y)
{
   y = 0.0; y(1) = x(2); y(2) = -x(1);
}
static void func_rzx(const Vector &x, Vector &y)
{
   y = 0.0; y(2) = x(0); y(0) = -x(2);
}

void HypreBoomerAMG::RecomputeRBMs()
{
   int nrbms;
   Array<HypreParVector*> gf_rbms;
   int dim = fespace->GetParMesh()->Dimension();

   for (int i = 0; i < rbms.Size(); i++)
   {
      HYPRE_ParVectorDestroy(rbms[i]);
   }

   if (dim == 2)
   {
      nrbms = 1;

      VectorFunctionCoefficient coeff_rxy(2, func_rxy);

      ParGridFunction rbms_rxy(fespace);
      rbms_rxy.ProjectCoefficient(coeff_rxy);

      rbms.SetSize(nrbms);
      gf_rbms.SetSize(nrbms);
      gf_rbms[0] = rbms_rxy.ParallelAverage();
   }
   else if (dim == 3)
   {
      nrbms = 3;

      VectorFunctionCoefficient coeff_rxy(3, func_rxy);
      VectorFunctionCoefficient coeff_ryz(3, func_ryz);
      VectorFunctionCoefficient coeff_rzx(3, func_rzx);

      ParGridFunction rbms_rxy(fespace);
      ParGridFunction rbms_ryz(fespace);
      ParGridFunction rbms_rzx(fespace);
      rbms_rxy.ProjectCoefficient(coeff_rxy);
      rbms_ryz.ProjectCoefficient(coeff_ryz);
      rbms_rzx.ProjectCoefficient(coeff_rzx);

      rbms.SetSize(nrbms);
      gf_rbms.SetSize(nrbms);
      gf_rbms[0] = rbms_rxy.ParallelAverage();
      gf_rbms[1] = rbms_ryz.ParallelAverage();
      gf_rbms[2] = rbms_rzx.ParallelAverage();
   }
   else
   {
      nrbms = 0;
      rbms.SetSize(nrbms);
   }

   // Transfer the RBMs from the ParGridFunction to the HYPRE_ParVector objects
   for (int i = 0; i < nrbms; i++)
   {
      rbms[i] = gf_rbms[i]->StealParVector();
      delete gf_rbms[i];
   }
}

void HypreBoomerAMG::SetElasticityOptions(ParFiniteElementSpace *fespace)
{
   // Save the finite element space to support multiple calls to SetOperator()
   this->fespace = fespace;

   // Make sure the systems AMG options are set
   int dim = fespace->GetParMesh()->Dimension();
   SetSystemsOptions(dim);

   // Nodal coarsening options (nodal coarsening is required for this solver)
   // See hypre's new_ij driver and the paper for descriptions.
   int nodal                 = 4; // strength reduction norm: 1, 3 or 4
   int nodal_diag            = 1; // diagonal in strength matrix: 0, 1 or 2
   int relax_coarse          = 8; // smoother on the coarsest grid: 8, 99 or 29

   // Elasticity interpolation options
   int interp_vec_variant    = 2; // 1 = GM-1, 2 = GM-2, 3 = LN
   int q_max                 = 4; // max elements per row for each Q
   int smooth_interp_vectors = 1; // smooth the rigid-body modes?

   // Optionally pre-process the interpolation matrix through iterative weight
   // refinement (this is generally applicable for any system)
   int interp_refine         = 1;

   HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
   HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
   HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
   HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
   HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
   HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
   HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

   RecomputeRBMs();
   HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());
}

HypreBoomerAMG::~HypreBoomerAMG()
{
   for (int i = 0; i < rbms.Size(); i++)
   {
      HYPRE_ParVectorDestroy(rbms[i]);
   }

   HYPRE_BoomerAMGDestroy(amg_precond);
}


HypreAMS::HypreAMS(HypreParMatrix &A, ParFiniteElementSpace *edge_fespace)
   : HypreSolver(&A)
{
   int cycle_type       = 13;
   int rlx_type         = 2;
   int rlx_sweeps       = 1;
   double rlx_weight    = 1.0;
   double rlx_omega     = 1.0;
   int amg_coarsen_type = 10;
   int amg_agg_levels   = 1;
   int amg_rlx_type     = 8;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;

   int dim = edge_fespace->GetMesh()->Dimension();
   int sdim = edge_fespace->GetMesh()->SpaceDimension();
   const FiniteElementCollection *edge_fec = edge_fespace->FEColl();

   bool trace_space, rt_trace_space;
   ND_Trace_FECollection *nd_tr_fec = NULL;
   trace_space = dynamic_cast<const ND_Trace_FECollection*>(edge_fec);
   rt_trace_space = dynamic_cast<const RT_Trace_FECollection*>(edge_fec);
   trace_space = trace_space || rt_trace_space;

   int p = 1;
   if (edge_fespace->GetNE() > 0)
   {
      if (trace_space)
      {
         p = edge_fespace->GetFaceOrder(0);
         if (dim == 2) { p++; }
      }
      else
      {
         p = edge_fespace->GetOrder(0);
      }
   }

   ParMesh *pmesh = edge_fespace->GetParMesh();
   if (rt_trace_space)
   {
      nd_tr_fec = new ND_Trace_FECollection(p, dim);
      edge_fespace = new ParFiniteElementSpace(pmesh, nd_tr_fec);
   }

   HYPRE_AMSCreate(&ams);

   HYPRE_AMSSetDimension(ams, sdim); // 2D H(div) and 3D H(curl) problems
   HYPRE_AMSSetTol(ams, 0.0);
   HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
   HYPRE_AMSSetCycleType(ams, cycle_type);
   HYPRE_AMSSetPrintLevel(ams, 1);

   // define the nodal linear finite element space associated with edge_fespace
   FiniteElementCollection *vert_fec;
   if (trace_space)
   {
      vert_fec = new H1_Trace_FECollection(p, dim);
   }
   else
   {
      vert_fec = new H1_FECollection(p, dim);
   }
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,
                                                                   vert_fec);

   // generate and set the vertex coordinates
   if (p == 1)
   {
      ParGridFunction x_coord(vert_fespace);
      ParGridFunction y_coord(vert_fespace);
      ParGridFunction z_coord(vert_fespace);
      double *coord;
      for (int i = 0; i < pmesh->GetNV(); i++)
      {
         coord = pmesh -> GetVertex(i);
         x_coord(i) = coord[0];
         y_coord(i) = coord[1];
         if (sdim == 3) { z_coord(i) = coord[2]; }
      }
      x = x_coord.ParallelProject();
      y = y_coord.ParallelProject();
      if (sdim == 2)
      {
         z = NULL;
         HYPRE_AMSSetCoordinateVectors(ams, *x, *y, NULL);
      }
      else
      {
         z = z_coord.ParallelProject();
         HYPRE_AMSSetCoordinateVectors(ams, *x, *y, *z);
      }
   }
   else
   {
      x = NULL;
      y = NULL;
      z = NULL;
   }

   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
   if (trace_space)
   {
      grad->AddTraceFaceInterpolator(new GradientInterpolator);
   }
   else
   {
      grad->AddDomainInterpolator(new GradientInterpolator);
   }
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   HYPRE_AMSSetDiscreteGradient(ams, *G);
   delete grad;

   // generate and set the Nedelec interpolation matrices
   Pi = Pix = Piy = Piz = NULL;
   if (p > 1)
   {
      ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, sdim, Ordering::byVDIM);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      if (trace_space)
      {
         id_ND->AddTraceFaceInterpolator(new IdentityInterpolator);
      }
      else
      {
         id_ND->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_ND->Assemble();
      id_ND->Finalize();

      if (cycle_type < 10)
      {
         Pi = id_ND->ParallelAssemble();
      }
      else
      {
         Array2D<HypreParMatrix *> Pi_blocks;
         id_ND->GetParBlocks(Pi_blocks);
         Pix = Pi_blocks(0,0);
         Piy = Pi_blocks(0,1);
         if (sdim == 3) { Piz = Pi_blocks(0,2); }
      }

      delete id_ND;

      HYPRE_ParCSRMatrix HY_Pi  = (Pi)  ? (HYPRE_ParCSRMatrix) *Pi  : NULL;
      HYPRE_ParCSRMatrix HY_Pix = (Pix) ? (HYPRE_ParCSRMatrix) *Pix : NULL;
      HYPRE_ParCSRMatrix HY_Piy = (Piy) ? (HYPRE_ParCSRMatrix) *Piy : NULL;
      HYPRE_ParCSRMatrix HY_Piz = (Piz) ? (HYPRE_ParCSRMatrix) *Piz : NULL;
      HYPRE_AMSSetInterpolations(ams, HY_Pi, HY_Pix, HY_Piy, HY_Piz);

      delete vert_fespace_d;
   }

   delete vert_fespace;
   delete vert_fec;

   if (rt_trace_space)
   {
      delete edge_fespace;
      delete nd_tr_fec;
   }

   // set additional AMS options
   HYPRE_AMSSetSmoothingOptions(ams, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
   HYPRE_AMSSetAlphaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                               theta, amg_interp_type, amg_Pmax);
   HYPRE_AMSSetBetaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                              theta, amg_interp_type, amg_Pmax);
}

HypreAMS::~HypreAMS()
{
   HYPRE_AMSDestroy(ams);

   delete x;
   delete y;
   delete z;

   delete G;
   delete Pi;
   delete Pix;
   delete Piy;
   delete Piz;
}

void HypreAMS::SetPrintLevel(int print_lvl)
{
   HYPRE_AMSSetPrintLevel(ams, print_lvl);
}

HypreADS::HypreADS(HypreParMatrix &A, ParFiniteElementSpace *face_fespace)
   : HypreSolver(&A)
{
   int cycle_type       = 11;
   int rlx_type         = 2;
   int rlx_sweeps       = 1;
   double rlx_weight    = 1.0;
   double rlx_omega     = 1.0;
   int amg_coarsen_type = 10;
   int amg_agg_levels   = 1;
   int amg_rlx_type     = 8;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
   int ams_cycle_type   = 14;

   const FiniteElementCollection *face_fec = face_fespace->FEColl();
   bool trace_space =
      (dynamic_cast<const RT_Trace_FECollection*>(face_fec) != NULL);
   int p = 1;
   if (face_fespace->GetNE() > 0)
   {
      if (trace_space)
      {
         p = face_fespace->GetFaceOrder(0) + 1;
      }
      else
      {
         p = face_fespace->GetOrder(0);
      }
   }

   HYPRE_ADSCreate(&ads);

   HYPRE_ADSSetTol(ads, 0.0);
   HYPRE_ADSSetMaxIter(ads, 1); // use as a preconditioner
   HYPRE_ADSSetCycleType(ads, cycle_type);
   HYPRE_ADSSetPrintLevel(ads, 1);

   // define the nodal and edge finite element spaces associated with face_fespace
   ParMesh *pmesh = (ParMesh *) face_fespace->GetMesh();
   FiniteElementCollection *vert_fec, *edge_fec;
   if (trace_space)
   {
      vert_fec = new H1_Trace_FECollection(p, 3);
      edge_fec = new ND_Trace_FECollection(p, 3);
   }
   else
   {
      vert_fec = new H1_FECollection(p, 3);
      edge_fec = new ND_FECollection(p, 3);
   }

   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,
                                                                   vert_fec);
   ParFiniteElementSpace *edge_fespace = new ParFiniteElementSpace(pmesh,
                                                                   edge_fec);

   // generate and set the vertex coordinates
   if (p == 1)
   {
      ParGridFunction x_coord(vert_fespace);
      ParGridFunction y_coord(vert_fespace);
      ParGridFunction z_coord(vert_fespace);
      double *coord;
      for (int i = 0; i < pmesh->GetNV(); i++)
      {
         coord = pmesh -> GetVertex(i);
         x_coord(i) = coord[0];
         y_coord(i) = coord[1];
         z_coord(i) = coord[2];
      }
      x = x_coord.ParallelProject();
      y = y_coord.ParallelProject();
      z = z_coord.ParallelProject();
      HYPRE_ADSSetCoordinateVectors(ads, *x, *y, *z);
   }
   else
   {
      x = NULL;
      y = NULL;
      z = NULL;
   }

   // generate and set the discrete curl
   ParDiscreteLinearOperator *curl;
   curl = new ParDiscreteLinearOperator(edge_fespace, face_fespace);
   if (trace_space)
   {
      curl->AddTraceFaceInterpolator(new CurlInterpolator);
   }
   else
   {
      curl->AddDomainInterpolator(new CurlInterpolator);
   }
   curl->Assemble();
   curl->Finalize();
   C = curl->ParallelAssemble();
   C->CopyColStarts(); // since we'll delete edge_fespace
   HYPRE_ADSSetDiscreteCurl(ads, *C);
   delete curl;

   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
   if (trace_space)
   {
      grad->AddTraceFaceInterpolator(new GradientInterpolator);
   }
   else
   {
      grad->AddDomainInterpolator(new GradientInterpolator);
   }
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   G->CopyColStarts(); // since we'll delete vert_fespace
   G->CopyRowStarts(); // since we'll delete edge_fespace
   HYPRE_ADSSetDiscreteGradient(ads, *G);
   delete grad;

   // generate and set the Nedelec and Raviart-Thomas interpolation matrices
   RT_Pi = RT_Pix = RT_Piy = RT_Piz = NULL;
   ND_Pi = ND_Pix = ND_Piy = ND_Piz = NULL;
   if (p > 1)
   {
      ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, 3, Ordering::byVDIM);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      if (trace_space)
      {
         id_ND->AddTraceFaceInterpolator(new IdentityInterpolator);
      }
      else
      {
         id_ND->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_ND->Assemble();
      id_ND->Finalize();

      if (ams_cycle_type < 10)
      {
         ND_Pi = id_ND->ParallelAssemble();
         ND_Pi->CopyColStarts(); // since we'll delete vert_fespace_d
         ND_Pi->CopyRowStarts(); // since we'll delete edge_fespace
      }
      else
      {
         Array2D<HypreParMatrix *> ND_Pi_blocks;
         id_ND->GetParBlocks(ND_Pi_blocks);
         ND_Pix = ND_Pi_blocks(0,0);
         ND_Piy = ND_Pi_blocks(0,1);
         ND_Piz = ND_Pi_blocks(0,2);
      }

      delete id_ND;

      ParDiscreteLinearOperator *id_RT;
      id_RT = new ParDiscreteLinearOperator(vert_fespace_d, face_fespace);
      if (trace_space)
      {
         id_RT->AddTraceFaceInterpolator(new NormalInterpolator);
      }
      else
      {
         id_RT->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_RT->Assemble();
      id_RT->Finalize();

      if (cycle_type < 10)
      {
         RT_Pi = id_RT->ParallelAssemble();
         RT_Pi->CopyColStarts(); // since we'll delete vert_fespace_d
      }
      else
      {
         Array2D<HypreParMatrix *> RT_Pi_blocks;
         id_RT->GetParBlocks(RT_Pi_blocks);
         RT_Pix = RT_Pi_blocks(0,0);
         RT_Piy = RT_Pi_blocks(0,1);
         RT_Piz = RT_Pi_blocks(0,2);
      }

      delete id_RT;

      HYPRE_ParCSRMatrix HY_RT_Pi, HY_RT_Pix, HY_RT_Piy, HY_RT_Piz;
      HY_RT_Pi  = (RT_Pi)  ? (HYPRE_ParCSRMatrix) *RT_Pi  : NULL;
      HY_RT_Pix = (RT_Pix) ? (HYPRE_ParCSRMatrix) *RT_Pix : NULL;
      HY_RT_Piy = (RT_Piy) ? (HYPRE_ParCSRMatrix) *RT_Piy : NULL;
      HY_RT_Piz = (RT_Piz) ? (HYPRE_ParCSRMatrix) *RT_Piz : NULL;
      HYPRE_ParCSRMatrix HY_ND_Pi, HY_ND_Pix, HY_ND_Piy, HY_ND_Piz;
      HY_ND_Pi  = (ND_Pi)  ? (HYPRE_ParCSRMatrix) *ND_Pi  : NULL;
      HY_ND_Pix = (ND_Pix) ? (HYPRE_ParCSRMatrix) *ND_Pix : NULL;
      HY_ND_Piy = (ND_Piy) ? (HYPRE_ParCSRMatrix) *ND_Piy : NULL;
      HY_ND_Piz = (ND_Piz) ? (HYPRE_ParCSRMatrix) *ND_Piz : NULL;
      HYPRE_ADSSetInterpolations(ads,
                                 HY_RT_Pi, HY_RT_Pix, HY_RT_Piy, HY_RT_Piz,
                                 HY_ND_Pi, HY_ND_Pix, HY_ND_Piy, HY_ND_Piz);

      delete vert_fespace_d;
   }

   delete vert_fec;
   delete vert_fespace;
   delete edge_fec;
   delete edge_fespace;

   // set additional ADS options
   HYPRE_ADSSetSmoothingOptions(ads, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
   HYPRE_ADSSetAMGOptions(ads, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                          theta, amg_interp_type, amg_Pmax);
   HYPRE_ADSSetAMSOptions(ads, ams_cycle_type, amg_coarsen_type, amg_agg_levels,
                          amg_rlx_type, theta, amg_interp_type, amg_Pmax);
}

HypreADS::~HypreADS()
{
   HYPRE_ADSDestroy(ads);

   delete x;
   delete y;
   delete z;

   delete G;
   delete C;

   delete RT_Pi;
   delete RT_Pix;
   delete RT_Piy;
   delete RT_Piz;

   delete ND_Pi;
   delete ND_Pix;
   delete ND_Piy;
   delete ND_Piz;
}

void HypreADS::SetPrintLevel(int print_lvl)
{
   HYPRE_ADSSetPrintLevel(ads, print_lvl);
}

HypreLOBPCG::HypreMultiVector::HypreMultiVector(int n, HypreParVector & v,
                                                mv_InterfaceInterpreter & interpreter)
   : hpv(NULL),
     nv(n)
{
   mv_ptr = mv_MultiVectorCreateFromSampleVector(&interpreter, nv,
                                                 (HYPRE_ParVector)v);

   HYPRE_ParVector* vecs = NULL;
   {
      mv_TempMultiVector* tmp =
         (mv_TempMultiVector*)mv_MultiVectorGetData(mv_ptr);
      vecs = (HYPRE_ParVector*)(tmp -> vector);
   }

   hpv = new HypreParVector*[nv];
   for (int i=0; i<nv; i++)
   {
      hpv[i] = new HypreParVector(vecs[i]);
   }
}

HypreLOBPCG::HypreMultiVector::~HypreMultiVector()
{
   if ( hpv != NULL )
   {
      for (int i=0; i<nv; i++)
      {
         delete hpv[i];
      }
      delete [] hpv;
   }

   mv_MultiVectorDestroy(mv_ptr);
}

void
HypreLOBPCG::HypreMultiVector::Randomize(HYPRE_Int seed)
{
   mv_MultiVectorSetRandom(mv_ptr, seed);
}

HypreParVector &
HypreLOBPCG::HypreMultiVector::GetVector(unsigned int i)
{
   MFEM_ASSERT((int)i < nv, "index out of range");

   return ( *hpv[i] );
}

HypreParVector **
HypreLOBPCG::HypreMultiVector::StealVectors()
{
   HypreParVector ** hpv_ret = hpv;

   hpv = NULL;

   mv_TempMultiVector * mv_tmp =
      (mv_TempMultiVector*)mv_MultiVectorGetData(mv_ptr);

   mv_tmp->ownsVectors = 0;

   for (int i=0; i<nv; i++)
   {
      hpv_ret[i]->SetOwnership(1);
   }

   return hpv_ret;
}

HypreLOBPCG::HypreLOBPCG(MPI_Comm c)
   : comm(c),
     myid(0),
     numProcs(1),
     nev(10),
     seed(75),
     glbSize(-1),
     part(NULL),
     multi_vec(NULL),
     x(NULL),
     subSpaceProj(NULL)
{
   MPI_Comm_size(comm,&numProcs);
   MPI_Comm_rank(comm,&myid);

   HYPRE_ParCSRSetupInterpreter(&interpreter);
   HYPRE_ParCSRSetupMatvec(&matvec_fn);
   HYPRE_LOBPCGCreate(&interpreter, &matvec_fn, &lobpcg_solver);
}

HypreLOBPCG::~HypreLOBPCG()
{
   delete multi_vec;
   delete x;
   delete [] part;

   HYPRE_LOBPCGDestroy(lobpcg_solver);
}

void
HypreLOBPCG::SetTol(double tol)
{
   HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
}

void
HypreLOBPCG::SetRelTol(double rel_tol)
{
#if MFEM_HYPRE_VERSION >= 21101
   HYPRE_LOBPCGSetRTol(lobpcg_solver, rel_tol);
#else
   MFEM_ABORT("This method requires HYPRE version >= 2.11.1");
#endif
}

void
HypreLOBPCG::SetMaxIter(int max_iter)
{
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver, max_iter);
}

void
HypreLOBPCG::SetPrintLevel(int logging)
{
   if (myid == 0)
   {
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, logging);
   }
}

void
HypreLOBPCG::SetPrecondUsageMode(int pcg_mode)
{
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcg_mode);
}

void
HypreLOBPCG::SetPreconditioner(Solver & precond)
{
   HYPRE_LOBPCGSetPrecond(lobpcg_solver,
                          (HYPRE_PtrToSolverFcn)this->PrecondSolve,
                          (HYPRE_PtrToSolverFcn)this->PrecondSetup,
                          (HYPRE_Solver)&precond);
}

void
HypreLOBPCG::SetOperator(Operator & A)
{
   int locSize = A.Width();

   if (HYPRE_AssumedPartitionCheck())
   {
      part = new HYPRE_Int[2];

      MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

      part[0] = part[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
   }
   else
   {
      part = new HYPRE_Int[numProcs+1];

      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part[1], 1, HYPRE_MPI_INT, comm);

      part[0] = 0;
      for (int i=0; i<numProcs; i++)
      {
         part[i+1] += part[i];
      }

      glbSize = part[numProcs];
   }

   if ( x != NULL )
   {
      delete x;
   }

   // Create a distributed vector without a data array.
   x = new HypreParVector(comm,glbSize,NULL,part);

   matvec_fn.MatvecCreate  = this->OperatorMatvecCreate;
   matvec_fn.Matvec        = this->OperatorMatvec;
   matvec_fn.MatvecDestroy = this->OperatorMatvecDestroy;

   HYPRE_LOBPCGSetup(lobpcg_solver,(HYPRE_Matrix)&A,NULL,NULL);
}

void
HypreLOBPCG::SetMassMatrix(Operator & M)
{
   matvec_fn.MatvecCreate  = this->OperatorMatvecCreate;
   matvec_fn.Matvec        = this->OperatorMatvec;
   matvec_fn.MatvecDestroy = this->OperatorMatvecDestroy;

   HYPRE_LOBPCGSetupB(lobpcg_solver,(HYPRE_Matrix)&M,NULL);
}

void
HypreLOBPCG::GetEigenvalues(Array<double> & eigs)
{
   // Initialize eigenvalues array with marker values
   eigs.SetSize(nev);

   for (int i=0; i<nev; i++)
   {
      eigs[i] = eigenvalues[i];
   }
}

HypreParVector &
HypreLOBPCG::GetEigenvector(unsigned int i)
{
   return multi_vec->GetVector(i);
}

void
HypreLOBPCG::SetInitialVectors(int num_vecs, HypreParVector ** vecs)
{
   // Initialize HypreMultiVector object if necessary
   if ( multi_vec == NULL )
   {
      MFEM_ASSERT(x != NULL, "In HypreLOBPCG::SetInitialVectors()");

      multi_vec = new HypreMultiVector(nev, *x, interpreter);
   }

   // Copy the vectors provided
   for (int i=0; i < min(num_vecs,nev); i++)
   {
      multi_vec->GetVector(i) = *vecs[i];
   }

   // Randomize any remaining vectors
   for (int i=min(num_vecs,nev); i < nev; i++)
   {
      multi_vec->GetVector(i).Randomize(seed);
   }

   // Ensure all vectors are in the proper subspace
   if ( subSpaceProj != NULL )
   {
      HypreParVector y(*x);
      y = multi_vec->GetVector(0);

      for (int i=1; i<nev; i++)
      {
         subSpaceProj->Mult(multi_vec->GetVector(i),
                            multi_vec->GetVector(i-1));
      }
      subSpaceProj->Mult(y,
                         multi_vec->GetVector(nev-1));
   }
}

void
HypreLOBPCG::Solve()
{
   // Initialize HypreMultiVector object if necessary
   if ( multi_vec == NULL )
   {
      MFEM_ASSERT(x != NULL, "In HypreLOBPCG::Solve()");

      multi_vec = new HypreMultiVector(nev, *x, interpreter);
      multi_vec->Randomize(seed);

      if ( subSpaceProj != NULL )
      {
         HypreParVector y(*x);
         y = multi_vec->GetVector(0);

         for (int i=1; i<nev; i++)
         {
            subSpaceProj->Mult(multi_vec->GetVector(i),
                               multi_vec->GetVector(i-1));
         }
         subSpaceProj->Mult(y, multi_vec->GetVector(nev-1));
      }
   }

   eigenvalues.SetSize(nev);
   eigenvalues = NAN;

   // Perform eigenmode calculation
   //
   // The eigenvalues are computed in ascending order (internally the
   // order is determined by the LAPACK routine 'dsydv'.)
   HYPRE_LOBPCGSolve(lobpcg_solver, NULL, *multi_vec, eigenvalues);
}

void *
HypreLOBPCG::OperatorMatvecCreate( void *A,
                                   void *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

HYPRE_Int
HypreLOBPCG::OperatorMatvec( void *matvec_data,
                             HYPRE_Complex alpha,
                             void *A,
                             void *x,
                             HYPRE_Complex beta,
                             void *y )
{
   MFEM_VERIFY(alpha == 1.0 && beta == 0.0, "values not supported");

   Operator *Aop = (Operator*)A;

   int width = Aop->Width();

   hypre_ParVector * xPar = (hypre_ParVector *)x;
   hypre_ParVector * yPar = (hypre_ParVector *)y;

   Vector xVec(xPar->local_vector->data, width);
   Vector yVec(yPar->local_vector->data, width);

   Aop->Mult( xVec, yVec );

   return 0;
}

HYPRE_Int
HypreLOBPCG::OperatorMatvecDestroy( void *matvec_data )
{
   return 0;
}

HYPRE_Int
HypreLOBPCG::PrecondSolve(void *solver,
                          void *A,
                          void *b,
                          void *x)
{
   Solver   *PC = (Solver*)solver;
   Operator *OP = (Operator*)A;

   int width = OP->Width();

   hypre_ParVector * bPar = (hypre_ParVector *)b;
   hypre_ParVector * xPar = (hypre_ParVector *)x;

   Vector bVec(bPar->local_vector->data, width);
   Vector xVec(xPar->local_vector->data, width);

   PC->Mult( bVec, xVec );

   return 0;
}

HYPRE_Int
HypreLOBPCG::PrecondSetup(void *solver,
                          void *A,
                          void *b,
                          void *x)
{
   return 0;
}

HypreAME::HypreAME(MPI_Comm comm)
   : myid(0),
     numProcs(1),
     nev(10),
     setT(false),
     ams_precond(NULL),
     eigenvalues(NULL),
     multi_vec(NULL),
     eigenvectors(NULL)
{
   MPI_Comm_size(comm,&numProcs);
   MPI_Comm_rank(comm,&myid);

   HYPRE_AMECreate(&ame_solver);
   HYPRE_AMESetPrintLevel(ame_solver, 0);
}

HypreAME::~HypreAME()
{
   if ( multi_vec )
   {
      hypre_TFree(multi_vec);
   }

   if ( eigenvectors )
   {
      for (int i=0; i<nev; i++)
      {
         delete eigenvectors[i];
      }
   }
   delete [] eigenvectors;

   if ( eigenvalues )
   {
      hypre_TFree(eigenvalues);
   }

   HYPRE_AMEDestroy(ame_solver);
}

void
HypreAME::SetNumModes(int num_eigs)
{
   nev = num_eigs;

   HYPRE_AMESetBlockSize(ame_solver, nev);
}

void
HypreAME::SetTol(double tol)
{
   HYPRE_AMESetTol(ame_solver, tol);
}

void
HypreAME::SetRelTol(double rel_tol)
{
#if MFEM_HYPRE_VERSION >= 21101
   HYPRE_AMESetRTol(ame_solver, rel_tol);
#else
   MFEM_ABORT("This method requires HYPRE version >= 2.11.1");
#endif
}

void
HypreAME::SetMaxIter(int max_iter)
{
   HYPRE_AMESetMaxIter(ame_solver, max_iter);
}

void
HypreAME::SetPrintLevel(int logging)
{
   if (myid == 0)
   {
      HYPRE_AMESetPrintLevel(ame_solver, logging);
   }
}

void
HypreAME::SetPreconditioner(HypreSolver & precond)
{
   ams_precond = &precond;
}

void
HypreAME::SetOperator(HypreParMatrix & A)
{
   if ( !setT )
   {
      HYPRE_Solver ams_precond_ptr = (HYPRE_Solver)*ams_precond;

      ams_precond->SetupFcn()(*ams_precond,A,NULL,NULL);

      HYPRE_AMESetAMSSolver(ame_solver, ams_precond_ptr);
   }

   HYPRE_AMESetup(ame_solver);
}

void
HypreAME::SetMassMatrix(HypreParMatrix & M)
{
   HYPRE_ParCSRMatrix parcsr_M = M;
   HYPRE_AMESetMassMatrix(ame_solver,(HYPRE_ParCSRMatrix)parcsr_M);
}

void
HypreAME::Solve()
{
   HYPRE_AMESolve(ame_solver);
}

void
HypreAME::GetEigenvalues(Array<double> & eigs)
{
   // Initialize eigenvalues array with marker values
   eigs.SetSize(nev); eigs = -1.0;

   if ( eigenvalues == NULL )
   {
      // Grab eigenvalues from AME
      HYPRE_AMEGetEigenvalues(ame_solver,&eigenvalues);
   }

   // Copy eigenvalues to eigs array
   for (int i=0; i<nev; i++)
   {
      eigs[i] = eigenvalues[i];
   }
}

void
HypreAME::createDummyVectors()
{
   if ( multi_vec == NULL )
   {
      HYPRE_AMEGetEigenvectors(ame_solver,&multi_vec);
   }

   eigenvectors = new HypreParVector*[nev];
   for (int i=0; i<nev; i++)
   {
      eigenvectors[i] = new HypreParVector(multi_vec[i]);
      eigenvectors[i]->SetOwnership(1);
   }

}

HypreParVector &
HypreAME::GetEigenvector(unsigned int i)
{
   if ( eigenvectors == NULL )
   {
      this->createDummyVectors();
   }

   return *eigenvectors[i];
}

HypreParVector **
HypreAME::StealEigenvectors()
{
   if ( eigenvectors == NULL )
   {
      this->createDummyVectors();
   }

   // Set the local pointers to NULL so that they won't be deleted later
   HypreParVector ** vecs = eigenvectors;
   eigenvectors = NULL;
   multi_vec = NULL;

   return vecs;
}

}

#endif
