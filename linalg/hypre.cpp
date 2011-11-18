// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifdef MFEM_USE_MPI

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

#include "linalg.hpp"
#include "../fem/fem.hpp"

HypreParVector::HypreParVector(int glob_size, int *col) : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,glob_size,col);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(int glob_size, double *_data, int *col)
   : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,glob_size,col);
   hypre_ParVectorSetDataOwner(x,1); // owns the seq vector
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),0);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
   // If hypre_ParVectorLocalVector(x) is non-NULL, hypre_ParVectorInitialize(x)
   // does not allocate memory!
   hypre_ParVectorInitialize(x);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParVector &y) : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD, y.x -> global_size,
                             y.x -> partitioning);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(HYPRE_ParVector y) : Vector()
{
   x = (hypre_ParVector *) y;
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 0;
}

HypreParVector::operator hypre_ParVector*() const
{
   return x;
}

HypreParVector::operator HYPRE_ParVector() const
{
   return (HYPRE_ParVector) x;
}

Vector * HypreParVector::GlobalVector()
{
   hypre_Vector *hv = hypre_ParVectorToVectorAll(*this);
   Vector *v = new Vector(hv->data, hv->size);
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
      mfem_error("HypreParVector::operator=");
#endif

   for (int i = 0; i < size; i++)
      data[i] = y.data[i];
   return *this;
}

void HypreParVector::SetData(double *_data)
{
   Vector::data = hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
}

int HypreParVector::Randomize(int seed)
{
   return hypre_ParVectorSetRandomValues(x,seed);
}

void HypreParVector::Print(const char *fname)
{
   hypre_ParVectorPrint(x,fname);
}

HypreParVector::~HypreParVector()
{
   if (own_ParVector)
      hypre_ParVectorDestroy(x);
}


double InnerProduct(HypreParVector *x, HypreParVector *y)
{
   return hypre_ParVectorInnerProd(*x, *y);
}

double InnerProduct(HypreParVector &x, HypreParVector &y)
{
   return hypre_ParVectorInnerProd(x, y);
}


HypreParMatrix::HypreParMatrix(int size, int *row, SparseMatrix *diag)
   : Operator(size)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, size, size, row, row,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd)    = hypre_CTAlloc(HYPRE_Int, diag->Size()+1);

   /* Don't need to call these, since they allocate memory only
      if it was not already allocated */
   // hypre_CSRMatrixInitialize(A->diag);
   // hypre_ParCSRMatrixInitialize(A);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;
}


HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               SparseMatrix *diag)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = hypre_CTAlloc(HYPRE_Int, diag->Size()+1);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               SparseMatrix *diag, SparseMatrix *offd,
                               int *cmap)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                offd->Width(), diag->NumNonZeroElems(),
                                offd->NumNonZeroElems());
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   hypre_CSRMatrixI(A->offd)    = offd->GetI();
   hypre_CSRMatrixJ(A->offd)    = offd->GetJ();
   hypre_CSRMatrixData(A->offd) = offd->GetData();
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(int *row, int *col, SparseMatrix *sm_a)
{
#ifdef MFEM_DEBUG
   if (sm_a == NULL)
      mfem_error("HypreParMatrix::HypreParMatrix: sm_a==NULL");
#endif

   hypre_CSRMatrix *csr_a;

   csr_a = hypre_CSRMatrixCreate(sm_a -> Size(), sm_a -> Width(),
                                 sm_a -> NumNonZeroElems());

   hypre_CSRMatrixSetDataOwner(csr_a,0);
   hypre_CSRMatrixI(csr_a)    = sm_a -> GetI();
   hypre_CSRMatrixJ(csr_a)    = sm_a -> GetJ();
   hypre_CSRMatrixData(csr_a) = sm_a -> GetData();
   hypre_CSRMatrixSetRownnz(csr_a);

   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD,csr_a,row,col);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();

   hypre_MatvecCommPkgCreate(A);
}

HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               Table *diag)
{
   int nnz = diag->Size_of_connections();
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                0, nnz, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,1);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();

   hypre_CSRMatrixData(A->diag) = hypre_TAlloc(double, nnz);
   for (int k = 0; k < nnz; k++)
      (hypre_CSRMatrixData(A->diag))[k] = 1.0;

   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = hypre_CTAlloc(HYPRE_Int, diag->Size()+1);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(MPI_Comm comm, int id, int np,
                               int *row, int *col,
                               int *i_diag, int *j_diag,
                               int *i_offd, int *j_offd,
                               int *cmap, int cmap_size)
{
   int diag_col, offd_col;

   if (HYPRE_AssumedPartitionCheck())
   {
      diag_col = i_diag[row[1]-row[0]];
      offd_col = i_offd[row[1]-row[0]];

      A = hypre_ParCSRMatrixCreate(comm, row[2], col[2], row, col,
                                   cmap_size, diag_col, offd_col);
   }
   else
   {
      diag_col = i_diag[row[id+1]-row[id]];
      offd_col = i_offd[row[id+1]-row[id]];

      A = hypre_ParCSRMatrixCreate(comm, row[np], col[np], row, col,
                                   cmap_size, diag_col, offd_col);
   }

   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   int i;

   double *a_diag = hypre_TAlloc(double, diag_col);
   for (i = 0; i < diag_col; i++)
      a_diag[i] = 1.0;

   double *a_offd = hypre_TAlloc(double, offd_col);
   for (i = 0; i < offd_col; i++)
      a_offd[i] = 1.0;

   hypre_CSRMatrixSetDataOwner(A->diag,1);
   hypre_CSRMatrixI(A->diag)    = i_diag;
   hypre_CSRMatrixJ(A->diag)    = j_diag;
   hypre_CSRMatrixData(A->diag) = a_diag;
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd)    = i_offd;
   hypre_CSRMatrixJ(A->offd)    = j_offd;
   hypre_CSRMatrixData(A->offd) = a_offd;
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(MPI_Comm comm, int nrows, int glob_nrows,
                               int glob_ncols, int *I, int *J, double *data,
                               int *rows, int *cols)
{
   // construct the local CSR matrix
   int nnz = I[nrows];
   hypre_CSRMatrix *local = hypre_CSRMatrixCreate(nrows, glob_ncols, nnz);
   hypre_CSRMatrixI(local) = I;
   hypre_CSRMatrixJ(local) = J;
   hypre_CSRMatrixData(local) = data;
   hypre_CSRMatrixRownnz(local) = NULL;
   hypre_CSRMatrixOwnsData(local) = 1;
   hypre_CSRMatrixNumRownnz(local) = nrows;

   int part_size, myid;
   if (HYPRE_AssumedPartitionCheck())
   {
      myid = 0;
      part_size = 2;
   }
   else
   {
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &part_size);
      part_size++;
   }

   // copy in the row and column partitionings
   HYPRE_Int *row_starts = hypre_TAlloc(HYPRE_Int, part_size);
   HYPRE_Int *col_starts = hypre_TAlloc(HYPRE_Int, part_size);
   for (int i = 0; i < part_size; i++)
   {
      row_starts[i] = rows[i];
      col_starts[i] = cols[i];
   }

   // construct the global ParCSR matrix
   A = hypre_ParCSRMatrixCreate(comm, glob_nrows, glob_ncols,
                                row_starts, col_starts, 0, 0, 0);
   hypre_ParCSRMatrixOwnsRowStarts(A) = 1;
   hypre_ParCSRMatrixOwnsColStarts(A) = 1;
   GenerateDiagAndOffd(local, A, col_starts[myid], col_starts[myid+1]-1);
   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_MatvecCommPkgCreate(A);

   // delete the local CSR matrix
   hypre_CSRMatrixI(local) = NULL;
   hypre_CSRMatrixJ(local) = NULL;
   hypre_CSRMatrixData(local) = NULL;
   hypre_CSRMatrixDestroy(local);

   CommPkg = NULL;
   X = Y = NULL;
   size = GetNumRows();
}

void HypreParMatrix::SetCommPkg(hypre_ParCSRCommPkg *comm_pkg)
{
   CommPkg = comm_pkg;

   if (hypre_ParCSRMatrixCommPkg(A))
      hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;
}

void HypreParMatrix::CheckCommPkg()
{
#ifdef MFEM_DEBUG
   if (CommPkg == NULL || CommPkg != hypre_ParCSRMatrixCommPkg(A))
      mfem_error("\nHypreParMatrix::CheckCommPkg()");
#endif
}

void HypreParMatrix::DestroyCommPkg()
{
   if (CommPkg == NULL)
      return;
   hypre_TFree(CommPkg->send_procs);
   hypre_TFree(CommPkg->send_map_starts);
   hypre_TFree(CommPkg->send_map_elmts);
   hypre_TFree(CommPkg->recv_procs);
   hypre_TFree(CommPkg->recv_vec_starts);
   if (CommPkg->send_mpi_types)
      hypre_TFree(CommPkg->send_mpi_types);
   if (CommPkg->recv_mpi_types)
      hypre_TFree(CommPkg->recv_mpi_types);
   if (hypre_ParCSRMatrixCommPkg(A) == CommPkg)
      hypre_ParCSRMatrixCommPkg(A) = NULL;
   delete CommPkg;
   CommPkg = NULL;
}

HypreParMatrix::operator hypre_ParCSRMatrix*()
{
   return (this) ? A : NULL;
}

HypreParMatrix::operator HYPRE_ParCSRMatrix()
{
   return (this) ? (HYPRE_ParCSRMatrix) A : (HYPRE_ParCSRMatrix) NULL;
}

hypre_ParCSRMatrix* HypreParMatrix::StealData()
{
   hypre_ParCSRMatrix *R = A;
   A = NULL;
   return R;
}

void HypreParMatrix::GetDiag(Vector &diag)
{
   int size=hypre_CSRMatrixNumRows(A->diag);
   diag.SetSize(size);
   for (int j = 0; j < size; j++)
   {
      diag(j) = A->diag->data[A->diag->i[j]];
#ifdef MFEM_DEBUG
      if (A->diag->j[A->diag->i[j]] != j)
         mfem_error("HypreParMatrix::GetDiag");
#endif
   }
}


HypreParMatrix * HypreParMatrix::Transpose()
{
   hypre_ParCSRMatrix * At;
   hypre_ParCSRMatrixTranspose(A, &At, 1);
   hypre_ParCSRMatrixSetNumNonzeros(At);

   hypre_MatvecCommPkgCreate(At);

   return new HypreParMatrix(At);
}

int HypreParMatrix::Mult(HypreParVector &x, HypreParVector &y,
                         double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a, A, x, b, y);
}

void HypreParMatrix::Mult(const Vector &x, Vector &y) const
{
   if (X == NULL)
   {
      X = new HypreParVector(GetGlobalNumCols(),
                             x.GetData(),
                             GetColStarts());
      Y = new HypreParVector(GetGlobalNumRows(),
                             y.GetData(),
                             GetRowStarts());
   }
   else
   {
      X -> SetData(x.GetData());
      Y -> SetData(y.GetData());
   }

   hypre_ParCSRMatrixMatvec(1.0, A, *X, 0.0, *Y);
}

void HypreParMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   if (X == NULL)
   {
      X = new HypreParVector(GetGlobalNumCols(),
                             x.GetData(),
                             GetColStarts());
      Y = new HypreParVector(GetGlobalNumRows(),
                             y.GetData(),
                             GetRowStarts());
   }
   else
   {
      X -> SetData(x.GetData());
      Y -> SetData(y.GetData());
   }

   hypre_ParCSRMatrixMatvecT(1.0, A, *X, 0.0, *Y);
}

int HypreParMatrix::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                         double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a,A,(hypre_ParVector *)x,b,(hypre_ParVector *)y);
}

int HypreParMatrix::MultTranspose(HypreParVector & x, HypreParVector & y,
                                  double a, double b)
{
   return hypre_ParCSRMatrixMatvecT(a,A,x,b,y);
}

void HypreParMatrix::Print(const char *fname, int offi, int offj)
{
   hypre_ParCSRMatrixPrintIJ(A,offi,offj,fname);
}

void HypreParMatrix::Read(const char *fname)
{
   if (A) hypre_ParCSRMatrixDestroy(A);
   int io,jo;
   hypre_ParCSRMatrixReadIJ(MPI_COMM_WORLD, fname, &io, &jo, &A);
   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);
}

HypreParMatrix::~HypreParMatrix()
{
   DestroyCommPkg();

   if (A)
   {
      if (hypre_ParCSRMatrixCommPkg(A))
         hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));
      hypre_ParCSRMatrixCommPkg(A) = NULL;

      if (hypre_CSRMatrixOwnsData(A->diag))
      {
         hypre_CSRMatrixDestroy(A->diag);
         A->diag = NULL;
      }
      else
      {
         if (hypre_CSRMatrixRownnz(A->diag))
            hypre_TFree(hypre_CSRMatrixRownnz(A->diag));
         hypre_TFree(A->diag);
         A->diag = NULL;
      }

      if (hypre_CSRMatrixOwnsData(A->offd))
      {
         hypre_CSRMatrixDestroy(A->offd);
         A->offd = NULL;
      }
      else
      {
         if (hypre_CSRMatrixRownnz(A->offd))
            hypre_TFree(hypre_CSRMatrixRownnz(A->offd));
      }
      hypre_ParCSRMatrixDestroy(A);
   }

   delete X;
   delete Y;
}

HypreParMatrix * ParMult(HypreParMatrix *A, HypreParMatrix *B)
{
   hypre_ParCSRMatrix * ab;
   ab = hypre_ParMatmul(*A,*B);

   hypre_MatvecCommPkgCreate(ab);

   return new HypreParMatrix(ab);
}

HypreParMatrix * RAP(HypreParMatrix *A, HypreParMatrix *P)
{
   int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
   hypre_ParCSRMatrix * rap;
   hypre_BoomerAMGBuildCoarseOperator(*P,*A,*P,&rap);

   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);
   if (!P_owns_its_col_starts)
   {
      /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
         from P (even if it does not own them)! */
      hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
      hypre_ParCSRMatrixSetColStartsOwner(rap,0);
   }
   return new HypreParMatrix(rap);
}

void EliminateBC(HypreParMatrix &A, HypreParMatrix &Ae,
                 Array<int> &ess_dof_list,
                 HypreParVector &x, HypreParVector &b)
{
   // b -= Ae*x
   Ae.Mult(x, b, -1.0, 1.0);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A);
   double *data = hypre_CSRMatrixData(A_diag);
   int    *I    = hypre_CSRMatrixI(A_diag);
#ifdef MFEM_DEBUG
   int    *J    = hypre_CSRMatrixJ(A_diag);
   int *I_offd  =
      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix *)A));
#endif

   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      b(r) = data[I[r]] * x(r);
#ifdef MFEM_DEBUG
      // Check that in the rows specified by the ess_dof_list, the matrix A has
      // only one entry -- the diagonal.
      if (I[r+1] != I[r]+1 || J[I[r]] != r || I_offd[r] != I_offd[r+1])
         mfem_error("EliminateBC (hypre.cpp)");
#endif
   }
}


HypreSolver::HypreSolver()
{
   size = 0;

   A = NULL;
   setup_called = 0;
   B = X = NULL;
}

HypreSolver::HypreSolver(HypreParMatrix *_A)
{
   size = _A -> GetNumRows();

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
      B = new HypreParVector(A -> GetGlobalNumRows(),
                             b.GetData(),
                             A -> GetRowStarts());
      X = new HypreParVector(A -> GetGlobalNumCols(),
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
   if (B) delete B;
   if (X) delete X;
}


HyprePCG::HyprePCG(HypreParMatrix &_A) : HypreSolver(&_A)
{
   MPI_Comm comm;

   print_level = 0;
   use_zero_initial_iterate = 0;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
}

void HyprePCG::SetTol(double tol)
{
   HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
}

void HyprePCG::SetMaxIter(int max_iter)
{
   HYPRE_ParCSRPCGSetMaxIter(pcg_solver, max_iter);
}

void HyprePCG::SetLogging(int logging)
{
   HYPRE_ParCSRPCGSetLogging(pcg_solver, logging);
}

void HyprePCG::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level);
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
      HYPRE_PCGSetRecomputeResidualP(pcg_solver, res_frequency);
   if (rtol > 0.0)
      HYPRE_PCGSetResidualTol(pcg_solver, rtol);
}

void HyprePCG::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   int time_index;
   int num_iterations;
   double final_res_norm;
   MPI_Comm comm;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!setup_called)
   {
      if (print_level > 0)
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRPCGSetup(pcg_solver, *A, b, x);
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
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
   }

   if (use_zero_initial_iterate)
      x = 0.0;

   HYPRE_ParCSRPCGSolve(pcg_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

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

   print_level = 0;
   use_zero_initial_iterate = 0;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRGMRESCreate(comm, &gmres_solver);
   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetTol(double tol)
{
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetMaxIter(int max_iter)
{
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
}

void HypreGMRES::SetKDim(int k_dim)
{
   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
}

void HypreGMRES::SetLogging(int logging)
{
   HYPRE_ParCSRGMRESSetLogging(gmres_solver, logging);
}

void HypreGMRES::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
   HYPRE_ParCSRGMRESSetPrintLevel(gmres_solver, print_level);
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
   int time_index;
   int num_iterations;
   double final_res_norm;
   MPI_Comm comm;

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

   if (use_zero_initial_iterate)
      x = 0.0;

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


HypreBoomerAMG::HypreBoomerAMG(HypreParMatrix &A) : HypreSolver(&A)
{
   int coarsen_type = 10;
   int agg_levels   = 1;
   int relax_type   = 8;
   int relax_sweeps = 1;
   double theta     = 0.25;
   int interp_type  = 6;
   int Pmax         = 4;
   int print_level  = 1;

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
}

void HypreBoomerAMG::SetSystemsOptions(int dim)
{
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

   // More robust options with respect to convergence
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);
}

HypreBoomerAMG::~HypreBoomerAMG()
{
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

   int p = edge_fespace->GetOrder(0);
   int dim = edge_fespace->GetMesh()->Dimension();

   HYPRE_AMSCreate(&ams);

   HYPRE_AMSSetDimension(ams, dim); // 2D H(div) and 3D H(curl) problems
   HYPRE_AMSSetTol(ams, 0.0);
   HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
   HYPRE_AMSSetCycleType(ams, cycle_type);
   HYPRE_AMSSetPrintLevel(ams, 1);

   // define the nodal linear finite element space associated with edge_fespace
   ParMesh *pmesh = (ParMesh *) edge_fespace->GetMesh();
   FiniteElementCollection *vert_fec = new H1_FECollection(p, dim);
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh, vert_fec);

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
         if (dim == 3)
            z_coord(i) = coord[2];
      }
      x = x_coord.ParallelAverage();
      y = y_coord.ParallelAverage();
      if (dim == 2)
      {
         z = NULL;
         HYPRE_AMSSetCoordinateVectors(ams, *x, *y, NULL);
      }
      else
      {
         z = z_coord.ParallelAverage();
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
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   HYPRE_AMSSetDiscreteGradient(ams, *G);
   delete grad;

   // generate and set the Nedelec interpolation matrices
   Pi = Pix = Piy = Piz = NULL;
   if (p > 1)
   {
      ParFiniteElementSpace *vert_fespace_d;
      if (cycle_type < 10)
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, dim,
                                                    Ordering::byVDIM);
      else
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, dim,
                                                    Ordering::byNODES);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      id_ND->AddDomainInterpolator(new IdentityInterpolator);
      id_ND->Assemble();

      if (cycle_type < 10)
      {
         id_ND->Finalize();
         Pi = id_ND->ParallelAssemble();
      }
      else
      {
         Array2D<HypreParMatrix *> Pi_blocks;
         id_ND->GetParBlocks(Pi_blocks);
         Pix = Pi_blocks(0,0);
         Piy = Pi_blocks(0,1);
         if (dim == 3)
            Piz = Pi_blocks(0,2);
      }

      HYPRE_AMSSetInterpolations(ams, *Pi, *Pix, *Piy, *Piz);

      delete vert_fespace_d;
   }

   delete vert_fespace;
   delete vert_fec;

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
   int amg_rlx_type     = 6;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
   int ams_cycle_type   = 14;

   int p = face_fespace->GetOrder(0);

   HYPRE_ADSCreate(&ads);

   HYPRE_ADSSetTol(ads, 0.0);
   HYPRE_ADSSetMaxIter(ads, 1); // use as a preconditioner
   HYPRE_ADSSetCycleType(ads, cycle_type);
   HYPRE_ADSSetPrintLevel(ads, 1);

   // define the nodal and edge finite element spaces associated with face_fespace
   ParMesh *pmesh = (ParMesh *) face_fespace->GetMesh();
   FiniteElementCollection *vert_fec   = new H1_FECollection(p, 3);
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh, vert_fec);
   FiniteElementCollection *edge_fec   = new ND_FECollection(p, 3);
   ParFiniteElementSpace *edge_fespace = new ParFiniteElementSpace(pmesh, edge_fec);

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
      x = x_coord.ParallelAverage();
      y = y_coord.ParallelAverage();
      z = z_coord.ParallelAverage();
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
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
   curl->Finalize();
   C = curl->ParallelAssemble();
   HYPRE_ADSSetDiscreteCurl(ads, *C);
   delete curl;

   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   HYPRE_ADSSetDiscreteGradient(ads, *G);
   delete grad;

   // generate and set the Nedelec and Raviart-Thomas interpolation matrices
   RT_Pi = RT_Pix = RT_Piy = RT_Piz = NULL;
   ND_Pi = ND_Pix = ND_Piy = ND_Piz = NULL;
   if (p > 1)
   {
      ParFiniteElementSpace *vert_fespace_d;

      if (ams_cycle_type < 10)
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                    Ordering::byVDIM);
      else
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                    Ordering::byNODES);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      id_ND->AddDomainInterpolator(new IdentityInterpolator);
      id_ND->Assemble();

      if (ams_cycle_type < 10)
      {
         id_ND->Finalize();
         ND_Pi = id_ND->ParallelAssemble();
      }
      else
      {
         Array2D<HypreParMatrix *> ND_Pi_blocks;
         id_ND->GetParBlocks(ND_Pi_blocks);
         ND_Pix = ND_Pi_blocks(0,0);
         ND_Piy = ND_Pi_blocks(0,1);
         ND_Piz = ND_Pi_blocks(0,2);
      }

      if (cycle_type < 10 && ams_cycle_type > 10)
      {
         delete vert_fespace_d;
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                    Ordering::byVDIM);
      }
      else if (cycle_type > 10 && ams_cycle_type < 10)
      {
         delete vert_fespace_d;
         vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                    Ordering::byNODES);
      }

      ParDiscreteLinearOperator *id_RT;
      id_RT = new ParDiscreteLinearOperator(vert_fespace_d, face_fespace);
      id_RT->AddDomainInterpolator(new IdentityInterpolator);
      id_RT->Assemble();

      if (cycle_type < 10)
      {
         id_RT->Finalize();
         RT_Pi = id_RT->ParallelAssemble();
      }
      else
      {
         Array2D<HypreParMatrix *> RT_Pi_blocks;
         id_RT->GetParBlocks(RT_Pi_blocks);
         RT_Pix = RT_Pi_blocks(0,0);
         RT_Piy = RT_Pi_blocks(0,1);
         RT_Piz = RT_Pi_blocks(0,2);
      }

      HYPRE_ADSSetInterpolations(ads,
                                 *RT_Pi, *RT_Pix, *RT_Piy, *RT_Piz,
                                 *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz);

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

#endif
