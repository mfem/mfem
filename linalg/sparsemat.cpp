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

// Implementation of sparse matrix

#include "linalg.hpp"
#include "../general/table.hpp"
#include "../general/sort_pairs.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace mfem
{

using namespace std;

SparseMatrix::SparseMatrix(int nrows, int ncols)
   : AbstractSparseMatrix(nrows, (ncols >= 0) ? ncols : nrows),
     I(NULL),
     J(NULL),
     A(NULL),
     Rows(new RowNode *[nrows]),
     current_row(-1),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     ownGraph(true),
     ownData(true),
     isSorted(false)
{
   for (int i = 0; i < nrows; i++)
   {
      Rows[i] = NULL;
   }

#ifdef MFEM_USE_MEMALLOC
   NodesMem = new RowNodeAlloc;
#endif
}

SparseMatrix::SparseMatrix(int *i, int *j, double *data, int m, int n)
   : AbstractSparseMatrix(m, n),
     I(i),
     J(j),
     A(data),
     Rows(NULL),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     ownGraph(true),
     ownData(true),
     isSorted(false)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
}

SparseMatrix::SparseMatrix(int *i, int *j, double *data, int m, int n,
                           bool ownij, bool owna, bool issorted)
   : AbstractSparseMatrix(m, n),
     I(i),
     J(j),
     A(data),
     Rows(NULL),
     ColPtrJ(NULL),
     ColPtrNode(NULL),
     ownGraph(ownij),
     ownData(owna),
     isSorted(issorted)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif

   if ( A == NULL )
   {
      ownData = true;
      int nnz = I[height];
      A = new double[ nnz ];
      for (int i=0; i<nnz; ++i)
      {
         A[i] = 0.0;
      }
   }
}

SparseMatrix::SparseMatrix(int nrows, int ncols, int rowsize)
   : AbstractSparseMatrix(nrows, ncols)
   , Rows(NULL)
   , ColPtrJ(NULL)
   , ColPtrNode(NULL)
   , ownGraph(true)
   , ownData(true)
   , isSorted(false)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   I = new int[nrows + 1];
   J = new int[nrows * rowsize];
   A = new double[nrows * rowsize];

   for (int i = 0; i <= nrows; i++)
   {
      I[i] = i * rowsize;
   }
}

SparseMatrix::SparseMatrix(const SparseMatrix &mat, bool copy_graph)
   : AbstractSparseMatrix(mat.Height(), mat.Width())
{
   if (mat.Finalized())
   {
      const int nnz = mat.I[height];
      if (copy_graph)
      {
         I = new int[height+1];
         J = new int[nnz];
         memcpy(I, mat.I, sizeof(int)*(height+1));
         memcpy(J, mat.J, sizeof(int)*nnz);
         ownGraph = true;
      }
      else
      {
         I = mat.I;
         J = mat.J;
         ownGraph = false;
      }
      A = new double[nnz];
      memcpy(A, mat.A, sizeof(double)*nnz);
      ownData = true;

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

      I = NULL;
      J = NULL;
      A = NULL;
      ownGraph = true;
      ownData = true;
   }

   current_row = -1;
   ColPtrJ = NULL;
   ColPtrNode = NULL;
   isSorted = mat.isSorted;
}

SparseMatrix::SparseMatrix(const Vector &v)
   : AbstractSparseMatrix(v.Size(), v.Size())
   , Rows(NULL)
   , ColPtrJ(NULL)
   , ColPtrNode(NULL)
   , ownGraph(true)
   , ownData(true)
   , isSorted(true)
{
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   I = new int[height + 1];
   J = new int[height];
   A = new double[height];

   for (int i = 0; i <= height; i++)
   {
      I[i] = i;
   }

   for (int r=0; r<height; r++)
   {
      J[r] = r;
      A[r] = v[r];
   }
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
   I = master.I;
   J = master.J;
   A = master.A;
   isSorted = master.isSorted;
}

void SparseMatrix::SetEmpty()
{
   height = width = 0;
   I = J = NULL;
   A = NULL;
   Rows = NULL;
   current_row = -1;
   ColPtrJ = NULL;
   ColPtrNode = NULL;
#ifdef MFEM_USE_MEMALLOC
   NodesMem = NULL;
#endif
   ownGraph = ownData = isSorted = false;
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
   int out=0;
   int rowSize=0;
   if (I)
   {
      for (int i=0; i < height; ++i)
      {
         rowSize = I[i+1]-I[i];
         out = (out > rowSize) ? out : rowSize;
      }
   }
   else
   {
      for (int i=0; i < height; ++i)
      {
         rowSize = RowSize(i);
         out = (out > rowSize) ? out : rowSize;
      }
   }

   return out;
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

double *SparseMatrix::GetRowEntries(const int row)
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   return A + I[row];
}

const double *SparseMatrix::GetRowEntries(const int row) const
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
   else if ( newWidth == -1)
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

   Array<Pair<int,double> > row;
   for (int j = 0, i = 0; i < height; i++)
   {
      int end = I[i+1];
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
      const double diag = A[j];
      for ( ; j > start; j--)
      {
         J[j] = J[j-1];
         A[j] = A[j-1];
      }
      J[start] = row;
      A[start] = diag;
   }
}

double &SparseMatrix::Elem(int i, int j)
{
   return operator()(i,j);
}

const double &SparseMatrix::Elem(int i, int j) const
{
   return operator()(i,j);
}

double &SparseMatrix::operator()(int i, int j)
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

const double &SparseMatrix::operator()(int i, int j) const
{
   static const double zero = 0.0;

   MFEM_ASSERT(i < height && i >= 0 && j < width && j >= 0,
               "Trying to access element outside of the matrix.  "
               << "height = " << height << ", "
               << "width = " << width << ", "
               << "i = " << i << ", "
               << "j = " << j);

   if (Finalized())
   {
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
   MFEM_VERIFY(height == width,
               "Matrix must be square, not height = " << height << ", width = " << width);
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   d.SetSize(height);

   int j, end;
   for (int i = 0; i < height; i++)
   {

      end = I[i+1];
      for (j = I[i]; j < end; j++)
      {
         if (J[j] == i)
         {
            d[i] = A[j];
            break;
         }
      }
      if (j == end)
      {
         d[i] = 0.;
      }
   }
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
      const double * val = this->GetRowEntries(r);

      for (int cj=0; cj<this->RowSize(r); cj++)
      {
         B(r, col[cj]) = val[cj];
      }
   }
}

void SparseMatrix::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void SparseMatrix::AddMult(const Vector &x, Vector &y, const double a) const
{
   MFEM_ASSERT(width == x.Size(),
               "Input vector size (" << x.Size() << ") must match matrix width (" << width
               << ")");
   MFEM_ASSERT(height == y.Size(),
               "Output vector size (" << y.Size() << ") must match matrix height (" << height
               << ")");

   int i, j, end;
   double *Ap = A, *yp = y.GetData();
   const double *xp = x.GetData();

   if (Ap == NULL)
   {
      //  The matrix is not finalized, but multiplication is still possible
      for (i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         double b = 0.0;
         for ( ; row != NULL; row = row->Prev)
         {
            b += row->Value * xp[row->Column];
         }
         *yp += a * b;
         yp++;
      }
      return;
   }

   int *Jp = J, *Ip = I;

   if (a == 1.0)
   {
#ifndef MFEM_USE_OPENMP
      for (i = j = 0; i < height; i++)
      {
         double d = 0.0;
         for (end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += d;
      }
#else
      #pragma omp parallel for private(j,end)
      for (i = 0; i < height; i++)
      {
         double d = 0.0;
         for (j = Ip[i], end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += d;
      }
#endif
   }
   else
   {
      for (i = j = 0; i < height; i++)
      {
         double d = 0.0;
         for (end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += a * d;
      }
   }
}

void SparseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void SparseMatrix::AddMultTranspose(const Vector &x, Vector &y,
                                    const double a) const
{
   MFEM_ASSERT(height == x.Size(),
               "Input vector size (" << x.Size() << ") must match matrix height (" << height
               << ")");
   MFEM_ASSERT(width == y.Size(),
               "Output vector size (" << y.Size() << ") must match matrix width (" << width
               << ")");

   int i, j, end;
   double *yp = y.GetData();

   if (A == NULL)
   {
      // The matrix is not finalized, but multiplication is still possible
      for (i = 0; i < height; i++)
      {
         RowNode *row = Rows[i];
         double b = a * x(i);
         for ( ; row != NULL; row = row->Prev)
         {
            yp[row->Column] += row->Value * b;
         }
      }
      return;
   }

   for (i = 0; i < height; i++)
   {
      double xi = a * x(i);
      end = I[i+1];
      for (j = I[i]; j < end; j++)
      {
         yp[J[j]] += A[j]*xi;
      }
   }
}

void SparseMatrix::PartMult(
   const Array<int> &rows, const Vector &x, Vector &y) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < rows.Size(); i++)
   {
      int r = rows[i];
      int end = I[r + 1];
      double a = 0.0;
      for (int j = I[r]; j < end; j++)
      {
         a += A[j] * x(J[j]);
      }
      y(r) = a;
   }
}

void SparseMatrix::PartAddMult(
   const Array<int> &rows, const Vector &x, Vector &y, const double a) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < rows.Size(); i++)
   {
      int r = rows[i];
      int end = I[r + 1];
      double val = 0.0;
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

   y.SetSize(Height());
   y = 0;

   for (int i = 0; i < Height(); i++)
   {
      int end = I[i+1];
      for (int j = I[i]; j < end; j++)
      {
         if (x[J[j]])
         {
            y[i] = x[J[j]];
            break;
         }
      }
   }
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

double SparseMatrix::InnerProduct(const Vector &x, const Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "x.Size() = " << x.Size()
               << " must be equal to Width() = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "y.Size() = " << y.Size()
               << " must be equal to Height() = " << Height());
   double prod = 0.0;
   for (int i = 0; i < height; i++)
   {
      double a = 0.0;
      if (A)
         for (int j = I[i], end = I[i+1]; j < end; j++)
         {
            a += A[j] * x(J[j]);
         }
      else
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
         {
            a += np->Value * x(np->Column);
         }
      prod += a * y(i);
   }

   return prod;
}

void SparseMatrix::GetRowSums(Vector &x) const
{
   for (int i = 0; i < height; i++)
   {
      double a = 0.0;
      if (A)
         for (int j = I[i], end = I[i+1]; j < end; j++)
         {
            a += A[j];
         }
      else
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
         {
            a += np->Value;
         }
      x(i) = a;
   }
}

double SparseMatrix::GetRowNorml1(int irow) const
{
   MFEM_VERIFY(irow < height,
               "row " << irow << " not in matrix with height " << height);

   double a = 0.0;
   if (A)
      for (int j = I[irow], end = I[irow+1]; j < end; j++)
      {
         a += fabs(A[j]);
      }
   else
      for (RowNode *np = Rows[irow]; np != NULL; np = np->Prev)
      {
         a += fabs(np->Value);
      }

   return a;
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

   I = new int[height+1];
   I[0] = 0;
   for (i = 1; i <= height; i++)
   {
      nr = 0;
      for (aux = Rows[i-1]; aux != NULL; aux = aux->Prev)
         if (!skip_zeros || aux->Value != 0.0)
         {
            nr++;
         }
      if (fix_empty_rows && !nr) { nr = 1; }
      I[i] = I[i-1] + nr;
   }

   nz = I[height];
   J = new int[nz];
   A = new double[nz];
   // Assume we're sorted until we find out otherwise
   isSorted = true;
   for (j = i = 0; i < height; i++)
   {
      int lastCol = -1;
      nr = 0;
      for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
      {
         if (!skip_zeros || aux->Value != 0.0)
         {
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
         int *bI = new int[nr + 1];
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
         b.J = new int[nnz];
         b.A = new double[nnz];
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

double SparseMatrix::IsSymmetric() const
{
   if (height != width)
   {
      return infinity();
   }

   double symm = 0.0;
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
      for (j = I[i]; j < I[i+1]; j++)
         if (J[j] < i)
         {
            A[j] += (*this)(J[j],i);
            A[j] *= 0.5;
            (*this)(J[j],i) = A[j];
         }
}

int SparseMatrix::NumNonZeroElems() const
{
   if (A != NULL)  // matrix is finalized
   {
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

double SparseMatrix::MaxNorm() const
{
   double m = 0.0;

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
         for (RowNode *n_p = Rows[i]; n_p != NULL; n_p = n_p->Prev)
         {
            m = std::max(m, std::abs(n_p->Value));
         }
   }
   return m;
}

int SparseMatrix::CountSmallElems(double tol) const
{
   int counter = 0;

   if (A)
   {
      const int nz = I[height];
      const double *Ap = A;

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

void SparseMatrix::EliminateRow(int row, const double sol, Vector &rhs)
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
         for (int jpos = I[i]; jpos != I[i+1]; ++jpos)
            if (cols[ J[jpos]] )
            {
               if (x && b)
               {
                  (*b)(i) -= A[jpos] * (*x)( J[jpos] );
               }
               A[jpos] = 0.0;
            }
   }
   else
   {
      for (int i = 0; i < height; i++)
         for (RowNode *aux = Rows[i]; aux != NULL; aux = aux->Prev)
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

void SparseMatrix::EliminateRowCol(int rc, const double sol, Vector &rhs,
                                   DiagonalPolicy dpolicy)
{
   int col;

   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         if ((col = J[j]) == rc)
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
         if ((col = aux->Column) == rc)
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
   int col;
   int num_rhs = rhs.Width();

   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);
   MFEM_ASSERT(sol.Size() == num_rhs, "solution size (" << sol.Size()
               << ") must match rhs width (" << num_rhs << ")");

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
      {
         if ((col = J[j]) == rc)
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
         if ((col = aux->Column) == rc)
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
   int col;

   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
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
            for (int k = I[col]; 1; k++)
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
   else
   {
      RowNode *aux, *node;

      for (aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         if ((col = aux->Column) == rc)
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

// This is almost identical to EliminateRowCol(int, int), except for
// the A[j] = value; and aux->Value = value; lines.
void SparseMatrix::EliminateRowColDiag(int rc, double value)
{
   int col;

   MFEM_ASSERT(rc < height && rc >= 0,
               "Row " << rc << " not in matrix of height " << height);

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
         {
            A[j] = value;
         }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
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
   else
   {
      RowNode *aux, *node;

      for (aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         if ((col = aux->Column) == rc)
         {
            aux->Value = value;
         }
         else
         {
            aux->Value = 0.0;
            for (node = Rows[col]; 1; node = node->Prev)
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

void SparseMatrix::EliminateRowCol(int rc, SparseMatrix &Ae,
                                   DiagonalPolicy dpolicy)
{
   int col;

   if (Rows)
   {
      RowNode *nd, *nd2;
      for (nd = Rows[rc]; nd != NULL; nd = nd->Prev)
      {
         if ((col = nd->Column) == rc)
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
         if ((col = J[j]) == rc)
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

void SparseMatrix::SetDiagIdentity()
{
   for (int i = 0; i < height; i++)
      if (I[i+1] == I[i]+1 && fabs(A[I[i]]) < 1e-16)
      {
         A[I[i]] = 1.0;
      }
}

void SparseMatrix::EliminateZeroRows(const double threshold)
{
   int i, j;
   double zero;

   for (i = 0; i < height; i++)
   {
      zero = 0.0;
      for (j = I[i]; j < I[i+1]; j++)
      {
         zero += fabs(A[j]);
      }
      if (zero <= threshold)
      {
         for (j = I[i]; j < I[i+1]; j++)
            if (J[j] == i)
            {
               A[j] = 1.0;
            }
            else
            {
               A[j] = 0.0;
            }
      }
   }
}

void SparseMatrix::Gauss_Seidel_forw(const Vector &x, Vector &y) const
{
   int c, i, s = height;
   double sum, *yp = y.GetData();
   const double *xp = x.GetData();

   if (A == NULL)
   {
      RowNode *diag_p, *n_p, **R = Rows;

      for (i = 0; i < s; i++)
      {
         sum = 0.0;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
            if ((c = n_p->Column) == i)
            {
               diag_p = n_p;
            }
            else
            {
               sum += n_p->Value * yp[c];
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
      int j, end, d, *Ip = I, *Jp = J;
      double *Ap = A;

      j = Ip[0];
      for (i = 0; i < s; i++)
      {
         end = Ip[i+1];
         sum = 0.0;
         d = -1;
         for ( ; j < end; j++)
            if ((c = Jp[j]) == i)
            {
               d = j;
            }
            else
            {
               sum += Ap[j] * yp[c];
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
   int i, c;
   double sum, *yp = y.GetData();
   const double *xp = x.GetData();

   if (A == NULL)
   {
      RowNode *diag_p, *n_p, **R = Rows;

      for (i = height-1; i >= 0; i--)
      {
         sum = 0.;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
            if ((c = n_p->Column) == i)
            {
               diag_p = n_p;
            }
            else
            {
               sum += n_p->Value * yp[c];
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
      int j, beg, d, *Ip = I, *Jp = J;
      double *Ap = A;

      j = Ip[height]-1;
      for (i = height-1; i >= 0; i--)
      {
         beg = Ip[i];
         sum = 0.;
         d = -1;
         for ( ; j >= beg; j--)
            if ((c = Jp[j]) == i)
            {
               d = j;
            }
            else
            {
               sum += Ap[j] * yp[c];
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

double SparseMatrix::GetJacobiScaling() const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   double sc = 1.0;
   for (int i = 0; i < height; i++)
   {
      int d = -1;
      double norm = 0.0;
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
         double a = 1.8 * fabs(A[d]) / norm;
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
                          double sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < height; i++)
   {
      int d = -1;
      double sum = b(i);
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
         x1(i) = sc * (sum / A[d]) + (1.0 - sc) * x0(i);
      }
      else
      {
         mfem_error("SparseMatrix::Jacobi(...) #2");
      }
   }
}

void SparseMatrix::DiagScale(const Vector &b, Vector &x, double sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   bool scale = (sc != 1.0);
   for (int i = 0, j = 0; i < height; i++)
   {
      int end = I[i+1];
      for ( ; true; j++)
      {
         MFEM_VERIFY(j != end, "Couldn't find diagonal in row. i = " << i
                     << ", j = " << j
                     << ", I[i+1] = " << end );
         if (J[j] == i)
         {
            MFEM_VERIFY(std::abs(A[j]) > 0.0, "Diagonal " << j << " must be nonzero");
            if (scale)
            {
               x(i) = sc * b(i) / A[j];
            }
            else
            {
               x(i) = b(i) / A[j];
            }
            break;
         }
      }
      j = end;
   }
   return;
}

void SparseMatrix::Jacobi2(const Vector &b, const Vector &x0, Vector &x1,
                           double sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < height; i++)
   {
      double resi = b(i), norm = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         resi -= A[j] * x0(J[j]);
         norm += fabs(A[j]);
      }
      if (norm > 0.0)
      {
         x1(i) = x0(i) + sc * resi / norm;
      }
      else
      {
         MFEM_ABORT("L1 norm of row " << i << " is zero.");
      }
   }
}

void SparseMatrix::Jacobi3(const Vector &b, const Vector &x0, Vector &x1,
                           double sc) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   for (int i = 0; i < height; i++)
   {
      double resi = b(i), sum = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         resi -= A[j] * x0(J[j]);
         sum  += A[j];
      }
      if (sum > 0.0)
      {
         x1(i) = x0(i) + sc * resi / sum;
      }
      else
      {
         MFEM_ABORT("sum of row " << i << " is zero.");
      }
   }
}

void SparseMatrix::AddSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                const DenseMatrix &subm, int skip_zeros)
{
   int i, j, gi, gj, s, t;
   double a;

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
            // if the element is zero do not assemble it unless this breaks
            // the symmetric structure
            if (&rows != &cols || subm(j, i) == 0.0)
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

void SparseMatrix::Set(const int i, const int j, const double A)
{
   double a = A;
   int gi, gj, s, t;

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
   _Set_(gi, gj, a);
}

void SparseMatrix::Add(const int i, const int j, const double A)
{
   int gi, gj, s, t;
   double a = A;

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
   double a;

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
         a = subm(i, j);
         if (skip_zeros && a == 0.0)
         {
            continue;
         }
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to insert a column " << gj << " outside the matrix width "
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
   double a;

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
         a = subm(j, i);
         if (skip_zeros && a == 0.0)
         {
            continue;
         }
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to insert a column " << gj << " outside the matrix width "
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
   double a;

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
               "Trying to insert a row " << gi << " outside the matrix height "
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
               "Trying to insert a row " << gi << " outside the matrix height "
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
      cols.MakeRef(J + j, I[gi+1]-j);
      srow.NewDataAndSize(A + j, cols.Size());
      MFEM_ASSERT(row >= 0, "Row not valid: " << row );
      return 1;
   }
}

void SparseMatrix::SetRow(const int row, const Array<int> &cols,
                          const Vector &srow)
{
   int gi, gj, s, t;
   double a;

   if ((gi=row) < 0) { gi = -1-gi, s = -1; }
   else { s = 1; }
   MFEM_ASSERT(gi < height,
               "Trying to insert a row " << gi << " outside the matrix height "
               << height);

   if (!Finalized())
   {
      SetColPtr(gi);
      for (int j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < width,
                     "Trying to insert a column " << gj << " outside the matrix"
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
                     "Trying to insert a column " << gj << " outside the matrix"
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
   double a;

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

void SparseMatrix::ScaleRow(const int row, const double scale)
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
   double scale;
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
               << B.height << ", B.width = " << width);

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

void SparseMatrix::Add(const double a, const SparseMatrix &B)
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

SparseMatrix &SparseMatrix::operator=(double a)
{
   if (Rows == NULL)
      for (int i = 0, nnz = I[height]; i < nnz; i++)
      {
         A[i] = a;
      }
   else
      for (int i = 0; i < height; i++)
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
         {
            node_p -> Value = a;
         }

   return (*this);
}

SparseMatrix &SparseMatrix::operator*=(double a)
{
   if (Rows == NULL)
      for (int i = 0, nnz = I[height]; i < nnz; i++)
      {
         A[i] *= a;
      }
   else
      for (int i = 0; i < height; i++)
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
         {
            node_p -> Value *= a;
         }

   return (*this);
}

void SparseMatrix::Print(std::ostream & out, int _width) const
{
   int i, j;

   if (A == NULL)
   {
      RowNode *nd;
      for (i = 0; i < height; i++)
      {
         out << "[row " << i << "]\n";
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++)
         {
            out << " (" << nd->Column << "," << nd->Value << ")";
            if ( !((j+1) % _width) )
            {
               out << '\n';
            }
         }
         if (j % _width)
         {
            out << '\n';
         }
      }
      return;
   }

   for (i = 0; i < height; i++)
   {
      out << "[row " << i << "]\n";
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << " (" << J[j] << "," << A[j] << ")";
         if ( !((j+1-I[i]) % _width) )
         {
            out << '\n';
         }
      }
      if ((j-I[i]) % _width)
      {
         out << '\n';
      }
   }
}

void SparseMatrix::PrintMatlab(std::ostream & out) const
{
   out << "% size " << height << " " << width << "\n";
   out << "% Non Zeros " << NumNonZeroElems() << "\n";
   int i, j;
   ios::fmtflags old_fmt = out.flags();
   out.setf(ios::scientific);
   std::streamsize old_prec = out.precision(14);

   for (i = 0; i < height; i++)
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << i+1 << " " << J[j]+1 << " " << A[j] << '\n';
      }
   out.precision(old_prec);
   out.flags(old_fmt);
}

void SparseMatrix::PrintMM(std::ostream & out) const
{
   int i, j;
   ios::fmtflags old_fmt = out.flags();
   out.setf(ios::scientific);
   std::streamsize old_prec = out.precision(14);

   out << "%%MatrixMarket matrix coordinate real general" << '\n'
       << "% Generated by MFEM" << '\n';

   out << height << " " << width << " " << NumNonZeroElems() << '\n';
   for (i = 0; i < height; i++)
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << i+1 << " " << J[j]+1 << " " << A[j] << '\n';
      }
   out.precision(old_prec);
   out.flags(old_fmt);
}

void SparseMatrix::PrintCSR(std::ostream & out) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   int i;

   out << height << '\n';  // number of rows

   for (i = 0; i <= height; i++)
   {
      out << I[i]+1 << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      out << J[i]+1 << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      out << A[i] << '\n';
   }
}

void SparseMatrix::PrintCSR2(std::ostream & out) const
{
   MFEM_VERIFY(Finalized(), "Matrix must be finalized.");

   int i;

   out << height << '\n'; // number of rows
   out << width << '\n';  // number of columns

   for (i = 0; i <= height; i++)
   {
      out << I[i] << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      out << J[i] << '\n';
   }

   for (i = 0; i < I[height]; i++)
   {
      out << A[i] << '\n';
   }
}

void SparseMatrix::PrintInfo(std::ostream &out) const
{
   const double MiB = 1024.*1024;
   int nnz = NumNonZeroElems();
   double pz = 100./nnz;
   int nz = CountSmallElems(0.0);
   double max_norm = MaxNorm();
   double symm = IsSymmetric();
   int nnf = CheckFinite();
   int ns12 = CountSmallElems(1e-12*max_norm);
   int ns15 = CountSmallElems(1e-15*max_norm);
   int ns18 = CountSmallElems(1e-18*max_norm);

   out <<
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
      out << "  Memory used by CSR          : " <<
          (sizeof(int)*(height+1+nnz)+sizeof(double)*nnz)/MiB << " MiB\n";
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
      out << "  Memory used by LIL          : " << used_mem/MiB << " MiB\n";
   }
}

void SparseMatrix::Destroy()
{
   if (I != NULL && ownGraph)
   {
      delete [] I;
   }
   if (J != NULL && ownGraph)
   {
      delete [] J;
   }
   if (A != NULL && ownData)
   {
      delete [] A;
   }

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

   if (ColPtrJ != NULL)
   {
      delete [] ColPtrJ;
   }
   if (ColPtrNode != NULL)
   {
      delete [] ColPtrNode;
   }
#ifdef MFEM_USE_MEMALLOC
   if (NodesMem != NULL)
   {
      delete NodesMem;
   }
#endif
}

int SparseMatrix::ActualWidth()
{
   int awidth = 0;
   if (A)
   {
      int *start_j = J;
      int *end_j = J + I[height];
      for (int *jptr = start_j; jptr != end_j; ++jptr)
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

void SparseMatrixFunction (SparseMatrix & S, double (*f)(double))
{
   int n = S.NumNonZeroElems();
   double * s = S.GetData();

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
   int m, n, nnz, *A_i, *A_j, *At_i, *At_j;
   double *A_data, *At_data;

   m      = A.Height(); // number of rows of A
   n      = A.Width();  // number of columns of A
   nnz    = A.NumNonZeroElems();
   A_i    = A.GetI();
   A_j    = A.GetJ();
   A_data = A.GetData();

   At_i = new int[n+1];
   At_j = new int[nnz];
   At_data = new double[nnz];

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

   return  new SparseMatrix (At_i, At_j, At_data, n, m);
}

SparseMatrix *TransposeAbstractSparseMatrix (const AbstractSparseMatrix &A,
                                             int useActualWidth)
{
   int i, j;
   int m, n, nnz, *At_i, *At_j;
   double *At_data;
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

   At_i = new int[n+1];
   At_j = new int[nnz];
   At_data = new double[nnz];

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
   int *A_i, *A_j, *B_i, *B_j, *C_i, *C_j, *B_marker;
   double *A_data, *B_data, *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   double a_entry, b_entry;
   SparseMatrix *C;

   nrowsA = A.Height();
   ncolsA = A.Width();
   nrowsB = B.Height();
   ncolsB = B.Width();

   MFEM_VERIFY(ncolsA == nrowsB,
               "number of columns of A (" << ncolsA
               << ") must equal number of rows of B (" << nrowsB << ")");

   A_i    = A.GetI();
   A_j    = A.GetJ();
   A_data = A.GetData();
   B_i    = B.GetI();
   B_j    = B.GetJ();
   B_data = B.GetData();

   B_marker = new int[ncolsB];

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   if (OAB == NULL)
   {
      C_i = new int[nrowsA+1];

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

      C_j    = new int[num_nonzeros];
      C_data = new double[num_nonzeros];

      C = new SparseMatrix (C_i, C_j, C_data, nrowsA, ncolsB);

      for (ib = 0; ib < ncolsB; ib++)
      {
         B_marker[ib] = -1;
      }
   }
   else
   {
      C = OAB;

      MFEM_VERIFY(nrowsA == C -> Height() && ncolsB == C -> Width(),
                  "Input matrix sizes do not match output sizes"
                  << " nrowsA = " << nrowsA
                  << ", C->Height() = " << C->Height()
                  << " ncolsB = " << ncolsB
                  << ", C->Width() = " << C->Width());

      // C_i    = C -> GetI(); // not used
      C_j    = C -> GetJ();
      C_data = C -> GetData();
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
   double *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   double a_entry, b_entry;
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

   C_i = new int[nrowsA+1];

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

   C_j    = new int[num_nonzeros];
   C_data = new double[num_nonzeros];

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
   DenseMatrix *_RAP = new DenseMatrix(R.Height(), AP->Width());
   Mult (R, *AP, *_RAP);
   delete AP;
   return _RAP;
}

DenseMatrix *RAP(DenseMatrix &A, const SparseMatrix &P)
{
   SparseMatrix *R  = Transpose(P);
   DenseMatrix  *RA = Mult(*R, A);
   DenseMatrix   AtP(*RA, 't');
   delete RA;
   DenseMatrix  *RAtP = Mult(*R, AtP);
   delete R;
   DenseMatrix * _RAP = new DenseMatrix(*RAtP, 't');
   delete RAtP;
   return _RAP;
}

SparseMatrix *RAP (const SparseMatrix &A, const SparseMatrix &R,
                   SparseMatrix *ORAP)
{
   SparseMatrix *P  = Transpose (R);
   SparseMatrix *AP = Mult (A, *P);
   delete P;
   SparseMatrix *_RAP = Mult (R, *AP, ORAP);
   delete AP;
   return _RAP;
}

SparseMatrix *RAP(const SparseMatrix &Rt, const SparseMatrix &A,
                  const SparseMatrix &P)
{
   SparseMatrix * R = Transpose(Rt);
   SparseMatrix * RA = Mult(*R,A);
   delete R;
   SparseMatrix * out = Mult(*RA, P);
   delete RA;
   return out;
}

SparseMatrix *Mult_AtDA (const SparseMatrix &A, const Vector &D,
                         SparseMatrix *OAtDA)
{
   int i, At_nnz, *At_j;
   double *At_data;

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

SparseMatrix * Add(double a, const SparseMatrix & A, double b,
                   const SparseMatrix & B)
{
   int nrows = A.Height();
   int ncols = A.Width();

   int * C_i = new int[nrows+1];
   int * C_j;
   double * C_data;

   int * A_i = A.GetI();
   int * A_j = A.GetJ();
   double * A_data = A.GetData();

   int * B_i = B.GetI();
   int * B_j = B.GetJ();
   double * B_data = B.GetData();

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

   C_j = new int[num_nonzeros];
   C_data = new double[num_nonzeros];

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
         double alpha, DenseMatrix &B)
{
   for (int r = 0; r < B.Height(); r++)
   {
      const int    * colA = A.GetRowColumns(r);
      const double * valA = A.GetRowEntries(r);
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
         C->AddMatrix(A(i,j), const_cast<DenseMatrix&>(B), i * mB, j * nB);
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
            const double * valB = B.GetRowEntries(r);

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
      const double * valA = A.GetRowEntries(r);

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
      const double * valA = A.GetRowEntries(ar);

      for (int aj=0; aj<A.RowSize(ar); aj++)
      {
         for (int br=0; br<mB; br++)
         {
            const int    * colB = B.GetRowColumns(br);
            const double * valB = B.GetRowEntries(br);

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

#ifdef MFEM_USE_MEMALLOC
   mfem::Swap(NodesMem, other.NodesMem);
#endif

   mfem::Swap(ownGraph, other.ownGraph);
   mfem::Swap(ownData, other.ownData);
   mfem::Swap(isSorted, other.isSorted);
}

}
