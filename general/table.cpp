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

// Implementation of data types Table.

#include "array.hpp"
#include "table.hpp"
#include "error.hpp"

#include "../general/mem_manager.hpp"
#include <iostream>
#include <iomanip>

namespace mfem
{

using namespace std;

Table::Table(const Table &table1,
             const Table &table2, int offset)
{
   MFEM_ASSERT(table1.size == table2.size,
               "Tables have different sizes can not merge.");
   size = table1.size;

   const int nnz = table1.I[size] + table2.I[size];
   I.SetSize(size+1);
   J.SetSize(nnz);

   I[0] = 0;
   Array<int> row;
   for (int i = 0; i < size; i++)
   {
      I[i+1] = I[i];

      table1.GetRow(i, row);
      for (int r = 0; r < row.Size(); r++,  I[i+1] ++)
      {
         J[ I[i+1] ] = row[r];
      }

      table2.GetRow(i, row);
      for (int r = 0; r < row.Size(); r++,  I[i+1] ++)
      {
         J[ I[i+1] ] = (row[r] < 0) ? row[r] - offset : row[r] + offset;
      }

   }
}

Table::Table(const Table &table1,
             const Table &table2, int offset2,
             const Table &table3, int offset3)
{
   MFEM_ASSERT(table1.size == table2.size,
               "Tables have different sizes can not merge.");
   MFEM_ASSERT(table1.size == table3.size,
               "Tables have different sizes can not merge.");
   size = table1.size;

   const int nnz = table1.I[size] + table2.I[size] + table3.I[size];
   I.SetSize(size+1);
   J.SetSize(nnz);

   I[0] = 0;
   Array<int> row;
   for (int i = 0; i < size; i++)
   {
      I[i+1] = I[i];

      table1.GetRow(i, row);
      for (int r = 0; r < row.Size(); r++,  I[i+1] ++)
      {
         J[ I[i+1] ] = row[r];
      }

      table2.GetRow(i, row);
      for (int r = 0; r < row.Size(); r++,  I[i+1] ++)
      {
         J[ I[i+1] ] = (row[r] < 0) ? row[r] - offset2 : row[r] + offset2;
      }

      table3.GetRow(i, row);
      for (int r = 0; r < row.Size(); r++,  I[i+1] ++)
      {
         J[ I[i+1] ] = (row[r] < 0) ? row[r] - offset3 : row[r] + offset3;
      }
   }
}

Table::Table (int dim, int connections_per_row)
{
   int i, j, sum = dim * connections_per_row;

   size = dim;
   I.SetSize(size+1);
   J.SetSize(sum);

   I[0] = 0;
   for (i = 1; i <= size; i++)
   {
      I[i] = I[i-1] + connections_per_row;
      for (j = I[i-1]; j < I[i]; j++) { J[j] = -1; }
   }
}

Table::Table(int nrows, int *partitioning)
{
   size = nrows;

   I.SetSize(size+1);
   J.SetSize(size);

   for (int i = 0; i < size; i++)
   {
      I[i] = i;
      J[i] = partitioning[i];
   }
   I[size] = size;
}

void Table::MakeI(int nrows)
{
   SetDims(nrows, 0);

   for (int i = 0; i <= nrows; i++)
   {
      I[i] = 0;
   }
}

void Table::MakeJ()
{
   int i, j, k;

   for (k = i = 0; i < size; i++)
   {
      j = I[i], I[i] = k, k += j;
   }

   J.SetSize(I[size]=k);
}

void Table::AddConnections(int r, const int *c, int nc)
{
   int *jp = J+I[r];

   for (int i = 0; i < nc; i++)
   {
      jp[i] = c[i];
   }
   I[r] += nc;
}

void Table::ShiftUpI()
{
   for (int i = size; i > 0; i--)
   {
      I[i] = I[i-1];
   }
   I[0] = 0;
}

void Table::SetSize(int dim, int connections_per_row)
{
   SetDims (dim, dim * connections_per_row);

   if (size > 0)
   {
      I[0] = 0;
      for (int i = 0, j = 0; i < size; i++)
      {
         int end = I[i] + connections_per_row;
         I[i+1] = end;
         for ( ; j < end; j++) { J[j] = -1; }
      }
   }
}

void Table::SetDims(int rows, int nnz)
{
   int j;

   j = (I) ? (I[size]) : (0);
   if (size != rows)
   {
      size = rows;
      (rows >= 0) ? I.SetSize(rows+1) : I.DeleteAll();
   }

   if (j != nnz)
   {
      (nnz > 0) ? J.SetSize(nnz) : J.DeleteAll();
   }

   if (size >= 0)
   {
      I[0] = 0;
      I[size] = nnz;
   }
}

int Table::operator()(int i, int j) const
{
   if ( i>=size || i<0 )
   {
      return -1;
   }

   int k, end = I[i+1];
   for (k = I[i]; k < end; k++)
   {
      if (J[k] == j)
      {
         return k;
      }
      else if (J[k] == -1)
      {
         return -1;
      }
   }
   return -1;
}

void Table::GetRow(int i, Array<int> &row) const
{
   MFEM_ASSERT(i >= 0 && i < size, "Row index " << i << " is out of range [0,"
               << size << ')');

   HostReadJ();
   HostReadI();

   row.SetSize(RowSize(i));
   row.Assign(GetRow(i));
}

void Table::SortRows()
{
   for (int r = 0; r < size; r++)
   {
      std::sort(J + I[r], J + I[r+1]);
   }
}

void Table::SetIJ(int *newI, int *newJ, int newsize)
{
   if (newsize >= 0)
   {
      size = newsize;
   }
   I.MakeRef(newI, size + 1, true);
   J.MakeRef(newJ, I[size], true);
}

int Table::Push(int i, int j)
{
   MFEM_ASSERT(i >=0 &&
               i<size, "Index out of bounds.  i = " << i << " size " << size);

   for (int k = I[i], end = I[i+1]; k < end; k++)
   {
      if (J[k] == j)
      {
         return k;
      }
      else if (J[k] == -1)
      {
         J[k] = j;
         return k;
      }
   }

   MFEM_ABORT("Reached end of loop unexpectedly: (i,j) = (" << i << ", " << j
              << ")");

   return -1;
}

void Table::Finalize()
{
   int i, j, end, sum = 0, n = 0, newI = 0;

   for (i=0; i<I[size]; i++)
   {
      if (J[i] != -1)
      {
         sum++;
      }
   }

   if (sum != I[size])
   {
      Array<int> NewJ(sum);

      for (i=0; i<size; i++)
      {
         end = I[i+1];
         for (j=I[i]; j<end; j++)
         {
            if (J[j] == -1) { break; }
            NewJ[ n++ ] = J[j];
         }
         I[i] = newI;
         newI = n;
      }
      I[size] = sum;

      mfem::Swap(J, NewJ);

      MFEM_ASSERT(sum == n, "sum = " << sum << ", n = " << n);
   }
}

void Table::MakeFromList(int nrows, const Array<Connection> &list)
{
   Clear();

   size = nrows;
   int nnz = list.Size();

   I.SetSize(size+1);
   J.SetSize(nnz);

   for (int i = 0, k = 0; i <= size; i++)
   {
      I[i] = k;
      while (k < nnz && list[k].from == i)
      {
         J[k] = list[k].to;
         k++;
      }
   }
}

int Table::Width() const
{
   int width = -1, nnz = (size >= 0) ? I[size] : 0;
   for (int k = 0; k < nnz; k++)
   {
      if (J[k] > width) { width = J[k]; }
   }
   return width + 1;
}

void Table::Print(std::ostream & os, int width) const
{
   int i, j;

   for (i = 0; i < size; i++)
   {
      os << "[row " << i << "]\n";
      for (j = I[i]; j < I[i+1]; j++)
      {
         os << setw(5) << J[j];
         if ( !((j+1-I[i]) % width) )
         {
            os << '\n';
         }
      }
      if ((j-I[i]) % width)
      {
         os << '\n';
      }
   }
}

void Table::PrintMatlab(std::ostream & os) const
{
   int i, j;

   for (i = 0; i < size; i++)
   {
      for (j = I[i]; j < I[i+1]; j++)
      {
         os << i << " " << J[j] << " 1. \n";
      }
   }

   os << flush;
}

void Table::Save(std::ostream &os) const
{
   os << size << '\n';

   for (int i = 0; i <= size; i++)
   {
      os << I[i] << '\n';
   }
   for (int i = 0, nnz = I[size]; i < nnz; i++)
   {
      os << J[i] << '\n';
   }
}

void Table::Load(std::istream &in)
{
   in >> size;
   I.SetSize(size+1);
   for (int i = 0; i <= size; i++)
   {
      in >> I[i];
   }
   int nnz = I[size];
   J.SetSize(nnz);
   for (int j = 0; j < nnz; j++)
   {
      in >> J[j];
   }
}

void Table::Clear()
{
   I.DeleteAll();
   J.DeleteAll();
   size = -1;
}

void Table::Copy(Table & copy) const
{
   copy = *this;
}

void Table::Swap(Table & other)
{
   mfem::Swap(*this, other);
}

std::size_t Table::MemoryUsage() const
{
   if (size < 0 || I == NULL) { return 0; }
   return (size+1 + I[size]) * sizeof(int);
}

void Transpose(const Table &A, Table &At, int ncols_A_)
{
   const int *i_A     = A.GetI();
   const int *j_A     = A.GetJ();
   const int  nrows_A = A.Size();
   const int  ncols_A = (ncols_A_ < 0) ? A.Width() : ncols_A_;
   const int  nnz_A   = i_A[nrows_A];

   At.SetDims (ncols_A, nnz_A);

   int *i_At = At.GetI();
   int *j_At = At.GetJ();

   for (int i = 0; i <= ncols_A; i++)
   {
      i_At[i] = 0;
   }
   for (int i = 0; i < nnz_A; i++)
   {
      i_At[j_A[i]+1]++;
   }
   for (int i = 1; i < ncols_A; i++)
   {
      i_At[i+1] += i_At[i];
   }

   for (int i = 0; i < nrows_A; i++)
   {
      for (int j = i_A[i]; j < i_A[i+1]; j++)
      {
         j_At[i_At[j_A[j]]++] = i;
      }
   }
   for (int i = ncols_A; i > 0; i--)
   {
      i_At[i] = i_At[i-1];
   }
   i_At[0] = 0;
}

Table * Transpose(const Table &A)
{
   Table * At = new Table;
   Transpose(A, *At);
   return At;
}

void Transpose(const Array<int> &A, Table &At, int ncols_A_)
{
   At.MakeI((ncols_A_ < 0) ? (A.Max() + 1) : ncols_A_);
   for (int i = 0; i < A.Size(); i++)
   {
      At.AddAColumnInRow(A[i]);
   }
   At.MakeJ();
   for (int i = 0; i < A.Size(); i++)
   {
      At.AddConnection(A[i], i);
   }
   At.ShiftUpI();
}

void Mult(const Table &A, const Table &B, Table &C)
{
   int  i, j, k, l, m;
   const int *i_A     = A.GetI();
   const int *j_A     = A.GetJ();
   const int *i_B     = B.GetI();
   const int *j_B     = B.GetJ();
   const int  nrows_A = A.Size();
   const int  nrows_B = B.Size();
   const int  ncols_A = A.Width();
   const int  ncols_B = B.Width();

   MFEM_VERIFY( ncols_A <= nrows_B, "Table size mismatch: ncols_A = " << ncols_A
                << ", nrows_B = " << nrows_B);

   Array<int> B_marker (ncols_B);

   for (i = 0; i < ncols_B; i++)
   {
      B_marker[i] = -1;
   }

   int counter = 0;
   for (i = 0; i < nrows_A; i++)
   {
      for (j = i_A[i]; j < i_A[i+1]; j++)
      {
         k = j_A[j];
         for (l = i_B[k]; l < i_B[k+1]; l++)
         {
            m = j_B[l];
            if (B_marker[m] != i)
            {
               B_marker[m] = i;
               counter++;
            }
         }
      }
   }

   C.SetDims (nrows_A, counter);

   for (i = 0; i < ncols_B; i++)
   {
      B_marker[i] = -1;
   }

   int *i_C = C.GetI();
   int *j_C = C.GetJ();
   counter = 0;
   for (i = 0; i < nrows_A; i++)
   {
      i_C[i] = counter;
      for (j = i_A[i]; j < i_A[i+1]; j++)
      {
         k = j_A[j];
         for (l = i_B[k]; l < i_B[k+1]; l++)
         {
            m = j_B[l];
            if (B_marker[m] != i)
            {
               B_marker[m] = i;
               j_C[counter] = m;
               counter++;
            }
         }
      }
   }
}


Table * Mult(const Table &A, const Table &B)
{
   Table * C = new Table;
   Mult(A,B,*C);
   return C;
}

STable::STable(int dim, int connections_per_row) :
   Table(dim, connections_per_row)
{}

int STable::operator()(int i, int j) const
{
   if (i < j)
   {
      return Table::operator()(i,j);
   }
   else
   {
      return Table::operator()(j,i);
   }
}

int STable::Push( int i, int j )
{
   if (i < j)
   {
      return Table::Push(i, j);
   }
   else
   {
      return Table::Push(j, i);
   }
}


DSTable::DSTable(int nrows)
{
   Rows = new Node*[nrows];
   for (int i = 0; i < nrows; i++)
   {
      Rows[i] = NULL;
   }
   NumRows = nrows;
   NumEntries = 0;
}

int DSTable::Push_(int r, int c)
{
   MFEM_ASSERT(r >= 0 && r < NumRows,
               "Row out of bounds: r = " << r << ", NumRows = " << NumRows);
   Node *n;
   for (n = Rows[r]; n != NULL; n = n->Prev)
   {
      if (n->Column == c)
      {
         return (n->Index);
      }
   }
#ifdef MFEM_USE_MEMALLOC
   n = NodesMem.Alloc ();
#else
   n = new Node;
#endif
   n->Column = c;
   n->Index  = NumEntries;
   n->Prev   = Rows[r];
   Rows[r]   = n;
   return (NumEntries++);
}

int DSTable::Index(int r, int c) const
{
   MFEM_ASSERT( r>=0, "Row index must be non-negative, not "<<r);
   if (r >= NumRows)
   {
      return (-1);
   }
   for (Node *n = Rows[r]; n != NULL; n = n->Prev)
   {
      if (n->Column == c)
      {
         return (n->Index);
      }
   }
   return (-1);
}

DSTable::~DSTable()
{
#ifdef MFEM_USE_MEMALLOC
   // NodesMem.Clear();  // this is done implicitly
#else
   for (int i = 0; i < NumRows; i++)
   {
      Node *na, *nb = Rows[i];
      while (nb != NULL)
      {
         na = nb;
         nb = nb->Prev;
         delete na;
      }
   }
#endif
   delete [] Rows;
}

}
