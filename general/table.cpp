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

// Implementation of data types Table.

#include "array.hpp"
#include "table.hpp"
#include "sort_pairs.hpp"
#include "error.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

// Include the METIS header, if using version 5. If using METIS 4, the needed
// declarations are inlined below, i.e. no header is needed.
#if defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
#include "metis.h"
#endif

namespace mfem
{

using namespace std;

Table::Table(const Table &table)
{
   size = table.size;
   if (size >= 0)
   {
      const int nnz = table.I[size];
      I = new int[size+1];
      J = new int[nnz];
      memcpy(I, table.I, sizeof(int)*(size+1));
      memcpy(J, table.J, sizeof(int)*nnz);
   }
   else
   {
      I = J = NULL;
   }
}

Table::Table (int dim, int connections_per_row)
{
   int i, j, sum = dim * connections_per_row;

   size = dim;
   I = new int[size+1];
   J = new int[sum];

   I[0] = 0;
   for (i = 1; i <= size; i++)
   {
      I[i] = I[i-1] + connections_per_row;
      for (j = I[i-1]; j < I[i]; j++) { J[j] = -1; }
   }
}

Table::Table (int nrows, int *partitioning)
{
   size = nrows;

   I = new int[size+1];
   J = new int[size];

   for (int i = 0; i < size; i++)
   {
      I[i] = i;
      J[i] = partitioning[i];
   }
   I[size] = size;
}

void Table::MakeI (int nrows)
{
   SetDims (nrows, 0);

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

   J = new int[I[size]=k];
}

void Table::AddConnections (int r, const int *c, int nc)
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
      if (I) { delete [] I; }
      I = (rows >= 0) ? (new int[rows+1]) : (NULL);
   }

   if (j != nnz)
   {
      if (J) { delete [] J; }
      J = (nnz > 0) ? (new int[nnz]) : (NULL);
   }

   if (size >= 0)
   {
      I[0] = 0;
      I[size] = nnz;
   }
}

int Table::operator() (int i, int j) const
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
   delete [] I;
   delete [] J;
   I = newI;
   J = newJ;
   if (newsize >= 0)
   {
      size = newsize;
   }
}

int Table::Push(int i, int j)
{
   MFEM_ASSERT( i >=0 && i<size, "Index out of bounds.  i = "<<i);

   for (int k = I[i], end = I[i+1]; k < end; k++)
      if (J[k] == j)
      {
         return k;
      }
      else if (J[k] == -1)
      {
         J[k] = j;
         return k;
      }

   MFEM_ABORT("Reached end of loop unexpectedly: (i,j) = (" << i << ", " << j
              << ")");

   return -1;
}

void Table::Finalize()
{
   int i, j, end, sum = 0, n = 0, newI = 0;

   for (i=0; i<I[size]; i++)
      if (J[i] != -1)
      {
         sum++;
      }

   if (sum != I[size])
   {
      int *NewJ = new int[sum];

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

      delete [] J;

      J = NewJ;

      MFEM_ASSERT(sum == n, "sum = " << sum << ", n = " << n);
   }
}

void Table::MakeFromList(int nrows, const Array<Connection> &list)
{
   Clear();

   size = nrows;
   int nnz = list.Size();

   I = new int[size+1];
   J = new int[nnz];

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

void Table::GetCMReordering(Array<int> &ordering, bool reverse) const
{
   if (size <= 0)
   {
      ordering.SetSize(0);
      return;
   }

   int num_el = size, stack_p, stack_top_p;
   ordering.SetSize(num_el);
   Array<Pair<int,int> > el_stack(num_el);
   Array<int> el_layer;
   el_layer.MakeRef(ordering);

   // Assuming that either all diagonal entries are present or none are present.

   // Choose starting element (for one connected component only).
   int el0 = 0, min_nbrs = RowSize(el0);
   for (int el = 1; el < num_el; el++)
   {
      const int num_nbrs = RowSize(el);
      if (num_nbrs < min_nbrs)
      {
         el0 = el;
         min_nbrs = num_nbrs;
      }
   }

   el_layer = -1;
   stack_p = stack_top_p = 0;
   for (int el = el0; stack_top_p < num_el; el=(el+1)%num_el)
   {
      if (el_layer[el] != -1) { continue; }

      // FIXME: choose starting element for this connected component.

      el_layer[el] = 0;
      el_stack[stack_top_p++] = Pair<int,int>(RowSize(el),el);

      int layer = 0, layer_start = stack_p;
      for ( ; stack_p < stack_top_p; stack_p++)
      {
         const int i = el_stack[stack_p].two;
         for (int j = I[i]; j < I[i+1]; j++)
         {
            int k = J[j];
            if (el_layer[k] == -1)
            {
               el_layer[k] = el_layer[i] + 1;
               el_stack[stack_top_p++] = Pair<int,int>(RowSize(k),k);
            }
         }
         if (stack_p+1 == stack_top_p ||
             layer < el_layer[el_stack[stack_p+1].two])
         {
            std::sort(&el_stack[layer_start], &el_stack[stack_p] + 1);
            layer++;
            layer_start = stack_p+1;
         }
      }
   }

   if (!reverse)
   {
      for (int i = 0; i < num_el; i++)
      {
         ordering[el_stack[i].two] = i;
      }
   }
   else
   {
      for (int i = 0; i < num_el; i++)
      {
         ordering[el_stack[num_el-1-i].two] = i;
      }
   }
}

#ifdef MFEM_USE_GECKO
void Table::GetGeckoReordering(const GeckoParameters &g_params,
                               Array<int> &ordering) const
{
   Gecko::Graph graph;

   // Run through all the elements and insert the nodes in the graph for them
   for (int elemid = 0; elemid < size; ++elemid)
   {
      graph.insert();
   }

   // Run through all the elems and insert arcs to the graph for each element
   // face Indices in Gecko are 1 based hence the +1 on the insertion
   for (int elemid = 0; elemid < size; ++elemid)
   {
      const int num_neigh = RowSize(elemid);
      const int *neighid = GetRow(elemid);
      for (int i = 0; i < num_neigh; ++i)
      {
         if (elemid != neighid[i])
         {
            graph.insert(elemid + 1,  neighid[i] + 1);
         }
      }
   }

   // Get the reordering from Gecko and copy it into the ordering Array<int>
   graph.order(g_params.functional,
               g_params.iterations,
               g_params.window,
               g_params.period,
               g_params.seed);
   ordering.DeleteAll();
   ordering.SetSize(size);
   Gecko::Node::Index NE = size;
   for (Gecko::Node::Index gnodeid = 1; gnodeid <= NE; ++gnodeid)
   {
      ordering[gnodeid - 1] = graph.rank(gnodeid);
   }
}
#endif // #ifdef MFEM_USE_GECKO

#ifdef MFEM_USE_METIS
#ifndef MFEM_USE_METIS_5
// METIS 4 prototypes
typedef int idxtype;
extern "C" {
   void METIS_EdgeND(int *, idxtype *, idxtype *, int *, int *, idxtype *,
                     idxtype *);
   void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *,
                     idxtype *);
}
#endif

void Table::GetMetisReordering(Array<int> &ordering, int type,
                               bool check_for_diag) const
{
   if (size <= 0)
   {
      ordering.SetSize(0);
      return;
   }

#ifndef MFEM_USE_METIS_5
   int numflag = 0;
   int options[8];
#else
   int err;
   int options[METIS_NOPTIONS];
#endif

#ifndef MFEM_USE_METIS_5
   options[0] = 0; // use the default options
#else
   METIS_SetDefaultOptions(options);
   // METIS_OPTION_CTYPE, METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP,
   // METIS_OPTION_NSEPS, METIS_OPTION_NITER, METIS_OPTION_UFACTOR,
   // METIS_OPTION_COMPRESS, METIS_OPTION_CCORDER, METIS_OPTION_SEED,
   // METIS_OPTION_PFACTOR, METIS_OPTION_NUMBERING, METIS_OPTION_DBGLVL
#endif
   int n = size, *mI = I, *mJ = J;

   // Check if we need to remove any diagonal entries.
   int num_diag = 0;
   if (check_for_diag)
   {
      for (int row = 0; row < n; row++)
      {
         for (int j = mI[row]; j < mI[row+1]; j++)
         {
            if (row == mJ[j]) { num_diag++; }
         }
      }
      if (num_diag)
      {
         // Remove the diagonal entries.
         mI = new int[n+1];
         mJ = new int[I[n]-num_diag];
         mI[0] = 0;
         for (int row = 0, counter = 0; row < n; row++)
         {
            for (int j = I[row]; j < I[row+1]; j++)
            {
               if (row != J[j]) { mJ[counter++] = J[j]; }
            }
            mI[row+1] = counter;
         }
      }
   }

   ordering.SetSize(n);
   Array<int> inv_ordering(n);

   if (type == 0 || type == 1)
   {
#ifndef MFEM_USE_METIS_5
      // Metis 4
      if (type == 0)
      {
         // From the manual: "This function computes fill reducing orderings of
         // sparse matrices using the multilevel nested dissection algorithm".
         // We create the reordering based on the element-to-element matrix as
         // defined by the method ElementToElementTable().
         METIS_NodeND(&n,
                      (idxtype *) mI,
                      (idxtype *) mJ,
                      &numflag,
                      options,
                      (idxtype *) inv_ordering.GetData(),
                      (idxtype *) ordering.GetData());
      }
      else
      {
         METIS_EdgeND(&n,
                      (idxtype *) mI,
                      (idxtype *) mJ,
                      &numflag,
                      options,
                      (idxtype *) inv_ordering.GetData(),
                      (idxtype *) ordering.GetData());
      }
#else // #ifndef MFEM_USE_METIS_5
      // Metis 5
      err = METIS_NodeND((idx_t *) &n,
                         (idx_t *) mI,
                         (idx_t *) mJ,
                         NULL,         // vwgt, NULL - equal weights
                         options,
                         (idx_t *) inv_ordering.GetData(),
                         (idx_t *) ordering.GetData());
      MFEM_VERIFY(err == METIS_OK, "error in METIS_NodeND");
#endif // #ifndef MFEM_USE_METIS_5
   }
   else
   {
      MFEM_ABORT("invalid parameter value: type = " << type);
   }
   if (num_diag)
   {
      delete [] mJ;
      delete [] mI;
   }
}
#endif // #ifdef MFEM_USE_METIS

void Table::Print(std::ostream & out, int width) const
{
   int i, j;

   for (i = 0; i < size; i++)
   {
      out << "[row " << i << "]\n";
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << setw(5) << J[j];
         if ( !((j+1-I[i]) % width) )
         {
            out << '\n';
         }
      }
      if ((j-I[i]) % width)
      {
         out << '\n';
      }
   }
}

void Table::PrintMatlab(std::ostream & out) const
{
   int i, j;

   for (i = 0; i < size; i++)
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << i << " " << J[j] << " 1. \n";
      }

   out << flush;
}

void Table::PrintOrderingStats(std::ostream &out) const
{
   out << "Table ordering statistics:\n";
   if (size <= 0)
   {
      out << "   (the Table is empty)\n";
      return;
   }
   const int num_conn = I[size];
   const int width = Width();
   out << "   number of rows        = " << size << '\n'
       << "   number of columns     = " << width << '\n'
       << "   number of connections = " << num_conn << '\n';

   const int bin_factor = 4, num_bins = 10;

   int max_jump = 0, min_jump = width;
   long sum_dist = 0, sum_jump = 0;

   int bins[2*num_bins+1];
   std::fill(bins, bins+2*num_bins+1, 0);

   for (int j = 1; j < num_conn; j++)
   {
      const int jump = J[j] - J[j-1];
      const int dist = std::abs(jump);
      max_jump = std::max(max_jump, jump);
      min_jump = std::min(min_jump, jump);
      sum_jump += jump;
      sum_dist += dist;
      // Put 'jump' in the appropriate bin.
      if (jump == 0)
      {
         bins[num_bins]++;
         continue;
      }
      for (int bin_id = 0, bin_max = bin_factor; true;
           bin_id++, bin_max *= bin_factor)
      {
         if (bin_id < num_bins-1)
         {
            if (dist < bin_max)
            {
               if (jump > 0) { bins[num_bins+1+bin_id]++; }
               else { bins[num_bins-1-bin_id]++; }
               break;
            }
         }
         else
         {
            if (jump > 0) { bins[2*num_bins]++; }
            else { bins[0]++; }
            break;
         }
      }
   }
   // Save precision and flags.
   streamsize old_prec = out.precision(4);
   ios_base::fmtflags old_flags = out.flags();
   out << fixed;
   out << "  jumps between consecutive column indices:"
       << "\n   minimal  = " << min_jump
       << "\n   maximal  = " << max_jump
       << "\n   average  = " << 1.*sum_jump/num_conn
       << "\n   avg dist = " << 1.*sum_dist/num_conn
       << "\n  distribution of the jumps, positive (+) and negative (-):";
   out << "\n                {0} : " << right << setw(8)
       << 100.*bins[num_bins]/num_conn << "% (" << bins[num_bins] << ")";
   for (int bin_id = 0, bin_min = 1; bin_id < num_bins;
        bin_id++, bin_min *= bin_factor)
   {
      out << "\n   [" << setw(6) << bin_min << ", ";
      if (bin_id < num_bins-1)
      {
         out << setw(6) << bin_min*bin_factor;
      }
      else
      {
         out << "     ∞";
      }
      const int n_neg = bins[num_bins-1-bin_id];
      const int n_pos = bins[num_bins+1+bin_id];
      out << ") : " << right << setw(8) << 100.*(n_neg+n_pos)/num_conn
          << "% = (+) " << setw(8)
          << 100.*n_pos/num_conn << "% + (-) " << setw(8)
          << 100.*n_neg/num_conn << "% ("
          << (n_neg+n_pos) << " = " << n_pos << " + " << n_neg << ")";
   }
   out << endl;
   // Restore precision and flags.
   out.precision(old_prec);
   out.flags(old_flags);
}

void Table::Save(std::ostream &out) const
{
   out << size << '\n';

   for (int i = 0; i <= size; i++)
   {
      out << I[i] << '\n';
   }
   for (int i = 0, nnz = I[size]; i < nnz; i++)
   {
      out << J[i] << '\n';
   }
}

void Table::Load(istream &in)
{
   delete [] I;
   delete [] J;

   in >> size;
   I = new int[size+1];
   for (int i = 0; i <= size; i++)
   {
      in >> I[i];
   }
   int nnz = I[size];
   J = new int[nnz];
   for (int j = 0; j < nnz; j++)
   {
      in >> J[j];
   }
}

void Table::Clear()
{
   delete [] I;
   delete [] J;
   size = -1;
   I = J = NULL;
}

void Table::Copy(Table & copy) const
{
   if (size >= 0)
   {
      int * i_copy = new int[size+1];
      int * j_copy = new int[I[size]];

      memcpy(i_copy, I, sizeof(int)*(size+1));
      memcpy(j_copy, J, sizeof(int)*I[size]);

      copy.SetIJ(i_copy, j_copy, size);
   }
   else
   {
      copy.Clear();
   }
}

void Table::Swap(Table & other)
{
   mfem::Swap(size, other.size);
   mfem::Swap(I, other.I);
   mfem::Swap(J, other.J);
}

long Table::MemoryUsage() const
{
   if (size < 0 || I == NULL) { return 0; }
   return (size+1 + I[size]) * sizeof(int);
}

Table::~Table ()
{
   if (I) { delete [] I; }
   if (J) { delete [] J; }
}

void Transpose (const Table &A, Table &At, int _ncols_A)
{
   const int *i_A     = A.GetI();
   const int *j_A     = A.GetJ();
   const int  nrows_A = A.Size();
   const int  ncols_A = (_ncols_A < 0) ? A.Width() : _ncols_A;
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
      for (int j = i_A[i]; j < i_A[i+1]; j++)
      {
         j_At[i_At[j_A[j]]++] = i;
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

void Transpose(const Array<int> &A, Table &At, int _ncols_A)
{
   At.MakeI((_ncols_A < 0) ? (A.Max() + 1) : _ncols_A);
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

void Mult (const Table &A, const Table &B, Table &C)
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


Table * Mult (const Table &A, const Table &B)
{
   Table * C = new Table;
   Mult(A,B,*C);
   return C;
}

STable::STable (int dim, int connections_per_row) :
   Table(dim, connections_per_row)
{}

int STable::operator() (int i, int j) const
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
