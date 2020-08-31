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

#include "../config/config.hpp"
#include "../general/error.hpp"

#ifdef MFEM_USE_MPI

#include "hypre_parcsr.hpp"
#include <limits>
#include <cmath>

namespace mfem
{
namespace internal
{

/*--------------------------------------------------------------------------
 *                        A*X = B style elimination
 *--------------------------------------------------------------------------*/

/*
  Function:  hypre_CSRMatrixEliminateAXB

  Eliminate the rows and columns of A corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal,
  subtract columns times X from B.
*/
void hypre_CSRMatrixEliminateAXB(hypre_CSRMatrix *A,
                                 HYPRE_Int nrows_to_eliminate,
                                 HYPRE_Int *rows_to_eliminate,
                                 hypre_Vector *X,
                                 hypre_Vector *B)
{
   HYPRE_Int  i, j;
   HYPRE_Int  irow, jcol, ibeg, iend, pos;
   HYPRE_Real a;

   HYPRE_Int  *Ai    = hypre_CSRMatrixI(A);
   HYPRE_Int  *Aj    = hypre_CSRMatrixJ(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);
   HYPRE_Int   nrows = hypre_CSRMatrixNumRows(A);

   HYPRE_Real *Xdata = hypre_VectorData(X);
   HYPRE_Real *Bdata = hypre_VectorData(B);

   /* eliminate the columns */
   for (i = 0; i < nrows; i++)
   {
      ibeg = Ai[i];
      iend = Ai[i+1];
      for (j = ibeg; j < iend; j++)
      {
         jcol = Aj[j];
         pos = hypre_BinarySearch(rows_to_eliminate, jcol, nrows_to_eliminate);
         if (pos != -1)
         {
            a = Adata[j];
            Adata[j] = 0.0;
            Bdata[i] -= a * Xdata[jcol];
         }
      }
   }

   /* remove the rows and set the diagonal equal to 1 */
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
      {
         if (Aj[j] == irow)
         {
            Adata[j] = 1.0;
         }
         else
         {
            Adata[j] = 0.0;
         }
      }
   }
}

/*
  Function:  hypre_CSRMatrixEliminateOffdColsAXB

  Eliminate the given sorted (!) list of columns of A, subtract them from B.
*/
void hypre_CSRMatrixEliminateOffdColsAXB(hypre_CSRMatrix *A,
                                         HYPRE_Int ncols_to_eliminate,
                                         HYPRE_Int *eliminate_cols,
                                         HYPRE_Real *eliminate_coefs,
                                         hypre_Vector *B)
{
   HYPRE_Int i, j;
   HYPRE_Int ibeg, iend, pos;
   HYPRE_Real a;

   HYPRE_Int *Ai = hypre_CSRMatrixI(A);
   HYPRE_Int *Aj = hypre_CSRMatrixJ(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);
   HYPRE_Int nrows = hypre_CSRMatrixNumRows(A);

   HYPRE_Real *Bdata = hypre_VectorData(B);

   for (i = 0; i < nrows; i++)
   {
      ibeg = Ai[i];
      iend = Ai[i+1];
      for (j = ibeg; j < iend; j++)
      {
         pos = hypre_BinarySearch(eliminate_cols, Aj[j], ncols_to_eliminate);
         if (pos != -1)
         {
            a = Adata[j];
            Adata[j] = 0.0;
            Bdata[i] -= a * eliminate_coefs[pos];
         }
      }
   }
}

/*
  Function:  hypre_CSRMatrixEliminateOffdRowsAXB

  Eliminate (zero) the given list of rows of A.
*/
void hypre_CSRMatrixEliminateOffdRowsAXB(hypre_CSRMatrix *A,
                                         HYPRE_Int  nrows_to_eliminate,
                                         HYPRE_Int *rows_to_eliminate)
{
   HYPRE_Int  *Ai    = hypre_CSRMatrixI(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);

   HYPRE_Int i, j;
   HYPRE_Int irow, ibeg, iend;

   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
      {
         Adata[j] = 0.0;
      }
   }
}

/*
  Function:  hypre_ParCSRMatrixEliminateAXB

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers,
  so that the solution to the system A*X = B is X_b for the given rows.

  The elimination is done as follows:

                    (input)                  (output)

                / A_ii | A_ib \          / A_ii |  0   \
            A = | -----+----- |   --->   | -----+----- |
                \ A_bi | A_bb /          \   0  |  I   /

                        / X_i \          / X_i \
                    X = | --- |   --->   | --- |  (no change)
                        \ X_b /          \ X_b /

                        / B_i \          / B_i - A_ib * X_b \
                    B = | --- |   --->   | ---------------- |
                        \ B_b /          \        X_b       /

*/
void hypre_ParCSRMatrixEliminateAXB(hypre_ParCSRMatrix *A,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
                                    hypre_ParVector *X,
                                    hypre_ParVector *B)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int diag_nrows  = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int offd_ncols  = hypre_CSRMatrixNumCols(offd);

   hypre_Vector *Xlocal = hypre_ParVectorLocalVector(X);
   hypre_Vector *Blocal = hypre_ParVectorLocalVector(B);

   HYPRE_Real   *Bdata  = hypre_VectorData(Blocal);
   HYPRE_Real   *Xdata  = hypre_VectorData(Xlocal);

   HYPRE_Int  num_offd_cols_to_elim;
   HYPRE_Int  *offd_cols_to_elim;
   HYPRE_Real *eliminate_coefs;

   /* figure out which offd cols should be eliminated and with what coef */
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *comm_pkg;
   HYPRE_Int num_sends;
   HYPRE_Int index, start;
   HYPRE_Int i, j, k, irow;

   HYPRE_Real *eliminate_row = mfem_hypre_CTAlloc(HYPRE_Real, diag_nrows);
   HYPRE_Real *eliminate_col = mfem_hypre_CTAlloc(HYPRE_Real, offd_ncols);
   HYPRE_Real *buf_data, coef;

   /* make sure A has a communication package */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* HACK: rows that shouldn't be eliminated are marked with quiet NaN;
      those that should are set to the boundary value from X; this is to
      avoid sending complex type (int+double) or communicating twice. */
   for (i = 0; i < diag_nrows; i++)
   {
      eliminate_row[i] = std::numeric_limits<HYPRE_Real>::quiet_NaN();
   }
   for (i = 0; i < num_rowscols_to_elim; i++)
   {
      irow = rowscols_to_elim[i];
      eliminate_row[irow] = Xdata[irow];
   }

   /* use a Matvec communication pattern to find (in eliminate_col)
      which of the local offd columns are to be eliminated */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   buf_data = mfem_hypre_CTAlloc(HYPRE_Real,
                                 hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                 num_sends));
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         buf_data[index++] = eliminate_row[k];
      }
   }
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg,
                                              buf_data, eliminate_col);

   /* do sequential part of the elimination while stuff is getting sent */
   hypre_CSRMatrixEliminateAXB(diag, num_rowscols_to_elim, rowscols_to_elim,
                               Xlocal, Blocal);

   /* finish the communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* received eliminate_col[], count offd columns to eliminate */
   num_offd_cols_to_elim = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         num_offd_cols_to_elim++;
      }
   }

   offd_cols_to_elim = mfem_hypre_CTAlloc(HYPRE_Int, num_offd_cols_to_elim);
   eliminate_coefs = mfem_hypre_CTAlloc(HYPRE_Real, num_offd_cols_to_elim);

   /* get a list of offd column indices and coefs */
   num_offd_cols_to_elim = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         offd_cols_to_elim[num_offd_cols_to_elim] = i;
         eliminate_coefs[num_offd_cols_to_elim] = coef;
         num_offd_cols_to_elim++;
      }
   }

   mfem_hypre_TFree(buf_data);
   mfem_hypre_TFree(eliminate_col);
   mfem_hypre_TFree(eliminate_row);

   /* eliminate the off-diagonal part */
   hypre_CSRMatrixEliminateOffdColsAXB(offd, num_offd_cols_to_elim,
                                       offd_cols_to_elim,
                                       eliminate_coefs, Blocal);

   hypre_CSRMatrixEliminateOffdRowsAXB(offd, num_rowscols_to_elim,
                                       rowscols_to_elim);

   /* set boundary values in the rhs */
   for (int i = 0; i < num_rowscols_to_elim; i++)
   {
      irow = rowscols_to_elim[i];
      Bdata[irow] = Xdata[irow];
   }

   mfem_hypre_TFree(offd_cols_to_elim);
   mfem_hypre_TFree(eliminate_coefs);
}


/*--------------------------------------------------------------------------
 *                        (A + Ae) style elimination
 *--------------------------------------------------------------------------*/

/*
  Function:  hypre_CSRMatrixElimCreate

  Prepare the Ae matrix: count nnz, initialize I, allocate J and data.
*/
void hypre_CSRMatrixElimCreate(hypre_CSRMatrix *A,
                               hypre_CSRMatrix *Ae,
                               HYPRE_Int nrows, HYPRE_Int *rows,
                               HYPRE_Int ncols, HYPRE_Int *cols,
                               HYPRE_Int *col_mark)
{
   HYPRE_Int  i, j, col;
   HYPRE_Int  A_beg, A_end;

   HYPRE_Int  *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j     = hypre_CSRMatrixJ(A);
   HYPRE_Int   A_rows  = hypre_CSRMatrixNumRows(A);

   hypre_CSRMatrixI(Ae) = mfem_hypre_TAlloc(HYPRE_Int, A_rows+1);

   HYPRE_Int  *Ae_i    = hypre_CSRMatrixI(Ae);
   HYPRE_Int   nnz     = 0;

   for (i = 0; i < A_rows; i++)
   {
      Ae_i[i] = nnz;

      A_beg = A_i[i];
      A_end = A_i[i+1];

      if (hypre_BinarySearch(rows, i, nrows) >= 0)
      {
         /* full row */
         nnz += A_end - A_beg;

         if (col_mark)
         {
            for (j = A_beg; j < A_end; j++)
            {
               col_mark[A_j[j]] = 1;
            }
         }
      }
      else
      {
         /* count columns */
         for (j = A_beg; j < A_end; j++)
         {
            col = A_j[j];
            if (hypre_BinarySearch(cols, col, ncols) >= 0)
            {
               nnz++;
               if (col_mark) { col_mark[col] = 1; }
            }
         }
      }
   }
   Ae_i[A_rows] = nnz;

   hypre_CSRMatrixJ(Ae) = mfem_hypre_TAlloc(HYPRE_Int, nnz);
   hypre_CSRMatrixData(Ae) = mfem_hypre_TAlloc(HYPRE_Real, nnz);
   hypre_CSRMatrixNumNonzeros(Ae) = nnz;
}

/*
  Function:  hypre_CSRMatrixEliminateRowsCols

  Eliminate rows and columns of A, store eliminated values in Ae.
  If 'diag' is nonzero, the eliminated diagonal of A is set to identity.
  If 'col_remap' is not NULL it specifies renumbering of columns of Ae.
*/
void hypre_CSRMatrixEliminateRowsCols(hypre_CSRMatrix *A,
                                      hypre_CSRMatrix *Ae,
                                      HYPRE_Int nrows, HYPRE_Int *rows,
                                      HYPRE_Int ncols, HYPRE_Int *cols,
                                      int diag, HYPRE_Int* col_remap)
{
   HYPRE_Int  i, j, k, col;
   HYPRE_Int  A_beg, Ae_beg, A_end;
   HYPRE_Real a;

   HYPRE_Int  *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j     = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data  = hypre_CSRMatrixData(A);
   HYPRE_Int   A_rows  = hypre_CSRMatrixNumRows(A);

   HYPRE_Int  *Ae_i    = hypre_CSRMatrixI(Ae);
   HYPRE_Int  *Ae_j    = hypre_CSRMatrixJ(Ae);
   HYPRE_Real *Ae_data = hypre_CSRMatrixData(Ae);

   for (i = 0; i < A_rows; i++)
   {
      A_beg = A_i[i];
      A_end = A_i[i+1];
      Ae_beg = Ae_i[i];

      if (hypre_BinarySearch(rows, i, nrows) >= 0)
      {
         /* eliminate row */
         for (j = A_beg, k = Ae_beg; j < A_end; j++, k++)
         {
            col = A_j[j];
            Ae_j[k] = col_remap ? col_remap[col] : col;
            a = (diag && col == i) ? 1.0 : 0.0;
            Ae_data[k] = A_data[j] - a;
            A_data[j] = a;
         }
      }
      else
      {
         /* eliminate columns */
         for (j = A_beg, k = Ae_beg; j < A_end; j++)
         {
            col = A_j[j];
            if (hypre_BinarySearch(cols, col, ncols) >= 0)
            {
               Ae_j[k] = col_remap ? col_remap[col] : col;
               Ae_data[k] = A_data[j];
               A_data[j] = 0.0;
               k++;
            }
         }
      }
   }
}

/*
  Eliminate rows of A, setting all entries in the eliminated rows to zero.
*/
void hypre_CSRMatrixEliminateRows(hypre_CSRMatrix *A,
                                  HYPRE_Int nrows, const HYPRE_Int *rows)
{
   HYPRE_Int  irow, i, j;
   HYPRE_Int  A_beg, A_end;

   HYPRE_Int  *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Real *A_data  = hypre_CSRMatrixData(A);

   for (i = 0; i < nrows; i++)
   {
      irow = rows[i];
      A_beg = A_i[irow];
      A_end = A_i[irow+1];
      /* eliminate row */
      for (j = A_beg; j < A_end; j++)
      {
         A_data[j] = 0.0;
      }
   }
}


/*
  Function:  hypre_ParCSRMatrixEliminateAAe

                    (input)                  (output)

                / A_ii | A_ib \          / A_ii |  0   \
            A = | -----+----- |   --->   | -----+----- |
                \ A_bi | A_bb /          \   0  |  I   /


                                         /   0  |   A_ib   \
                                    Ae = | -----+--------- |
                                         \ A_bi | A_bb - I /

*/
void hypre_ParCSRMatrixEliminateAAe(hypre_ParCSRMatrix *A,
                                    hypre_ParCSRMatrix **Ae,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
                                    int ignore_rows)
{
   HYPRE_Int i, j, k;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int A_diag_ncols  = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int A_offd_ncols  = hypre_CSRMatrixNumCols(A_offd);

   *Ae = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                  hypre_ParCSRMatrixGlobalNumRows(A),
                                  hypre_ParCSRMatrixGlobalNumCols(A),
                                  hypre_ParCSRMatrixRowStarts(A),
                                  hypre_ParCSRMatrixColStarts(A),
                                  0, 0, 0);

   hypre_ParCSRMatrixSetRowStartsOwner(*Ae, 0);
   hypre_ParCSRMatrixSetColStartsOwner(*Ae, 0);

   hypre_CSRMatrix *Ae_diag = hypre_ParCSRMatrixDiag(*Ae);
   hypre_CSRMatrix *Ae_offd = hypre_ParCSRMatrixOffd(*Ae);
   HYPRE_Int Ae_offd_ncols;

   HYPRE_Int  num_offd_cols_to_elim;
   HYPRE_Int  *offd_cols_to_elim;

   HYPRE_Int  *A_col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int  *Ae_col_map_offd;

   HYPRE_Int  *col_mark;
   HYPRE_Int  *col_remap;

   /* figure out which offd cols should be eliminated */
   {
      hypre_ParCSRCommHandle *comm_handle;
      hypre_ParCSRCommPkg *comm_pkg;
      HYPRE_Int num_sends, *int_buf_data;
      HYPRE_Int index, start;

      HYPRE_Int *eliminate_diag_col = mfem_hypre_CTAlloc(HYPRE_Int, A_diag_ncols);
      HYPRE_Int *eliminate_offd_col = mfem_hypre_CTAlloc(HYPRE_Int, A_offd_ncols);

      /* make sure A has a communication package */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }

      /* which of the local rows are to be eliminated */
      for (i = 0; i < A_diag_ncols; i++)
      {
         eliminate_diag_col[i] = 0;
      }
      for (i = 0; i < num_rowscols_to_elim; i++)
      {
         eliminate_diag_col[rowscols_to_elim[i]] = 1;
      }

      /* use a Matvec communication pattern to find (in eliminate_col)
         which of the local offd columns are to be eliminated */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = mfem_hypre_CTAlloc(
                        HYPRE_Int,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = eliminate_diag_col[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                 int_buf_data, eliminate_offd_col);

      /* eliminate diagonal part, overlapping it with communication */
      if (ignore_rows)
      {
         hypre_CSRMatrixElimCreate(A_diag, Ae_diag,
                                   0, nullptr,
                                   num_rowscols_to_elim, rowscols_to_elim,
                                   NULL);

         hypre_CSRMatrixEliminateRowsCols(A_diag, Ae_diag,
                                          0, nullptr,
                                          num_rowscols_to_elim, rowscols_to_elim,
                                          1, NULL);
      }
      else
      {
         hypre_CSRMatrixElimCreate(A_diag, Ae_diag,
                                   num_rowscols_to_elim, rowscols_to_elim,
                                   num_rowscols_to_elim, rowscols_to_elim,
                                   NULL);

         hypre_CSRMatrixEliminateRowsCols(A_diag, Ae_diag,
                                          num_rowscols_to_elim, rowscols_to_elim,
                                          num_rowscols_to_elim, rowscols_to_elim,
                                          1, NULL);
      }

      hypre_CSRMatrixReorder(Ae_diag);

      /* finish the communication */
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* received eliminate_col[], count offd columns to eliminate */
      num_offd_cols_to_elim = 0;
      for (i = 0; i < A_offd_ncols; i++)
      {
         if (eliminate_offd_col[i]) { num_offd_cols_to_elim++; }
      }

      offd_cols_to_elim = mfem_hypre_CTAlloc(HYPRE_Int, num_offd_cols_to_elim);

      /* get a list of offd column indices and coefs */
      num_offd_cols_to_elim = 0;
      for (i = 0; i < A_offd_ncols; i++)
      {
         if (eliminate_offd_col[i])
         {
            offd_cols_to_elim[num_offd_cols_to_elim++] = i;
         }
      }

      mfem_hypre_TFree(int_buf_data);
      mfem_hypre_TFree(eliminate_offd_col);
      mfem_hypre_TFree(eliminate_diag_col);
   }

   /* eliminate the off-diagonal part */
   col_mark = mfem_hypre_CTAlloc(HYPRE_Int, A_offd_ncols);
   col_remap = mfem_hypre_CTAlloc(HYPRE_Int, A_offd_ncols);

   if (ignore_rows)
   {
      hypre_CSRMatrixElimCreate(A_offd, Ae_offd,
                                0, nullptr,
                                num_offd_cols_to_elim, offd_cols_to_elim,
                                col_mark);

      for (i = k = 0; i < A_offd_ncols; i++)
      {
         if (col_mark[i]) { col_remap[i] = k++; }
      }

      hypre_CSRMatrixEliminateRowsCols(A_offd, Ae_offd,
                                       0, nullptr,
                                       num_offd_cols_to_elim, offd_cols_to_elim,
                                       0, col_remap);
   }
   else
   {
      hypre_CSRMatrixElimCreate(A_offd, Ae_offd,
                                num_rowscols_to_elim, rowscols_to_elim,
                                num_offd_cols_to_elim, offd_cols_to_elim,
                                col_mark);

      for (i = k = 0; i < A_offd_ncols; i++)
      {
         if (col_mark[i]) { col_remap[i] = k++; }
      }

      hypre_CSRMatrixEliminateRowsCols(A_offd, Ae_offd,
                                       num_rowscols_to_elim, rowscols_to_elim,
                                       num_offd_cols_to_elim, offd_cols_to_elim,
                                       0, col_remap);
   }

   /* create col_map_offd for Ae */
   Ae_offd_ncols = 0;
   for (i = 0; i < A_offd_ncols; i++)
   {
      if (col_mark[i]) { Ae_offd_ncols++; }
   }

   Ae_col_map_offd  = mfem_hypre_CTAlloc(HYPRE_Int, Ae_offd_ncols);

   Ae_offd_ncols = 0;
   for (i = 0; i < A_offd_ncols; i++)
   {
      if (col_mark[i])
      {
         Ae_col_map_offd[Ae_offd_ncols++] = A_col_map_offd[i];
      }
   }

   hypre_ParCSRMatrixColMapOffd(*Ae) = Ae_col_map_offd;
   hypre_CSRMatrixNumCols(Ae_offd) = Ae_offd_ncols;

   mfem_hypre_TFree(col_remap);
   mfem_hypre_TFree(col_mark);
   mfem_hypre_TFree(offd_cols_to_elim);

   hypre_ParCSRMatrixSetNumNonzeros(*Ae);
   hypre_MatvecCommPkgCreate(*Ae);
}


// Eliminate rows from the diagonal and off-diagonal blocks of the matrix
void hypre_ParCSRMatrixEliminateRows(hypre_ParCSRMatrix *A,
                                     HYPRE_Int num_rows_to_elim,
                                     const HYPRE_Int *rows_to_elim)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixEliminateRows(A_diag, num_rows_to_elim, rows_to_elim);
   hypre_CSRMatrixEliminateRows(A_offd, num_rows_to_elim, rows_to_elim);
}


/*--------------------------------------------------------------------------
 *                               Split
 *--------------------------------------------------------------------------*/

void hypre_CSRMatrixSplit(hypre_CSRMatrix *A,
                          HYPRE_Int nr, HYPRE_Int nc,
                          HYPRE_Int *row_block_num, HYPRE_Int *col_block_num,
                          hypre_CSRMatrix **blocks)
{
   HYPRE_Int i, j, k, bi, bj;

   HYPRE_Int* A_i = hypre_CSRMatrixI(A);
   HYPRE_Int* A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex* A_data = hypre_CSRMatrixData(A);

   HYPRE_Int A_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int A_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Int *num_rows = mfem_hypre_CTAlloc(HYPRE_Int, nr);
   HYPRE_Int *num_cols = mfem_hypre_CTAlloc(HYPRE_Int, nc);

   HYPRE_Int *block_row = mfem_hypre_TAlloc(HYPRE_Int, A_rows);
   HYPRE_Int *block_col = mfem_hypre_TAlloc(HYPRE_Int, A_cols);

   for (i = 0; i < A_rows; i++)
   {
      block_row[i] = num_rows[row_block_num[i]]++;
   }
   for (j = 0; j < A_cols; j++)
   {
      block_col[j] = num_cols[col_block_num[j]]++;
   }

   /* allocate the blocks */
   for (i = 0; i < nr; i++)
   {
      for (j = 0; j < nc; j++)
      {
         hypre_CSRMatrix *B = hypre_CSRMatrixCreate(num_rows[i], num_cols[j], 0);
         hypre_CSRMatrixI(B) = mfem_hypre_CTAlloc(HYPRE_Int, num_rows[i] + 1);
         blocks[i*nc + j] = B;
      }
   }

   /* count block row nnz */
   for (i = 0; i < A_rows; i++)
   {
      bi = row_block_num[i];
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         bj = col_block_num[A_j[j]];
         hypre_CSRMatrix *B = blocks[bi*nc + bj];
         hypre_CSRMatrixI(B)[block_row[i] + 1]++;
      }
   }

   /* count block nnz */
   for (k = 0; k < nr*nc; k++)
   {
      hypre_CSRMatrix *B = blocks[k];
      HYPRE_Int* B_i = hypre_CSRMatrixI(B);

      HYPRE_Int nnz = 0, rs;
      for (int k = 1; k <= hypre_CSRMatrixNumRows(B); k++)
      {
         rs = B_i[k], B_i[k] = nnz, nnz += rs;
      }

      hypre_CSRMatrixJ(B) = mfem_hypre_TAlloc(HYPRE_Int, nnz);
      hypre_CSRMatrixData(B) = mfem_hypre_TAlloc(HYPRE_Complex, nnz);
      hypre_CSRMatrixNumNonzeros(B) = nnz;
   }

   /* populate blocks */
   for (i = 0; i < A_rows; i++)
   {
      bi = row_block_num[i];
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         k = A_j[j];
         bj = col_block_num[k];
         hypre_CSRMatrix *B = blocks[bi*nc + bj];
         HYPRE_Int *bii = hypre_CSRMatrixI(B) + block_row[i] + 1;
         hypre_CSRMatrixJ(B)[*bii] = block_col[k];
         hypre_CSRMatrixData(B)[*bii] = A_data[j];
         (*bii)++;
      }
   }

   mfem_hypre_TFree(block_col);
   mfem_hypre_TFree(block_row);

   mfem_hypre_TFree(num_cols);
   mfem_hypre_TFree(num_rows);
}


void hypre_ParCSRMatrixSplit(hypre_ParCSRMatrix *A,
                             HYPRE_Int nr, HYPRE_Int nc,
                             hypre_ParCSRMatrix **blocks,
                             int interleaved_rows, int interleaved_cols)
{
   HYPRE_Int i, j, k;

   MPI_Comm comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *Adiag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *Aoffd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   HYPRE_Int local_rows = hypre_CSRMatrixNumRows(Adiag);
   HYPRE_Int local_cols = hypre_CSRMatrixNumCols(Adiag);
   HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(Aoffd);

   hypre_assert(local_rows % nr == 0 && local_cols % nc == 0);
   hypre_assert(global_rows % nr == 0 && global_cols % nc == 0);

   HYPRE_Int block_rows = local_rows / nr;
   HYPRE_Int block_cols = local_cols / nc;
   HYPRE_Int num_blocks = nr * nc;

   /* mark local rows and columns with block number */
   HYPRE_Int *row_block_num = mfem_hypre_TAlloc(HYPRE_Int, local_rows);
   HYPRE_Int *col_block_num = mfem_hypre_TAlloc(HYPRE_Int, local_cols);

   for (i = 0; i < local_rows; i++)
   {
      row_block_num[i] = interleaved_rows ? (i % nr) : (i / block_rows);
   }
   for (i = 0; i < local_cols; i++)
   {
      col_block_num[i] = interleaved_cols ? (i % nc) : (i / block_cols);
   }

   /* determine the block numbers for offd columns */
   HYPRE_Int* offd_col_block_num = mfem_hypre_TAlloc(HYPRE_Int, offd_cols);
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *int_buf_data;
   {
      /* make sure A has a communication package */
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }

      /* calculate the final global column numbers for each block */
      HYPRE_Int *count = mfem_hypre_CTAlloc(HYPRE_Int, nc);
      HYPRE_Int *block_global_col = mfem_hypre_TAlloc(HYPRE_Int, local_cols);
      HYPRE_Int first_col = hypre_ParCSRMatrixFirstColDiag(A) / nc;
      for (i = 0; i < local_cols; i++)
      {
         block_global_col[i] = first_col + count[col_block_num[i]]++;
      }
      mfem_hypre_TFree(count);

      /* use a Matvec communication pattern to determine offd_col_block_num */
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = mfem_hypre_CTAlloc(
                        HYPRE_Int,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      HYPRE_Int start, index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = col_block_num[k] + nc*block_global_col[k];
         }
      }
      mfem_hypre_TFree(block_global_col);

      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 offd_col_block_num);
   }

   /* create the block matrices */
   HYPRE_Int num_procs = 1;
   if (!hypre_ParCSRMatrixAssumedPartition(A))
   {
      hypre_MPI_Comm_size(comm, &num_procs);
   }

   HYPRE_Int *row_starts = mfem_hypre_TAlloc(HYPRE_Int, num_procs+1);
   HYPRE_Int *col_starts = mfem_hypre_TAlloc(HYPRE_Int, num_procs+1);
   for (i = 0; i <= num_procs; i++)
   {
      row_starts[i] = hypre_ParCSRMatrixRowStarts(A)[i] / nr;
      col_starts[i] = hypre_ParCSRMatrixColStarts(A)[i] / nc;
   }

   for (i = 0; i < num_blocks; i++)
   {
      blocks[i] = hypre_ParCSRMatrixCreate(comm,
                                           global_rows/nr, global_cols/nc,
                                           row_starts, col_starts, 0, 0, 0);
   }

   /* split diag part */
   hypre_CSRMatrix **csr_blocks = mfem_hypre_TAlloc(hypre_CSRMatrix*, nr*nc);
   hypre_CSRMatrixSplit(Adiag, nr, nc, row_block_num, col_block_num,
                        csr_blocks);

   for (i = 0; i < num_blocks; i++)
   {
      mfem_hypre_TFree(hypre_ParCSRMatrixDiag(blocks[i]));
      hypre_ParCSRMatrixDiag(blocks[i]) = csr_blocks[i];
   }

   /* finish communication, receive offd_col_block_num */
   hypre_ParCSRCommHandleDestroy(comm_handle);
   mfem_hypre_TFree(int_buf_data);

   /* decode global offd column numbers */
   HYPRE_Int* offd_global_col = mfem_hypre_TAlloc(HYPRE_Int, offd_cols);
   for (i = 0; i < offd_cols; i++)
   {
      offd_global_col[i] = offd_col_block_num[i] / nc;
      offd_col_block_num[i] %= nc;
   }

   /* split offd part */
   hypre_CSRMatrixSplit(Aoffd, nr, nc, row_block_num, offd_col_block_num,
                        csr_blocks);

   for (i = 0; i < num_blocks; i++)
   {
      mfem_hypre_TFree(hypre_ParCSRMatrixOffd(blocks[i]));
      hypre_ParCSRMatrixOffd(blocks[i]) = csr_blocks[i];
   }

   mfem_hypre_TFree(csr_blocks);
   mfem_hypre_TFree(col_block_num);
   mfem_hypre_TFree(row_block_num);

   /* update block col-maps */
   for (int bi = 0; bi < nr; bi++)
   {
      for (int bj = 0; bj < nc; bj++)
      {
         hypre_ParCSRMatrix *block = blocks[bi*nc + bj];
         hypre_CSRMatrix *block_offd = hypre_ParCSRMatrixOffd(block);
         HYPRE_Int block_offd_cols = hypre_CSRMatrixNumCols(block_offd);

         HYPRE_Int *block_col_map = mfem_hypre_TAlloc(HYPRE_Int,
                                                      block_offd_cols);
         for (i = j = 0; i < offd_cols; i++)
         {
            HYPRE_Int bn = offd_col_block_num[i];
            if (bn == bj) { block_col_map[j++] = offd_global_col[i]; }
         }
         hypre_assert(j == block_offd_cols);

         hypre_ParCSRMatrixColMapOffd(block) = block_col_map;
      }
   }

   mfem_hypre_TFree(offd_global_col);
   mfem_hypre_TFree(offd_col_block_num);

   /* finish the new matrices, make them own all the stuff */
   for (i = 0; i < num_blocks; i++)
   {
      hypre_ParCSRMatrixSetNumNonzeros(blocks[i]);
      hypre_MatvecCommPkgCreate(blocks[i]);

      hypre_ParCSRMatrixOwnsData(blocks[i]) = 1;

      /* only the first block will own the row/col_starts */
      hypre_ParCSRMatrixOwnsRowStarts(blocks[i]) = !i;
      hypre_ParCSRMatrixOwnsColStarts(blocks[i]) = !i;
   }
}

/* Based on hypre_CSRMatrixMatvec in hypre's csr_matvec.c */
void hypre_CSRMatrixAbsMatvec(hypre_CSRMatrix *A,
                              HYPRE_Real alpha,
                              HYPRE_Real *x,
                              HYPRE_Real beta,
                              HYPRE_Real *y)
{
   HYPRE_Real       *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Real       *x_data = x;
   HYPRE_Real       *y_data = y;

   HYPRE_Real        temp, tempx;

   HYPRE_Int         i, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      for (i = 0; i < num_rows; i++)
      {
         y_data[i] *= beta;
      }
      return;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
         for (i = 0; i < num_rows; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
         for (i = 0; i < num_rows; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += abs(A)*x
    *-----------------------------------------------------------------*/

   /* use rownnz pointer to do the abs(A)*x multiplication
      when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
      for (i = 0; i < num_rownnz; i++)
      {
         m = A_rownnz[i];

         tempx = 0;
         for (jj = A_i[m]; jj < A_i[m+1]; jj++)
         {
            tempx += std::abs(A_data[jj])*x_data[A_j[jj]];
         }
         y_data[m] += tempx;
      }
   }
   else
   {
      for (i = 0; i < num_rows; i++)
      {
         tempx = 0;
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
         {
            tempx += std::abs(A_data[jj])*x_data[A_j[jj]];
         }
         y_data[i] += tempx;
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
      for (i = 0; i < num_rows; i++)
      {
         y_data[i] *= alpha;
      }
   }

}

/* Based on hypre_CSRMatrixMatvecT in hypre's csr_matvec.c */
void hypre_CSRMatrixAbsMatvecT(hypre_CSRMatrix *A,
                               HYPRE_Real alpha,
                               HYPRE_Real *x,
                               HYPRE_Real beta,
                               HYPRE_Real *y)
{
   HYPRE_Real       *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Real       *x_data = x;
   HYPRE_Real       *y_data = y;

   HYPRE_Int         i, j, jj;

   HYPRE_Real        temp;

   if (alpha == 0.0)
   {
      for (i = 0; i < num_cols; i++)
      {
         y_data[i] *= beta;
      }
      return;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
         for (i = 0; i < num_cols; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
         for (i = 0; i < num_cols; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += abs(A)^T*x
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         j = A_j[jj];
         y_data[j] += std::abs(A_data[jj]) * x_data[i];
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
      for (i = 0; i < num_cols; i++)
      {
         y_data[i] *= alpha;
      }
   }
}

/* Based on hypre_CSRMatrixMatvec in hypre's csr_matvec.c */
void hypre_CSRMatrixBooleanMatvec(hypre_CSRMatrix *A,
                                  HYPRE_Bool alpha,
                                  HYPRE_Bool *x,
                                  HYPRE_Bool beta,
                                  HYPRE_Bool *y)
{
   /* HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A); */
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Bool       *x_data = x;
   HYPRE_Bool       *y_data = y;

   HYPRE_Bool        temp, tempx;

   HYPRE_Int         i, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
      {
         y_data[i] = y_data[i] && beta;
      }
      return;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   if (beta == 0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
      {
         y_data[i] = 0;
      }
   }
   else
   {
      /* beta is true -> no change to y_data */
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

   /* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i,jj,m,tempx) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rownnz; i++)
      {
         m = A_rownnz[i];

         tempx = 0;
         for (jj = A_i[m]; jj < A_i[m+1]; jj++)
         {
            /* tempx = tempx || ((A_data[jj] != 0.0) && x_data[A_j[jj]]); */
            tempx = tempx || x_data[A_j[jj]];
         }
         y_data[m] = y_data[m] || tempx;
      }
   }
   else
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i,jj,temp) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
      {
         temp = 0;
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
         {
            /* temp = temp || ((A_data[jj] != 0.0) && x_data[A_j[jj]]); */
            temp = temp || x_data[A_j[jj]];
         }
         y_data[i] = y_data[i] || temp;
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
   /* alpha is true */
}

/* Based on hypre_CSRMatrixMatvecT in hypre's csr_matvec.c */
void hypre_CSRMatrixBooleanMatvecT(hypre_CSRMatrix *A,
                                   HYPRE_Bool alpha,
                                   HYPRE_Bool *x,
                                   HYPRE_Bool beta,
                                   HYPRE_Bool *y)
{
   /* HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A); */
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Bool       *x_data = x;
   HYPRE_Bool       *y_data = y;

   HYPRE_Int         i, j, jj;

   /*-----------------------------------------------------------------------
    * y = beta*y
    *-----------------------------------------------------------------------*/

   if (beta == 0)
   {
      for (i = 0; i < num_cols; i++)
      {
         y_data[i] = 0;
      }
   }
   else
   {
      /* beta is true -> no change to y_data */
   }

   /*-----------------------------------------------------------------------
    * Check if (alpha == 0)
    *-----------------------------------------------------------------------*/

   if (alpha == 0)
   {
      return;
   }

   /* alpha is true */

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   for (i = 0; i < num_rows; i++)
   {
      if (x_data[i] != 0)
      {
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
         {
            j = A_j[jj];
            /* y_data[j] += A_data[jj] * x_data[i]; */
            y_data[j] = 1;
         }
      }
   }
}

/* Based on hypre_ParCSRCommHandleCreate in hypre's par_csr_communication.c. The
   input variable job controls the communication type: 1=Matvec, 2=MatvecT. */
hypre_ParCSRCommHandle *
hypre_ParCSRCommHandleCreate_bool(HYPRE_Int            job,
                                  hypre_ParCSRCommPkg *comm_pkg,
                                  HYPRE_Bool          *send_data,
                                  HYPRE_Bool          *recv_data)
{
   HYPRE_Int                  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int                  num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm                   comm      = hypre_ParCSRCommPkgComm(comm_pkg);

   hypre_ParCSRCommHandle    *comm_handle;
   HYPRE_Int                  num_requests;
   hypre_MPI_Request         *requests;

   HYPRE_Int                  i, j;
   HYPRE_Int                  my_id, num_procs;
   HYPRE_Int                  ip, vec_start, vec_len;

   num_requests = num_sends + num_recvs;
   requests = mfem_hypre_CTAlloc(hypre_MPI_Request, num_requests);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   j = 0;
   switch (job)
   {
      case  1:
      {
         HYPRE_Bool *d_send_data = (HYPRE_Bool *) send_data;
         HYPRE_Bool *d_recv_data = (HYPRE_Bool *) recv_data;
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1)-vec_start;
            hypre_MPI_Irecv(&d_recv_data[vec_start], vec_len, HYPRE_MPI_BOOL,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            hypre_MPI_Isend(&d_send_data[vec_start], vec_len, HYPRE_MPI_BOOL,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  2:
      {
         HYPRE_Bool *d_send_data = (HYPRE_Bool *) send_data;
         HYPRE_Bool *d_recv_data = (HYPRE_Bool *) recv_data;
         for (i = 0; i < num_sends; i++)
         {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            hypre_MPI_Irecv(&d_recv_data[vec_start], vec_len, HYPRE_MPI_BOOL,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1)-vec_start;
            hypre_MPI_Isend(&d_send_data[vec_start], vec_len, HYPRE_MPI_BOOL,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }
   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = mfem_hypre_CTAlloc(hypre_ParCSRCommHandle, 1);

   hypre_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_ParCSRCommHandleSendData(comm_handle)    = send_data;
   hypre_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   hypre_ParCSRCommHandleRequests(comm_handle)    = requests;

   return comm_handle;
}

/* Based on hypre_ParCSRMatrixMatvec in par_csr_matvec.c */
void hypre_ParCSRMatrixAbsMatvec(hypre_ParCSRMatrix *A,
                                 HYPRE_Real alpha,
                                 HYPRE_Real *x,
                                 HYPRE_Real beta,
                                 HYPRE_Real *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int          num_sends, i, j, index;

   HYPRE_Real        *x_tmp, *x_buf;

   x_tmp = mfem_hypre_CTAlloc(HYPRE_Real, num_cols_offd);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf = mfem_hypre_CTAlloc(
              HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for ( ; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         x_buf[index++] = x[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, x_buf, x_tmp);

   hypre_CSRMatrixAbsMatvec(diag, alpha, x, beta, y);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (num_cols_offd)
   {
      hypre_CSRMatrixAbsMatvec(offd, alpha, x_tmp, 1.0, y);
   }

   mfem_hypre_TFree(x_buf);
   mfem_hypre_TFree(x_tmp);
}

/* Based on hypre_ParCSRMatrixMatvecT in par_csr_matvec.c */
void hypre_ParCSRMatrixAbsMatvecT(hypre_ParCSRMatrix *A,
                                  HYPRE_Real alpha,
                                  HYPRE_Real *x,
                                  HYPRE_Real beta,
                                  HYPRE_Real *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix        *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix        *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real             *y_tmp;
   HYPRE_Real             *y_buf;

   HYPRE_Int               num_cols_offd = hypre_CSRMatrixNumCols(offd);

   HYPRE_Int               i, j, jj, end, num_sends;

   y_tmp = mfem_hypre_TAlloc(HYPRE_Real, num_cols_offd);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf = mfem_hypre_CTAlloc(
              HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   if (num_cols_offd)
   {
#if MFEM_HYPRE_VERSION >= 21100
      if (A->offdT)
      {
         // offdT is optional. Used only if it's present.
         hypre_CSRMatrixAbsMatvec(A->offdT, alpha, x, 0., y_tmp);
      }
      else
#endif
      {
         hypre_CSRMatrixAbsMatvecT(offd, alpha, x, 0., y_tmp);
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, y_tmp, y_buf);

#if MFEM_HYPRE_VERSION >= 21100
   if (A->diagT)
   {
      // diagT is optional. Used only if it's present.
      hypre_CSRMatrixAbsMatvec(A->diagT, alpha, x, beta, y);
   }
   else
#endif
   {
      hypre_CSRMatrixAbsMatvecT(diag, alpha, x, beta, y);
   }

   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_sends; i++)
   {
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
      for (j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); j < end; j++)
      {
         jj = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         y[jj] += y_buf[j];
      }
   }

   mfem_hypre_TFree(y_buf);
   mfem_hypre_TFree(y_tmp);
}

/* Based on hypre_ParCSRMatrixMatvec in par_csr_matvec.c */
void hypre_ParCSRMatrixBooleanMatvec(hypre_ParCSRMatrix *A,
                                     HYPRE_Bool alpha,
                                     HYPRE_Bool *x,
                                     HYPRE_Bool beta,
                                     HYPRE_Bool *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int          num_sends, i, j, index;

   HYPRE_Bool        *x_tmp, *x_buf;

   x_tmp = mfem_hypre_CTAlloc(HYPRE_Bool, num_cols_offd);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf = mfem_hypre_CTAlloc(
              HYPRE_Bool, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for ( ; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         x_buf[index++] = x[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate_bool(1, comm_pkg, x_buf, x_tmp);

   hypre_CSRMatrixBooleanMatvec(diag, alpha, x, beta, y);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (num_cols_offd)
   {
      hypre_CSRMatrixBooleanMatvec(offd, alpha, x_tmp, 1, y);
   }

   mfem_hypre_TFree(x_buf);
   mfem_hypre_TFree(x_tmp);
}

/* Based on hypre_ParCSRMatrixMatvecT in par_csr_matvec.c */
void hypre_ParCSRMatrixBooleanMatvecT(hypre_ParCSRMatrix *A,
                                      HYPRE_Bool alpha,
                                      HYPRE_Bool *x,
                                      HYPRE_Bool beta,
                                      HYPRE_Bool *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix        *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix        *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Bool             *y_tmp;
   HYPRE_Bool             *y_buf;

   HYPRE_Int               num_cols_offd = hypre_CSRMatrixNumCols(offd);

   HYPRE_Int               i, j, jj, end, num_sends;

   y_tmp = mfem_hypre_TAlloc(HYPRE_Bool, num_cols_offd);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf = mfem_hypre_CTAlloc(
              HYPRE_Bool, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   if (num_cols_offd)
   {
#if MFEM_HYPRE_VERSION >= 21100
      if (A->offdT)
      {
         // offdT is optional. Used only if it's present.
         hypre_CSRMatrixBooleanMatvec(A->offdT, alpha, x, 0, y_tmp);
      }
      else
#endif
      {
         hypre_CSRMatrixBooleanMatvecT(offd, alpha, x, 0, y_tmp);
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate_bool(2, comm_pkg, y_tmp, y_buf);

#if MFEM_HYPRE_VERSION >= 21100
   if (A->diagT)
   {
      // diagT is optional. Used only if it's present.
      hypre_CSRMatrixBooleanMatvec(A->diagT, alpha, x, beta, y);
   }
   else
#endif
   {
      hypre_CSRMatrixBooleanMatvecT(diag, alpha, x, beta, y);
   }

   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_sends; i++)
   {
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
      for (j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); j < end; j++)
      {
         jj = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         y[jj] = y[jj] || y_buf[j];
      }
   }

   mfem_hypre_TFree(y_buf);
   mfem_hypre_TFree(y_tmp);
}

HYPRE_Int
hypre_CSRMatrixSum(hypre_CSRMatrix *A,
                   HYPRE_Complex    beta,
                   hypre_CSRMatrix *B)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);

   HYPRE_Int         ia, j, pos;
   HYPRE_Int        *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      return -1; /* error: incompatible matrix dimensions */
   }

   marker = mfem_hypre_CTAlloc(HYPRE_Int, ncols_A);
   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   for (ia = 0; ia < nrows_A; ia++)
   {
      for (j = A_i[ia]; j < A_i[ia+1]; j++)
      {
         marker[A_j[j]] = j;
      }

      for (j = B_i[ia]; j < B_i[ia+1]; j++)
      {
         pos = marker[B_j[j]];
         if (pos < A_i[ia])
         {
            return -2; /* error: found an entry in B that is not present in A */
         }
         A_data[pos] += beta * B_data[j];
      }
   }

   mfem_hypre_TFree(marker);
   return 0;
}

hypre_ParCSRMatrix *
hypre_ParCSRMatrixAdd(hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix *B)
{
   MPI_Comm            comm   = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_cmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int           A_cmap_size = hypre_CSRMatrixNumCols(A_offd);
   hypre_CSRMatrix    *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix    *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int          *B_cmap = hypre_ParCSRMatrixColMapOffd(B);
   HYPRE_Int           B_cmap_size = hypre_CSRMatrixNumCols(B_offd);
   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix    *C_diag;
   hypre_CSRMatrix    *C_offd;
   HYPRE_Int          *C_cmap;
   HYPRE_Int           im;
   HYPRE_Int           cmap_differ;

   /* Check if A_cmap and B_cmap are the same. */
   cmap_differ = 0;
   if (A_cmap_size != B_cmap_size)
   {
      cmap_differ = 1; /* A and B have different cmap_size */
   }
   else
   {
      for (im = 0; im < A_cmap_size; im++)
      {
         if (A_cmap[im] != B_cmap[im])
         {
            cmap_differ = 1; /* A and B have different cmap arrays */
            break;
         }
      }
   }

   if ( cmap_differ == 0 )
   {
      /* A and B have the same column mapping for their off-diagonal blocks so
         we can sum the diagonal and off-diagonal blocks separately and reduce
         temporary memory usage. */

      /* Add diagonals, off-diagonals, copy cmap. */
      C_diag = hypre_CSRMatrixAdd(A_diag, B_diag);
      if (!C_diag)
      {
         return NULL; /* error: A_diag and B_diag have different dimensions */
      }
      C_offd = hypre_CSRMatrixAdd(A_offd, B_offd);
      if (!C_offd)
      {
         hypre_CSRMatrixDestroy(C_diag);
         return NULL; /* error: A_offd and B_offd have different dimensions */
      }
      /* copy A_cmap -> C_cmap */
      C_cmap = mfem_hypre_TAlloc(HYPRE_Int, A_cmap_size);
      for (im = 0; im < A_cmap_size; im++)
      {
         C_cmap[im] = A_cmap[im];
      }

      C = hypre_ParCSRMatrixCreate(comm,
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParCSRMatrixGlobalNumCols(A),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParCSRMatrixColStarts(A),
                                   hypre_CSRMatrixNumCols(C_offd),
                                   hypre_CSRMatrixNumNonzeros(C_diag),
                                   hypre_CSRMatrixNumNonzeros(C_offd));

      /* In C, destroy diag/offd (allocated by Create) and replace them with
      C_diag/C_offd. */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
      hypre_ParCSRMatrixDiag(C) = C_diag;
      hypre_ParCSRMatrixOffd(C) = C_offd;

      hypre_ParCSRMatrixColMapOffd(C) = C_cmap;
   }
   else
   {
      /* A and B have different column mappings for their off-diagonal blocks so
      we need to use the column maps to create full-width CSR matrices. */

      int  ierr = 0;
      hypre_CSRMatrix * csr_A;
      hypre_CSRMatrix * csr_B;
      hypre_CSRMatrix * csr_C_temp;

      /* merge diag and off-diag portions of A */
      csr_A = hypre_MergeDiagAndOffd(A);

      /* merge diag and off-diag portions of B */
      csr_B = hypre_MergeDiagAndOffd(B);

      /* add A and B */
      csr_C_temp = hypre_CSRMatrixAdd(csr_A,csr_B);

      /* delete CSR versions of A and B */
      ierr += hypre_CSRMatrixDestroy(csr_A);
      ierr += hypre_CSRMatrixDestroy(csr_B);

      /* create a new empty ParCSR matrix to contain the sum */
      C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParCSRMatrixGlobalNumCols(A),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParCSRMatrixColStarts(A),
                                   0, 0, 0);

      /* split C into diag and off-diag portions */
      /* FIXME: GenerateDiagAndOffd() uses an int array of size equal to the
         number of columns in csr_C_temp which is the global number of columns
         in A and B. This does not scale well. */
      ierr += GenerateDiagAndOffd(csr_C_temp, C,
                                  hypre_ParCSRMatrixFirstColDiag(A),
                                  hypre_ParCSRMatrixLastColDiag(A));

      /* delete CSR version of C */
      ierr += hypre_CSRMatrixDestroy(csr_C_temp);

      MFEM_VERIFY(ierr == 0, "");
   }

   /* hypre_ParCSRMatrixSetNumNonzeros(A); */

   /* Make sure that the first entry in each row is the diagonal one. */
   hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(C));

   /* C owns diag, offd, and cmap. */
   hypre_ParCSRMatrixSetDataOwner(C, 1);
   /* C does not own row and column starts. */
   hypre_ParCSRMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRMatrixSetColStartsOwner(C, 0);

   return C;
}

HYPRE_Int
hypre_ParCSRMatrixSum(hypre_ParCSRMatrix *A,
                      HYPRE_Complex       beta,
                      hypre_ParCSRMatrix *B)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int ncols_B_offd  = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_Int error;

   error = hypre_CSRMatrixSum(A_diag, beta, B_diag);
   if (ncols_B_offd > 0) /* treat B_offd as zero if it has no columns */
   {
      error = error ? error : hypre_CSRMatrixSum(A_offd, beta, B_offd);
   }

   return error;
}

HYPRE_Int
hypre_CSRMatrixSetConstantValues(hypre_CSRMatrix *A,
                                 HYPRE_Complex    value)
{
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int      A_nnz  = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      ia;

   for (ia = 0; ia < A_nnz; ia++)
   {
      A_data[ia] = value;
   }

   return 0;
}

HYPRE_Int
hypre_ParCSRMatrixSetConstantValues(hypre_ParCSRMatrix *A,
                                    HYPRE_Complex       value)
{
   hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixDiag(A), value);
   hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixOffd(A), value);

   return 0;
}

} // namespace mfem::internal

} // namespace mfem

#endif // MFEM_USE_MPI
