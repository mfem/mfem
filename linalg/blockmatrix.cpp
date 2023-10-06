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

#include "../general/array.hpp"
#include "../general/globals.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "blockvector.hpp"
#include "blockmatrix.hpp"
#include <vector>

namespace mfem
{

BlockMatrix::BlockMatrix(const Array<int> & offsets):
   AbstractSparseMatrix(offsets.Last()),
   owns_blocks(false),
   nRowBlocks(offsets.Size()-1),
   nColBlocks(offsets.Size()-1),
   Aij(nRowBlocks, nColBlocks)
{
   row_offsets.MakeRef(const_cast< Array<int>& >(offsets));
   col_offsets.MakeRef(const_cast< Array<int>& >(offsets));
   Aij = (SparseMatrix *)NULL;
}

BlockMatrix::BlockMatrix(const Array<int> & row_offsets_,
                         const Array<int> & col_offsets_):
   AbstractSparseMatrix(row_offsets_.Last(), col_offsets_.Last()),
   owns_blocks(false),
   nRowBlocks(row_offsets_.Size()-1),
   nColBlocks(col_offsets_.Size()-1),
   Aij(nRowBlocks, nColBlocks)
{
   row_offsets.MakeRef(const_cast< Array<int>& >(row_offsets_));
   col_offsets.MakeRef(const_cast< Array<int>& >(col_offsets_));
   Aij = (SparseMatrix *)NULL;
}


BlockMatrix::~BlockMatrix()
{
   if (owns_blocks)
   {
      for (SparseMatrix ** it = Aij.GetRow(0);
           it != Aij.GetRow(0)+(Aij.NumRows()*Aij.NumCols()); ++it)
      {
         delete *it;
      }
   }
}

void BlockMatrix::SetBlock(int i, int j, SparseMatrix * mat)
{
#ifdef MFEM_DEBUG
   if (nRowBlocks <= i || nColBlocks <= j)
   {
      mfem_error("BlockMatrix::SetBlock #0");
   }

   if (mat->Height() != row_offsets[i+1] - row_offsets[i])
   {
      mfem_error("BlockMatrix::SetBlock #1");
   }

   if (mat->Width() != col_offsets[j+1] - col_offsets[j])
   {
      mfem_error("BlockMatrix::SetBlock #2");
   }
#endif
   Aij(i,j) = mat;
}

SparseMatrix & BlockMatrix::GetBlock(int i, int j)
{
#ifdef MFEM_DEBUG
   if (nRowBlocks <= i || nColBlocks <= j)
   {
      mfem_error("BlockMatrix::Block #0");
   }

   if (IsZeroBlock(i,j))
   {
      mfem_error("BlockMatrix::Block #1");
   }
#endif
   return *Aij(i,j);
}

const SparseMatrix & BlockMatrix::GetBlock(int i, int j) const
{
#ifdef MFEM_DEBUG
   if (nRowBlocks <= i || nColBlocks <= j)
   {
      mfem_error("BlockMatrix::Block const #0");
   }

   if (IsZeroBlock(i,j))
   {
      mfem_error("BlockMatrix::Block const #1");
   }
#endif

   return *Aij(i,j);
}


int BlockMatrix::NumNonZeroElems() const
{
   int nnz_elem = 0;
   for (int jcol = 0; jcol != nColBlocks; ++jcol)
   {
      for (int irow = 0; irow != nRowBlocks; ++irow)
      {
         if (Aij(irow,jcol))
         {
            nnz_elem+= Aij(irow,jcol)->NumNonZeroElems();
         }
      }
   }
   return nnz_elem;
}


double& BlockMatrix::Elem (int i, int j)
{
   int iloc, iblock;
   int jloc, jblock;

   findGlobalRow(i, iblock, iloc);
   findGlobalCol(j, jblock, jloc);

   if (IsZeroBlock(iblock, jblock))
   {
      mfem_error("BlockMatrix::Elem");
   }

   return Aij(iblock, jblock)->Elem(iloc, jloc);
}

const double& BlockMatrix::Elem (int i, int j) const
{
   static const double zero = 0.0;
   int iloc, iblock;
   int jloc, jblock;

   findGlobalRow(i, iblock, iloc);
   findGlobalCol(j, jblock, jloc);

   if (IsZeroBlock(iblock, jblock))
   {
      return zero;
   }
   return static_cast<const SparseMatrix *>(Aij(iblock, jblock))->Elem(iloc, jloc);
}

int BlockMatrix::RowSize(const int i) const
{
   int rowsize = 0;

   int iblock, iloc;
   findGlobalRow(i, iblock, iloc);

   for (int jblock = 0; jblock < nColBlocks; ++jblock)
   {
      if (Aij(iblock,jblock) != NULL)
      {
         rowsize += Aij(iblock,jblock)->RowSize(iloc);
      }
   }

   return rowsize;
}

int BlockMatrix::GetRow(const int row, Array<int> &cols, Vector &srow) const
{
   int iblock, iloc, rowsize;
   findGlobalRow(row, iblock, iloc);
   rowsize = RowSize(row);
   cols.SetSize(rowsize);
   srow.SetSize(rowsize);

   Array<int> bcols;
   Vector bsrow;

   int * it_cols = cols.GetData();
   double *it_srow = srow.GetData();

   for (int jblock = 0; jblock < nColBlocks; ++jblock)
   {
      if (Aij(iblock,jblock) != NULL)
      {
         Aij(iblock,jblock)->GetRow(iloc, bcols, bsrow);
         for (int i = 0; i < bcols.Size(); ++i)
         {
            *(it_cols++) = bcols[i] + col_offsets[jblock];
            *(it_srow++) = bsrow(i);
         }
      }
   }

   return 0;
}

void BlockMatrix::EliminateRowCol(int rc, DiagonalPolicy dpolicy)
{
   // Find the block to which the dof belongs and its local number
   int idx, iiblock;
   for (iiblock = 0; iiblock < nRowBlocks; ++iiblock)
   {
      idx = rc - row_offsets[iiblock];
      if (idx < 0 ) { break; }
   }
   iiblock--;
   idx = rc - row_offsets[iiblock];

   // Asserts
   MFEM_ASSERT(nRowBlocks == nColBlocks,
               "BlockMatrix::EliminateRowCol: nRowBlocks != nColBlocks");

   MFEM_ASSERT(row_offsets[iiblock] == col_offsets[iiblock],
               "BlockMatrix::EliminateRowCol: row_offsets["
               << iiblock << "] != col_offsets["<<iiblock<<"]");

   MFEM_ASSERT(Aij(iiblock, iiblock),
               "BlockMatrix::EliminateRowCol: Null diagonal block");

   // Apply the constraint idx to the iiblock
   for (int jjblock = 0; jjblock < nRowBlocks; ++jjblock)
   {
      if (iiblock == jjblock) { continue; }
      if (Aij(iiblock,jjblock)) { Aij(iiblock,jjblock)->EliminateRow(idx); }
   }
   for (int jjblock = 0; jjblock < nRowBlocks; ++jjblock)
   {
      if (iiblock == jjblock) { continue; }
      if (Aij(jjblock,iiblock)) { Aij(jjblock,iiblock)->EliminateCol(idx); }
   }
   Aij(iiblock, iiblock)->EliminateRowCol(idx,dpolicy);
}

void BlockMatrix::EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol,
                                  Vector & rhs)
{
   if (nRowBlocks != nColBlocks)
   {
      mfem_error("BlockMatrix::EliminateRowCol: nRowBlocks != nColBlocks");
   }

   for (int iiblock = 0; iiblock < nRowBlocks; ++iiblock)
   {
      if (row_offsets[iiblock] != col_offsets[iiblock])
      {
         mfem::out << "BlockMatrix::EliminateRowCol: row_offsets["
                   << iiblock << "] != col_offsets["<<iiblock<<"]\n";
         mfem_error();
      }
   }

   // We also have to do the same for each Aij
   Array<int> block_dofs;
   Vector block_sol, block_rhs;

   for (int iiblock = 0; iiblock < nRowBlocks; ++iiblock)
   {
      int dsize = row_offsets[iiblock+1] - row_offsets[iiblock];
      block_dofs.MakeRef(ess_bc_dofs.GetData()+row_offsets[iiblock], dsize);
      block_sol.SetDataAndSize(sol.GetData()+row_offsets[iiblock], dsize);
      block_rhs.SetDataAndSize(rhs.GetData()+row_offsets[iiblock], dsize);

      if (Aij(iiblock, iiblock))
      {
         for (int i = 0; i < block_dofs.Size(); ++i)
         {
            if (block_dofs[i])
            {
               Aij(iiblock, iiblock)->EliminateRowCol(i,block_sol(i), block_rhs);
            }
         }
      }
      else
      {
         for (int i = 0; i < block_dofs.Size(); ++i)
         {
            if (block_dofs[i])
            {
               mfem_error("BlockMatrix::EliminateRowCol: Null diagonal block \n");
            }
         }
      }

      for (int jjblock = 0; jjblock < nRowBlocks; ++jjblock)
      {
         if (jjblock != iiblock && Aij(iiblock, jjblock))
         {
            for (int i = 0; i < block_dofs.Size(); ++i)
            {
               if (block_dofs[i])
               {
                  Aij(iiblock, jjblock)->EliminateRow(i);
               }
            }
         }
         if (jjblock != iiblock && Aij(jjblock, iiblock))
         {
            block_rhs.SetDataAndSize(rhs.GetData()+row_offsets[jjblock],
                                     row_offsets[jjblock+1] - row_offsets[jjblock]);
            Aij(jjblock, iiblock)->EliminateCols(block_dofs, &block_sol, &block_rhs);
         }
      }
   }
}

void BlockMatrix::EliminateRowCols(const Array<int> & vdofs, BlockMatrix *Ae,
                                   DiagonalPolicy dpolicy)
{
   MFEM_VERIFY(Ae,
               "BlockMatrix::EliminateRowCols: Elimination matrix pointer is null");
   MFEM_VERIFY(nRowBlocks == nColBlocks,
               "BlockMatrix::EliminateRowCols supported only for"
               "nRowBlocks = nColBlocks");

   std::vector<Array<int>> cols(nRowBlocks);
   std::vector<Array<int>> rows(nRowBlocks);
   SparseMatrix * tmp = nullptr;

   for (int k = 0; k < vdofs.Size(); k++)
   {
      int vdof = (vdofs[k]) >=0 ? vdofs[k] : -1 - vdofs[k];
      // find block
      int iblock, dof;
      findGlobalCol(vdof,iblock,dof);
      cols[iblock].Append(dof);
      tmp = &GetBlock(iblock,iblock);
      if (tmp)
      {
         tmp->EliminateRowCol(dof,Ae->GetBlock(iblock,iblock), dpolicy);
      }
   }

   // Eliminate col from off-diagonal blocks
   for (int j = 0; j<nColBlocks; j++)
   {
      if (!cols[j].Size()) { continue; }
      int blocksize = col_offsets[j+1] - col_offsets[j];
      Array<int> colmarker(blocksize); colmarker = 0;
      for (int i = 0; i < cols[j].Size(); i++) { colmarker[cols[j][i]] = 1; }
      for (int i = 0; i<nRowBlocks; i++)
      {
         if (i == j) { continue; }
         tmp = &GetBlock(i,j);
         if (tmp) { tmp->EliminateCols(colmarker,Ae->GetBlock(i,j)); }
         for (int k = 0; k < cols[j].Size(); k++)
         {
            tmp = &GetBlock(j,i);
            if (tmp) { tmp->EliminateRow(cols[j][k]); }
         }
      }
   }
}

void BlockMatrix::EliminateZeroRows(const double threshold)
{
   MFEM_VERIFY(nRowBlocks == nColBlocks, "not a square matrix");

   for (int iblock = 0; iblock < nRowBlocks; ++iblock)
   {
      if (Aij(iblock,iblock))
      {
         double norm;
         for (int i = 0; i < Aij(iblock, iblock)->NumRows(); ++i)
         {
            norm = 0.;
            for (int jblock = 0; jblock < nColBlocks; ++jblock)
               if (Aij(iblock,jblock))
               {
                  norm += Aij(iblock,jblock)->GetRowNorml1(i);
               }

            if (norm <= threshold)
            {
               for (int jblock = 0; jblock < nColBlocks; ++jblock)
               {
                  if (Aij(iblock,jblock))
                  {
                     Aij(iblock,jblock)->EliminateRow(
                        i, (iblock==jblock) ? DIAG_ONE : DIAG_ZERO);
                  }
               }
            }
         }
      }
      else
      {
         double norm;
         for (int i = 0; i < row_offsets[iblock+1] - row_offsets[iblock]; ++i)
         {
            norm = 0.;
            for (int jblock = 0; jblock < nColBlocks; ++jblock)
            {
               if (Aij(iblock,jblock))
               {
                  norm += Aij(iblock,jblock)->GetRowNorml1(i);
               }
            }

            MFEM_VERIFY(!(norm <= threshold), "diagonal block is NULL:"
                        " iblock = " << iblock << ", i = " << i << ", norm = "
                        << norm);
         }
      }
   }
}

void BlockMatrix::Finalize(int skip_zeros, bool fix_empty_rows)
{
   for (int iblock = 0; iblock < nRowBlocks; ++iblock)
   {
      for (int jblock = 0; jblock < nColBlocks; ++jblock)
      {
         if (!Aij(iblock,jblock)) { continue; }
         if (!Aij(iblock,jblock)->Finalized())
         {
            Aij(iblock,jblock)->Finalize(skip_zeros, fix_empty_rows);
         }
      }
   }
}

void BlockMatrix::Mult(const Vector & x, Vector & y) const
{
   if (x.GetData() == y.GetData())
   {
      mfem_error("Error: x and y can't point to the same data \n");
   }

   MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
               << ") must match matrix width (" << width << ")");
   MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
               << ") must match matrix height (" << height << ")");
   y = 0.;
   AddMult(x, y, 1.0);
}

void BlockMatrix::AddMult(const Vector & x, Vector & y, const double val) const
{
   if (x.GetData() == y.GetData())
   {
      mfem_error("Error: x and y can't point to the same data \n");
   }

   Vector xblockview, yblockview;

   for (int iblock = 0; iblock != nRowBlocks; ++iblock)
   {
      yblockview.SetDataAndSize(y.GetData() + row_offsets[iblock],
                                row_offsets[iblock+1] - row_offsets[iblock]);

      for (int jblock = 0; jblock != nColBlocks; ++jblock)
      {
         if (Aij(iblock, jblock) != NULL)
         {
            xblockview.SetDataAndSize(
               x.GetData() + col_offsets[jblock],
               col_offsets[jblock+1] - col_offsets[jblock]);

            Aij(iblock, jblock)->AddMult(xblockview, yblockview, val);
         }
      }
   }
}

void BlockMatrix::MultTranspose(const Vector & x, Vector & y) const
{
   if (x.GetData() == y.GetData())
   {
      mfem_error("Error: x and y can't point to the same data \n");
   }

   y = 0.;
   AddMultTranspose(x, y, 1.0);
}

void BlockMatrix::AddMultTranspose(const Vector & x, Vector & y,
                                   const double val) const
{
   if (x.GetData() == y.GetData())
   {
      mfem_error("Error: x and y can't point to the same data \n");
   }

   Vector xblockview, yblockview;

   for (int iblock = 0; iblock != nColBlocks; ++iblock)
   {
      yblockview.SetDataAndSize(y.GetData() + col_offsets[iblock],
                                col_offsets[iblock+1] - col_offsets[iblock]);

      for (int jblock = 0; jblock != nRowBlocks; ++jblock)
      {
         if (Aij(jblock, iblock) != NULL)
         {
            xblockview.SetDataAndSize(
               x.GetData() + row_offsets[jblock],
               row_offsets[jblock+1] - row_offsets[jblock]);

            Aij(jblock, iblock)->AddMultTranspose(xblockview, yblockview, val);
         }
      }
   }
}

void BlockMatrix::PartMult(const Array<int> &rows, const Vector &x,
                           Vector &y) const
{
   Array<int> cols;
   Vector srow;
   for (int i = 0; i<rows.Size(); i++)
   {
      int dof = (rows[i]>=0) ? rows[i] : -1-rows[i];
      GetRow(dof,cols,srow);

      double s=0.0;
      for (int k = 0; k <cols.Size(); k++)
      {
         s += srow[k] * x[cols[k]];
      }
      y[dof] = s;
   }
}
void BlockMatrix::PartAddMult(const Array<int> &rows, const Vector &x,
                              Vector &y,
                              const double a) const
{
   Array<int> cols;
   Vector srow;
   for (int i = 0; i<rows.Size(); i++)
   {
      int dof = (rows[i]>=0) ? rows[i] : -1-rows[i];
      GetRow(dof,cols,srow);

      double s=0.0;
      for (int k = 0; k <cols.Size(); k++)
      {
         s += srow[k] * x[cols[k]];
      }
      y[dof] += a * s;
   }
}


SparseMatrix * BlockMatrix::CreateMonolithic() const
{
   int nnz = NumNonZeroElems();

   int * i_amono = Memory<int>(row_offsets[nRowBlocks]+2);
   int * j_amono = Memory<int>(nnz);
   double * data = Memory<double>(nnz);

   for (int i = 0; i < row_offsets[nRowBlocks]+2; i++)
   {
      i_amono[i] = 0;
   }

   int * i_amono_construction = i_amono+1;

   for (int iblock = 0; iblock != nRowBlocks; ++iblock)
   {
      for (int irow(row_offsets[iblock]); irow < row_offsets[iblock+1]; ++irow)
      {
         int local_row = irow - row_offsets[iblock];
         int ind = i_amono_construction[irow];
         for (int jblock = 0; jblock < nColBlocks; ++jblock)
         {
            if (Aij(iblock,jblock) != NULL)
               ind += Aij(iblock, jblock)->GetI()[local_row+1]
                      - Aij(iblock, jblock)->GetI()[local_row];
         }
         i_amono_construction[irow+1] = ind;
      }
   }

   // Fill in the jarray and copy the data
   for (int iblock = 0; iblock != nRowBlocks; ++iblock)
   {
      for (int jblock = 0; jblock != nColBlocks; ++jblock)
      {
         if (Aij(iblock,jblock) != NULL)
         {
            int nrow = row_offsets[iblock+1]-row_offsets[iblock];
            int * i_aij = Aij(iblock, jblock)->GetI();
            int * j_aij = Aij(iblock, jblock)->GetJ();
            double * data_aij = Aij(iblock, jblock)->GetData();
            int *i_it = i_amono_construction+row_offsets[iblock];

            int loc_start_index = 0;
            int loc_end_index = 0;
            int glob_start_index = 0;

            int shift(col_offsets[jblock]);
            for (int * i_it_aij(i_aij+1); i_it_aij != i_aij+nrow+1; ++i_it_aij)
            {
               glob_start_index = *i_it;

#ifdef MFEM_DEBUG
               if (glob_start_index > nnz)
               {
                  mfem::out<<"glob_start_index = " << glob_start_index << "\n";
                  mfem::out<<"Block:" << iblock << " " << jblock << "\n";
                  mfem::out<<std::endl;
               }
#endif
               loc_end_index = *(i_it_aij);
               for (int cnt = 0; cnt < loc_end_index-loc_start_index; cnt++)
               {
                  data[glob_start_index+cnt] = data_aij[loc_start_index+cnt];
                  j_amono[glob_start_index+cnt] = j_aij[loc_start_index+cnt] + shift;
               }

               *i_it += loc_end_index-loc_start_index;
               ++i_it;
               loc_start_index = loc_end_index;
            }
         }
      }
   }

   return new SparseMatrix(i_amono, j_amono, data, row_offsets[nRowBlocks],
                           col_offsets[nColBlocks]);
}

void BlockMatrix::PrintMatlab(std::ostream & os) const
{

   Vector row_data;
   Array<int> row_ind;
   int nnz_elem = NumNonZeroElems();
   os<<"% size " << row_offsets.Last() << " " << col_offsets.Last() << "\n";
   os<<"% Non Zeros " << nnz_elem << "\n";
   int i, j;
   std::ios::fmtflags old_fmt = os.flags();
   os.setf(std::ios::scientific);
   std::streamsize old_prec = os.precision(14);
   for (i = 0; i < row_offsets.Last(); i++)
   {
      GetRow(i, row_ind, row_data);
      for (j = 0; j < row_ind.Size(); j++)
      {
         os << i+1 << " " << row_ind[j]+1 << " " << row_data[j] << std::endl;
      }
   }
   // Write a zero entry at (m,n) to make sure MATLAB doesn't shrink the matrix
   os << row_offsets.Last() << " " << col_offsets.Last () << " 0.0\n";

   os.precision(old_prec);
   os.flags(old_fmt);
}

BlockMatrix * Transpose(const BlockMatrix & A)
{
   BlockMatrix * At = new BlockMatrix(A.ColOffsets(), A.RowOffsets());
   At->owns_blocks = 1;

   for (int irowAt = 0; irowAt < At->NumRowBlocks(); ++irowAt)
   {
      for (int jcolAt = 0; jcolAt < At->NumColBlocks(); ++jcolAt)
      {
         if (!A.IsZeroBlock(jcolAt, irowAt))
         {
            At->SetBlock(irowAt, jcolAt, Transpose(A.GetBlock(jcolAt, irowAt)));
         }
      }
   }
   return At;
}

BlockMatrix * Mult(const BlockMatrix & A, const BlockMatrix & B)
{
   BlockMatrix * C= new BlockMatrix(A.RowOffsets(), B.ColOffsets());
   C->owns_blocks = 1;
   Array<SparseMatrix *> CijPieces(A.NumColBlocks());

   for (int irowC = 0; irowC < A.NumRowBlocks(); ++irowC)
   {
      for (int jcolC = 0; jcolC < B.NumColBlocks(); ++jcolC)
      {
         CijPieces.SetSize(0, static_cast<SparseMatrix *>(NULL));
         for (int k = 0; k < A.NumColBlocks(); ++k)
         {
            if (!A.IsZeroBlock(irowC, k) && !B.IsZeroBlock(k, jcolC))
            {
               CijPieces.Append(Mult(A.GetBlock(irowC, k), B.GetBlock(k, jcolC)));
            }
         }

         if (CijPieces.Size() > 1)
         {
            C->SetBlock(irowC, jcolC, Add(CijPieces));
            for (SparseMatrix ** it = CijPieces.GetData();
                 it != CijPieces.GetData()+CijPieces.Size(); ++it)
            {
               delete *it;
            }
         }
         else if (CijPieces.Size() == 1)
         {
            C->SetBlock(irowC, jcolC, CijPieces[0]);
         }
      }
   }

   return C;
}

}
