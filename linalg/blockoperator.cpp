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


#include "../general/array.hpp"
#include "operator.hpp"
#include "blockvector.hpp"
#include "blockoperator.hpp"

namespace mfem
{

BlockOperator::BlockOperator(const Array<int> & offsets)
   : Operator(offsets.Last()),
     owns_blocks(0),
     nRowBlocks(offsets.Size() - 1),
     nColBlocks(offsets.Size() - 1),
     row_offsets(0),
     col_offsets(0),
     op(nRowBlocks, nRowBlocks),
     coef(nRowBlocks, nColBlocks)
{
   op = static_cast<Operator *>(NULL);
   row_offsets.MakeRef(offsets);
   col_offsets.MakeRef(offsets);
}

BlockOperator::BlockOperator(const Array<int> & row_offsets_,
                             const Array<int> & col_offsets_)
   : Operator(row_offsets_.Last(), col_offsets_.Last()),
     owns_blocks(0),
     nRowBlocks(row_offsets_.Size()-1),
     nColBlocks(col_offsets_.Size()-1),
     row_offsets(0),
     col_offsets(0),
     op(nRowBlocks, nColBlocks),
     coef(nRowBlocks, nColBlocks)
{
   op = static_cast<Operator *>(NULL);
   row_offsets.MakeRef(row_offsets_);
   col_offsets.MakeRef(col_offsets_);
}

void BlockOperator::SetDiagonalBlock(int iblock, Operator *op, double c)
{
   SetBlock(iblock, iblock, op, c);
}

void BlockOperator::SetBlock(int iRow, int iCol, Operator *opt, double c)
{
   if (owns_blocks && op(iRow, iCol))
   {
      delete op(iRow, iCol);
   }
   op(iRow, iCol) = opt;
   coef(iRow, iCol) = c;

   MFEM_VERIFY(row_offsets[iRow+1] - row_offsets[iRow] == opt->NumRows() &&
               col_offsets[iCol+1] - col_offsets[iCol] == opt->NumCols(),
               "incompatible Operator dimensions");
}

// Operator application
void BlockOperator::Mult (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),col_offsets);
   yblock.Update(y,row_offsets);

   for (int iRow=0; iRow < nRowBlocks; ++iRow)
   {
      tmp.SetSize(row_offsets[iRow+1] - row_offsets[iRow]);
      for (int jCol=0; jCol < nColBlocks; ++jCol)
      {
         if (op(iRow,jCol))
         {
            op(iRow,jCol)->Mult(xblock.GetBlock(jCol), tmp);
            yblock.GetBlock(iRow).Add(coef(iRow,jCol), tmp);
         }
      }
   }

   for (int iRow=0; iRow < nRowBlocks; ++iRow)
   {
      yblock.GetBlock(iRow).SyncAliasMemory(y);
   }

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
   for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
}

// Action of the transpose operator
void BlockOperator::MultTranspose (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),row_offsets);
   yblock.Update(y,col_offsets);

   for (int iRow=0; iRow < nColBlocks; ++iRow)
   {
      tmp.SetSize(col_offsets[iRow+1] - col_offsets[iRow]);
      for (int jCol=0; jCol < nRowBlocks; ++jCol)
      {
         if (op(jCol,iRow))
         {
            op(jCol,iRow)->MultTranspose(xblock.GetBlock(jCol), tmp);
            yblock.GetBlock(iRow).Add(coef(jCol,iRow), tmp);
         }
      }
   }

   for (int iRow=0; iRow < nColBlocks; ++iRow)
   {
      yblock.GetBlock(iRow).SyncAliasMemory(y);
   }

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
   for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
}

BlockOperator::~BlockOperator()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nRowBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nColBlocks; ++jCol)
         {
            delete op(jCol,iRow);
         }
      }
   }
}

//-----------------------------------------------------------------------
BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(
   const Array<int> & offsets_):
   Solver(offsets_.Last()),
   owns_blocks(0),
   nBlocks(offsets_.Size() - 1),
   offsets(0),
   op(nBlocks)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockDiagonalPreconditioner::SetDiagonalBlock(int iblock, Operator *opt)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == opt->Height() &&
               offsets[iblock+1] - offsets[iblock] == opt->Width(),
               "incompatible Operator dimensions");

   if (owns_blocks && op[iblock])
   {
      delete op[iblock];
   }
   op[iblock] = opt;
}

// Operator application
void BlockDiagonalPreconditioner::Mult (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),offsets);
   yblock.Update(y,offsets);

   for (int i=0; i<nBlocks; ++i)
   {
      if (op[i])
      {
         op[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }
   }

   for (int i=0; i<nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(y);
   }

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
   for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
}

// Action of the transpose operator
void BlockDiagonalPreconditioner::MultTranspose (const Vector & x,
                                                 Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),offsets);
   yblock.Update(y,offsets);

   for (int i=0; i<nBlocks; ++i)
   {
      if (op[i])
      {
         (op[i])->MultTranspose(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }
   }

   for (int i=0; i<nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(y);
   }

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
   for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
}

BlockDiagonalPreconditioner::~BlockDiagonalPreconditioner()
{
   if (owns_blocks)
   {
      for (int i=0; i<nBlocks; ++i)
      {
         delete op[i];
      }
   }
}

BlockLowerTriangularPreconditioner::BlockLowerTriangularPreconditioner(
   const Array<int> & offsets_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     op(nBlocks, nBlocks)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockLowerTriangularPreconditioner::SetDiagonalBlock(int iblock,
                                                          Operator *op)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");

   SetBlock(iblock, iblock, op);
}

void BlockLowerTriangularPreconditioner::SetBlock(int iRow, int iCol,
                                                  Operator *opt)
{
   MFEM_VERIFY(iRow >= iCol,"cannot set block in upper triangle");
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == opt->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == opt->NumCols(),
               "incompatible Operator dimensions");

   op(iRow, iCol) = opt;
}

// Operator application
void BlockLowerTriangularPreconditioner::Mult (const Vector & x,
                                               Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   y = 0.0;
   for (int iRow=0; iRow < nBlocks; ++iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=0; jCol < iRow; ++jCol)
      {
         if (op(iRow,jCol))
         {
            op(iRow,jCol)->Mult(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->Mult(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

// Action of the transpose operator
void BlockLowerTriangularPreconditioner::MultTranspose (const Vector & x,
                                                        Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   y = 0.0;
   for (int iRow=nBlocks-1; iRow >=0; --iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=iRow+1; jCol < nBlocks; ++jCol)
      {
         if (op(jCol,iRow))
         {
            op(jCol,iRow)->MultTranspose(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->MultTranspose(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

BlockLowerTriangularPreconditioner::~BlockLowerTriangularPreconditioner()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nBlocks; ++jCol)
         {
            delete op(jCol,iRow);
         }
      }
   }
}

BlockUpperTriangularPreconditioner::BlockUpperTriangularPreconditioner(
   const Array<int> & offsets_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     op(nBlocks, nBlocks)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockUpperTriangularPreconditioner::SetDiagonalBlock(int iblock,
                                                          Operator *op)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");

   SetBlock(iblock, iblock, op);
}

void BlockUpperTriangularPreconditioner::SetBlock(int iRow, int iCol,
                                                  Operator *opt)
{
   MFEM_VERIFY(iRow <= iCol,"cannot set block in lower triangle");
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == opt->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == opt->NumCols(),
               "incompatible Operator dimensions");

   op(iRow, iCol) = opt;
}

// Operator application
void BlockUpperTriangularPreconditioner::MultTranspose (const Vector & x,
                                               Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   y = 0.0;
   for (int iRow=0; iRow < nBlocks; ++iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=0; jCol < iRow; ++jCol)
      {
         if (op(iRow,jCol))
         {
            op(iRow,jCol)->MultTranspose(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->MultTranspose(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

// Action of the transpose operator
void BlockUpperTriangularPreconditioner::Mult(const Vector & x,
                                                        Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);

   y = 0.0;
   for (int iRow=nBlocks-1; iRow >=0; --iRow)
   {
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=iRow+1; jCol < nBlocks; ++jCol)
      {
         if (op(jCol,iRow))
         {
            op(jCol,iRow)->Mult(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (op(iRow,iRow))
      {
         op(iRow,iRow)->Mult(tmp2, yblock.GetBlock(iRow));
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;
      }
   }
}

BlockUpperTriangularPreconditioner::~BlockUpperTriangularPreconditioner()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nBlocks; ++jCol)
         {
            delete op(jCol,iRow);
         }
      }
   }
}

BlockLDUPreconditioner::BlockLDUPreconditioner(
   const Array<int> & offsets_, int schur_index_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     op(nBlocks, nBlocks),
     schur_index(schur_index_)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockLDUPreconditioner::SetDiagonalBlock(int iblock, Operator *op)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");
   MFEM_VERIFY(iblock >=0 && iblock <= 1,
               "Only valid for 2x2 block matrices");

   SetBlock(iblock, iblock, op);
}

void BlockLDUPreconditioner::SetBlock(int iRow, int iCol, Operator *opt)
{
   MFEM_VERIFY(iRow >=0 && iRow <= 1, "Only valid for 2x2 block matrices");
   MFEM_VERIFY(iCol >=0 && iCol <= 1, "Only valid for 2x2 block matrices");
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == opt->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == opt->NumCols(),
               "incompatible Operator dimensions");

   op(iRow, iCol) = opt;
}

// Action of the transpose operator
void BlockLDUPreconditioner::Mult(const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   yblock.Update(y.GetData(),offsets);
   xblock.Update(x.GetData(),offsets);
   y = 0.0;

   // Schur-complement in the (1,1) block (i.e., (0,0) block w/ zero indexing)
   // TODO

   // Schur-complement in the (2,2) block (i.e., (1,1) block w/ zero indexing)
   // A^{-1} = [I, -A11^{-1}A12; 0, I] * [I, 0; 0, S22^{-2}] * 
   //          [I, 0; -A21, I] * [A11^{-1}, 0; 0, I]
   op(0,0) -> Mult(xblock.GetBlock(0), yblock.GetBlock(0));
   
   tmp.SetSize(offsets[2] - offsets[1]);
   op(1,0) -> Mult(yblock.GetBlock(0), tmp);
   tmp *= -1;
   tmp += xblock.GetBlock(1);

   op(1,1) -> Mult(tmp, yblock.GetBlock(1));

   // BUG IN THIS SECTION
   tmp3.SetSize(offsets[1] - offsets[0]);
   tmp2.SetSize(offsets[1] - offsets[0]);
   tmp2 = 0.0;
   op(0,1) -> Mult(yblock.GetBlock(1), tmp3);
   op(0,0) -> Mult(tmp3, tmp2);
   tmp = yblock.GetBlock(0);
   yblock.GetBlock(0) = tmp - tmp2; // <--- this line
}

BlockLDUPreconditioner::~BlockLDUPreconditioner()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nBlocks; ++jCol)
         {
            delete op(jCol,iRow);
         }
      }
   }
}

}
