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
     coef(nRowBlocks, nColBlocks),
     vr(nRowBlocks),
     vc(nColBlocks)
{
   op = static_cast<Operator *>(NULL);
   row_offsets.MakeRef(offsets);
   col_offsets.MakeRef(offsets);

   for (int i=0; i<nRowBlocks; ++i)
   {
      vr[i].SetSize(row_offsets[i+1] - row_offsets[i]);
   }
   for (int i=0; i<nColBlocks; ++i)
   {
      vc[i].SetSize(col_offsets[i+1] - col_offsets[i]);
   }
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
     coef(nRowBlocks, nColBlocks),
     vr(nRowBlocks),
     vc(nColBlocks)
{
   op = static_cast<Operator *>(NULL);
   row_offsets.MakeRef(row_offsets_);
   col_offsets.MakeRef(col_offsets_);

   for (int i=0; i<nRowBlocks; ++i)
   {
      vr[i].SetSize(row_offsets[i+1] - row_offsets[i]);
   }
   for (int i=0; i<nColBlocks; ++i)
   {
      vc[i].SetSize(col_offsets[i+1] - col_offsets[i]);
   }
}

void BlockOperator::SetDiagonalBlock(int iblock, Operator *op, double c)
{
   SetBlock(iblock, iblock, op, c);
}

void BlockOperator::SetBlock(int iRow, int iCol, Operator *opt, double c)
{
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

   /*
   // Dylan version
   y = 0.0;

   for (int iRow=0; iRow < nRowBlocks; ++iRow)
   {
      vr[iRow] = 0.0;
      for (int jCol=0; jCol < nColBlocks; ++jCol)
      {
         if (op(iRow,jCol))
         {
            vc[jCol].SetOffset(1.0, x, col_offsets[jCol]);
            op(iRow,jCol)->Mult(vc[jCol], vr[iRow]);
            y.AddOffset(coef(iRow,jCol), vr[iRow], row_offsets[iRow]);
         }
      }
   }
   */
   /*
   // Master version
   yblock.Update(y.GetData(),row_offsets);
   xblock.Update(x.GetData(),col_offsets);

   y = 0.0;
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
   */

   // Veselin version
   yblock.Update(y,row_offsets);
   xblock.Update(const_cast<Vector&>(x),col_offsets);

   y = 0.0;
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
}

// Action of the transpose operator
void BlockOperator::MultTranspose (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   y = 0.0;

   xblock.Update(x.GetData(),row_offsets);
   yblock.Update(y.GetData(),col_offsets);

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
   op(nBlocks),
   vr(nBlocks),
   vc(nBlocks)
{
   op = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);

   for (int i=0; i<nBlocks; ++i)
   {
      vr[i].SetSize(offsets[i+1] - offsets[i]);
      vc[i].SetSize(offsets[i+1] - offsets[i]);
   }
}

void BlockDiagonalPreconditioner::SetDiagonalBlock(int iblock, Operator *opt)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == opt->Height() &&
               offsets[iblock+1] - offsets[iblock] == opt->Width(),
               "incompatible Operator dimensions");

   op[iblock] = opt;
}

// Operator application
void BlockDiagonalPreconditioner::Mult (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   y = 0.0;

   for (int i=0; i<nBlocks; ++i)
   {
      if (op[i])
      {
         vc[i].SetOffset(1.0, x, offsets[i]);
         op[i]->Mult(vc[i], vr[i]);
         y.AddOffset(1.0, vr[i], offsets[i]);
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }
   }
}

// Action of the transpose operator
void BlockDiagonalPreconditioner::MultTranspose (const Vector & x,
                                                 Vector & y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   yblock.Update(y.GetData(), offsets);
   xblock.Update(x.GetData(), offsets);

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

}
