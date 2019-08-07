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
   
   const bool use_dev = x.UseDevice() || y.UseDevice();
   yblock.Update(y,row_offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   xblock.Update(x,col_offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);
   tmp.UseDevice(use_dev);

   y = 0.0;
   for (int iRow=0; iRow < nRowBlocks; ++iRow)
   {
      yblock.GetBlock(iRow).SyncAliasMemory(yblock);
      tmp.SetSize(row_offsets[iRow+1] - row_offsets[iRow]);
      for (int jCol=0; jCol < nColBlocks; ++jCol)
      {
         if (op(iRow,jCol))
         {
            xblock.GetBlock(jCol).SyncAliasMemory(xblock); // TODO: This shouldn't have to be done...
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

   const bool use_dev = x.UseDevice() || y.UseDevice();
   xblock.Update(x,row_offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);
   yblock.Update(y,col_offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   tmp.UseDevice(use_dev);

   for (int iRow=0; iRow < nColBlocks; ++iRow)
   {
      yblock.GetBlock(iRow).SyncAliasMemory(yblock);
      tmp.SetSize(col_offsets[iRow+1] - col_offsets[iRow]);
      for (int jCol=0; jCol < nRowBlocks; ++jCol)
      {
         if (op(jCol,iRow))
         {
            xblock.GetBlock(jCol).SyncAliasMemory(xblock);
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

   op[iblock] = opt;
}

// Operator application
void BlockDiagonalPreconditioner::Mult (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   const bool use_dev = x.UseDevice() || y.UseDevice();
   yblock.Update(y, offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   xblock.Update(x, offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);

   for (int i=0; i<nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(yblock);
      xblock.GetBlock(i).SyncAliasMemory(xblock); // TODO: This shouldn't have to be done...
      if (op[i])
      {
         op[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
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

   const bool use_dev = x.UseDevice() || y.UseDevice();
   yblock.Update(y, offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   xblock.Update(x, offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);

   for (int i=0; i<nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(yblock);
      xblock.GetBlock(i).SyncAliasMemory(xblock); // TODO: This shouldn't have to be done
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

   const bool use_dev = x.UseDevice() || y.UseDevice();
   yblock.Update(y,offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   xblock.Update(x,offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);
   tmp.UseDevice(use_dev);
   tmp2.UseDevice(use_dev);

   y = 0.0;
   for (int iRow=0; iRow < nBlocks; ++iRow)
   {
      xblock.GetBlock(iRow).SyncAliasMemory(xblock); // TODO: This shouldn't have to be done...
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=0; jCol < iRow; ++jCol)
      {
         if (op(iRow,jCol))
         {
            yblock.GetBlock(jCol).SyncAliasMemory(yblock);
            op(iRow,jCol)->Mult(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      yblock.GetBlock(iRow).SyncAliasMemory(yblock);
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

   const bool use_dev = x.UseDevice() || y.UseDevice();
   yblock.Update(y,offsets);
   yblock.UseDevice(use_dev);
   yblock.ReadWrite(use_dev);
   xblock.Update(x,offsets);
   xblock.UseDevice(use_dev);
   xblock.Read(use_dev);
   tmp.UseDevice(use_dev);
   tmp2.UseDevice(use_dev);

   y = 0.0;
   for (int iRow=nBlocks-1; iRow >=0; --iRow)
   {
      xblock.GetBlock(iRow).SyncAliasMemory(xblock); // TODO: This shouldn't have to be done...
      tmp.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow+1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);
      for (int jCol=iRow+1; jCol < nBlocks; ++jCol)
      {
         if (op(jCol,iRow))
         {
            yblock.GetBlock(jCol).SyncAliasMemory(yblock);
            op(jCol,iRow)->MultTranspose(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      yblock.GetBlock(iRow).SyncAliasMemory(yblock);
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
