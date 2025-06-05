// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
     row_offsets(offsets),
     col_offsets(offsets),
     op(nRowBlocks, nRowBlocks),
     coef(nRowBlocks, nColBlocks)
{
   op = static_cast<Operator *>(NULL);
}

BlockOperator::BlockOperator(const Array<int> & row_offsets_,
                             const Array<int> & col_offsets_)
   : Operator(row_offsets_.Last(), col_offsets_.Last()),
     owns_blocks(0),
     nRowBlocks(row_offsets_.Size()-1),
     nColBlocks(col_offsets_.Size()-1),
     row_offsets(row_offsets_),
     col_offsets(col_offsets_),
     op(nRowBlocks, nColBlocks),
     coef(nRowBlocks, nColBlocks)
{
   op = static_cast<Operator *>(NULL);
}

void BlockOperator::SetDiagonalBlock(int iblock, Operator *opt, real_t c)
{
   SetBlock(iblock, iblock, opt, c);
}

void BlockOperator::SetBlock(int iRow, int iCol, Operator *opt, real_t c)
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
         if (op(iRow,jCol) && coef(iRow,jCol) != 0.)
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
         if (op(jCol,iRow) && coef(jCol,iRow) != 0.)
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
}

BlockOperator::~BlockOperator()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nRowBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nColBlocks; ++jCol)
         {
            delete op(iRow,jCol);
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
   ops(nBlocks)
{
   ops = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockDiagonalPreconditioner::SetDiagonalBlock(int iblock, Operator *op)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");

   if (owns_blocks && ops[iblock])
   {
      delete ops[iblock];
   }
   ops[iblock] = op;
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
      if (ops[i])
      {
         ops[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
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
      if (ops[i])
      {
         (ops[i])->MultTranspose(xblock.GetBlock(i), yblock.GetBlock(i));
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
}

BlockDiagonalPreconditioner::~BlockDiagonalPreconditioner()
{
   if (owns_blocks)
   {
      for (int i=0; i<nBlocks; ++i)
      {
         delete ops[i];
      }
   }
}

BlockLowerTriangularPreconditioner::BlockLowerTriangularPreconditioner(
   const Array<int> & offsets_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     ops(nBlocks, nBlocks)
{
   ops = static_cast<Operator *>(NULL);
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
                                                  Operator *op)
{
   MFEM_VERIFY(iRow >= iCol,"cannot set block in upper triangle");
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == op->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == op->NumCols(),
               "incompatible Operator dimensions");

   ops(iRow, iCol) = op;
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
         if (ops(iRow,jCol))
         {
            ops(iRow,jCol)->Mult(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (ops(iRow,iRow))
      {
         ops(iRow,iRow)->Mult(tmp2, yblock.GetBlock(iRow));
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
         if (ops(jCol,iRow))
         {
            ops(jCol,iRow)->MultTranspose(yblock.GetBlock(jCol), tmp);
            tmp2 -= tmp;
         }
      }
      if (ops(iRow,iRow))
      {
         ops(iRow,iRow)->MultTranspose(tmp2, yblock.GetBlock(iRow));
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
            delete ops(jCol,iRow);
         }
      }
   }
}

BlockTriangularSymmetricPreconditioner::BlockTriangularSymmetricPreconditioner(
   const Array<int> & offsets_)
   : Solver(offsets_.Last()),
     owns_blocks(0),
     nBlocks(offsets_.Size() - 1),
     offsets(0),
     ops(nBlocks, nBlocks),
     coef(nBlocks, nBlocks)
{
   ops = static_cast<Operator *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockTriangularSymmetricPreconditioner::SetDiagonalBlock(int iblock,
                                                              Operator *op, real_t c)
{
   MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == op->Height() &&
               offsets[iblock+1] - offsets[iblock] == op->Width(),
               "incompatible Operator dimensions");

   SetBlock(iblock, iblock, op);
   coef(iblock, iblock) = c;
}

void BlockTriangularSymmetricPreconditioner::SetBlock(int iRow, int iCol,
                                                      Operator *op, real_t c)
{
   MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == op->NumRows() &&
               offsets[iCol+1] - offsets[iCol] == op->NumCols(),
               "incompatible Operator dimensions");

   ops(iRow, iCol) = op;
   coef(iRow, iCol) = c;
}

// Operator application

void BlockTriangularSymmetricPreconditioner::ForwardPass(const Vector & x,
                                                         Vector & y) const
{
   // Forward sweep: Solve for y1, then y2
   for (int iRow = 0; iRow < nBlocks; ++iRow)
   {
      tmp.SetSize(offsets[iRow + 1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow + 1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);  // tmp2 = xblock(iRow)

      // Process the lower triangular part (jCol < iRow)
      for (int jCol = 0; jCol < iRow; ++jCol)
      {
         if (ops(iRow, jCol))
         {
            ops(iRow, jCol)->Mult(yblock.GetBlock(jCol),
                                  tmp);  // tmp = A(iRow,jCol) * yblock(jCol)
            tmp*= coef(iRow,jCol);  // tmp *= c
            tmp2 -= tmp;  // tmp2 -= A(iRow, jCol) * yblock(jCol)
         }
      }

      // Apply the diagonal block
      if (ops(iRow, iRow))
      {
         ops(iRow, iRow)->Mult(tmp2,
                               yblock.GetBlock(iRow));
         yblock.GetBlock(iRow) *= coef(iRow,
                                       iRow);  // yblock(iRow) = A(iRow,iRow)^-1 * tmp2
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;  // If no diagonal operator, set yblock directly
      }
   }
}

void BlockTriangularSymmetricPreconditioner::BackwardPass(const Vector & x,
                                                          Vector & y) const
{
   // Backward sweep: Adjust y1 based on y2
   for (int iRow = nBlocks - 1; iRow >= 0; --iRow)
   {
      tmp.SetSize(offsets[iRow + 1] - offsets[iRow]);
      tmp2.SetSize(offsets[iRow + 1] - offsets[iRow]);
      tmp2 = 0.0;
      tmp2 += xblock.GetBlock(iRow);  // tmp2 = yblock(iRow) from forward sweep

      // Process the upper triangular part (jCol > iRow)
      for (int jCol = iRow + 1; jCol < nBlocks; ++jCol)
      {
         if (ops(iRow, jCol))
         {
            ops(iRow, jCol)->Mult(yblock.GetBlock(jCol),
                                  tmp);  // tmp = A(iRow,jCol) * yblock(jCol)
            tmp *= coef(iRow,jCol);  // tmp *= c
            tmp2 -= tmp;  // tmp2 -= A(iRow,jCol) * yblock(jCol)
         }
      }

      // Reapply diagonal block to correct y1
      if (ops(iRow, iRow))
      {
         ops(iRow, iRow)->Mult(tmp2,
                               yblock.GetBlock(iRow));  // Final correction for yblock(iRow)
         yblock.GetBlock(iRow) *= coef(iRow,
                                       iRow);  // yblock(iRow) = A(iRow,iRow)^-1 * tmp2
      }
      else
      {
         yblock.GetBlock(iRow) = tmp2;  // If no diagonal operator, set yblock directly
      }
   }
}



void BlockTriangularSymmetricPreconditioner::Mult(const Vector & x,
                                                  Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   // Update block views of the vectors y and x using offsets
   yblock.Update(y.GetData(), offsets);
   xblock.Update(x.GetData(), offsets);

   // Initialize y to zero
   y = 0.0;
   ForwardPass(x,y);

   // Update Residual
   r.SetSize(x.Size());
   r = 0.0; r+=x;
   Op->AddMult(y,r,-1.0);

   Vector y1(y);
   yblock.Update(y1.GetData(), offsets);
   xblock.Update(r.GetData(), offsets);
   BackwardPass(r,y1);
   y+=y1;
}

BlockTriangularSymmetricPreconditioner::~BlockTriangularSymmetricPreconditioner()
{
   if (owns_blocks)
   {
      for (int iRow=0; iRow < nBlocks; ++iRow)
      {
         for (int jCol=0; jCol < nBlocks; ++jCol)
         {
            delete ops(jCol,iRow);
         }
      }
   }
}



}
