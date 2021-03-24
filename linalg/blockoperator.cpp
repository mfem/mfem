// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

SchurComplimentOperator::SchurComplimentOperator(Solver & _AInv, Operator & _B,
                                                 Operator & _C, Operator & _D)
   : Operator(),
     A(NULL), B(&_B), C(&_C), D(&_D), AInv(&_AInv), DInv(NULL),
     sizeA(AInv->Height()), sizeD(D->Height())
{
   height = sizeD;
   width  = height;

   rhs.SetSize(sizeD);

   y2.SetSize(sizeD);
   x1.SetSize(sizeA);
   rhs1.SetSize(sizeA);
}

SchurComplimentOperator::SchurComplimentOperator(Operator & _A, Operator & _B,
                                                 Operator & _C, Solver & _DInv)
   : A(&_A), B(&_B), C(&_C), D(NULL), AInv(NULL), DInv(&_DInv),
     sizeA(A->Height()), sizeD(DInv->Height())
{
   height = sizeA;
   width  = height;

   rhs.SetSize(sizeA);

   y1.SetSize(sizeA);
   x2.SetSize(sizeD);
   rhs2.SetSize(sizeD);
}

const Vector & SchurComplimentOperator::GetRHSVector(const Vector & a,
                                                     const Vector & b)
{
   if (DInv)
   {
      DInv->Mult(b, x2);
      B->Mult(x2, rhs);
      rhs *= -1.0;
      rhs.Add(1.0, a);
   }
   else
   {
      AInv->Mult(a, x1);
      C->Mult(x1, rhs);
      rhs *= -1.0;
      rhs.Add(1.0, b);
   }

   return rhs;
}

void SchurComplimentOperator::Mult(const Vector & x, Vector & y) const
{
   if (DInv)
   {
      A->Mult(x, y);

      C->Mult(x, rhs2);
      DInv->Mult(rhs2, x2);
      B->Mult(x2, y1);

      y.Add(-1.0, y1);
   }
   else
   {
      D->Mult(x, y);

      B->Mult(x, rhs1);
      AInv->Mult(rhs1, x1);
      C->Mult(x1, y2);

      y.Add(-1.0, y2);
   }
}

void SchurComplimentOperator::Solve(const Vector & b, const Vector & x,
                                    Vector & y)
{
   if (DInv)
   {
      C->Mult(x, rhs2);
      rhs2 *= -1.0;
      rhs2.Add(1.0, b);
      DInv->Mult(rhs2, y);
   }
   else
   {
      B->Mult(x, rhs1);
      rhs1 *= -1.0;
      rhs1.Add(1.0, b);
      AInv->Mult(rhs1, y);
   }
}

BlockDiagonalMultiplicativePreconditioner::BlockDiagonalMultiplicativePreconditioner(
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

void BlockDiagonalMultiplicativePreconditioner::SetDiagonalBlock(int iblock, Operator *opt)
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
void BlockDiagonalMultiplicativePreconditioner::Mult (const Vector & x, Vector & y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),offsets);
   yblock.Update(y,offsets);

   Vector r(x);
   Vector z(x.Size()); z = 0.0;

   // BlockVector rblock(offsets); rblock = 0.0;
   // BlockVector zblock(offsets); zblock = 0.0;
   // rblock += r;
   // zblock += z;

   // BlockVector zaux(offsets);
   int n1 = offsets[1];
   int n2 = offsets[2]- offsets[1];
   Array<int> map1(n1), map2(n2);
   for (int i = 0; i<n1; i++) map1[i] = i;
   for (int i = 0; i<n2; i++) map2[i] = n1+i;

   Vector r1(n1), z1(n1);
   Vector r2(n2), z2(n2);
   // ------------------------------
   r.GetSubVector(map1,r1);
   op[0]->Mult(r1, z1);
   Vector zaux(z.Size()); zaux = 0.0;
   zaux.SetSubVector(map1,z1);
   z+=zaux;
   // r = r - Az
   Vector raux(z.Size());
   oper->Mult(zaux,raux);
   r -= raux;
   r.GetSubVector(map2,r2);
   op[1]->Mult(r2, z2);
   zaux = 0.0;
   zaux.AddElementVector(map2,z2);
   z+=zaux;
   // ------------------------------
   oper->Mult(zaux,raux);
   r -= raux;
   r.GetSubVector(map2,r2);
   op[1]->Mult(r2, z2);
   zaux = 0.0;
   zaux.AddElementVector(map2,z2);
   z +=zaux;
   oper->Mult(zaux,raux);
   r -= raux;
   r.GetSubVector(map1,r1);
   op[0]->Mult(r1, z1);
   zaux = 0.0;
   zaux.AddElementVector(map1,z1);
   z+=zaux;

   y = z;


   // for (int i=0; i<nBlocks; ++i)
   // {
   //    op[i]->Mult(rblock.GetBlock(i), zblock.GetBlock(i));

   //    // update residual
   //    // r = r - Az
   //    oper->Mult(zblock,zaux);
   //    rblock -= zaux;
   //    yblock.GetBlock(i) += zblock.GetBlock(i);
   // }
   // for (int i=nBlocks-1; i>=0; i--)
   // {
   //       op[i]->Mult(rblock.GetBlock(i), zblock.GetBlock(i));
   // //    // update residual
   // //    // r = r - Az
   //    yblock.GetBlock(i) += zblock.GetBlock(i);
   //    if (i>0)
   //    {
   //       oper->Mult(zblock,zaux);
   //       rblock -= zaux;
   //    }
   // }

   // for (int i=0; i<nBlocks; ++i)
   // {
   //    yblock.GetBlock(i).SyncAliasMemory(y);
   // }

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
   for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
}

// Action of the transpose operator
void BlockDiagonalMultiplicativePreconditioner::MultTranspose (const Vector & x,
                                                 Vector & y) const
{
    MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   std::cout << "Mult Transpose" << std::endl; std::cin.get();

   x.Read();
   y.Write(); y = 0.0;

   xblock.Update(const_cast<Vector&>(x),offsets);
   yblock.Update(y,offsets);

   for (int i=0; i<nBlocks; ++i)
   {
      if (op[i])
      {
         op[i]->MultTranspose(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }

      yblock.GetBlock(i).SyncAliasMemory(y);
      // update residual
      Vector xaux(y);
      Vector yaux(y.Size());
      oper->Mult(xaux,yaux);
      y-=yaux;
   }
   for (int i=nBlocks-1; i>=0; i--)
   {
      if (op[i])
      {
         op[i]->MultTranspose(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }

      yblock.GetBlock(i).SyncAliasMemory(y);
      // update residual
      Vector xaux(y);
      Vector yaux(y.Size());
      oper->Mult(xaux,yaux);
      y-=yaux;
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

BlockDiagonalMultiplicativePreconditioner::~BlockDiagonalMultiplicativePreconditioner()
{
   if (owns_blocks)
   {
      for (int i=0; i<nBlocks; ++i)
      {
         delete op[i];
      }
   }
}

// // Operator application
// void BlockDiagonalMultiplicativePreconditioner::Mult(const Vector & x, Vector & y) const
// {
//    MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
//    MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

//    x.Read();
//    y.Write(); y = 0.0;

//    xblock.Update(const_cast<Vector&>(x),offsets);
//    yblock.Update(y,offsets);

//    for (int i=0; i<nBlocks; ++i)
//    {
//       if (op[i])
//       {
//          op[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
//       }
//       else
//       {
//          yblock.GetBlock(i) = xblock.GetBlock(i);
//       }

//       yblock.GetBlock(i).SyncAliasMemory(y);
//       // update residual
//       Vector xaux(y);
//       Vector yaux(y.Size());
//       oper->Mult(xaux,yaux);
//       y-=yaux;
//    }
//    for (int i=nBlocks-1; i>=0; i--)
//    {
//       if (op[i])
//       {
//          op[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
//       }
//       else
//       {
//          yblock.GetBlock(i) = xblock.GetBlock(i);
//       }

//       yblock.GetBlock(i).SyncAliasMemory(y);
//       // update residual
//       Vector xaux(y);
//       Vector yaux(y.Size());
//       oper->Mult(xaux,yaux);
//       y-=yaux;
//    }
//    for (int i=0; i<nBlocks; ++i)
//    {
//       yblock.GetBlock(i).SyncAliasMemory(y);
//    }

//    // Destroy alias vectors to prevent dangling aliases when the base vectors
//    // are deleted
//    for (int i=0; i < xblock.NumBlocks(); ++i) { xblock.GetBlock(i).Destroy(); }
//    for (int i=0; i < yblock.NumBlocks(); ++i) { yblock.GetBlock(i).Destroy(); }
// }
}
