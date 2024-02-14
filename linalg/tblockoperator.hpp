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

#ifndef MFEM_TBLOCKOPERATOR
#define MFEM_TBLOCKOPERATOR

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "operator.hpp"
#include "blockvector.hpp"

namespace mfem
{

//! @class TBlockOperator
/**
 * \brief A template to handle Block systems in a matrix-free implementation.
 *
 * Usage:
 * - Use one of the constructors to define the block structure.
 * - Use SetDiagonalBlock or SetBlock to fill the TBlockOperator
 * - Use the method Mult and MultTranspose to apply the operator to a vector.
 *
 * If a block is not set, it is assumed to be a zero block.
 */
template<class T> class TBlockOperator : public Operator
{
public:
   //! Constructor for TBlockOperator%s with the same block-structure for rows and
   //! columns.
   /**
    *  offsets: offsets that mark the start of each row/column block (size
    *  nRowBlocks+1).
    */
   TBlockOperator(const Array<int> & offsets);
   //! Constructor for general TBlockOperator%s.
   /**
    *  row_offsets: offsets that mark the start of each row block (size
    *  nRowBlocks+1).  col_offsets: offsets that mark the start of each column
    *  block (size nColBlocks+1).
    */
   TBlockOperator(const Array<int> & row_offsets,
                  const Array<int> & col_offsets);

   /// Copy assignment is not supported
   TBlockOperator &operator=(const TBlockOperator &) = delete;

   /// Move assignment is not supported
   TBlockOperator &operator=(TBlockOperator &&) = delete;

   //! Add block op in the block-entry (iblock, iblock).
   /**
    * iblock: The block will be inserted in location (iblock, iblock).
    * op: the Operator to be inserted.
    * c: optional scalar multiple for this block.
    */
   void SetDiagonalBlock(int iblock, T *op, double c = 1.0);
   //! Add a block op in the block-entry (iblock, jblock).
   /**
    * irow, icol: The block will be inserted in location (irow, icol).
    * op: the Operator to be inserted.
    * c: optional scalar multiple for this block.
    */
   void SetBlock(int iRow, int iCol, T *op, double c = 1.0);

   //! Return the number of row blocks
   int NumRowBlocks() const { return nRowBlocks; }
   //! Return the number of column blocks
   int NumColBlocks() const { return nColBlocks; }

   //! Check if block (i,j) is a zero block
   int IsZeroBlock(int i, int j) const { return (op(i,j)==NULL) ? 1 : 0; }
   //! Return a reference to block i,j
   T& GetBlock(int i, int j)
   { MFEM_VERIFY(op(i,j), ""); return *op(i,j); }
   //! Return a reference to block i,j
   const T& GetBlock(int i, int j) const
   { MFEM_VERIFY(op(i,j), ""); return *op(i,j); }
   //! Return the coefficient for block i,j
   double GetBlockCoef(int i, int j) const
   { MFEM_VERIFY(op(i,j), ""); return coef(i,j); }
   //! Set the coefficient for block i,j
   void SetBlockCoef(int i, int j, double c)
   { MFEM_VERIFY(op(i,j), ""); coef(i,j) = c; }

   //! Return the row offsets for block starts
   Array<int> & RowOffsets() { return row_offsets; }
   //! Read only access to the row offsets for block starts
   const Array<int> & RowOffsets() const { return row_offsets; }
   //! Return the columns offsets for block starts
   Array<int> & ColOffsets() { return col_offsets; }
   //! Read only access to the columns offsets for block starts
   const Array<int> & ColOffsets() const { return col_offsets; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const;

   virtual ~TBlockOperator();

   //! Controls the ownership of the blocks: if nonzero, TBlockOperator will
   //! delete all blocks that are set (non-NULL); the default value is zero.
   int owns_blocks;

   virtual Type GetType() const { return MFEM_Block_Operator; }

protected:
   //! Number of block rows
   int nRowBlocks;
   //! Number of block columns
   int nColBlocks;
   //! Row offsets for the starting position of each block
   Array<int> row_offsets;
   //! Column offsets for the starting position of each block
   Array<int> col_offsets;
   //! 2D array that stores each block of the operator.
   Array2D<T *> op;
   //! 2D array that stores a coefficient for each block of the operator.
   Array2D<double> coef;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
   mutable BlockVector xblock;
   mutable BlockVector yblock;
   mutable Vector tmp;
};

//-----------------------------------------------------------------------

template<class T>
TBlockOperator<T>::TBlockOperator(const Array<int> & offsets)
   : Operator(offsets.Last()),
     owns_blocks(0),
     nRowBlocks(offsets.Size() - 1),
     nColBlocks(offsets.Size() - 1),
     row_offsets(offsets),
     col_offsets(offsets),
     op(nRowBlocks, nRowBlocks),
     coef(nRowBlocks, nColBlocks)
{
   op = static_cast<T*>(NULL);
}

template<class T>
TBlockOperator<T>::TBlockOperator(const Array<int> & row_offsets_,
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
   op = static_cast<T*>(NULL);
}

template<class T>
void TBlockOperator<T>::SetDiagonalBlock(int iblock, T *opt,
                                         double c)
{
   SetBlock(iblock, iblock, opt, c);
}

template<class T>
void TBlockOperator<T>::SetBlock(int iRow, int iCol, T *opt,
                                 double c)
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
template<class T>
void TBlockOperator<T>::Mult (const Vector & x, Vector & y) const
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
}

// Action of the transpose operator
template<class T>
void TBlockOperator<T>::MultTranspose (const Vector & x, Vector & y) const
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
}

template<class T>
TBlockOperator<T>::~TBlockOperator()
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

}

#endif /* MFEM_TBLOCKOPERATOR */
