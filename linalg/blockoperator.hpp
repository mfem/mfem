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

#ifndef MFEM_BLOCKOPERATOR
#define MFEM_BLOCKOPERATOR

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "operator.hpp"
#include "blockvector.hpp"

namespace mfem
{

//! @class BlockOperator
/**
 * \brief A class to handle Block systems in a matrix-free implementation.
 *
 * Usage:
 * - Use one of the constructors to define the block structure.
 * - Use SetDiagonalBlock or SetBlock to fill the BlockOperator
 * - Use the method Mult and MultTranspose to apply the operator to a vector.
 *
 * If a block is not set, it is assumed to be a zero block
 */
class BlockOperator : public Operator
{
public:
   //! Constructor for BlockOperators with the same block-structure for rows and
   //! columns.
   /**
    *  offsets: offsets that mark the start of each row/column block (size
    *  nRowBlocks+1).  Note: BlockOperator will not own/copy the data contained
    *  in offsets.
    */
   BlockOperator(const Array<int> & offsets);
   //! Constructor for general BlockOperators.
   /**
    *  row_offsets: offsets that mark the start of each row block (size
    *  nRowBlocks+1).  col_offsets: offsets that mark the start of each column
    *  block (size nColBlocks+1).  Note: BlockOperator will not own/copy the
    *  data contained in offsets.
    */
   BlockOperator(const Array<int> & row_offsets, const Array<int> & col_offsets);

   //! Add block op in the block-entry (iblock, iblock).
   /**
    * iblock: The block will be inserted in location (iblock, iblock).
    * op: the Operator to be inserted.
    */
   void SetDiagonalBlock(int iblock, Operator *op);
   //! Add a block op in the block-entry (iblock, jblock).
   /**
    * irow, icol: The block will be inserted in location (irow, icol).
    * op: the Operator to be inserted.
    */
   void SetBlock(int iRow, int iCol, Operator *op);

   //! Return the number of row blocks
   int NumRowBlocks() const { return nRowBlocks; }
   //! Return the number of column blocks
   int NumColBlocks() const { return nColBlocks; }

   //! Check if block (i,j) is a zero block
   int IsZeroBlock(int i, int j) const { return (op(i,j)==NULL) ? 1 : 0; }
   //! Return a reference to block i,j
   Operator & GetBlock(int i, int j)
   { MFEM_VERIFY(op(i,j), ""); return *op(i,j); }

   //! Return the row offsets for block starts
   Array<int> & RowOffsets() { return row_offsets; }
   //! Return the columns offsets for block starts
   Array<int> & ColOffsets() { return col_offsets; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const;

   ~BlockOperator();

   //! Controls the ownership of the blocks: if nonzero, BlockOperator will
   //! delete all blocks that are set (non-NULL); the default value is zero.
   int owns_blocks;

private:
   //! Number of block rows
   int nRowBlocks;
   //! Number of block columns
   int nColBlocks;
   //! Row offsets for the starting position of each block
   Array<int> row_offsets;
   //! Column offsets for the starting position of each block
   Array<int> col_offsets;
   //! 2D array that stores each block of the operator.
   Array2D<Operator *> op;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
   mutable BlockVector xblock;
   mutable BlockVector yblock;
   mutable Vector tmp;
};

//! @class BlockDiagonalPreconditioner
/**
 * \brief A class to handle Block diagonal preconditioners in a matrix-free implementation.
 *
 * Usage:
 * - Use the constructors to define the block structure
 * - Use SetDiagonalBlock to fill the BlockOperator
 * - Use the method Mult and MultTranspose to apply the operator to a vector.
 *
 * If a block is not set, it is assumed it is an identity block
 *
 */
class BlockDiagonalPreconditioner : public Solver
{
public:
   //! Constructor that specifies the block structure
   BlockDiagonalPreconditioner(const Array<int> & offsets);
   //! Add a square block op in the block-entry (iblock, iblock).
   /**
    * iblock: The block will be inserted in location (iblock, iblock).
    * op: the Operator to be inserted.
    */
   void SetDiagonalBlock(int iblock, Operator *op);
   //! This method is present since required by the abstract base class Solver
   virtual void SetOperator(const Operator &op) { }

   //! Return the number of blocks
   int NumBlocks() const { return nBlocks; }

   //! Return a reference to block i,i.
   Operator & GetDiagonalBlock(int iblock)
   { MFEM_VERIFY(op[iblock], ""); return *op[iblock]; }

   //! Return the offsets for block starts
   Array<int> & Offsets() { return offsets; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const;

   ~BlockDiagonalPreconditioner();

   //! Controls the ownership of the blocks: if nonzero,
   //! BlockDiagonalPreconditioner will delete all blocks that are set
   //! (non-NULL); the default value is zero.
   int owns_blocks;

private:
   //! Number of Blocks
   int nBlocks;
   //! Offsets for the starting position of each block
   Array<int> offsets;
   //! 1D array that stores each block of the operator.
   Array<Operator *> op;
   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable BlockVector xblock;
   mutable BlockVector yblock;
};

}

#endif /* MFEM_BLOCKOPERATOR */
