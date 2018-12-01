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

#ifndef MFEM_BLOCKVECTOR
#define MFEM_BLOCKVECTOR

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "vector.hpp"

namespace mfem
{

//! @class BlockVector
/**
 * \brief A class to handle Vectors in a block fashion
 *
 * All data is contained in Vector::data, while blockVector is just a viewer for
 * this data.
 *
 */
class BlockVector: public Vector
{
protected:

   //! Number of blocks in the blockVector
   int numBlocks;
   //! Offset for each block start. (length numBlocks+1)
   /**
    * blockOffsets[i+1] - blockOffsets[i] is the size of block i.
    *
    * This array is not owned.
    */
   const int *blockOffsets;
   //! array of Vector objects used to extract blocks without allocating memory.
   /** This array is owned. */
   Vector *blocks;

   void SetBlocks();

public:
   //! empty constructor
   BlockVector();

   //! Constructor
   /**
    * bOffsets is an array of integers (length nBlocks+1) that tells the offsets
    * of each block start.
    */
   BlockVector(const Array<int> & bOffsets);

   //! Copy constructor
   BlockVector(const BlockVector & block);

   //! View constructor
   /*
    * data is an array of double of length at least blockOffsets[numBlocks] that
    * contain all the values of the monolithic vector.  bOffsets is an array of
    * integers (length nBlocks+1) that tells the offsets of each block start.
    * nBlocks is the number of blocks.
    */
   BlockVector(double *data, const Array<int> & bOffsets);

   //! Assignment operator. this and original must have the same block structure.
   BlockVector & operator=(const BlockVector & original);
   //! Set each entry of this equal to val
   BlockVector & operator=(double val);

   //! Destructor
   ~BlockVector();

   //! Get the i-th vector in the block.
   Vector & GetBlock(int i) { return blocks[i]; }
   //! Get the i-th vector in the block (const version).
   const Vector & GetBlock(int i) const { return blocks[i]; }

   //! Get the i-th vector in the block
   void GetBlockView(int i, Vector & blockView);

   int BlockSize(int i) { return blockOffsets[i+1] - blockOffsets[i];}

   //! Update method
   /**
    * data is an array of double of length at least blockOffsets[numBlocks] that
    * contain all the values of the monolithic vector.  bOffsets is an array of
    * integers (length nBlocks+1) that tells the offsets of each block start.
    * nBlocks is the number of blocks.
    */
   void Update(double *data, const Array<int> & bOffsets);

   /// Update a BlockVector with new @a bOffsets and make sure it owns its data.
   /** The block-vector will be re-allocated if either:
       - the offsets @a bOffsets are different from the current offsets, or
       - currently, the block-vector does not own its data. */
   void Update(const Array<int> &bOffsets);
};

}

#endif /* MFEM_BLOCKVECTOR */
