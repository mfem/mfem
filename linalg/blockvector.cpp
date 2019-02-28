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
#include "vector.hpp"
#include "blockvector.hpp"

namespace mfem
{

void BlockVector::SetBlocks()
{
   for (int i = 0; i < numBlocks; ++i)
   {
      blocks[i].NewDataAndSize(data+blockOffsets[i],
                               blockOffsets[i+1]-blockOffsets[i]);
   }
}

BlockVector::BlockVector():
   Vector(),
   numBlocks(0),
   blockOffsets(NULL),
   blocks(NULL)
{

}

//! Standard constructor
BlockVector::BlockVector(const Array<int> & bOffsets):
   Vector(bOffsets.Last()),
   numBlocks(bOffsets.Size()-1),
   blockOffsets(bOffsets.GetData())
{
   blocks = new Vector[numBlocks];
   SetBlocks();
}

//! Copy constructor
BlockVector::BlockVector(const BlockVector & v):
   Vector(v),
   numBlocks(v.numBlocks),
   blockOffsets(v.blockOffsets)
{
   blocks = new Vector[numBlocks];
   SetBlocks();
}

//! View constructor
BlockVector::BlockVector(double *data, const Array<int> & bOffsets):
   Vector(data, bOffsets.Last()),
   numBlocks(bOffsets.Size()-1),
   blockOffsets(bOffsets.GetData())
{
   blocks = new Vector[numBlocks];
   SetBlocks();
}

void BlockVector::Update(double *data, const Array<int> & bOffsets)
{
   NewDataAndSize(data, bOffsets.Last());
   blockOffsets = bOffsets.GetData();
   if (numBlocks != bOffsets.Size()-1)
   {
      delete [] blocks;
      numBlocks = bOffsets.Size()-1;
      blocks = new Vector[numBlocks];
   }
   SetBlocks();
}

void BlockVector::Update(const Array<int> &bOffsets)
{
   blockOffsets = bOffsets.GetData();
   if (OwnsData())
   {
      // check if 'bOffsets' agree with the 'blocks'
      if (bOffsets.Size() == numBlocks+1)
      {
         if (numBlocks == 0) { return; }
         if (Size() == bOffsets.Last())
         {
            for (int i = numBlocks - 1; true; i--)
            {
               if (i < 0) { return; }
               if (blocks[i].Size() != bOffsets[i+1] - bOffsets[i]) { break; }
               MFEM_ASSERT(blocks[i].GetData() == data + bOffsets[i],
                           "invalid blocks[" << i << ']');
            }
         }
      }
   }
   else
   {
      Destroy();
   }
   SetSize(bOffsets.Last());
   if (numBlocks != bOffsets.Size()-1)
   {
      delete [] blocks;
      numBlocks = bOffsets.Size()-1;
      blocks = new Vector[numBlocks];
   }
   SetBlocks();
}

BlockVector & BlockVector::operator=(const BlockVector & original)
{
   if (numBlocks!=original.numBlocks)
   {
      mfem_error("Number of Blocks don't match in BlockVector::operator=");
   }

   for (int i(0); i <= numBlocks; ++i)
      if (blockOffsets[i]!=original.blockOffsets[i])
      {
         mfem_error("Size of Blocks don't match in BlockVector::operator=");
      }

   Vector::operator=(original.GetData());

   return *this;
}

BlockVector & BlockVector::operator=(double val)
{
   Vector::operator=(val);
   return *this;
}

//! Destructor
BlockVector::~BlockVector()
{
   delete [] blocks;
}

void BlockVector::GetBlockView(int i, Vector & blockView)
{
   blockView.NewDataAndSize(data+blockOffsets[i],
                            blockOffsets[i+1]-blockOffsets[i]);
}

}
