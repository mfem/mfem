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
#include "vector.hpp"
#include "blockvector.hpp"

namespace mfem
{

void BlockVector::SetBlocks()
{
   for (int i = 0; i < numBlocks; ++i)
   {
      blocks[i].NewMemoryAndSize(
         Memory<double>(data, blockOffsets[i], BlockSize(i)),
         BlockSize(i), true);
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

BlockVector::BlockVector(const Array<int> & bOffsets, MemoryType mt)
   : Vector(bOffsets.Last(), mt),
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
   Update(bOffsets, data.GetMemoryType());
}

void BlockVector::Update(const Array<int> &bOffsets, MemoryType mt)
{
   blockOffsets = bOffsets.GetData();
   if (OwnsData() && data.GetMemoryType() == mt)
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
   SetSize(bOffsets.Last(), mt);
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
   {
      if (blockOffsets[i]!=original.blockOffsets[i])
      {
         mfem_error("Size of Blocks don't match in BlockVector::operator=");
      }
   }

   Vector::operator=(original);

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
   blockView.NewMemoryAndSize(
      Memory<double>(data, blockOffsets[i], BlockSize(i)),
      BlockSize(i), true);
}

}
