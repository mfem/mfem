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

BlockVector::BlockVector():
   Vector(),
   numBlocks(0),
   blockOffsets(NULL),
   tmp_block(0)
{

}

//! Standard constructor
BlockVector::BlockVector(const Array<int> & bOffsets):
   Vector(bOffsets.Last()),
   numBlocks(bOffsets.Size()-1),
   blockOffsets(bOffsets.GetData()),
   tmp_block(numBlocks)
{
   for (int i = 0; i < numBlocks; ++i)
   {
      tmp_block[i] =  new Vector((double *)data + blockOffsets[i],
                                 blockOffsets[i+1] - blockOffsets[i]);
   }
}

//! Copy constructor
BlockVector::BlockVector(const BlockVector & v):
   Vector(v),
   numBlocks(v.numBlocks),
   blockOffsets(v.blockOffsets),
   tmp_block(numBlocks)
{
   for (int i = 0; i < numBlocks; ++i)
   {
      tmp_block[i] =  new Vector((double *)data + blockOffsets[i],
                                 blockOffsets[i+1] - blockOffsets[i]);
   }
}

//! View constructor
BlockVector::BlockVector(double *data, const Array<int> & bOffsets):
   Vector(data, bOffsets.Last()),
   numBlocks(bOffsets.Size()-1),
   blockOffsets(bOffsets.GetData()),
   tmp_block(numBlocks)
{
   for (int i = 0; i < numBlocks; ++i)
   {
      tmp_block[i] =  new Vector(data + blockOffsets[i],
                                 blockOffsets[i+1] - blockOffsets[i]);
   }
}

void BlockVector::Update(double *data, const Array<int> & bOffsets)
{
   NewDataAndSize(data, bOffsets.Last());
   blockOffsets = bOffsets.GetData();
   numBlocks = bOffsets.Size()-1;

   int oldNumBlocks = tmp_block.Size();
   for (int i = numBlocks; i < oldNumBlocks; ++i)
   {
      delete tmp_block[i];
   }

   tmp_block.SetSize(numBlocks);
   for (int i = oldNumBlocks; i < numBlocks; ++i)
   {
      tmp_block[i] =  new Vector(data + blockOffsets[i],
                                 blockOffsets[i+1] - blockOffsets[i]);
   }
}

void BlockVector::Update(const double *data, const Array<int> & bOffsets)
{
   double *d = const_cast<double *>(data);
   Update(d, bOffsets);
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

   original.data.Copy(data);

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
   for (int i = 0; i < tmp_block.Size(); ++i)
   {
      delete tmp_block[i];
   }
}

Vector & BlockVector::GetBlock(int i)
{
   tmp_block[i]->NewDataAndSize((double *)data + blockOffsets[i],
                                blockOffsets[i+1] - blockOffsets[i]);
   return *(tmp_block[i]);
}

const Vector &  BlockVector::GetBlock(int i) const
{
   tmp_block[i]->NewDataAndSize((const double *)data + blockOffsets[i],
                                blockOffsets[i+1] - blockOffsets[i]);
   return *(tmp_block[i]);
}

void BlockVector::GetBlockView(int i, Vector & blockView)
{
   blockView.NewDataAndSize((double *)data + blockOffsets[i],
                            blockOffsets[i+1] - blockOffsets[i]);
}

}
