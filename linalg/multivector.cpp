// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "multivector.hpp"

namespace mfem
{

MultiVector::MultiVector(const Array<int> &vector_sizes)
{
   SetSizes(vector_sizes);
}

MultiVector::MultiVector(const Array<int> &vector_sizes, MemoryType mt)
{
   SetSizes(vector_sizes, mt);
}

MultiVector::MultiVector(Vector &base, const Array<int> &vector_sizes)
{
   MakeRef(base, vector_sizes);
}

void MultiVector::SetSizes(const Array<int> &vector_sizes)
{
   blocks.resize(vector_sizes.Size());
   for (int i = 0; i < vector_sizes.Size(); i++)
   {
      operator[](i).SetSize(vector_sizes[i]);
   }
}

void MultiVector::SetSizes(const Array<int> &vector_sizes, MemoryType mt)
{
   blocks.resize(vector_sizes.Size());
   for (int i = 0; i < vector_sizes.Size(); i++)
   {
      operator[](i).SetSize(vector_sizes[i], mt);
   }
}

Array<int> MultiVector::GetSizes() const
{
   int num_blocks = NumBlocks();
   Array<int> sizes(num_blocks);
   for (int i = 0; i < num_blocks; i++)
   {
      sizes[i] = (*this)[i].Size();
   }
   return sizes;
}

void MultiVector::MakeRef(Vector &base, const Array<int> &vector_sizes)
{
   blocks.resize(vector_sizes.Size());
   for (int offset = 0, i = 0; i < vector_sizes.Size(); i++)
   {
      blocks[i].emplace<0>(base, offset, vector_sizes[i]);
      offset += vector_sizes[i];
   }
}

MultiVector MultiVector::GetBlockViews(const Array<int> &blk_idx)
{
   if (blk_idx.Size() == 0) { return MultiVector(); }
   MFEM_VERIFY(blk_idx.Max() < NumBlocks() && blk_idx.Min() >= 0,
               "Block index out of bounds.");
   MultiVector v(blk_idx.Size());
   for (int i=0; i<blk_idx.Size(); i++)
   {
      v.MakeRef(i, (*this)[blk_idx[i]]);
   }
   return v;
}

MultiVector MultiVector::GetBlockViews(const Array<int> &blk_idx) const
{
   if (blk_idx.Size() == 0) { return MultiVector(); }
   MFEM_VERIFY(blk_idx.Max() < NumBlocks() && blk_idx.Min() >= 0,
               "Block index out of bounds.");
   MultiVector v(blk_idx.Size());
   for (int i=0; i<blk_idx.Size(); i++)
   {
      v.MakeRef(i, (*this)[blk_idx[i]]);
   }
   return v;
}

} // namespace mfem
