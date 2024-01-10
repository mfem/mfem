// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "workspace.hpp"

namespace mfem
{

WorkspaceVector::WorkspaceVector(WorkspaceChunk &chunk_, int n)
   : Vector(chunk_.GetData(), chunk_.GetOffset(), n),
     chunk(chunk_),
     size_in_chunk(n)
{
   UseDevice(true);
}

WorkspaceVector::WorkspaceVector(WorkspaceVector &&other)
   : Vector(std::move( other)), chunk(other.chunk)
{
   mfem::out << "WorkspaceVector move ctor\n";
   other.size_in_chunk = 0;
}

WorkspaceVector::~WorkspaceVector()
{
   chunk.FreeCapacity(size_in_chunk);
}

WorkspaceVector WorkspaceChunk::NewVector(int n)
{
   MFEM_ASSERT(HasCapacityFor(n), "Requested vector is too large.");
   WorkspaceVector vector(*this, n);
   offset += n;
   return vector;
}

void Workspace::ClearEmptyChunks()
{
   while (!chunks.empty())
   {
      WorkspaceChunk &front_chunk = chunks.front();
      if (front_chunk.GetOffset() == 0)
      {
         current_capacity -= front_chunk.GetCapacity();
         chunks.pop_front();
      }
      else
      {
         break;
      }
   }
}

WorkspaceChunk &Workspace::NewChunk(int requested_size)
{
   const int new_chunk_size = [&]()
   {
      if (total_requested_capacity - current_capacity >= requested_size)
      {
         return total_requested_capacity - current_capacity;
      }
      else
      {
         return requested_size;
      }
   }();
   current_capacity += new_chunk_size;
   total_requested_capacity = std::max(total_requested_capacity, current_capacity);
   chunks.emplace_front(new_chunk_size);
   return chunks.front();
}

WorkspaceVector Workspace::NewVector(int n)
{
   // If the front chunk has capacity, get the new vector from there
   if (!chunks.empty())
   {
      WorkspaceChunk &chunk = chunks.front();
      if (chunk.HasCapacityFor(n))
      {
         return chunk.NewVector(n);
      }
   }
   // Otherwise, we need to add a new chunk (first getting rid of any empty
   // chunks that are too small for the requested size)
   ClearEmptyChunks();
   return NewChunk(n).NewVector(n);
}

} // namespace mfem
