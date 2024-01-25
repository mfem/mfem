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

WorkspaceVector::WorkspaceVector(internal::WorkspaceChunk &chunk_, int n)
   : Vector(chunk_.GetData(), chunk_.GetOffset(), n),
     chunk(chunk_)
{
   UseDevice(true);
}

WorkspaceVector::WorkspaceVector(WorkspaceVector &&other)
   : Vector(std::move(other)), chunk(other.chunk)
{
   other.moved_from = true;
}

WorkspaceVector::~WorkspaceVector()
{
   if (!moved_from) { chunk.FreeVector(); }
}

namespace internal
{

WorkspaceChunk::WorkspaceChunk(int capacity)
   : data(capacity), original_capacity(capacity)
{ }

WorkspaceVector WorkspaceChunk::NewVector(int n)
{
   MFEM_ASSERT(HasCapacityFor(n), "Requested vector is too large.");
   WorkspaceVector vector(*this, n);
   offset += n;
   vector_count += 1;
   return vector;
}

void WorkspaceChunk::FreeVector()
{
   MFEM_ASSERT(vector_count >= 0, "");
   vector_count -= 1;
   // If the chunk is completely empty, we can reclaim all of the memory and
   // allow new allocations (before it is completely empty, we cannot reclaim
   // memory because we don't track the specific regions that are freed).
   if (vector_count == 0)
   {
      offset = 0;
      // If we are not the front chunk, deallocate the backing memory. This
      // chunk will be consolidated later anyway.
      if (!front) { data.Destroy(); }
   }
}

} // namespace internal

Workspace &Workspace::Instance()
{
   static Workspace ws;
   return ws;
}

void Workspace::ConsolidateAndEnsureAvailable(int requested_size)
{
   int n_empty = 0;
   int empty_capacity = 0;
   // Merge all empty chunks at the beginning of the list
   auto it = chunks.begin();
   while (it != chunks.end() && it->IsEmpty())
   {
      empty_capacity += it->GetOriginalCapacity();
      ++it;
      ++n_empty;
   }

   // If we have multiple empty chunks at the beginning of the list, we need
   // to merge them. Also, if the front chunk is empty, but not big enough,
   // we need to replace it, so we remove it here.
   if (n_empty > 1 || requested_size > empty_capacity)
   {
      chunks.erase_after(chunks.before_begin(), it);
   }

   const int capacity = chunks.empty() ? -1 :
                        chunks.front().GetAvailableCapacity();

   const int min_chunk_size = std::max(requested_size, empty_capacity);

   if (min_chunk_size > capacity)
   {
      if (!chunks.empty()) { chunks.front().SetFront(false); }
      chunks.emplace_front(min_chunk_size);
   }
}

WorkspaceVector Workspace::NewVector(int n)
{
   Workspace &ws = Instance();
   ws.ConsolidateAndEnsureAvailable(n);
   return ws.chunks.front().NewVector(n);
}

void Workspace::Reserve(int n)
{
   Instance().ConsolidateAndEnsureAvailable(n);
}

void Workspace::Clear()
{
   Instance().chunks.clear();
}

} // namespace mfem
