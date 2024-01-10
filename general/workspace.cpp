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

WorkspaceChunk::WorkspaceChunk(int capacity) : data(capacity), offset(0)
{
   mfem::out << "Allocating new WorkspaceChunk.\n";
}

WorkspaceVector WorkspaceChunk::NewVector(int n)
{
   MFEM_ASSERT(HasCapacityFor(n), "Requested vector is too large.");
   WorkspaceVector vector(*this, n);
   offset += n;
   return vector;
}

void Workspace::Consolidate(int requested_size)
{
   int n_empty = 0;
   int empty_capacity = 0;
   // Merge all empty chunks at the beginning of the list
   auto it = chunks.begin();
   while (it != chunks.end() && it->IsEmpty())
   {
      empty_capacity += it->GetAvailableCapacity();
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

   if (requested_size > capacity)
   {
      chunks.emplace_front(std::max(requested_size, empty_capacity));
   }
}

WorkspaceVector Workspace::NewVector(int n)
{
   Consolidate(n);
   return chunks.front().NewVector(n);
}

Workspace::~Workspace()
{
   int nchunks = 0;
   int total_capacity = 0;

   for (auto &chunk : chunks)
   {
      ++nchunks;
      total_capacity += chunk.GetCapacity();
   }

   mfem::out << "Number of chunks currently in workspace: "
             << nchunks << '\n';
   mfem::out << "Total capacity: " << total_capacity << '\n';
}

} // namespace mfem
