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

#ifndef MFEM_WORKSPACE_HPP
#define MFEM_WORKSPACE_HPP

#include "../linalg/vector.hpp"
#include <forward_list>

namespace mfem
{

class WorkspaceVector : public Vector
{
   class WorkspaceChunk &chunk;
   int size_in_chunk;
public:
   /// Create a new
   WorkspaceVector(WorkspaceChunk &chunk_, int n);
   // Cannot copy from one WorkspaceVector to another
   WorkspaceVector(WorkspaceVector &other) = delete;
   WorkspaceVector& operator=(WorkspaceVector &other) = delete;
   // Can move WorkspaceVector%s, just need to make sure that the moved-from
   // WorkspaceVector has zero size, so that when it is destroyed, no capacity
   // is released to the chunk.
   WorkspaceVector(WorkspaceVector &&other);
   // No move assignment operator
   WorkspaceVector& operator=(WorkspaceVector &&other) = delete;
   // All other operator=, inherit from Vector
   using Vector::operator=;
   ~WorkspaceVector();
};

class WorkspaceChunk
{
   Vector data;
   int offset;

public:
   WorkspaceChunk(int capacity) : data(capacity), offset(0)
   {
      mfem::out << "Allocating new WorkspaceChunk.\n";
   }

   int GetCapacity() const { return data.Size(); }

   int GetOffset() const { return offset; }

   void FreeCapacity(int n)
   {
      MFEM_ASSERT(offset >= n, "");
      offset -= n;
   }

   Vector &GetData() { return data; }

   bool HasCapacityFor(int n) const { return offset + n <= data.Size(); }

   WorkspaceVector NewVector(int n);
};

class Workspace
{
   int current_capacity = 0;
   int total_requested_capacity = 0;
   std::forward_list<WorkspaceChunk> chunks;

   Workspace() = default;
   void ClearEmptyChunks();
   WorkspaceChunk &NewChunk(int requested_size);
public:
   static Workspace &Instance()
   {
      static Workspace ws;
      return ws;
   }
   WorkspaceVector NewVector(int n);
   void Clear() { chunks.clear(); }
};

} // namespace mfem

#endif
