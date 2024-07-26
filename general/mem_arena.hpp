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

#ifndef MFEM_MEM_ARENA_HPP
#define MFEM_MEM_ARENA_HPP

#include "mem_manager.hpp"
#include "mem_internal.hpp"

#include <forward_list>
#include <vector>

namespace mfem
{

namespace internal
{

/// This is an internal class used by Workspace.
class WorkspaceChunk
{
   /// The data used as a base pointer.
   Memory data;
   /// The offset (in bytes) of the next available memory in this chunk.
   size_t offset = 0;
   /// How many pointers have been allocated in this chunk.
   size_t ptr_count = 0; // <- Can remove this?
   /// Is the chunk in the front of the list?
   bool front = true;
   /// Whether the memory has been freed (prevent double-free).
   bool dealloced = false;

   std::vector<void*> ptr_stack;
public:
   /// Create a WorkspaceChunk with the given @a capacity.
   WorkspaceChunk(size_t capacity);
   /// Destroy a WorkspaceChunk (free the memory)
   ~WorkspaceChunk();
   /// @brief Return the available capacity (i.e. the largest vector that will
   /// fit in this chunk).
   size_t GetAvailableCapacity() const { return data.bytes - offset; }
   /// @brief Returns the original capacity of the chunk.
   ///
   /// If the chunk is not in the front of the list and all of its vectors are
   /// freed, it may deallocate its data, so the capacity becomes zero. The
   /// "original capacity" remains unchained.
   size_t GetCapacity() const { return data.bytes; }
   /// Return the data offset.
   size_t GetOffset() const { return offset; }
   /// Sets whether the chunk is in the front of the list
   void SetFront(bool front_) { front = front_; }
   /// Returns true if this chunk can fit a new vector of size @a n.
   bool HasCapacityFor(size_t n) const { return n <= GetAvailableCapacity(); }
   /// Returns true if this chunk is empty.
   bool IsEmpty() const { return ptr_stack.empty(); }
   /// Clear all deallocated pointers from the top of the stack.
   void ClearDeallocated();
   /// Returns the backing data pointer.
   // void *GetData() { return data; }
   /// Allocates a buffer of size nbytes, returns the associated pointer.
   void *NewPointer(size_t nbytes)
   {
      MFEM_ASSERT(HasCapacityFor(nbytes), "Requested pointer is too large.");
      void *ptr = (char *)data.h_ptr + offset;
      offset += nbytes;
      ptr_count += 1;
      ptr_stack.push_back(ptr);
      return ptr;
   }

   size_t GetPointerCount() const { return ptr_count; }

   void *GetDevicePointer(void *h_ptr);
};

class Workspace
{
   friend class WorkspaceHostMemorySpace;
   friend class WorkspaceDeviceMemorySpace;

   /// Chunks of storage to hold the vectors.
   std::forward_list<internal::WorkspaceChunk> chunks;
   /// Map from pointers to chunks
   std::unordered_map<void*,WorkspaceControlBlock> map;
   /// @brief Consolidate the chunks (merge consecutive empty chunks), and
   /// ensure that the front chunk has sufficient available capacity for a
   /// buffer of @a requested_size.
   void ConsolidateAndEnsureAvailable(size_t requested_size)
   {
      size_t n_empty = 0;
      size_t empty_capacity = 0;
      // Merge all empty chunks at the beginning of the list
      auto it = chunks.begin();
      while (it != chunks.end() && it->IsEmpty())
      {
         empty_capacity += it->GetCapacity();
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

      const size_t min_chunk_size = std::max(requested_size, empty_capacity);
      bool add_new_chunk = false;
      if (chunks.empty())
      {
         add_new_chunk = true;
      }
      else
      {
         add_new_chunk = min_chunk_size > chunks.front().GetAvailableCapacity();
      }

      if (add_new_chunk)
      {
         if (!chunks.empty()) { chunks.front().SetFront(false); }
         chunks.emplace_front(min_chunk_size);
      }
   }
   static Workspace &Instance()
   {
      static Workspace ws;
      return ws;
   }
};

class WorkspaceHostMemorySpace : public HostMemorySpace
{
public:
   void Alloc(void **ptr, size_t nbytes) override
   {
      Workspace &ws = Workspace::Instance();
      ws.ConsolidateAndEnsureAvailable(nbytes);
      internal::WorkspaceChunk &front_chunk = ws.chunks.front();
      void *new_ptr = front_chunk.NewPointer(nbytes);
      maps->workspace.emplace(new_ptr, WorkspaceControlBlock(front_chunk, nbytes));
      *ptr = new_ptr;

      size_t nchunks = std::distance(std::begin(ws.chunks), std::end(ws.chunks));
      mfem::out << "===========================================================\n";
      mfem::out << nchunks << " chunks\n";
      int i = 0;
      for (auto it = ws.chunks.begin(); it != ws.chunks.end(); )
      {
         auto &c = *it;
         mfem::out << "   Chunk " << i << '\n';
         mfem::out << "   Size:      " << c.GetCapacity() << '\n';
         mfem::out << "   Vectors:   " << c.GetPointerCount() << '\n';
         mfem::out << "   Available: " << c.GetAvailableCapacity() << '\n';
         ++it;
         ++i;
         if (it != ws.chunks.end())
         {
            mfem::out << "   --------------------------------------------------\n";
         }
      }
   }

   void Dealloc(Memory &mem) override
   {
      auto it = maps->workspace.find(mem.h_ptr);
      auto &control = it->second;
      control.deallocated = true; // Mark as deallocated
      control.chunk.ClearDeallocated();
   }
};

class WorkspaceDeviceMemorySpace : public DeviceMemorySpace
{
public:
   void Alloc(Memory &base) override;
   void Dealloc(Memory &base) override;
   void *HtoH(void *dst, const void *src, size_t bytes);
   void *HtoD(void *dst, const void *src, size_t bytes) override;
   void *DtoD(void* dst, const void* src, size_t bytes) override;
   void *DtoH(void *dst, const void *src, size_t bytes) override;
};

} // namespace internal

} // namespace mfem

#endif
