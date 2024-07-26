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

#include "mem_arena.hpp"

namespace mfem
{

namespace internal
{

struct ArenaControlBlock
{
   class ArenaChunk &chunk;
   size_t bytes;
   bool deallocated;
   ArenaControlBlock(class ArenaChunk &chunk_, size_t bytes_)
      : chunk(chunk_), bytes(bytes_), deallocated(false)
   { }
};

std::unordered_map<void*,ArenaControlBlock> arena_map;

Memory NewMemory(size_t nbytes)
{
   MFEM_ASSERT(ctrl != nullptr, "");
   const MemoryType h_mt = MemoryManager::GetHostMemoryType();
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   void *h_ptr;
   ctrl->Host(h_mt)->Alloc(&h_ptr, nbytes);
   return Memory(h_ptr, nbytes, h_mt, d_mt);
}

ArenaChunk::ArenaChunk(size_t capacity)
   : data(NewMemory(capacity))
{
   ptr_stack.reserve(16); // <- Can adjust this
}

ArenaChunk::~ArenaChunk()
{
   if (!dealloced)
   {
      if (data.d_ptr)
      {
         ctrl->Device(data.d_mt)->Dealloc(data);
      }
      ctrl->Host(data.h_mt)->Dealloc(data);
   }
}

void ArenaChunk::ClearDeallocated()
{
   ptr_count -= 1;
   if (ptr_count == 0)
   {
      offset = 0;
      for (void *ptr : ptr_stack) { arena_map.erase(ptr); }
      ptr_stack.clear();
      // If we are not the front chunk, deallocate the backing memory. This
      // chunk will be consolidated later anyway.
      if (!front)
      {
         // data.Delete();
         if (data.d_ptr)
         {
            ctrl->Device(data.d_mt)->Dealloc(data);
         }
         ctrl->Host(data.h_mt)->Dealloc(data);
         dealloced = true;
      }
      return;
   }

   auto end = ptr_stack.end();
   auto ptr_it = ptr_stack.rbegin();
   for (; ptr_it != ptr_stack.rend(); ++ptr_it)
   {
      auto it = arena_map.find(*ptr_it);
      MFEM_ASSERT(it != arena_map.end(), "");
      auto &control = it->second;
      if (!control.deallocated) { break; }
   }
   auto begin = ptr_it.base();
   if (begin != end)
   {
      std::cout << "Reclaiming " << offset - ((char*)*begin - (char*)data.h_ptr)
                << " bytes from " << (end - begin) << " pointers.\n";
      offset = (char*)*begin - (char*)data.h_ptr;
   }

   for (auto it = begin; it != end; ++it)
   {
      arena_map.erase(*it);
   }
   ptr_stack.erase(begin, end);

   MFEM_ASSERT(ptr_stack.size() >= ptr_count, "");
}

void *ArenaChunk::NewPointer(size_t nbytes)
{
   MFEM_ASSERT(HasCapacityFor(nbytes), "Requested pointer is too large.");
   void *ptr = (char *)data.h_ptr + offset;
   offset += nbytes;
   ptr_count += 1;
   ptr_stack.push_back(ptr);
   return ptr;
}

void *ArenaChunk::GetDevicePointer(void *h_ptr)
{
   if (data.d_ptr == nullptr)
   {
      if (IsDeviceMemory(data.d_mt))
      {
         ctrl->Device(data.d_mt)->Alloc(data);
      }
      else
      {
         return h_ptr;
      }
   }
   const size_t ptr_offset = ((char*)h_ptr) - ((char*)data.h_ptr);
   return ((char*)data.d_ptr) + ptr_offset;
}

void ArenaHostMemorySpace::ConsolidateAndEnsureAvailable(
   size_t requested_size)
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

void ArenaHostMemorySpace::Alloc(void **ptr, size_t nbytes)
{
   // Round up requested size to multiple of alignment.
   const size_t r = nbytes % alignment;
   const size_t nbytes_aligned = r ? (nbytes + alignment - r) : nbytes;

   ConsolidateAndEnsureAvailable(nbytes_aligned);
   *ptr = chunks.front().NewPointer(nbytes_aligned);
   arena_map.emplace(*ptr, ArenaControlBlock(chunks.front(), nbytes_aligned));

   // Debug output:
   size_t nchunks = std::distance(std::begin(chunks), std::end(chunks));
   mfem::out << "===========================================================\n";
   mfem::out << nchunks << " chunks\n";
   int i = 0;
   for (auto it = chunks.begin(); it != chunks.end(); )
   {
      auto &c = *it;
      mfem::out << "   Chunk " << i << '\n';
      mfem::out << "   Size:      " << c.GetCapacity() << '\n';
      mfem::out << "   Vectors:   " << c.GetPointerCount() << '\n';
      mfem::out << "   Available: " << c.GetAvailableCapacity() << '\n';
      ++it;
      ++i;
      if (it != chunks.end())
      {
         mfem::out << "   --------------------------------------------------\n";
      }
   }
}

void ArenaHostMemorySpace::Dealloc(Memory &mem)
{
   auto it = arena_map.find(mem.h_ptr);
   auto &control = it->second;
   control.deallocated = true; // Mark as deallocated
   control.chunk.ClearDeallocated();
}

void ArenaDeviceMemorySpace::Alloc(Memory &base)
{
   auto &chunk = arena_map.find(base.h_ptr)->second.chunk;
   base.d_ptr = chunk.GetDevicePointer(base.h_ptr);
}

void ArenaDeviceMemorySpace::Dealloc(Memory &base) { /* no-op */ }

void *ArenaDeviceMemorySpace::HtoD(void *dst, const void *src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   return ctrl->Device(d_mt)->HtoD(dst, src, bytes);
}

void *ArenaDeviceMemorySpace::DtoD(void* dst, const void* src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   return ctrl->Device(d_mt)->DtoD(dst, src, bytes);
}

void *ArenaDeviceMemorySpace::DtoH(void *dst, const void *src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   return ctrl->Device(d_mt)->DtoH(dst, src, bytes);
}

} // namespace internal

} // namespace mfem
