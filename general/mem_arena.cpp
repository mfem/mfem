// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   ArenaChunk &chunk;
   size_t bytes;
   bool deallocated;
   ArenaControlBlock(ArenaChunk &chunk_, size_t bytes_)
      : chunk(chunk_), bytes(bytes_), deallocated(false)
   { }
};

static std::unordered_map<void*,ArenaControlBlock> *arena_map = nullptr;

static Memory NewMemory(size_t nbytes)
{
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
   if (data.d_ptr) { ctrl->Device(data.d_mt)->Dealloc(data); }
   if (data.h_ptr) { ctrl->Host(data.h_mt)->Dealloc(data); }
}

void ArenaChunk::Dealloc(ArenaControlBlock &ptr_control)
{
   ptr_control.deallocated = true;
   ptr_count -= 1;

   // If ptr_count is zero, then every allocation in thus chunk has been
   // deallocated. We can clear any metadata associated with allocations in this
   // chunk, and reset the offset to zero.
   if (ptr_count == 0)
   {
      offset = 0;
      for (void *ptr : ptr_stack) { arena_map->erase(ptr); }
      ptr_stack.clear();
      // If this is the front chunk, don't deallocate the backing memory --- it
      // can potentially be reused next time an arena allocation is requested.
      //
      // If, on the other hand, this is not the front chunk, the backing memory
      // cannot be reused (it will first need to be consolidated with the chunks
      // that come before it), and so we release the memory immediately.
      if (!front)
      {
         // The pointers are set to NULL after they are deallocated to prevent
         // double-free when the ArenaChunk is destroyed.
         if (data.d_ptr)
         {
            ctrl->Device(data.d_mt)->Dealloc(data);
            data.d_ptr = nullptr;
         }
         ctrl->Host(data.h_mt)->Dealloc(data);
         data.h_ptr = nullptr;
      }
      return;
   }

   // This chunk is not empty, but we still can reclaim deallocated memory that
   // is at the top of the pointer stack.
   //
   // We start searching at the top of the stack and find all consecutive
   // deallocated pointers, and then reclaim this (potentially empty) range of
   // memory.
   auto end = ptr_stack.end();
   auto ptr_it = ptr_stack.rbegin();
   while (ptr_it != ptr_stack.rend())
   {
      const ArenaControlBlock &control = arena_map->at(*ptr_it);
      if (!control.deallocated) { break; }
      ++ptr_it;
   }
   auto begin = ptr_it.base();

   // Sanity check:
   MFEM_ASSERT(ptr_stack.size() - (end - begin) >= ptr_count, "");

   // If the range is not empty, reclaim the memory by shifting the offset, and
   // delete the associated pointer metadata.
   if (begin != end)
   {
      // Debug output:
      // mfem::out << "Reclaiming " << offset - ((char*)*begin - (char*)data.h_ptr)
      //           << " bytes from " << (end - begin) << " pointers.\n";
      offset = (char*)*begin - (char*)data.h_ptr;

      for (auto it = begin; it != end; ++it)
      {
         arena_map->erase(*it);
      }
      ptr_stack.erase(begin, end);
   }
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

void ArenaChunk::ReleaseDevice()
{
   data.d_ptr = nullptr;
}

ArenaHostMemorySpace::ArenaHostMemorySpace()
{
   MFEM_ASSERT(arena_map == nullptr, "internal error");
   arena_map = new std::unordered_map<void*,ArenaControlBlock>;
}

ArenaHostMemorySpace::~ArenaHostMemorySpace()
{
   delete arena_map;
   arena_map = nullptr;
   if (!Device::IsConfigured())
   {
      // For an explanation why we do this, see the documentation of
      // ArenaChunk::ReleaseDevice().
      for (auto &chunk : chunks) { chunk.ReleaseDevice(); }
   }
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
   // If requesting zero-size allocation, allocate the minimum positive number
   // of bytes (will round up to alignment).
   if (nbytes == 0) { ++nbytes; }

   // Round up requested size to multiple of alignment.
   const size_t r = nbytes % alignment;
   const size_t nbytes_aligned = r ? (nbytes + alignment - r) : nbytes;

   ConsolidateAndEnsureAvailable(nbytes_aligned);
   // FIXME: this does not ensure the pointer is aligned at 256 bytes !!!
   *ptr = chunks.front().NewPointer(nbytes_aligned);
   arena_map->emplace(*ptr, ArenaControlBlock(chunks.front(), nbytes_aligned));

   // Debug output:
   // size_t nchunks = std::distance(std::begin(chunks), std::end(chunks));
   // mfem::out << "===========================================================\n";
   // mfem::out << nchunks << " chunks\n";
   // int i = 0;
   // for (auto it = chunks.begin(); it != chunks.end(); )
   // {
   //    auto &c = *it;
   //    mfem::out << "   Chunk " << i << '\n';
   //    mfem::out << "   Size:      " << c.GetCapacity() << '\n';
   //    mfem::out << "   Vectors:   " << c.GetPointerCount() << '\n';
   //    mfem::out << "   Available: " << c.GetAvailableCapacity() << '\n';
   //    ++it;
   //    ++i;
   //    if (it != chunks.end())
   //    {
   //       mfem::out << "   --------------------------------------------------\n";
   //    }
   // }
}

void ArenaHostMemorySpace::Dealloc(Memory &mem)
{
   MFEM_ASSERT(arena_map != nullptr, "internal error");
   ArenaControlBlock &control = arena_map->at(mem.h_ptr);
   control.chunk.Dealloc(control);
}

void ArenaDeviceMemorySpace::Alloc(Memory &base)
{
   MFEM_ASSERT(arena_map != nullptr, "internal error");
   auto &chunk = arena_map->at(base.h_ptr).chunk;
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
