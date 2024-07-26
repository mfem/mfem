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

Memory NewMemory(size_t nbytes)
{
   MFEM_ASSERT(ctrl != nullptr, "");
   const MemoryType h_mt = MemoryManager::GetHostMemoryType();
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   void *h_ptr;
   ctrl->Host(h_mt)->Alloc(&h_ptr, nbytes);
   return Memory(h_ptr, nbytes, h_mt, d_mt);
}

WorkspaceChunk::WorkspaceChunk(size_t capacity)
   : data(NewMemory(capacity))
{
   ptr_stack.reserve(16); // <- Can adjust this
}

WorkspaceChunk::~WorkspaceChunk()
{
   // if (!dealloced)
   if (ctrl && !dealloced)
   {
      // data.Delete();
      if (data.d_ptr)
      {
         ctrl->Device(data.d_mt)->Dealloc(data);
      }
      ctrl->Host(data.h_mt)->Dealloc(data);
   }
}

void WorkspaceChunk::ClearDeallocated()
{
   ptr_count -= 1;
   if (ptr_count == 0)
   {
      offset = 0;
      for (void *ptr : ptr_stack) { maps->workspace.erase(ptr); }
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
      auto it = maps->workspace.find(*ptr_it);
      MFEM_ASSERT(it != maps->workspace.end(), "");
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
      maps->workspace.erase(*it);
   }
   ptr_stack.erase(begin, end);

   MFEM_ASSERT(ptr_stack.size() >= ptr_count, "");
}

// void WorkspaceChunk::FreePointer(void *ptr, size_t capacity)
// {
//    MFEM_ASSERT(ptr_count >= 0, "");
//    ptr_count -= 1;
//    // If the chunk is completely empty, we can reclaim all of the memory and
//    // allow new allocations (before it is completely empty, we cannot reclaim
//    // memory because we don't track the specific regions that are freed).
//    if (ptr_count == 0)
//    {
//       offset = 0;
//       // If we are not the front chunk, deallocate the backing memory. This
//       // chunk will be consolidated later anyway.
//       if (!front)
//       {
//          if (data.d_ptr)
//          {
//             ctrl->Device(data.d_mt)->Dealloc(data);
//          }
//          ctrl->Host(data.h_mt)->Dealloc(data.h_ptr);
//          dealloced = true;
//       }
//    }
//    else
//    {
//       // If the vector being freed is the most recent vector allocated (i.e. if
//       // the vector is freed in stack/LIFO order), then we can reclaim its
//       // memory by moving the offset.
//       if ((char*)ptr + capacity == (char*)data.h_ptr + offset)
//       {
//          offset -= capacity;
//       }
//    }
// }

void *WorkspaceChunk::GetDevicePointer(void *h_ptr)
{
   if (data.d_ptr == nullptr)
   {
      ctrl->Device(data.d_mt)->Alloc(data);
   }
   const size_t ptr_offset = ((char*)h_ptr) - ((char*)data.h_ptr);
   return ((char*)data.d_ptr) + ptr_offset;
}

void WorkspaceDeviceMemorySpace::Alloc(Memory &base)
{
   auto &chunk = maps->workspace.find(base.h_ptr)->second.chunk;
   base.d_ptr = chunk.GetDevicePointer(base.h_ptr);
}

void WorkspaceDeviceMemorySpace::Dealloc(Memory &base) { /* no-op */ }

void *WorkspaceDeviceMemorySpace::HtoH(void *dst, const void *src, size_t bytes)
{
   MFEM_ABORT("");
   return std::memcpy(dst, src, bytes);
}

void *WorkspaceDeviceMemorySpace::HtoD(void *dst, const void *src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   if (IsHostMemory(d_mt)) { return HtoH(dst, src, bytes); }
   return ctrl->Device(d_mt)->HtoD(dst, src, bytes);
}

void *WorkspaceDeviceMemorySpace::DtoD(void* dst, const void* src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   if (IsHostMemory(d_mt)) { return HtoH(dst, src, bytes); }
   return ctrl->Device(d_mt)->DtoD(dst, src, bytes);
}

void *WorkspaceDeviceMemorySpace::DtoH(void *dst, const void *src, size_t bytes)
{
   const MemoryType d_mt = MemoryManager::GetDeviceMemoryType();
   if (IsHostMemory(d_mt)) { return HtoH(dst, src, bytes); }
   return ctrl->Device(d_mt)->DtoH(dst, src, bytes);
}

} // namespace internal

} // namespace mfem
