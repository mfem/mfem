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

#include "../general/forall.hpp"

#include <cstring> // std::memcpy

#include <list>
#include <unordered_map>
#include <algorithm> // std::max

namespace mfem
{

#ifdef MFEM_USE_HIP
#define MFEM_GPU(...) Hip ## __VA_ARGS__
#else
#define MFEM_GPU(...) Cu ## __VA_ARGS__
#endif

MemoryType GetMemoryType(MemoryClass mc)
{
   switch (mc)
   {
      case MemoryClass::HOST:     return MemoryType::HOST;
      case MemoryClass::HOST_32:  return MemoryType::HOST_32;
      case MemoryClass::HOST_64:  return MemoryType::HOST_64;
      case MemoryClass::CUDA:     return MemoryType::CUDA;
      case MemoryClass::CUDA_UVM: return MemoryType::CUDA_UVM;
   }
   return MemoryType::HOST;
}

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //          |   HOST     HOST_32   HOST_64    CUDA    CUDA_UVM
   // ---------+--------------------------------------------------
   //   HOST   |   HOST     HOST_32   HOST_64    CUDA    CUDA_UVM
   //  HOST_32 |  HOST_32   HOST_32   HOST_64    CUDA    CUDA_UVM
   //  HOST_64 |  HOST_64   HOST_64   HOST_64    CUDA    CUDA_UVM
   //   CUDA   |   CUDA      CUDA      CUDA      CUDA    CUDA_UVM
   // CUDA_UVM | CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM  CUDA_UVM

   // Using the enumeration ordering:
   //    HOST < HOST_32 < HOST_64 < CUDA < CUDA_UVM,
   // the above table is simply: a*b = max(a,b).

   return std::max(mc1, mc2);
}


namespace internal
{

/// Forward declaration of the Alias structure
struct Alias;

/// Memory class that holds:
///   - a boolean telling which memory space is being used
///   - the size in bytes of this memory region,
///   - the host and the device pointer.
struct Memory
{
   bool host;
   const std::size_t bytes;
   void *const h_ptr;
   void *d_ptr;
   Memory(void* const h, const std::size_t size):
      host(true), bytes(size), h_ptr(h), d_ptr(nullptr) {}
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *const mem;
   const long offset;
   unsigned long counter;
};

typedef std::unordered_map<const void*, Memory> MemoryMap;
// TODO: use 'Alias' or 'const Alias' as the mapped type in the AliasMap instead
// of 'Alias*'
typedef std::unordered_map<const void*, Alias*> AliasMap;

struct Ledger
{
   MemoryMap memories;
   AliasMap aliases;
};

} // namespace mfem::internal

static internal::Ledger *maps;

MemoryManager::MemoryManager()
{
   exists = true;
   maps = new internal::Ledger();
}

MemoryManager::~MemoryManager()
{
   if (exists) { Destroy(); }
}

void MemoryManager::Destroy()
{
   MFEM_VERIFY(exists, "MemoryManager has been destroyed already!");
   for (auto& n : maps->memories)
   {
      internal::Memory &mem = n.second;
      if (mem.d_ptr) { MFEM_GPU(MemFree)(mem.d_ptr); }
   }
   for (auto& n : maps->aliases)
   {
      delete n.second;
   }
   delete maps;
   exists = false;
}

void* MemoryManager::Insert(void *ptr, const std::size_t bytes)
{
   if (ptr == NULL)
   {
      MFEM_VERIFY(bytes == 0, "Trying to add NULL with size " << bytes);
      return NULL;
   }
   auto res = maps->memories.emplace(ptr, internal::Memory(ptr, bytes));
   if (res.second == false)
   {
      mfem_error("Trying to add an already present address!");
   }
   return ptr;
}

void MemoryManager::InsertDevice(void *ptr, void *h_ptr, size_t bytes)
{
   MFEM_VERIFY(ptr != NULL, "cannot register NULL device pointer");
   MFEM_VERIFY(h_ptr != NULL, "internal error");
   auto res = maps->memories.emplace(h_ptr, internal::Memory(h_ptr, bytes));
   if (res.second == false)
   {
      mfem_error("Trying to add an already present address!");
   }
   res.first->second.d_ptr = ptr;
}

void *MemoryManager::Erase(void *ptr, bool free_dev_ptr)
{
   if (!ptr) { return ptr; }
   auto mem_map_iter = maps->memories.find(ptr);
   if (mem_map_iter == maps->memories.end())
   {
      mfem_error("Trying to erase an unknown pointer!");
   }
   internal::Memory &mem = mem_map_iter->second;
   if (mem.d_ptr && free_dev_ptr) { MFEM_GPU(MemFree)(mem.d_ptr); }
   maps->memories.erase(mem_map_iter);
   return ptr;
}

bool MemoryManager::IsKnown(const void *ptr)
{
   return maps->memories.find(ptr) != maps->memories.end();
}

void *MemoryManager::GetDevicePtr(const void *ptr, size_t bytes, bool copy_data)
{
   if (!ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   internal::Memory &base = maps->memories.at(ptr);
   if (!base.d_ptr)
   {
      MFEM_GPU(MemAlloc)(&base.d_ptr, base.bytes);
   }
   if (copy_data)
   {
      MFEM_ASSERT(bytes <= base.bytes, "invalid copy size");
      MFEM_GPU(MemcpyHtoD)(base.d_ptr, ptr, bytes);
      base.host = false;
   }
   return base.d_ptr;
}

void MemoryManager::InsertAlias(const void *base_ptr, void *alias_ptr,
                                bool base_is_alias)
{
   long offset = static_cast<const char*>(alias_ptr) -
                 static_cast<const char*>(base_ptr);
   if (!base_ptr)
   {
      MFEM_VERIFY(offset == 0,
                  "Trying to add alias to NULL at offset " << offset);
      return;
   }
   if (base_is_alias)
   {
      const internal::Alias *alias = maps->aliases.at(base_ptr);
      base_ptr = alias->mem->h_ptr;
      offset += alias->offset;
   }
   internal::Memory &mem = maps->memories.at(base_ptr);
   auto res = maps->aliases.emplace(alias_ptr, nullptr);
   if (res.second == false) // alias_ptr was already in the map
   {
      if (res.first->second->mem != &mem || res.first->second->offset != offset)
      {
         mfem_error("alias already exists with different base/offset!");
      }
      else
      {
         res.first->second->counter++;
      }
   }
   else
   {
      res.first->second = new internal::Alias{&mem, offset, 1};
   }
}

void MemoryManager::EraseAlias(void *alias_ptr)
{
   if (!alias_ptr) { return; }
   auto alias_map_iter = maps->aliases.find(alias_ptr);
   if (alias_map_iter == maps->aliases.end())
   {
      mfem_error("alias not found");
   }
   internal::Alias *alias = alias_map_iter->second;
   if (--alias->counter) { return; }
   // erase the alias from the alias map:
   maps->aliases.erase(alias_map_iter);
   delete alias;
}

void *MemoryManager::GetAliasDevicePtr(const void *alias_ptr, size_t bytes,
                                       bool copy_data)
{
   if (!alias_ptr)
   {
      MFEM_VERIFY(bytes == 0, "Trying to access NULL with size " << bytes);
      return NULL;
   }
   auto &alias_map = maps->aliases;
   auto alias_map_iter = alias_map.find(alias_ptr);
   if (alias_map_iter == alias_map.end())
   {
      mfem_error("alias not found");
   }
   const internal::Alias *alias = alias_map_iter->second;
   internal::Memory &base = *alias->mem;
   MFEM_ASSERT((char*)base.h_ptr + alias->offset == alias_ptr,
               "internal error");
   if (!base.d_ptr)
   {
      MFEM_GPU(MemAlloc)(&base.d_ptr, base.bytes);
   }
   if (copy_data)
   {
      MFEM_GPU(MemcpyHtoD)((char*)base.d_ptr + alias->offset, alias_ptr, bytes);
      base.host = false;
   }
   return (char*)base.d_ptr + alias->offset;
}

static void PullKnown(internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes, bool copy_data)
{
   internal::Memory &base = maps->memories.at(ptr);
   MFEM_ASSERT(base.h_ptr == ptr, "internal error");
   // There are cases where it is OK if base.d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && base.d_ptr)
   {
      MFEM_GPU(MemcpyDtoH)(base.h_ptr, base.d_ptr, bytes);
      base.host = true;
   }
}

static void PullAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes, bool copy_data)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   MFEM_ASSERT((char*)alias->mem->h_ptr + alias->offset == ptr,
               "internal error");
   // There are cases where it is OK if alias->mem->d_ptr is not allocated yet:
   // for example, when requesting read-write access on host to memory created
   // as device memory.
   if (copy_data && alias->mem->d_ptr)
   {
      MFEM_GPU(MemcpyDtoH)(const_cast<void*>(ptr),
                           static_cast<char*>(alias->mem->d_ptr) + alias->offset,
                           bytes);
   }
}

void MemoryManager::RegisterCheck(void *ptr)
{
   if (ptr != NULL)
   {
      if (!IsKnown(ptr))
      {
         mfem_error("Pointer is not registered!");
      }
   }
}

void MemoryManager::PrintPtrs(void)
{
   for (const auto& n : maps->memories)
   {
      const internal::Memory &mem = n.second;
      mfem::out << std::endl
                << "key " << n.first << ", "
                << "host " << mem.host << ", "
                << "h_ptr " << mem.h_ptr << ", "
                << "d_ptr " << mem.d_ptr;
   }
   mfem::out << std::endl;
}

// Static private MemoryManager methods used by class Memory

void *MemoryManager::New_(void *h_ptr, std::size_t size, MemoryType mt,
                          unsigned &flags)
{
   // TODO: save the types of the pointers ...
   flags = Mem::REGISTERED | Mem::OWNS_INTERNAL;
   switch (mt)
   {
      case MemoryType::HOST: return nullptr; // case is handled outside

      case MemoryType::HOST_32:
      case MemoryType::HOST_64:
         mfem_error("New_(): aligned host types are not implemented yet");
         return nullptr;

      case MemoryType::CUDA:
         mm.Insert(h_ptr, size);
         flags = flags | Mem::OWNS_HOST | Mem::OWNS_DEVICE | Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryType::CUDA_UVM:
         mfem_error("New_(): CUDA UVM allocation is not implemented yet");
         return nullptr;
   }
   return nullptr;
}

void *MemoryManager::Register_(void *ptr, void *h_ptr, std::size_t capacity,
                               MemoryType mt, bool own, bool alias,
                               unsigned &flags)
{
   // TODO: save the type of the registered pointer ...
   MFEM_VERIFY(alias == false, "cannot register an alias!");
   flags = flags | (Mem::REGISTERED | Mem::OWNS_INTERNAL);
   if (IsHostMemory(mt))
   {
      mm.Insert(ptr, capacity);
      flags = (own ? flags | Mem::OWNS_HOST : flags & ~Mem::OWNS_HOST) |
              Mem::OWNS_DEVICE | Mem::VALID_HOST;
      return ptr;
   }
   MFEM_VERIFY(mt == MemoryType::CUDA, "Only CUDA pointers are supported");
   mm.InsertDevice(ptr, h_ptr, capacity);
   flags = (own ? flags | Mem::OWNS_DEVICE : flags & ~Mem::OWNS_DEVICE) |
           Mem::OWNS_HOST | Mem::VALID_DEVICE;
   return h_ptr;
}

void MemoryManager::Alias_(void *base_h_ptr, std::size_t offset,
                           std::size_t size, unsigned base_flags,
                           unsigned &flags)
{
   // TODO: store the 'size' in the MemoryManager?
   mm.InsertAlias(base_h_ptr, (char*)base_h_ptr + offset,
                  base_flags & Mem::ALIAS);
   flags = (base_flags | Mem::ALIAS | Mem::OWNS_INTERNAL) &
           ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
}

MemoryType MemoryManager::Delete_(void *h_ptr, unsigned flags)
{
   // TODO: this logic needs to be updated when support for HOST_32 and HOST_64
   // memory types is added.

   MFEM_ASSERT(!(flags & Mem::OWNS_DEVICE) || (flags & Mem::OWNS_INTERNAL),
               "invalid Memory state");
   if (mm.exists && (flags & Mem::OWNS_INTERNAL))
   {
      if (flags & Mem::ALIAS)
      {
         mm.EraseAlias(h_ptr);
      }
      else
      {
         mm.Erase(h_ptr, flags & Mem::OWNS_DEVICE);
      }
   }
   return MemoryType::HOST;
}

void *MemoryManager::ReadWrite_(void *h_ptr, MemoryClass mc,
                                std::size_t size, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, size, true); }
            else { PullKnown(maps, h_ptr, size, true); }
         }
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::CUDA:
      {
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, need_copy);
         }
         return mm.GetDevicePtr(h_ptr, size, need_copy);
      }

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

const void *MemoryManager::Read_(void *h_ptr, MemoryClass mc,
                                 std::size_t size, unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         if (!(flags & Mem::VALID_HOST))
         {
            if (flags & Mem::ALIAS) { PullAlias(maps, h_ptr, size, true); }
            else { PullKnown(maps, h_ptr, size, true); }
         }
         flags = flags | Mem::VALID_HOST;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64
         return h_ptr;

      case MemoryClass::CUDA:
      {
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         const bool need_copy = !(flags & Mem::VALID_DEVICE);
         flags = flags | Mem::VALID_DEVICE;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, need_copy);
         }
         return mm.GetDevicePtr(h_ptr, size, need_copy);
      }

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

void *MemoryManager::Write_(void *h_ptr, MemoryClass mc, std::size_t size,
                            unsigned &flags)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_32:
         // TODO: check that the host pointer is MemoryType::HOST_32 or
         // MemoryType::HOST_64

         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::HOST_64:
         // TODO: check that the host pointer is MemoryType::HOST_64

         flags = (flags | Mem::VALID_HOST) & ~Mem::VALID_DEVICE;
         return h_ptr;

      case MemoryClass::CUDA:
         // TODO: check that the device pointer is MemoryType::CUDA or
         // MemoryType::CUDA_UVM

         flags = (flags | Mem::VALID_DEVICE) & ~Mem::VALID_HOST;

         // TODO: add support for UVM
         if (flags & Mem::ALIAS)
         {
            return mm.GetAliasDevicePtr(h_ptr, size, false);
         }
         return mm.GetDevicePtr(h_ptr, size, false);

      case MemoryClass::CUDA_UVM:
         // TODO: check that the host+device pointers are MemoryType::CUDA_UVM

         // Do we need to update the validity flags?

         return h_ptr; // the host and device pointers are the same
   }
   return nullptr;
}

void MemoryManager::SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                               size_t alias_size, unsigned base_flags,
                               unsigned &alias_flags)
{
   // This is called only when (base_flags & Mem::REGISTERED) is true.
   // Note that (alias_flags & REGISTERED) may not be true.
   MFEM_ASSERT(alias_flags & Mem::ALIAS, "not an alias");
   if ((base_flags & Mem::VALID_HOST) && !(alias_flags & Mem::VALID_HOST))
   {
      PullAlias(maps, alias_h_ptr, alias_size, true);
   }
   if ((base_flags & Mem::VALID_DEVICE) && !(alias_flags & Mem::VALID_DEVICE))
   {
      if (!(alias_flags & Mem::REGISTERED))
      {
         mm.InsertAlias(base_h_ptr, alias_h_ptr, base_flags & Mem::ALIAS);
         alias_flags = (alias_flags | Mem::REGISTERED | Mem::OWNS_INTERNAL) &
                       ~(Mem::OWNS_HOST | Mem::OWNS_DEVICE);
      }
      mm.GetAliasDevicePtr(alias_h_ptr, alias_size, true);
   }
   alias_flags = (alias_flags & ~(Mem::VALID_HOST | Mem::VALID_DEVICE)) |
                 (base_flags & (Mem::VALID_HOST | Mem::VALID_DEVICE));
}

MemoryType MemoryManager::GetMemoryType_(void *h_ptr, unsigned flags)
{
   // TODO: support other memory types
   if (flags & Mem::VALID_DEVICE) { return MemoryType::CUDA; }
   return MemoryType::HOST;
}

void MemoryManager::Copy_(void *dest_h_ptr, const void *src_h_ptr,
                          std::size_t size, unsigned src_flags,
                          unsigned &dest_flags)
{
   // Type of copy to use based on the src and dest validity flags:
   //            |       src
   //            |  h  |  d  |  hd
   // -----------+-----+-----+------
   //         h  | h2h   d2h   h2h
   //  dest   d  | h2d   d2d   d2d
   //        hd  | h2h   d2d   d2d

   const bool src_on_host =
      (src_flags & Mem::VALID_HOST) &&
      (!(src_flags & Mem::VALID_DEVICE) ||
       ((dest_flags & Mem::VALID_HOST) && !(dest_flags & Mem::VALID_DEVICE)));
   const bool dest_on_host =
      (dest_flags & Mem::VALID_HOST) &&
      (!(dest_flags & Mem::VALID_DEVICE) ||
       ((src_flags & Mem::VALID_HOST) && !(src_flags & Mem::VALID_DEVICE)));
   const void *src_d_ptr = src_on_host ? NULL :
                           ((src_flags & Mem::ALIAS) ?
                            mm.GetAliasDevicePtr(src_h_ptr, size, false) :
                            mm.GetDevicePtr(src_h_ptr, size, false));
   if (dest_on_host)
   {
      if (src_on_host)
      {
         if (dest_h_ptr != src_h_ptr && size != 0)
         {
            MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                        (char*)src_h_ptr + size <= dest_h_ptr,
                        "data overlaps!");
            std::memcpy(dest_h_ptr, src_h_ptr, size);
         }
      }
      else
      {
         MFEM_GPU(MemcpyDtoH)(dest_h_ptr, src_d_ptr, size);
      }
   }
   else
   {
      void *dest_d_ptr = (dest_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dest_h_ptr, size, false) :
                         mm.GetDevicePtr(dest_h_ptr, size, false);
      if (src_on_host)
      {
         MFEM_GPU(MemcpyHtoD)(dest_d_ptr, src_h_ptr, size);
      }
      else
      {
         MFEM_GPU(MemcpyDtoD)(dest_d_ptr, src_d_ptr, size);
      }
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}

void MemoryManager::CopyToHost_(void *dest_h_ptr, const void *src_h_ptr,
                                std::size_t size, unsigned src_flags)
{
   const bool src_on_host = src_flags & Mem::VALID_HOST;
   if (src_on_host)
   {
      if (dest_h_ptr != src_h_ptr && size != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                     (char*)src_h_ptr + size <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, size);
      }
   }
   else
   {
      const void *src_d_ptr = (src_flags & Mem::ALIAS) ?
                              mm.GetAliasDevicePtr(src_h_ptr, size, false) :
                              mm.GetDevicePtr(src_h_ptr, size, false);
      MFEM_GPU(MemcpyDtoH)(dest_h_ptr, src_d_ptr, size);
   }
}

void MemoryManager::CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                                  std::size_t size, unsigned &dest_flags)
{
   const bool dest_on_host = dest_flags & Mem::VALID_HOST;
   if (dest_on_host)
   {
      if (dest_h_ptr != src_h_ptr && size != 0)
      {
         MFEM_ASSERT((char*)dest_h_ptr + size <= src_h_ptr ||
                     (char*)src_h_ptr + size <= dest_h_ptr,
                     "data overlaps!");
         std::memcpy(dest_h_ptr, src_h_ptr, size);
      }
   }
   else
   {
      void *dest_d_ptr = (dest_flags & Mem::ALIAS) ?
                         mm.GetAliasDevicePtr(dest_h_ptr, size, false) :
                         mm.GetDevicePtr(dest_h_ptr, size, false);
      MFEM_GPU(MemcpyHtoD)(dest_d_ptr, src_h_ptr, size);
   }
   dest_flags = dest_flags &
                ~(dest_on_host ? Mem::VALID_DEVICE : Mem::VALID_HOST);
}


void MemoryPrintFlags(unsigned flags)
{
   typedef Memory<int> Mem;
   mfem::out
         <<   "   registered    = " << bool(flags & Mem::REGISTERED)
         << "\n   owns host     = " << bool(flags & Mem::OWNS_HOST)
         << "\n   owns device   = " << bool(flags & Mem::OWNS_DEVICE)
         << "\n   owns internal = " << bool(flags & Mem::OWNS_INTERNAL)
         << "\n   valid host    = " << bool(flags & Mem::VALID_HOST)
         << "\n   valid device  = " << bool(flags & Mem::VALID_DEVICE)
         << "\n   alias         = " << bool(flags & Mem::ALIAS)
         << "\n   device flag   = " << bool(flags & Mem::USE_DEVICE)
         << std::endl;
}


MemoryManager mm;
bool MemoryManager::exists = false;

} // namespace mfem
