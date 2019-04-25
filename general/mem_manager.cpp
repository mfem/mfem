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

namespace mfem
{

namespace internal
{

/// Forward declaration of the Alias structure
struct Alias;

/// Memory class that holds:
///   - a boolean telling which memory space is being used
///   - the size in bytes of this memory region,
///   - the host and the device pointer,
///   - a list of all aliases seen using this region (used only to free them).
struct Memory
{
   bool host;
   const std::size_t bytes;
   void *const h_ptr;
   void *d_ptr;
   std::list<const void*> aliases;
   Memory(void* const h, const std::size_t size):
      host(true), bytes(size), h_ptr(h), d_ptr(nullptr), aliases() {}
};

/// Alias class that holds the base memory region and the offset
struct Alias
{
   Memory *const mem;
   const long offset;
};

typedef std::unordered_map<const void*, Memory> MemoryMap;
typedef std::unordered_map<const void*, const Alias*> AliasMap;

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
   enabled = true;
   maps = new internal::Ledger();
}

MemoryManager::~MemoryManager()
{
   delete maps;
   exists = false;
}

void* MemoryManager::Insert(void *ptr, const std::size_t bytes)
{
   if (!UsingMM()) { return ptr; }
   const bool known = IsKnown(ptr);
   if (known)
   {
      mfem_error("Trying to add an already present address!");
   }
   maps->memories.emplace(ptr, internal::Memory(ptr, bytes));
   return ptr;
}

void *MemoryManager::Erase(void *ptr)
{
   if (!UsingMM()) { return ptr; }
   if (!ptr) { return ptr; }
   const bool known = IsKnown(ptr);
   if (!known)
   {
      mfem_error("Trying to erase an unknown pointer!");
   }
   internal::Memory &mem = maps->memories.at(ptr);
   if (mem.d_ptr) { CuMemFree(mem.d_ptr); }
   for (const void *alias : mem.aliases)
   {
      maps->aliases.erase(maps->aliases.find(alias));
   }
   mem.aliases.clear();
   maps->memories.erase(maps->memories.find(ptr));
   return ptr;
}

void MemoryManager::SetHostDevicePtr(void *h_ptr, void *d_ptr, const bool host)
{
   internal::Memory &base = maps->memories.at(h_ptr);
   base.d_ptr = d_ptr;
   base.host = host;
}

bool MemoryManager::IsKnown(const void *ptr)
{
   return maps->memories.find(ptr) != maps->memories.end();
}

bool MemoryManager::IsOnHost(const void *ptr)
{
   return maps->memories.at(ptr).host;
}

std::size_t MemoryManager::Bytes(const void *ptr)
{
   return maps->memories.at(ptr).bytes;
}

void *MemoryManager::GetDevicePtr(const void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const size_t bytes = base.bytes;
   if (!base.d_ptr)
   {
      CuMemAlloc(&base.d_ptr, bytes);
      CuMemcpyHtoD(base.d_ptr, ptr, bytes);
      base.host = false;
   }
   return base.d_ptr;
}

// Looks if ptr is an alias of one memory
static const void* AliasBaseMemory(const internal::Ledger *maps,
                                   const void *ptr)
{
   for (internal::MemoryMap::const_iterator mem = maps->memories.begin();
        mem != maps->memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      if (b_ptr > ptr) { continue; }
      const void *end = static_cast<const char*>(b_ptr) + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return nullptr;
}

bool MemoryManager::IsAlias(const void *ptr)
{
   const internal::AliasMap::const_iterator found = maps->aliases.find(ptr);
   if (found != maps->aliases.end()) { return true; }
   MFEM_ASSERT(!IsKnown(ptr), "Ptr is an already known address!");
   const void *base = AliasBaseMemory(maps, ptr);
   if (!base) { return false; }
   internal::Memory &mem = maps->memories.at(base);
   const long offset = static_cast<const char*>(ptr) -
                       static_cast<const char*> (base);
   const internal::Alias *alias = new internal::Alias{&mem, offset};
   maps->aliases.emplace(ptr, alias);
   mem.aliases.push_back(ptr);
   return true;
}

static inline bool MmDeviceIniFilter(void)
{
   if (!mm.UsingMM()) { return true; }
   if (!mm.IsEnabled()) { return true; }
   if (!Device::IsAvailable()) { return true; }
   if (!Device::IsConfigured()) { return true; }
   return false;
}

// Turn a known address into the right host or device address. Alloc, Push, or
// Pull it if necessary.
static void *PtrKnown(internal::Ledger *maps, void *ptr)
{
   internal::Memory &base = maps->memories.at(ptr);
   const bool ptr_on_host = base.host;
   const std::size_t bytes = base.bytes;
   const bool run_on_device = Device::Allows(Backend::DEVICE_MASK);
   if (ptr_on_host && !run_on_device) { return ptr; }
   if (bytes==0) { mfem_error("PtrKnown bytes==0"); }
   if (!base.d_ptr) { CuMemAlloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (!ptr_on_host && run_on_device) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (!ptr_on_host && !run_on_device) // Pull
   {
      CuMemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   if (!(ptr_on_host && run_on_device)) { mfem_error("PtrKnown !(host && gpu)"); }
   CuMemcpyHtoD(base.d_ptr, ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// Turn an alias into the right host or device address. Alloc, Push, or Pull it
// if necessary.
static void *PtrAlias(internal::Ledger *maps, void *ptr)
{
   const bool gpu = Device::Allows(Backend::DEVICE_MASK);
   const internal::Alias *alias = maps->aliases.at(ptr);
   const internal::Memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (bytes==0) { mfem_error("PtrAlias bytes==0"); }
   if (!base->d_ptr) { CuMemAlloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      CuMemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
   CuMemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

void *MemoryManager::Ptr(void *ptr)
{
   if (ptr==NULL) { return NULL; };
   if (MmDeviceIniFilter()) { return ptr; }
   if (IsKnown(ptr)) { return PtrKnown(maps, ptr); }
   if (IsAlias(ptr)) { return PtrAlias(maps, ptr); }
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mfem_error("Trying to use unknown pointer on the DEVICE!");
   }
   return ptr;
}

const void *MemoryManager::Ptr(const void *ptr)
{
   return static_cast<const void*>(Ptr(const_cast<void*>(ptr)));
}

static void PushKnown(internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   internal::Memory &base = maps->memories.at(ptr);
   if (!base.d_ptr) { CuMemAlloc(&base.d_ptr, base.bytes); }
   CuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

static void PushAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   CuMemcpyHtoD(dst, ptr, bytes);
}

void MemoryManager::Push(const void *ptr, const std::size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (IsKnown(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (IsAlias(ptr)) { return PushAlias(maps, ptr, bytes); }
   if (Device::Allows(Backend::DEVICE_MASK))
   { mfem_error("Unknown pointer to push to!"); }
}

static void PullKnown(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Memory &base = maps->memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   CuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

static void PullAlias(const internal::Ledger *maps,
                      const void *ptr, const std::size_t bytes)
{
   const internal::Alias *alias = maps->aliases.at(ptr);
   const bool host = alias->mem->host;
   if (host) { return; }
   if (!ptr) { mfem_error("PullAlias !ptr"); }
   if (!alias->mem->d_ptr) { mfem_error("PullAlias !alias->mem->d_ptr"); }
   CuMemcpyDtoH(const_cast<void*>(ptr),
                static_cast<char*>(alias->mem->d_ptr) + alias->offset,
                bytes);
}

void MemoryManager::Pull(const void *ptr, const std::size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (IsKnown(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (IsAlias(ptr)) { return PullAlias(maps, ptr, bytes); }
   if (Device::Allows(Backend::DEVICE_MASK))
   { mfem_error("Unknown pointer to pull from!"); }
}

void* MemoryManager::Memcpy(void *dst, const void *src,
                            const std::size_t bytes, const bool async)
{
   void *d_dst = Ptr(dst);
   void *d_src = const_cast<void*>(Ptr(src));
   if (bytes == 0) { return dst; }
   const bool run_on_host = !Device::Allows(Backend::DEVICE_MASK);
   if (run_on_host) { return std::memcpy(dst, src, bytes); }
   if (!async) { return CuMemcpyDtoD(d_dst, d_src, bytes); }
   return CuMemcpyDtoDAsync(d_dst, d_src, bytes);
}

void MemoryManager::RegisterCheck(void *ptr)
{
   if (ptr != NULL && UsingMM())
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
}

void MemoryManager::GetAll(void)
{
   for (const auto& n : maps->memories)
   {
      const void *ptr = n.first;
      Ptr(ptr);
   }
}

MemoryManager mm;
bool MemoryManager::exists = false;

} // namespace mfem
