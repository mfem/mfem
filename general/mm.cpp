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

#include "../general/okina.hpp"

#include <bitset>
#include <cassert>

namespace mfem
{

namespace mm
{

MemoryManager& getInstance()
{
   static MemoryManager* s_instance = new MemoryManager();
   return *s_instance;
}

// TODO This wraps the d_ptr -- check if this works
// OccaMemory occaPtr(const void *a) {
// void *d_ptr = getInstance().getDevicePtr(a);
// return occaWrapMemory(config::GetOccaDevice(), d_ptr, bytes);
// }

static inline bool useMM()
{
   const bool usingMM = config::UsingMM();
   const bool devNotDisabled = !config::DeviceDisabled();
   const bool devHasBeenEnabled = config::DeviceHasBeenEnabled();
   return usingMM && devHasBeenEnabled && devNotDisabled;
}

void* ptr(void *a)
{
   if (useMM())
   {
      return getInstance().getPtr(a);
   }
   else
   {
      return a;
   }
}

const void* ptr(const void *a)
{
   return ptr(const_cast<void*>(a));
}

OccaMemory occaPtr(const void *a)
{
   if (config::UsingMM())
   {
      return getInstance().getOccaPtr(a);
   }
   else
   {
      OccaMemory o_ptr = occaWrapMemory(config::GetOccaDevice(),
                                        const_cast<void*>(a), 0);
      return o_ptr;
   }
}

void push(const void *ptr, const std::size_t bytes)
{
   if (useMM())
   {
      getInstance().pushData(ptr, bytes);
   }
}

void pull(const void *ptr, const std::size_t bytes)
{
   if (useMM())
   {
      getInstance().pullData(ptr, bytes);
   }
}

void memcpy(void *dst, const void *src,
            const std::size_t bytes, const bool async)
{
   if (bytes > 0)
   {
      getInstance().copyData(dst, src, bytes, async);
   }
}

} // namespace mm


// ********** DefaultMemoryManager **********

// *****************************************************************************
// * Tests if ptr is a known address
// *****************************************************************************
static bool Known(const DefaultMemoryManager::ledger &maps, const void *ptr)
{
   const DefaultMemoryManager::memory_map::const_iterator
   found = maps.memories.find(ptr);
   const bool known = found != maps.memories.end();
   if (known) { return true; }
   return false;
}

// *****************************************************************************
bool DefaultMemoryManager::Known(const void *ptr)
{
   return mfem::Known(maps,ptr);
}

// *****************************************************************************
// * Looks if ptr is an alias of one memory
// *****************************************************************************
static const void* IsAlias(const DefaultMemoryManager::ledger &maps,
                           const void *ptr)
{
   MFEM_ASSERT(!Known(maps, ptr), "Ptr is an already known address!");
   for (DefaultMemoryManager::memory_map::const_iterator
        mem = maps.memories.begin();
        mem != maps.memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      if (b_ptr > ptr) { continue; }
      const void *end = static_cast<const char*>(b_ptr) + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return nullptr;
}

// *****************************************************************************
static const void* InsertAlias(DefaultMemoryManager::ledger &maps,
                               const void *base, const void *ptr)
{
   DefaultMemoryManager::memory &mem = maps.memories.at(base);
   const long offset = static_cast<const char*>(ptr) -
                       static_cast<const char*> (base);
   const DefaultMemoryManager::alias *alias = new DefaultMemoryManager::alias{&mem, offset};
   maps.aliases.emplace(ptr, alias);
#ifdef MFEM_DEBUG_MM
   {
      mem.aliases.sort();
      for (const mm::alias *a : mem.aliases)
      {
         if (a->mem == &mem )
         {
            if (a->offset == offset)
            {
               mfem_error("a->offset == offset");
            }
         }
      }
   }
#endif
   mem.aliases.push_back(alias);
   return ptr;
}

// *****************************************************************************
// * Tests if ptr is an alias address
// *****************************************************************************
static bool Alias(DefaultMemoryManager::ledger &maps, const void *ptr)
{
   const DefaultMemoryManager::alias_map::const_iterator found =
      maps.aliases.find(ptr);
   const bool alias = found != maps.aliases.end();
   if (alias) { return true; }
   const void *base = IsAlias(maps, ptr);
   if (!base) { return false; }
   InsertAlias(maps, base, ptr);
   return true;
}

// *****************************************************************************
bool DefaultMemoryManager::Alias(const void *ptr)
{
   return mfem::Alias(maps,ptr);
}

// *****************************************************************************
static void DumpMode(void)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   static std::bitset<8+1> mode;
   std::bitset<8+1> cfg;
   cfg.set(config::UsingMM()?8:0);
   cfg.set(config::DeviceHasBeenEnabled()?7:0);
   cfg.set(config::DeviceEnabled()?6:0);
   cfg.set(config::DeviceDisabled()?5:0);
   cfg.set(config::UsingHost()?4:0);
   cfg.set(config::UsingDevice()?3:0);
   cfg.set(config::UsingCuda()?2:0);
   cfg.set(config::UsingOcca()?1:0);
   cfg>>=1;
   if (cfg==mode) { return; }
   mode=cfg;
   printf("\033[1K\r[0x%lx] %sMM %sHasBeenEnabled %sEnabled %sDisabled "
          "%sHOST %sDEVICE %sCUDA %sOCCA\033[m", mode.to_ulong(),
          config::UsingMM()?"\033[32m":"\033[31m",
          config::DeviceHasBeenEnabled()?"\033[32m":"\033[31m",
          config::DeviceEnabled()?"\033[32m":"\033[31m",
          config::DeviceDisabled()?"\033[32m":"\033[31m",
          config::UsingHost()?"\033[32m":"\033[31m",
          config::UsingDevice()?"\033[32m":"\033[31m",
          config::UsingCuda()?"\033[32m":"\033[31m",
          config::UsingOcca()?"\033[32m":"\033[31m");
}

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void DefaultMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
   if (!config::UsingMM()) { return; }
   const bool known = Known(ptr);
   if (known)
   {
      mfem_error("Trying to add an already present address!");
   }
   DumpMode();
   // ex1p comes with (bytes==0)
   maps.memories.emplace(ptr, memory(ptr, bytes));
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void DefaultMemoryManager::removeAddress(void *ptr)
{
   if (!config::UsingMM()) { return; }
   const bool known = Known(ptr);
   if (!known) { mfem_error("Trying to erase an unknown pointer!"); }
   memory &mem = maps.memories.at(ptr);
   for (const alias* const alias : mem.aliases)
   {
      maps.aliases.erase(alias);
   }
   mem.aliases.clear();
   maps.memories.erase(ptr);
}

// *****************************************************************************
static inline bool MmDeviceIniFilter(void)
{
   if (!config::UsingMM()) { return true; }
   if (config::DeviceDisabled()) { return true; }
   if (!config::DeviceHasBeenEnabled()) { return true; }
   if (config::UsingOcca()) { mfem_error("config::UsingOcca()"); }
   return false;
}

// *****************************************************************************
// * Turn a known address to the right host or device one
// * Alloc, Push or Pull it if required
// *****************************************************************************
static void *PtrKnown(DefaultMemoryManager::ledger &maps, void *ptr)
{
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   const bool device = !host;
   const size_t bytes = base.bytes;
   const bool gpu = config::UsingDevice();
   if (host && !gpu) { return ptr; }
   if (bytes==0) { mfem_error("PtrKnown bytes==0"); }
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, bytes); }
   if (!base.d_ptr) { mfem_error("PtrKnown !base->d_ptr"); }
   if (device &&  gpu) { return base.d_ptr; }
   if (!ptr) { mfem_error("PtrKnown !ptr"); }
   if (device && !gpu) // Pull
   {
      mfem::cuMemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrKnown !(host && gpu)"); }
   cuMemcpyHtoD(base.d_ptr, ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// *****************************************************************************
// * Turn an alias to the right host or device one
// * Alloc, Push or Pull it if required
// *****************************************************************************
static void *PtrAlias(DefaultMemoryManager::ledger &maps, void *ptr)
{
   const bool gpu = config::UsingDevice();
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   assert(alias->offset >0);
   const DefaultMemoryManager::memory *base = alias->mem;
   assert(base);
   const bool host = base->host;
   const bool device = !base->host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (bytes==0) { mfem_error("PtrAlias bytes==0"); }
   if (!base->d_ptr) { cuMemAlloc(&(alias->mem->d_ptr), bytes); }
   if (!base->d_ptr) { mfem_error("PtrAlias !base->d_ptr"); }
   void *a_ptr = static_cast<char*>(base->d_ptr) + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (!base->h_ptr) { mfem_error("PtrAlias !base->h_ptr"); }
   if (device && !gpu) // Pull
   {
      mfem::cuMemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   if (!(host && gpu)) { mfem_error("PtrAlias !(host && gpu)"); }
   mfem::cuMemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* DefaultMemoryManager::getPtr(void *ptr)
{
   if (MmDeviceIniFilter()) { return ptr; }
   if (Known(ptr)) { return PtrKnown(maps, ptr); }
   if (Alias(ptr)) { return PtrAlias(maps, ptr); }
   if (config::UsingDevice())
   {
      mfem_error("Trying to use unknown pointer on the DEVICE!");
   }
   return ptr;
}

// *****************************************************************************
OccaMemory DefaultMemoryManager::getOccaPtr(const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   if (!config::UsingMM())
   {
      OccaMemory o_ptr = occaWrapMemory(occaDevice, (void *)ptr, 0);
      return o_ptr;
   }
   const bool known = Known(ptr);
   // if (!known) { BUILTIN_TRAP; }
   if (!known) { mfem_error("occaMemory"); }
   MFEM_ASSERT(known, "Unknown address!");
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const std::size_t bytes = base.bytes;
   const bool gpu = config::UsingDevice();
   MFEM_ASSERT(config::usingOcca(), "Using OCCA memory without OCCA mode!");
   if (!base.d_ptr)
   {
      base.host = false; // This address is no more on the host
      if (gpu)
      {
         cuMemAlloc(&base.d_ptr, bytes);
         void *stream = config::Stream();
         cuMemcpyHtoDAsync(base.d_ptr, base.h_ptr, bytes, stream);
      }
      else
      {
         base.o_ptr = occaDeviceMalloc(occaDevice, bytes);
         base.d_ptr = occaMemoryPtr(base.o_ptr);
         occaCopyFrom(base.o_ptr, base.h_ptr);
      }
   }
   if (gpu)
   {
      return occaWrapMemory(occaDevice, base.d_ptr, bytes);
   }
   return base.o_ptr;
}

// *****************************************************************************

static void PushKnown(DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, base.bytes); }
   mfem::cuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PushAlias(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   mfem::cuMemcpyHtoD(dst, ptr, bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   if (bytes==0) { mfem_error("Push bytes==0"); }
   if (MmDeviceIniFilter()) { return; }
   if (Known(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PushAlias(maps, ptr, bytes); }
   if (config::UsingDevice()) { mfem_error("Unknown pointer to push to!"); }
}

// *****************************************************************************
static void PullKnown(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   assert(base.h_ptr);
   assert(base.d_ptr);
   mfem::cuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PullAlias(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   const bool host = alias->mem->host;
   if (host) { return; }
   if (!ptr) { mfem_error("PullAlias !ptr"); }
   if (!alias->mem->d_ptr) { mfem_error("PullAlias !alias->mem->d_ptr"); }
   mfem::cuMemcpyDtoH(const_cast<void*>(ptr),
                      static_cast<char*>(alias->mem->d_ptr) + alias->offset,
                      bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (Known(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PullAlias(maps, ptr, bytes); }
   if (config::UsingDevice()) { mfem_error("Unknown pointer to pull from!"); }
}

// *****************************************************************************
// * Data will be pushed/pulled before the copy happens on the H or the D
// *****************************************************************************
void DefaultMemoryManager::copyData(void *dst, const void *src, const size_t bytes,
                                    const bool async)
{
   void *d_dst = mm::ptr(dst);
   const void *d_src = mm::ptr(src);
   const bool host = config::UsingHost();
   if (bytes == 0) { return; }
   if (host) { std::memcpy(dst, src, bytes); }
   if (!async)
   {
      mfem::cuMemcpyDtoD(d_dst, const_cast<void*>(d_src), bytes);
   }
   else
   {
      mfem::cuMemcpyDtoDAsync(d_dst, const_cast<void*>(d_src),
                              bytes, config::Stream());
   }
}

// *****************************************************************************
static OccaMemory occaMemory(DefaultMemoryManager::ledger &maps, const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   if (!config::UsingMM())
   {
      OccaMemory o_ptr = occaWrapMemory(occaDevice, const_cast<void*>(ptr), 0);
      return o_ptr;
   }
   const bool known = mm::Known(ptr);
   if (!known) { mfem_error("occaMemory: Unknown address!"); }
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const size_t bytes = base.bytes;
   const bool gpu = config::UsingDevice();
   if (!config::UsingOcca()) { mfem_error("Using OCCA without support!"); }
   if (!base.d_ptr)
   {
      base.host = false; // This address is no more on the host
      if (gpu)
      {
         cuMemAlloc(&base.d_ptr, bytes);
         void *stream = config::Stream();
         cuMemcpyHtoDAsync(base.d_ptr, base.h_ptr, bytes, stream);
      }
      else
      {
         base.o_ptr = occaDeviceMalloc(occaDevice, bytes);
         base.d_ptr = occaMemoryPtr(base.o_ptr);
         occaCopyFrom(base.o_ptr, base.h_ptr);
      }
   }
   if (gpu)
   {
      return occaWrapMemory(occaDevice, base.d_ptr, bytes);
   }
   return base.o_ptr;
}

} // namespace mfem
