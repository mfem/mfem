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

#include "okina.hpp"
#include <bitset>

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
   const bool usingMM = config::usingMM();
   const bool gpuNotDisabled = !config::gpuDisabled();
   const bool gpuHasBeenEnabled = config::gpuHasBeenEnabled();
   return usingMM && gpuHasBeenEnabled && gpuNotDisabled;
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
   if (config::usingMM())
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
      const void *end = (char*)b_ptr + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return NULL;
}

// *****************************************************************************
static const void* InsertAlias(DefaultMemoryManager::ledger &maps,
                               const void *base, const void *ptr)
{
   DefaultMemoryManager::memory &mem = maps.memories.at(base);
   const std::size_t offset = (char *)ptr - (char *)base;
   const DefaultMemoryManager::alias *alias =
      new DefaultMemoryManager::alias{&mem, offset};
   maps.aliases.emplace(ptr, alias);
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
/*static void DumpMode(void)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   static std::bitset<9+1> mode;
   std::bitset<9+1> cfg;
   cfg.set(config::usingMM()?9:0);
   cfg.set(config::gpuHasBeenEnabled()?8:0);
   cfg.set(config::gpuEnabled()?7:0);
   cfg.set(config::gpuDisabled()?6:0);
   cfg.set(config::usingCpu()?5:0);
   cfg.set(config::usingGpu()?4:0);
   cfg.set(config::usingPA()?3:0);
   cfg.set(config::usingCuda()?2:0);
   cfg.set(config::usingOcca()?1:0);
   cfg>>=1;
   if (cfg==mode) { return; }
   mode=cfg;
   dbg("\033[1K\r[0x%x] %sMM %sHasBeenEnabled %sEnabled %sDisabled \
%sCPU %sGPU %sPA %sCUDA %sOCCA", mode.to_ulong(),
       config::usingMM()?"\033[32m":"\033[31m",
       config::gpuHasBeenEnabled()?"\033[32m":"\033[31m",
       config::gpuEnabled()?"\033[32m":"\033[31m",
       config::gpuDisabled()?"\033[32m":"\033[31m",
       config::usingCpu()?"\033[32m":"\033[31m",
       config::usingGpu()?"\033[32m":"\033[31m",
       config::usingPA()?"\033[32m":"\033[31m",
       config::usingCuda()?"\033[32m":"\033[31m",
       config::usingOcca()?"\033[32m":"\033[31m");
       }*/

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void DefaultMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
   const bool known = Known(ptr);
   if (known)
   {
      mfem_error("Trying to insert a non-MM pointer!");
   }
   MFEM_ASSERT(!known, "Trying to add already present address!");
   //dbg("\033[33m%p \033[35m(%ldb)", ptr, bytes);
   //DumpMode();
   maps.memories.emplace(ptr, memory(ptr, bytes));
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void DefaultMemoryManager::removeAddress(void *ptr)
{
   const bool known = Known(ptr);
   if (!known)
   {
      mfem_error("Trying to remove an unknown address!");
   }
   MFEM_ASSERT(known, "Trying to remove an unknown address!");
   memory &mem = maps.memories.at(ptr);
   //dbg("\033[33m %p \033[35m(%ldb)", ptr, mem.bytes);
   for (const alias* const alias : mem.aliases)
   {
      maps.aliases.erase(alias);
   }
   mem.aliases.clear();
   maps.memories.erase(ptr);
}

// *****************************************************************************
static void* PtrKnown(DefaultMemoryManager::ledger &maps, void *ptr)
{
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   const bool device = !host;
   const std::size_t bytes = base.bytes;
   const bool gpu = config::usingGpu();
   if (host && !gpu) { return ptr; }
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, bytes); }
   if (device &&  gpu) { return base.d_ptr; }
   if (device && !gpu) // Pull
   {
      cuMemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   assert(host && gpu);
   cuMemcpyHtoD(base.d_ptr, ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// *****************************************************************************
static void* PtrAlias(DefaultMemoryManager::ledger &maps, void *ptr)
{
   const bool gpu = config::usingGpu();
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   const DefaultMemoryManager::memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (!base->d_ptr) { cuMemAlloc(&alias->mem->d_ptr, bytes); }
   void *a_ptr = (char*)base->d_ptr + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (device && !gpu) // Pull
   {
      assert(base->d_ptr);
      cuMemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   assert(host && gpu);
   cuMemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* DefaultMemoryManager::getPtr(void *ptr)
{
   if (Known(ptr)) { return PtrKnown(maps, ptr); }
   if (Alias(ptr)) { return PtrAlias(maps, ptr); }
   if (config::usingGpu())
   {
      mfem_error("Trying to use unknown pointer on the GPU!");
   }
   return ptr;
}

// *****************************************************************************
OccaMemory DefaultMemoryManager::getOccaPtr(const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   if (!config::usingMM())
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
   const bool gpu = config::usingGpu();
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
   cuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PushAlias(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   cuMemcpyHtoD((char*)alias->mem->d_ptr + alias->offset, ptr, bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   if (Known(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PushAlias(maps, ptr, bytes); }
   mfem_error("Unknown address!");
}

// *****************************************************************************
static void PullKnown(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   cuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PullAlias(const DefaultMemoryManager::ledger &maps,
                      const void *ptr, const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   const bool host = alias->mem->host;
   if (host) { return; }
   cuMemcpyDtoH((void *)ptr, (char*)alias->mem->d_ptr + alias->offset, bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   if (Known(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PullAlias(maps, ptr, bytes); }
   mfem_error("Unknown address!");
}

// *****************************************************************************
void DefaultMemoryManager::copyData(void *dst, const void *src,
                                    const std::size_t bytes, const bool async)
{
   if (config::usingCpu())
   {
      std::memcpy(dst, src, bytes);
   }
   else
   {
      const void *d_src = mm::ptr(src);
      void *d_dst = mm::ptr(dst);
      if (!async)
      {
         cuMemcpyDtoD(d_dst, (void *)d_src, bytes);
      }
      else
      {
         cuMemcpyDtoDAsync(d_dst, (void *)d_src, bytes, config::Stream());
      }
   }
}

} // namespace mfem
