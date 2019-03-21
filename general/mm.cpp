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

// *****************************************************************************
// * Tests if ptr is a known address
// *****************************************************************************
static bool Known(const mm::ledger &maps, const void *ptr)
{
   const mm::memory_map::const_iterator found = maps.memories.find(ptr);
   const bool known = found != maps.memories.end();
   if (known) { return true; }
   return false;
}

// *****************************************************************************
bool mm::Known(const void *ptr)
{
   return mfem::Known(maps,ptr);
}

// *****************************************************************************
// * Looks if ptr is an alias of one memory
// *****************************************************************************
static const void* IsAlias(const mm::ledger &maps, const void *ptr)
{
   MFEM_ASSERT(!Known(maps, ptr), "Ptr is an already known address!");
   for (mm::memory_map::const_iterator mem = maps.memories.begin();
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
static const void* InsertAlias(mm::ledger &maps,
                               const void *base,
                               const void *ptr)
{
   mm::memory &mem = maps.memories.at(base);
   const long offset = static_cast<const char*>(ptr) -
      static_cast<const char*> (base);
   const mm::alias *alias = new mm::alias{&mem, offset};
   maps.aliases.emplace(ptr, alias);
#ifdef MFEM_DEBUG_MM
   {
      mem.aliases.sort();
      for (const mm::alias *a : mem.aliases)
      {
         if (a->mem == &mem )
         {
            if (a->offset == offset){
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
static bool Alias(mm::ledger &maps, const void *ptr)
{
   const mm::alias_map::const_iterator found = maps.aliases.find(ptr);
   const bool alias = found != maps.aliases.end();
   if (alias) { return true; }
   const void *base = IsAlias(maps, ptr);
   if (!base) { return false; }
   InsertAlias(maps, base, ptr);
   return true;
}

// *****************************************************************************
bool mm::Alias(const void *ptr)
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
void* mm::Insert(void *ptr, const size_t bytes)
{
   if (!config::UsingMM()) { return ptr; }
   const bool known = Known(ptr);
   if (known)
   {
      mfem_error("Trying to add an already present address!");
   }
   DumpMode();
   // ex1p comes with (bytes==0)
   maps.memories.emplace(ptr, memory(ptr, bytes));
   return ptr;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void *mm::Erase(void *ptr)
{
   if (!config::UsingMM()) { return ptr; }
   const bool known = Known(ptr);
   if (!known) { mfem_error("Trying to erase an unknown pointer!"); }
   memory &mem = maps.memories.at(ptr);
   for (const alias* const alias : mem.aliases)
   {
      maps.aliases.erase(alias);
   }
   mem.aliases.clear();
   maps.memories.erase(ptr);
   return ptr;
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
static void *PtrKnown(mm::ledger &maps, void *ptr)
{
   mm::memory &base = maps.memories.at(ptr);
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
static void *PtrAlias(mm::ledger &maps, void *ptr)
{
   const bool gpu = config::UsingDevice();
   const mm::alias *alias = maps.aliases.at(ptr);
   assert(alias->offset >0);
   const mm::memory *base = alias->mem;
   assert(base);
   const bool host = base->host;
   const bool device = !base->host;
   const size_t bytes = base->bytes;
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
// * If the pointer is NULL the companion pointer
// * will be too.
// *****************************************************************************
void *mm::Ptr(void *ptr)
{
   if (MmDeviceIniFilter()) { return ptr; }
   if (ptr==NULL) { return NULL;};
   if (Known(ptr)) { return PtrKnown(maps, ptr); }
   if (Alias(ptr)) { return PtrAlias(maps, ptr); }
   if (config::UsingDevice())
   {
     mfem_error("Trying to use unknown pointer on the DEVICE!");
   }
   return ptr;
}

// *****************************************************************************
const void *mm::Ptr(const void *ptr)
{
   return static_cast<const void*>(Ptr(const_cast<void*>(ptr)));
}

// *****************************************************************************
static void PushKnown(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   mm::memory &base = maps.memories.at(ptr);
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, base.bytes); }
   mfem::cuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PushAlias(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   const mm::alias *alias = maps.aliases.at(ptr);
   void *dst = static_cast<char*>(alias->mem->d_ptr) + alias->offset;
   mfem::cuMemcpyHtoD(dst, ptr, bytes);
}

// *****************************************************************************
void mm::Push(const void *ptr, const size_t bytes)
{
   if (bytes==0) { mfem_error("Push bytes==0"); }
   if (MmDeviceIniFilter()) { return; }
   if (Known(ptr)) { return PushKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PushAlias(maps, ptr, bytes); }
   if (config::UsingDevice()) { mfem_error("Unknown pointer to push to!"); }
}

// *****************************************************************************
static void PullKnown(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   const mm::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   if (host) { return; }
   assert(base.h_ptr);
   assert(base.d_ptr);
   mfem::cuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PullAlias(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   const mm::alias *alias = maps.aliases.at(ptr);
   const bool host = alias->mem->host;
   if (host) { return; }
   if (!ptr) { mfem_error("PullAlias !ptr"); }
   if (!alias->mem->d_ptr) { mfem_error("PullAlias !alias->mem->d_ptr"); }
   mfem::cuMemcpyDtoH(const_cast<void*>(ptr),
                      static_cast<char*>(alias->mem->d_ptr) + alias->offset,
                      bytes);
}

// *****************************************************************************
void mm::Pull(const void *ptr, const size_t bytes)
{
   if (MmDeviceIniFilter()) { return; }
   if (Known(ptr)) { return PullKnown(maps, ptr, bytes); }
   if (Alias(ptr)) { return PullAlias(maps, ptr, bytes); }
   if (config::UsingDevice()) { mfem_error("Unknown pointer to pull from!"); }
}

// *****************************************************************************
// * Data will be pushed/pulled before the copy happens on the H or the D
// *****************************************************************************
void* mm::memcpy(void *dst, const void *src, const size_t bytes,
                 const bool async)
{
   void *d_dst = mm::ptr(dst);
   const void *d_src = mm::ptr(src);
   const bool host = config::UsingHost();
   if (bytes == 0) return dst;
   if (host) { return std::memcpy(dst, src, bytes); }
   if (!async)
   {
      return mfem::cuMemcpyDtoD(d_dst, const_cast<void*>(d_src), bytes);
   }
   return mfem::cuMemcpyDtoDAsync(d_dst, const_cast<void*>(d_src),
                                  bytes, config::Stream());
}

// *****************************************************************************
static OccaMemory occaMemory(mm::ledger &maps, const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   if (!config::UsingMM())
   {
      OccaMemory o_ptr = occaWrapMemory(occaDevice, const_cast<void*>(ptr), 0);
      return o_ptr;
   }
   const bool known = mm::known(ptr);
   if (!known) { mfem_error("occaMemory: Unknown address!"); }
   mm::memory &base = maps.memories.at(ptr);
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

// *****************************************************************************
void RegisterCheck(void *ptr)
{
  if(ptr != NULL && config::UsingMM()){
    if(!mm::known(ptr))
      {
        assert(0 && "Pointer is not registered!");
        mfem_error("Trying use an unregistered pointer!");
      }
  }
}

// *****************************************************************************
OccaMemory mm::Memory(const void *ptr) { return occaMemory(maps, ptr); }

} // namespace mfem
