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
// * Looks if ptr is an alias of one memory
// *****************************************************************************
static const void* IsAlias(const mm::ledger &maps, const void *ptr)
{
   MFEM_ASSERT(!Known(maps, ptr), "Ptr is an already known address!");
   for (mm::memory_map::const_iterator mem = maps.memories.begin();
        mem != maps.memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      assert(b_ptr==mem->second.h_ptr);
      if (b_ptr > ptr) { continue; }
      const void *end = (char*)b_ptr + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return NULL;
}

// *****************************************************************************
static const void* InsertAlias(mm::ledger &maps,
                               const void *base,
                               const void *ptr)
{
   mm::memory &mem = maps.memories.at(base);
   const size_t offset = (char *)ptr - (char *)base;
   const mm::alias *alias = new mm::alias{&mem, offset};
   dbg("\033[33m%p < (\033[37m%ld) < \033[33m%p", base, offset, ptr);
   maps.aliases.emplace(ptr, alias);
   { // Sanity checks
      mem.aliases.sort();
      for (const mm::alias *a : mem.aliases)
      {
         if (a->mem == &mem ){
            assert(a->offset != offset);
         }
      }
   }
   // Add this alias to the memory
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
static void DumpMode(void)
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
   if (cfg==mode) return;
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
}

// *****************************************************************************
static void Dump(const mm::ledger &maps)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   const mm::memory_map &mem = maps.memories;
   const mm::alias_map  &als = maps.aliases;
   size_t k = 0;
   size_t l = 0;
   printf("\n\033[35mvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv");
   for (mm::memory_map::const_iterator m = mem.begin(); m != mem.end(); m++)
   {
      const void *h_ptr = m->first;
      assert(h_ptr == m->second.h_ptr);
      const size_t bytes = m->second.bytes;
      const void *d_ptr = m->second.d_ptr;
      if (!d_ptr)
      {
         const bool kB = bytes>1024;
         printf("\n[%ld] \033[33m%p \033[35m(%ld%s)", k, h_ptr,
                kB?bytes/1024:bytes,
                kB?"\033[1mk\033[0;35m":"");
      }
      else
      {
         assert(false);
         printf("\n[%ld] \033[33m%p \033[35m (%ld) \033[32 -> %p",
                k, h_ptr, bytes, d_ptr);
      }
      
      for (const mm::alias *alias : m->second.aliases)
      {
         const size_t offset = alias->offset;
         const void *base = alias->mem->h_ptr;
         assert(base);
         const void *ptr = (char*)base + offset;
         printf("\n\t[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p",
                l, base, offset, ptr);
         assert(((char*)base + offset)==ptr);
         // check
         maps.aliases.at(ptr);
         l++;
      }
      fflush(0);
      k++;
   }
   k = 0;
   for (mm::alias_map::const_iterator a = als.begin(); a != als.end(); a++)
   {
      const void *ptr = a->first;
      const size_t offset = a->second->offset;
      const void *base = a->second->mem->h_ptr;
      assert(base);
      printf("\n[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p",
             k, base, offset, ptr);
      fflush(0);
      assert(((char*)base + offset)==ptr);
      k++;
   }
   printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
   fflush(0);
}

// *****************************************************************************
// * WARNING, as all aliases are not removed, this Assert will fail
// *****************************************************************************
static void Assert(const mm::ledger &maps)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   const mm::memory_map &memories = maps.memories;
   const mm::alias_map  &aliases = maps.aliases;
   size_t nb_mems = 0;
   size_t nb_aliases = 0;
   for (mm::memory_map::const_iterator m = memories.begin(); m != memories.end(); m++)
   {
      const void *h_ptr = m->first;
      assert(h_ptr == m->second.h_ptr);
      //const size_t bytes = m->second.bytes;
      //const void *d_ptr = m->second.d_ptr;
      //for (const mm::alias *alias : m->second.aliases)
      for (auto a = m->second.aliases.begin(); a != m->second.aliases.end(); a++)
      {
         const mm::alias *alias = *a;
         const size_t offset = alias->offset;
         const void *base = alias->mem->h_ptr;
         assert(base);
         const void *ptr = (char*)base + offset;
         //dbg("\n\t[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p", nb_aliases, base, offset, ptr);
         assert(((char*)base + offset)==ptr);
         // check it exists
         maps.aliases.at(ptr);
         nb_aliases++;
      }
      nb_mems++;
   }
   const size_t nb_aliases_in_mems = nb_aliases;
   nb_aliases = 0;
   for (mm::alias_map::const_iterator a = aliases.begin(); a != aliases.end(); a++)
   {
      const void *ptr = a->first;
      const size_t offset = a->second->offset;
      const void *base = a->second->mem->h_ptr;
      assert(base);
#warning no assert(((char*)base + offset)==ptr);
      /*
      if (((char*)base + offset)!=ptr){
         dbg("\033[33m%p ?<? (\033[37m%ld) ?<? \033[33m%p",
             base, offset, ptr);
      }
      assert(((char*)base + offset)==ptr);
      */
      nb_aliases++;
   }
#warning no assert(nb_aliases==nb_aliases_in_mems)
   //assert(nb_aliases==nb_aliases_in_mems);
}

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void* mm::Insert(void *ptr, const size_t bytes)
{
   if (!config::usingMM()) { return ptr; }
   if (config::gpuDisabled()) { return ptr; }
   //Assert(maps);
   const bool known = Known(maps, ptr);
   if (known) { BUILTIN_TRAP; }
   MFEM_ASSERT(!known, "Trying to add already present address!");
   //dbg("\033[33m%p \033[35m(%ldb)", ptr, bytes);
   assert(ptr);
   DumpMode();
   maps.memories.emplace(ptr, memory(ptr, bytes));
   return ptr;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void *mm::Erase(void *ptr)
{
   if (!config::usingMM()) { return ptr; }
   if (config::gpuDisabled()) { return ptr; }
   const bool known = Known(maps, ptr);
   if (!known) { BUILTIN_TRAP; }
   if (!known) { mfem_error("Trying to remove an unknown address!"); }
   MFEM_ASSERT(known, "Trying to remove an unknown address!");
   assert(ptr);
   DumpMode();
   Assert(maps);
   memory &mem = maps.memories.at(ptr);
   //dbg("\033[33m %p \033[35m(%ldb)", ptr, mem.bytes);
   //dbg("\033[33m BEFORE:");
   //Dump(maps);
   //for (const mm::alias* alias : mem.aliases)
   for (auto alias = mem.aliases.begin(); alias != mem.aliases.end(); alias++)
   {
      //const size_t offset = alias->offset;
      //const void *base = alias->mem->h_ptr;
      //assert(base);
      //dbg("\t\033[33m maps.aliases.erase %p <- %ld", base, offset);
      //mem.aliases.erase(alias);
      maps.aliases.erase(*alias);
      //dbg("\t\033[33m delete alias %p", alias);
      //delete *alias;
   }
   //dbg("\033[33m mem.aliases.clear");
   //mem.aliases.clear();
   //dbg("\033[33m maps.memories.erase %p", ptr);
   maps.memories.erase(ptr);
   //Assert(maps);
   //dbg("\033[33m AFTER:");
   //DumpMaps(maps);
   return ptr;
}

// *****************************************************************************
// * Turn a known address to the right host or device one
// * Alloc, Push or Pull it if required
// *****************************************************************************
static void* PtrKnown(mm::ledger &maps, void *ptr)
{
   mm::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   const bool device = !host;
   const size_t bytes = base.bytes;
   const bool gpu = config::usingGpu();
   if (host && !gpu) { return ptr; }
   assert(false);
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
// * Turn an alias to the right host or device one
// * Alloc, Push or Pull it if required
// *****************************************************************************
static void* PtrAlias(mm::ledger &maps, void *ptr)
{
   const bool gpu = config::usingGpu();
   const mm::alias *alias = maps.aliases.at(ptr);
   const mm::memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const size_t bytes = base->bytes;
   assert(base);
   if (not host){
      dbg("\033[1;33m%p < (\033[37m%ld) < \033[33m%p",
          base, alias->offset, ptr);
      Dump(maps);
   }
   assert(host);
   if (host && !gpu) { return ptr; }
   assert(false);
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
void* mm::Ptr(void *ptr)
{
   if (!config::usingMM()) { return ptr; }
   if (config::gpuDisabled()) { return ptr; }
   if (!config::gpuHasBeenEnabled()) { return ptr; }
   if (Known(maps, ptr)) { return PtrKnown(maps, ptr); }
   dbg("\033[1;31m %p asked but unknown\033[35m", ptr);
   const bool alias = Alias(maps, ptr); // Alias always returns true
   if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("Unknown address!"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PtrAlias(maps, ptr);
}

// *****************************************************************************
const void* mm::Ptr(const void *ptr)
{
   return (const void *) Ptr((void *)ptr);
}

// *****************************************************************************
static OccaMemory occaMemory(mm::ledger &maps, const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   if (!config::usingMM())
   {
      OccaMemory o_ptr = occaWrapMemory(occaDevice, (void *)ptr, 0);
      return o_ptr;
   }
   const bool known = Known(maps, ptr);
   // if (!known) { BUILTIN_TRAP; }
   if (!known) { mfem_error("occaMemory"); }
   MFEM_ASSERT(known, "Unknown address!");
   mm::memory &base = maps.memories.at(ptr);
   const size_t bytes = base.bytes;
   const bool gpu = config::usingGpu();
   const bool occa = config::usingOcca();
   MFEM_ASSERT(occa, "Using OCCA memory without OCCA mode!");
   if (!base.d_ptr)
   {
      assert(false);
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
OccaMemory mm::Memory(const void *ptr)
{
   return occaMemory(maps, ptr);
}

// *****************************************************************************
static void PushKnown(mm::ledger &maps, const void *ptr, const size_t bytes)
{
   assert(false);
   mm::memory &base = maps.memories.at(ptr);
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, base.bytes); }
   cuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PushAlias(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   assert(false);
   const mm::alias *alias = maps.aliases.at(ptr);
   cuMemcpyHtoD((char*)alias->mem->d_ptr + alias->offset, ptr, bytes);
}

// *****************************************************************************
void mm::Push(const void *ptr, const size_t bytes)
{
   if (!config::usingMM()) { return; }
   if (!config::gpuEnabled()) { return; }
   if (!config::gpuHasBeenEnabled()) { return; }
   assert(false);
   if (Known(maps, ptr)) { return PushKnown(maps, ptr, bytes); }
   assert(!config::usingOcca());
   const bool alias = Alias(maps, ptr);
   // if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("Unknown address!"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PushAlias(maps, ptr, bytes);
}

// *****************************************************************************
static void PullKnown(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   const mm::memory &base = maps.memories.at(ptr);
   cuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PullAlias(const mm::ledger &maps, const void *ptr,
                      const size_t bytes)
{
   const mm::alias *alias = maps.aliases.at(ptr);
   cuMemcpyDtoH((void *)ptr, (char*)alias->mem->d_ptr + alias->offset, bytes);
}

// *****************************************************************************
void mm::Pull(const void *ptr, const size_t bytes)
{
   if (!config::usingMM()) { return; }
   if (!config::gpuEnabled()) { return; }
   if (!config::gpuHasBeenEnabled()) { return; }
   assert(false);
   if (Known(maps, ptr)) { return PullKnown(maps, ptr, bytes); }
   assert(!config::usingOcca());
   const bool alias = Alias(maps, ptr);
   // if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("Unknown address!"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PullAlias(maps, ptr, bytes);
}

// *****************************************************************************
// * Data will be pushed/pulled before the copy happens on the H or the D
// *****************************************************************************
static void* d2d(void *dst, const void *src, const size_t bytes,
                 const bool async)
{
   const bool cpu = config::usingCpu();
   if (cpu) { return std::memcpy(dst, src, bytes); }
   GET_PTR(src);
   GET_PTR(dst);
   if (!async) { return cuMemcpyDtoD(d_dst, (void *)d_src, bytes); }
   return cuMemcpyDtoDAsync(d_dst, (void *)d_src, bytes, config::Stream());
}

// *****************************************************************************
void* mm::memcpy(void *dst, const void *src, const size_t bytes,
                 const bool async)
{
   if (bytes == 0)
   {
      return dst;
   }
   else
   {
      return d2d(dst, src, bytes, async);
   }
}

} // namespace mfem
