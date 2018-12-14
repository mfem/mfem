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

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * Tests if adrs is a known address
// *****************************************************************************
static bool Known(const mm_t *maps, const void *adrs)
{
   const auto found = maps->memories->find(adrs);
   const bool known = found != maps->memories->end();
   if (known) return true;
   return false;
}

// *****************************************************************************
// * Looks if adrs is in one mapping's range
// * Returns base address if it is a hit, NULL otherwise
// *****************************************************************************
static const void* IsAlias(const mm_t *maps, const void *adrs)
{
   for(mm_iterator_t mem = maps->memories->begin();
       mem != maps->memories->end(); mem++) {
      const void *b_adrs = mem->first;
      if (b_adrs > adrs) continue;
      const void *end = (char*)b_adrs + mem->second.bytes;
      if (adrs < end) return b_adrs;
   }
   return NULL;
}

// *****************************************************************************
static const void* InsertAlias(const mm_t *maps,
                               const void *base,
                               const void *adrs)
{
   alias_t &alias = maps->aliases->operator[](adrs);
   alias.base = base;
   assert(adrs > base);
   alias.offset = (char*)adrs - (char*)base;
   const size_t offset = alias.offset;
   memory_t &mem = maps->memories->at(base);
   const size_t b_bytes = mem.bytes;
   assert(offset < b_bytes);
   dbg("\033[33m%p < (\033[37m%ld) < \033[33m%p < (\033[37m%ld)",
       base, offset, adrs, b_bytes);
   // Add alias info in the base
   mem.aliases.push_back(adrs);
   return adrs;
}

// *****************************************************************************
// * Tests if adrs is an alias address
// *****************************************************************************
static bool Alias(const mm_t *maps, const void *adrs)
{
   const auto found = maps->aliases->find(adrs);
   const bool alias = found != maps->aliases->end();
   if (alias) return true;
   MFEM_ASSERT(not Known(maps, adrs),
               "[ERROR] Alias is a known base address!");
   // Test if it is in a memory range
   const void *base = IsAlias(maps, adrs);
   if (not base) return false;
   assert(base != adrs);
   InsertAlias(maps, base, adrs);
   return true;   
}

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void* mm::Insert(void *adrs, const size_t bytes)
{
   const bool known = Known(maps, adrs);
   MFEM_ASSERT(not known, "[ERROR] Trying to add already present address!");
   dbg("\033[33m%p \033[35m(%ldb)", adrs, bytes);
   memory_t &mem = memories->operator[](adrs);
   mem.host = true;
   mem.bytes = bytes;
   mem.h_adrs = adrs;
   mem.d_adrs = NULL;
   mem.aliases.clear();
   return adrs;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void *mm::Erase(void *adrs)
{
   const bool known = Known(maps, adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to remove an unknown address!");
   const memory_t &memory = memories->at(adrs);
   dbg(" \033[31m%p \033[35m(%ldb)", mem.h_adrs, mem.bytes);
   for (const void *alias : memory.aliases) {
      dbg("\033[31;7mAlias @ %p", alias);
      aliases->erase(alias);
   }   
   memories->erase(adrs);
   return adrs;
}

// *****************************************************************************
static void* AdrsKnown(const mm_t *maps, void* adrs){
   const bool cuda = config::Cuda();
   memory_t &base = maps->memories->at(adrs);
   const bool host = base.host;
   const bool device = not base.host;
   const size_t bytes = base.bytes;
   void *d_adrs = base.d_adrs;
   if (host and not cuda) { return adrs; }
   if (not d_adrs)
   {
      MFEM_ASSERT(config::Nvcc(), "Trying to run without CUDA support!");
      okMemAlloc(&base.d_adrs, bytes);
      d_adrs = base.d_adrs;
   }
   // Now d_adrs can be used
   if (device and cuda) { return d_adrs; }
   // Pull
   if (device and not cuda)
   {
      okMemcpyDtoH(adrs, d_adrs, bytes);
      base.host = true;
      return adrs;
   }
   // Push
   assert(host and cuda);
   okMemcpyHtoD(d_adrs, adrs, bytes);
   base.host = false;
   return d_adrs;   
}

// *****************************************************************************
static void* AdrsAlias(mm_t *maps, void* adrs){
   const bool cuda = config::Cuda();
   const alias_t alias = maps->aliases->at(adrs);
   memory_t &base = maps->memories->at(alias.base);
   const bool host = base.host;
   const bool device = not base.host;
   const size_t bytes = base.bytes;
   void *d_adrs = base.d_adrs;
   void *h_adrs = base.h_adrs;
   const size_t offset = alias.offset;
   if (host and not cuda) { return adrs; }
   if (not d_adrs)
   {
      MFEM_ASSERT(config::Nvcc(), "Trying to run without CUDA support!");
      okMemAlloc(&base.d_adrs, bytes);
      d_adrs = base.d_adrs;
   }
   void *alias_result = (char*)d_adrs + offset;
   if (device and cuda) { return alias_result; }
   // Pull
   if (device and not cuda)
   {
      okMemcpyDtoH(h_adrs, d_adrs, bytes);
      base.host = true;
      assert(false);
      return adrs;
   }
   // Push
   assert(host and cuda);
   okMemcpyHtoD(d_adrs, h_adrs, bytes);
   base.host = false;   
   return alias_result;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* mm::Adrs(void *adrs)
{
   constexpr bool nvcc = config::Nvcc();
   if (not nvcc) return adrs;
   const bool known = Known(maps, adrs);
   if (known) return AdrsKnown(maps, adrs);
   const bool alias = Alias(maps, adrs);
   if (not alias) { BUILTIN_TRAP; }   
   MFEM_ASSERT(alias, "Unknown address!");
   return AdrsAlias(maps, adrs);
   /*
   //const bool occa = config::Occa();
   // If it hasn't been seen, alloc it in the device
   if (is_not_device_ready and occa)
   {
   assert(false);
   dbg("is_not_device_ready and OCCA");
   const size_t bytes = mm.bytes;
   if (bytes>0) { okMemAlloc(&mm.d_adrs, bytes); }
   void *stream = config::Stream();
   okMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, bytes, stream);
   mm.host = false; // This address is no more on the host
   }
   */
}

// *****************************************************************************
const void* mm::Adrs(const void *adrs){
   return (const void*) Adrs((void*)adrs);
}

// *****************************************************************************
static void PushKnown(mm_t *maps, const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   memory_t &base = maps->memories->at(adrs);
   void *d_adrs = base.d_adrs;
   if (not d_adrs){
      MFEM_ASSERT(config::Nvcc(), "Trying to run without CUDA support!");
      okMemAlloc(&base.d_adrs, base.bytes);
      d_adrs = base.d_adrs;
      assert(d_adrs);
   }
   okMemcpyHtoD(d_adrs, adrs, bytes==0?base.bytes:bytes);
   base.host = false;
}

// *****************************************************************************
static void PushAlias(const mm_t *maps, const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   assert(bytes > 0);
   const alias_t &alias = maps->aliases->at(adrs);
   memory_t &base = maps->memories->at(alias.base);
   void *d_adrs = base.d_adrs;
   assert(d_adrs);
   void *a_d_adrs = (char*)d_adrs + alias.offset;
   okMemcpyHtoD(a_d_adrs, adrs, bytes);
}

// *****************************************************************************
void mm::Push(const void *adrs, const size_t bytes)
{
   const bool known = Known(maps, adrs);
   if (known) return PushKnown(maps, adrs, bytes);
   const bool alias = Alias(maps, adrs);
   if (not alias) { BUILTIN_TRAP; }
   MFEM_ASSERT(alias, "Unknown address!");
   return PushAlias(maps, adrs, bytes);
}

// *****************************************************************************
static void PullKnown(const mm_t *maps, const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   assert(bytes > 0);
   memory_t &base = maps->memories->at(adrs);
   const bool host = base.host;
   const void *d_adrs = base.d_adrs;
   if (host){ return; }
   assert(d_adrs);
   okMemcpyDtoH(base.h_adrs, d_adrs, bytes==0?base.bytes:bytes);
   base.host = true;
}

// *****************************************************************************
static void PullAlias(const mm_t *maps, const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   assert(bytes > 0);
   const alias_t &alias = maps->aliases->at(adrs);
   memory_t &base = maps->memories->at(alias.base);
   void *d_adrs = base.d_adrs;
   assert(d_adrs);
   void *a_d_adrs = (char*)d_adrs + alias.offset;
   okMemcpyDtoH((void*)adrs, a_d_adrs, bytes);
}

// *****************************************************************************
void mm::Pull(const void *adrs, const size_t bytes)
{
   const bool known = Known(maps, adrs);
   if (known) return PullKnown(maps, adrs, bytes);
   const bool alias = Alias(maps, adrs);
   if (not alias) { BUILTIN_TRAP; }
   MFEM_ASSERT(alias, "Unknown address!");
   return PullAlias(maps, adrs, bytes);   /*
   if (config::Occa())
   {
      okCopyTo(Memory(adrs), (void*)mm.h_adrs);
      return;
   }
   MFEM_ASSERT(false, "[ERROR] Should not be there!");
   */
}

// *****************************************************************************
__attribute__((unused))
static OccaMemory Memory(const mm_t *maps, const void *adrs)
{
   const bool present = Known(maps, adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool occa = config::Occa();
   MFEM_ASSERT(occa, "[ERROR] Using OCCA memory without OCCA mode!");
   memory_t &mm = maps->memories->at(adrs);
   const bool cuda = config::Cuda();
   const size_t bytes = mm.bytes;
   OccaDevice device = config::OccaGetDevice();
   if (not mm.d_adrs)
   {
      mm.host = false; // This address is no more on the host
      if (cuda)
      {
         okMemAlloc(&mm.d_adrs, bytes);
         void *stream = config::Stream();
         okMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, bytes, stream);
      }
      else
      {
         mm.o_adrs = okDeviceMalloc(device, bytes);
         mm.d_adrs = okMemoryPtr(mm.o_adrs);
         okCopyFrom(mm.o_adrs, mm.h_adrs);
      }
   }
   if (cuda)
   {
      return okWrapMemory(device, mm.d_adrs, bytes);
   }
   return mm.o_adrs;
}

// *****************************************************************************
__attribute__((unused))
static void Dump(mm_t *maps){
   if (!getenv("DBG")) return;
   memory_map_t *mem = maps->memories;
   alias_map_t  *als = maps->aliases;
   size_t k = 0;
   for(memory_map_t::iterator m = mem->begin(); m != mem->end(); m++) {
      const void *h_adrs = m->first;
      assert(h_adrs == m->second.h_adrs);
      const size_t bytes = m->second.bytes;
      const void *d_adrs = m->second.d_adrs;
      if (not d_adrs){
         printf("\n[%ld] \033[33m%p \033[35m(%ld)", k, h_adrs, bytes);
      }else{
         printf("\n[%ld] \033[33m%p \033[35m (%ld) \033[32 -> %p", k, h_adrs, bytes, d_adrs);
      }
      fflush(0);
      k++;
   }
   k = 0;
   for(alias_map_t::iterator a = als->begin(); a != als->end(); a++) {
      const void *adrs = a->first;
      const size_t offset = a->second.offset;
      const void *base = a->second.base;
      printf("\n[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p",k , base, offset, adrs);
      fflush(0);
      k++;
   }
}

// *****************************************************************************
static void* d2d(void *dst, const void *src, size_t bytes, const bool async){
   GET_ADRS(src);
   GET_ADRS(dst);
   if (not async) { okMemcpyDtoD(d_dst, (void*)d_src, bytes); }
   else { okMemcpyDtoDAsync(d_dst, (void*)d_src, bytes, config::Stream()); }
   return dst;
}

// *****************************************************************************
// * Logic should be looked at to do all H2H, H2D, D2D and D2H
// *****************************************************************************
void* mm::memcpy(void *dst, const void *src, size_t bytes, const bool async)
{
   assert(bytes>0);
   if (bytes==0) { return dst; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dst, src, bytes); }
   return d2d(dst, src, bytes, async);
}

// *****************************************************************************
} // namespace mfem
