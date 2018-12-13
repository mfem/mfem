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
// * Memory & Alias setup
// *****************************************************************************
void mm::Setup(){
   if (memories) return;
   assert(not aliases);
   memories = new memory_map_t();
   aliases = new alias_map_t();
}

// *****************************************************************************
// * Tests if adrs is a known address
// *****************************************************************************
bool mm::Known(const void *adrs)
{
   const auto found = memories->find(adrs);
   const bool known = found != memories->end();
   if (known) return true;
   return false;
}

// *****************************************************************************
// * Tests if adrs is an alias address
// *****************************************************************************
bool mm::Alias(const void *adrs)
{
   const auto found = aliases->find(adrs);
   const bool alias = found != aliases->end();
   if (alias) return true;
   // Make sure it is not a known address
   MFEM_ASSERT(not Known(adrs), "[ERROR] Alias is a known base address!");
   // Test if it is in a memory range
   const void *base = Range(adrs);
   if (not base) return false;
   assert(base != adrs);
   InsertAlias(base, adrs);
   return true;   
}

// *****************************************************************************
// * Looks if adrs is in one mapping's range
// * Returns base address if it is a hit, NULL otherwise
// *****************************************************************************
const void *mm::Range(const void *adrs)
{
   for(memory_map_t::iterator m = memories->begin(); m != memories->end(); m++) {
      const void *b_adrs = m->first;
      if (b_adrs > adrs) continue;
      const void *end = (char*)b_adrs + m->second.bytes;
      if (adrs < end) return b_adrs;
   }
   return NULL;
}

// *****************************************************************************
const void* mm::InsertAlias(const void *base, const void *adrs)
{
   alias_t &alias = aliases->operator[](adrs);
   alias.base = base;
   alias.adrs = adrs;
   assert(adrs > base);
   alias.offset = (char*)adrs - (char*)base;
   const size_t offset = alias.offset;
   memory_t &mem = memories->at(base);
   const size_t b_bytes = mem.bytes;
   assert(offset < b_bytes);
   dbg("\033[33m%p < (\033[37m%ld) < \033[33m%p < (\033[37m%ld)",
       base, offset, adrs, b_bytes);
   // Add alias info in the base
   mem.aliases.push_back(adrs);
   return adrs;
}

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void* mm::Insert(void *adrs, const size_t bytes)
{
   if (not memories) { Setup(); }
   const bool known = Known(adrs);
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
   const bool known = Known(adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to remove an unknown address!");
   const memory_t &mem = memories->at(adrs);
   dbg(" \033[31m%p \033[35m(%ldb)", mem.h_adrs, mem.bytes);
   // Removing aliases of this base
   for (const void *a : mem.aliases) {
      dbg("\033[31;7mAlias @ %p", a);
      aliases->erase(a);
   }   
   memories->erase(adrs);
   return adrs;
}

// *****************************************************************************
void* mm::AdrsKnown(void* adrs){
   const bool cuda = config::Cuda();
   memory_t &base = memories->at(adrs);
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
void* mm::AdrsAlias(void* adrs){
   const bool cuda = config::Cuda();
   const alias_t alias = aliases->at(adrs);
   memory_t &base = memories->at(alias.base);
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
   const bool known = Known(adrs);
   if (known) return AdrsKnown(adrs);
   const bool alias = Alias(adrs);
   if (not alias) { BUILTIN_TRAP; }   
   MFEM_ASSERT(alias, "Unknown address!");
   return AdrsAlias(adrs);
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
void mm::PushKnown(const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   memory_t &base = memories->at(adrs);
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
void mm::PushAlias(const void *adrs, const size_t bytes){
   const bool cuda = config::Cuda();
   if (not cuda) { return; }
   assert(bytes > 0);
   const alias_t &alias = aliases->at(adrs);
   memory_t &base = memories->at(alias.base);
   void *d_adrs = base.d_adrs;
   assert(d_adrs);
   void *a_d_adrs = (char*)d_adrs + alias.offset;
   okMemcpyHtoD(a_d_adrs, adrs, bytes);
}

// *****************************************************************************
void mm::Push(const void *adrs, const size_t bytes)
{
   const bool cuda = config::Cuda();
   const bool known = Known(adrs);
   if (known) return PushKnown(adrs, bytes);
   const bool alias = Alias(adrs);
   if (not alias) { BUILTIN_TRAP; }
   MFEM_ASSERT(alias, "Unknown address!");
   return PushAlias(adrs, bytes);
}

// *****************************************************************************
void mm::Pull(const void *adrs, const size_t given_bytes)
{
   const bool cuda = config::Cuda();
   const bool known = Known(adrs);
   const bool alias = known ? false : Alias(adrs);
   assert(alias ^ known);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to PULL an unknown address!");
      
   if (known and not alias){
      if (not cuda) {
         // dbg("known, not CUDA, return");
         return;
      }
      memory_t &base = memories->operator[](adrs);      
      const size_t bytes = given_bytes >0 ? given_bytes :base.bytes;
      const bool host = base.host;
      if (host and not base.d_adrs){
         //dbg("host, known, not base.d_adrs, return");
         return;
      }
      if (not host){
         //dbg("known, not host, okMemcpyDtoH & return");
         assert(base.d_adrs);
         okMemcpyDtoH(base.h_adrs, base.d_adrs, bytes);
         base.host = true;
         return;
      }else{
         //dbg("known, base.d_adrs, but in host, Nothing to do...");
         //assert(false);
         return;
      }
   }

   if (alias){
      if (not cuda) { return; }
      assert(given_bytes > 0);
      const alias_t &a = aliases->operator[](adrs);
      void *b_d_adrs = memories->operator[](a.base).d_adrs;
      assert(b_d_adrs);
      void *a_d_adrs = (void*) ((char*)b_d_adrs + a.offset);
      //dbg("alias, cuda, okMemcpyDtoH & return");
      okMemcpyDtoH((void*)a.adrs, a_d_adrs, given_bytes);
      return;
   }
   
   MFEM_ASSERT(false, "[ERROR] Should not be there!");

/*
   if (config::Cuda())
   {
      okMemcpyDtoH((void*)mm.h_adrs, mm.d_adrs, mm.bytes);
      return;
   }
   if (config::Occa())
   {
      okCopyTo(Memory(adrs), (void*)mm.h_adrs);
      return;
      }
   MFEM_ASSERT(false, "[ERROR] Should not be there!");
*/
}

// *****************************************************************************
OccaMemory mm::Memory(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool occa = config::Occa();
   MFEM_ASSERT(occa, "[ERROR] Using OCCA memory without OCCA mode!");
   memory_t &mm = memories->operator[](adrs);
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
void mm::Dump(){
   if (!getenv("DBG")) return;
   memory_map_t *mem = memories;
   alias_map_t  *als = aliases;
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
      assert(adrs == a->second.adrs);
      const size_t offset = a->second.offset;
      const void *base = a->second.base;
      printf("\n[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p",k , base, offset, adrs);
      fflush(0);
      k++;
   }
}

// *****************************************************************************
} // namespace mfem
