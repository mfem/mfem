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
#include <list>

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * Memory & Alias setup
// *****************************************************************************
void mm::Setup(){
   if (memories) return;
   memories = new memory_map_t();
   aliases = new alias_map_t();
}

// *****************************************************************************
// * Tests if adrs is a known address
// * If insert_if_in_range is set, it will insert this knew range
// *****************************************************************************
bool mm::Known(const void *adrs)
{
   const bool known = memories->find(adrs) != memories->end();
   if (known) return true;
   return false;
}

// *****************************************************************************
// * Looks if adrs is in one mapping's range
// * Returns base address if it is a hit, NULL otherwise
// *****************************************************************************
const void *mm::Range(const void *adrs)
{
   if (Known(adrs)) return NULL;
   for(memory_map_t::iterator m = memories->begin(); m != memories->end(); m++) {
      const void *b_adrs = m->first;
      if (b_adrs > adrs) continue;
      const void *end = (char*)b_adrs + m->second.bytes;
      if (adrs < end) return b_adrs;
   }
   return NULL;
}

// *****************************************************************************
// * Tests if adrs is an alias address
// *****************************************************************************
bool mm::Alias(const void *adrs)
{
   // Look for an alias
   const bool alias =  aliases->find(adrs) != aliases->end();
   if (alias) return true;
   // Test if it is in a memory range
   const void *base = Range(adrs);
   if (not base) return false;
   assert(base != adrs);
   InsertAlias(base, adrs);
   return true;   
}

// *****************************************************************************
const void* mm::InsertAlias(const void *base, const void *adrs)
{
   alias_t &alias = aliases->operator[](adrs);
   //dbg("\033[32m[Known] Insert new alias: %p < \033[7m%p", base, adrs);
   alias.base = (void*) base;
   alias.adrs = (void*) adrs;
   assert(adrs > base);
   alias.offset = (char*)adrs - (char*)base;
   return adrs;
}

// *****************************************************************************
// * Adds an address 
// * Warning: size can be 0 like from mfem::GroupTopology::Create
// *****************************************************************************
void* mm::Insert(const void *h_adrs,
                 const size_t size, const size_t size_of_T,
                 const void *h_base)
{
   if (not memories or not aliases) { Setup(); }
   assert(not Alias(h_adrs));
   const bool known = Known(h_adrs);
   if (known) {
      BUILTIN_TRAP;
      MFEM_ASSERT(false, "[ERROR] Trying to add already present address!");
   }
   memory_t &mem = memories->operator[](h_adrs);
   mem.host = true;
   mem.bytes = size*size_of_T;
   //dbg("New %p (%ldb)", h_adrs, mem.bytes);
   assert(mem.bytes < 0x1000l);
   mem.h_adrs = (void*) h_adrs;
   mem.d_adrs = NULL;
   return mem.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' rangers
// *****************************************************************************
void *mm::Erase(void *adrs)
{
   //push();
   const bool known = Known(adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to remove an unknown address!");
   memory_t &mem = memories->operator[](adrs);
   //dbg("\033[31mMemory @ %p", mem.h_adrs);
   {  // Scanning aliases to remove the ones using this base
      std::list<void*> aliases_to_remove;
      for(alias_map_t::iterator a = aliases->begin(); a != aliases->end(); a++) {
         const void *base = a->second.base;
         if (adrs != base) continue;
         aliases_to_remove.push_back(a->second.adrs);
      }
      for (void *a : aliases_to_remove) {
         //dbg("\033[31;7mAlias @ %p", a);
         aliases->erase(a);
      }
   }   
   memories->erase(adrs);
   return adrs;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   const bool cuda = config::Cuda();
   //const bool occa = config::Occa();
   const bool known = Known(adrs);
   const bool alias = Alias(adrs);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to convert unknown address!");
   
   const bool host = known ?
      memories->operator[](adrs).host :
      memories->operator[](aliases->operator[](adrs).base).host;
   
   void *h_adrs = (void*) adrs;

   void *b_d_adrs = known ?
      memories->operator[](adrs).d_adrs : 
      memories->operator[](aliases->operator[](adrs).base).d_adrs;
   
   // Just return asked known host address if not in CUDA mode
   if (host and not cuda) {
      //dbg("[H,!C] Returning HOST_@");
      return h_adrs;
   }   
   
   if (not host and cuda and b_d_adrs and known and not alias) {
      //dbg("[!H,C,D] Returning CUDA_@");
      //dbg("[!H,C,D] %p ", b_d_adrs);
      return b_d_adrs;
   }

   if (not host and not cuda and b_d_adrs and known) {
      const size_t bytes = memories->operator[](adrs).bytes;
      //dbg("[!H,!C,D] \033[7mPulling & Returning HOST_@");
      okMemcpyDtoH(h_adrs, b_d_adrs, bytes);
      memories->operator[](adrs).host = true; // Tell base is on CPU
      return h_adrs;
   }
   
   if (host and cuda and b_d_adrs and known) {
      const size_t bytes = memories->operator[](adrs).bytes;
      //dbg("[H,C,D] \033[7mPushing & Returning CUDA_@");
      //dbg("[H,C,D] %p => %p %ld", h_adrs, b_d_adrs, bytes);
      okMemcpyHtoD(b_d_adrs, h_adrs, bytes);
      memories->operator[](adrs).host = false; // Tell base is on GPU
      return b_d_adrs;
   }
   
   // If it hasn't been seen, and we are a alias,
   // the base should be alloc'ed and pushed!
   if (not b_d_adrs and alias){
      dbg("[!D,R]");
      assert(false);
      // NVCC sanity check
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      alias_t &alias = aliases->operator[](adrs);
      dbg("\033[7mDevice Alias: @%p < @%p", alias.base, alias.adrs);
       
      const void *b_adrs = alias.base;
      // First, make sure we have an existing base address
      assert(b_adrs);
      // then, make sure base is known, without inserting it
      assert(Known(b_adrs));
      // Let's grab our base map item
      memory_t &base = memories->operator[](b_adrs);
      const size_t base_bytes = base.bytes;
      // Treat the case base is *NOT* on GPU
      if (not base.d_adrs){
         dbg("Base is \033[7mNOT on GPU");
         assert(base_bytes>0);
         okMemAlloc(&base.d_adrs, base_bytes);
         base.host = false; // Tell base is on GPU
         dbg("Base is now on GPU h_%p => d_%p",base.h_adrs, base.d_adrs);
         okMemcpyHtoD(base.d_adrs, base.h_adrs, base.bytes);
         //Push(h_adrs);
      }else{
         dbg("Base is \033[7malready on GPU @%p", base.d_adrs);
      }
      const size_t offset = (char*)alias.adrs - (char*)alias.base;
      void *result = (void*) ((char*)base.d_adrs + offset);
      dbg("offset %ld (0x%x), returning  @ %p", offset, offset, result);
      return result;
   }
   
   // If it hasn't been seen, alloc it in the device
   const bool device_not_ready = b_d_adrs == NULL;
   if (device_not_ready)
   {
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      memory_t &m = memories->operator[](adrs);
      const size_t bytes = m.bytes;
      // Allocate on device
      if (bytes>0) { okMemAlloc(&m.d_adrs, bytes); }
      //dbg("[!D] %p, bytes=%ld => %p",m.h_adrs,bytes,m.d_adrs);
      assert(m.d_adrs);
      //dbg("[!D] okMemcpyHtoD(%p, %p, %ld)",m.d_adrs, m.h_adrs, m.bytes);
      assert(m.bytes==bytes);
      okMemcpyHtoD(m.d_adrs, m.h_adrs, m.bytes);
      m.host = false; // Now this address is on GPU
      return m.d_adrs;
   }
   
      /*
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
   
   // Otherwise, just return known device pointer
   if (known) {
      dbg("Returning base DEV_@");
      return b_d_adrs;
   }
   //dbg("Returning alias DEV_@");
   assert(alias);
   const alias_t &a = aliases->operator[](adrs);
   assert(a.base);
   assert(a.adrs);
   const size_t offset = (char*)a.adrs - (char*)a.base;
   void *alias_result = (void*) ((char*)b_d_adrs + offset);
   dbg("Returning alias DEV_@ %p",alias_result);
   return alias_result;
}

// *****************************************************************************
void mm::Push(const void *adrs, const size_t given_bytes)
{
   const bool cuda = config::Cuda();
   const bool alias = Alias(adrs);
   const bool known = Known(adrs);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to PUSH an unknown address!");
   
   if (known and not alias){
      if (not cuda) {
         //dbg("known, not CUDA, return");
         return;
      }
      memory_t &base = memories->operator[](adrs);
      if (not base.d_adrs){
         //dbg("Allocating base on GPU");
         const size_t base_bytes = base.bytes;
         assert(base_bytes>0);
         okMemAlloc(&base.d_adrs, base_bytes);
      }
      assert(base.d_adrs);
      if (given_bytes==0){
         const size_t bytes = base.bytes;
         okMemcpyHtoD(base.d_adrs, base.h_adrs, bytes);
      }else{
         okMemcpyHtoD(base.d_adrs, base.h_adrs, given_bytes);
      }
      base.host = false; // Tell base is on GPU
      return;
   }
   
   if (alias){
      if (not cuda) { return; }
      assert(given_bytes > 0);
      const alias_t &a = aliases->operator[](adrs);
      void *b_d_adrs = memories->operator[](a.base).d_adrs;
      assert(b_d_adrs);
      void *a_d_adrs = (void*) ((char*)b_d_adrs + a.offset);
      okMemcpyHtoD(a_d_adrs, a.adrs, given_bytes);
      return;
   }
   
   MFEM_ASSERT(false, "[ERROR] Should not be there!");
/*   if (not mm.host) {
      dbg("\033[33;1mAlready on device!");
      return;
   }
   const bool cuda = config::Cuda();
   if (not cuda){
      dbg("\033[33;1mNo device ready!");
      return;
   }
   dbg("\033[31;1mHtoD");
   if (not mm.d_adrs){
      dbg("\033[31;1mNO @, getting one for you!");
      const void *d_adrs = Adrs(adrs);
      assert(d_adrs);
      MFEM_ASSERT(Known(adrs), "[ERROR] PUSH address!");
      mm = memory->operator[](adrs);
      }
*/
}

// *****************************************************************************
void mm::Pull(const void *adrs, const size_t given_bytes)
{
   const bool cuda = config::Cuda();
   const bool alias = Alias(adrs);
   const bool known = Known(adrs);
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
      okMemcpyDtoH(a.adrs, a_d_adrs, given_bytes);
      return;
   }
   
   MFEM_ASSERT(false, "[ERROR] Should not be there!");
/*
   dbg("\033[31;1mDtoH");
   if (not mm.d_adrs){
      dbg("\033[31;1mNO @, getting one for you!");
      const void *d_adrs = Adrs(adrs);
      assert(d_adrs);
      MFEM_ASSERT(Known(adrs), "[ERROR] PULL address!");
      mm = memory->operator[](adrs);
      }
*/

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
// * Sync:        !cuda  | cuda
// *        host | Push  | Push
// *       !host | Pull  | Pull
// *****************************************************************************
void mm::Sync_p(const void *adrs)
{
   push();
   const bool known = Known(adrs);
   const bool alias = Alias(adrs);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to SYNC an unknown address!");
   if (known){
      const memory_t &m = memories->operator[](adrs);
      if (m.host) { Push(adrs); return; }
      Pull(adrs);
      return;
   }
   assert(alias);
   assert(false);
}

// *****************************************************************************
void* mm::memcpy(void *dest, const void *src, size_t bytes)
{
   //BUILTIN_TRAP;
   //assert(false);
   return mm::D2D(dest, src, bytes, false);
}

// ******************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async)
{
   BUILTIN_TRAP;
   assert(false);
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kH2D(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async)
{
   BUILTIN_TRAP;
   assert(false);
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2H(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async)
{
   //BUILTIN_TRAP;
   //assert(false);
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   assert(false);
   return mfem::kD2D(dest, src, bytes, async);
}

// *****************************************************************************
} // namespace mfem
