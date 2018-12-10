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
// * Looks if adrs is in one mapping's range
// * Returns base address if it is a hit, NULL otherwise
// *****************************************************************************
void *mm::Range(const void *adrs)
{
   {
      const auto found = memories->find(adrs);
      const bool known = found != memories->end();
      assert(not known);
   }
   for(memory_map_t::iterator a = memories->begin(); a != memories->end(); a++) {
      const void *h_adrs = a->first;
      if (h_adrs > adrs) continue;
      const size_t bytes = a->second.bytes;
      const void *end = (char*)h_adrs + bytes;
      if (adrs <= end) return a->second.h_adrs;
   }
   return NULL;
}

// *****************************************************************************
// * Tests if adrs is an alias address
// *****************************************************************************
bool mm::Alias(const void *adrs, const bool insert_if_is_alias){
   push();
   // Look for an alias
   const bool alias =  aliases->find(adrs) != aliases->end();
   if (alias) return true;
   // If we are not asked to add it, just return we don't know this address
   if (not insert_if_is_alias) return false;
   // Add this alias
   const void *base = Range(adrs);
   // Its base address should be valid
   assert(base);
   dbg("\033[32m[Known] Found new alias: %p < \033[7m%p", base, adrs);
   assert(base <= adrs);
   InsertAlias(base, adrs);
   return true;   
}

// *****************************************************************************
// * Tests if adrs is a known address
// * If insert_if_in_range is set, it will insert this knew range
// *****************************************************************************
bool mm::Known(const void *adrs)
{
   push();
   const bool known = memories->find(adrs) != memories->end();
   if (known) return true;
   return false;
}

// *****************************************************************************
const void* mm::InsertAlias(const void *base, const void *adrs){
   push();
   alias_t &alias = aliases->operator[](adrs);
   dbg("New alias @ %p", adrs);
   alias.base = (void*) base;
   alias.adrs = (void*) adrs;
   return adrs;
}

// *****************************************************************************
// * Adds an address 
// * Warning: size can be 0 like from mfem::GroupTopology::Create
// *****************************************************************************
void* mm::Insert(const void *h_adrs,
                 const size_t size, const size_t size_of_T,
                 const char *file, const int line,
                 const void *h_base)
{
   push();
   if (not memories or not aliases) { Setup(); }
   if (h_base) {
      dbg("=> InsertAlias");
      return (void*) InsertAlias(h_adrs, h_base);
   }
   const bool known = Known(h_adrs);
   if (known) {
      BUILTIN_TRAP;
      MFEM_ASSERT(false, "[ERROR] Trying to add already present address!");
   }
   memory_t &mem = memories->operator[](h_adrs);
   mem.host = true;
   mem.bytes = size*size_of_T;
   dbg("New %p (%ldb)", h_adrs, mem.bytes);
   mem.h_adrs = (void*) h_adrs;
   mem.d_adrs = NULL;
   return mem.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' rangers
// *****************************************************************************
void *mm::Erase(void *adrs)
{
   push();
   const bool known = Known(adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to remove an unknown address!");
   memory_t &mem = memories->operator[](adrs);
   dbg("\033[31m@ %p", mem.h_adrs);
   {  // Scanning aliases to remove the ones using this base
      std::list<void*> aliases_to_remove;
      for(alias_map_t::iterator a = aliases->begin(); a != aliases->end(); a++) {
         const void *base = a->second.base;
         if (adrs != base) continue;
         aliases_to_remove.push_back(a->second.adrs);
      }
      for (void *a : aliases_to_remove) {
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
   push();
   const bool cuda = config::Cuda();
   //const bool occa = config::Occa();
   const bool known = Known(adrs);
   const bool alias = known ? false : Alias(adrs,true);
   
   if (not known and not alias) { BUILTIN_TRAP; }
   MFEM_ASSERT(known or alias, "[ERROR] Trying to convert unknown address!");
 
   const bool host = known ?
      memories->operator[](adrs).host :
      memories->operator[](aliases->operator[](adrs).base).host ;
   
   void *h_adrs = (void*) adrs;
   
   void *d_adrs = known ?
      memories->operator[](adrs).d_adrs :
      memories->operator[](aliases->operator[](adrs).base).d_adrs ;
   
   // Just return asked known host address if not in CUDA mode
   if (host and not cuda) {
      dbg("[H,!C] Returning HOST_@");
      return h_adrs;
   }
   
   
   if (not host and not cuda and d_adrs) {
      dbg("[!H,!C,D] Returning HOST_@");
      Pull(adrs);
      return h_adrs;
   }
   
   
   if (not host and cuda and d_adrs) {
      dbg("[!H,C,D] Returning CUDA_@");
      return d_adrs;
   }
   /*
   if (mm.host and cuda and mm.d_adrs) {
      dbg("[H,C,D] Push & Returning CUDA_@");
      Push(adrs);
      return mm.d_adrs;
   }
   */
   
   // If it hasn't been seen, and we are a ranger,
   // the base should be alloc'ed and pushed!
   /*if (not mm.d_adrs and mm.ranged){
      dbg("[!D,R] Base Alloc + CPY");
      dbg("\033[7mDevice Ranger: @%p < @%p", mm.b_adrs, mm.h_adrs);
      // NVCC sanity check
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      const void *b_adrs = mm.b_adrs;
      // First, make sure we have an existing base address
      assert(b_adrs); // redondant with if's mm.ranged test
      // then, make sure base is known, without inserting it
      assert(Known(b_adrs));
      // Let's grab our base map item
      memory_t &base = memory->operator[](b_adrs);
      const size_t base_bytes = base.bytes;
      const size_t range_bytes = mm.bytes;
      assert(base_bytes >= range_bytes);
      const size_t offset = base_bytes - range_bytes;
      dbg("offset: %ld, 0x%x", offset, offset);
      // base should at least already be on GPU!
      //assert(base.d_adrs);
      // Treat the case base is *NOT* on GPU
      if (not base.d_adrs){
         dbg("Base is \033[7mNOT on GPU");
         assert(base_bytes>0);
         okMemAlloc(&base.d_adrs, base_bytes);
         base.host = false; // Tell base is on GPU
         // We are the only ranger of this base
         assert(base.n_rangers==1);
         dbg("Base is now on GPU h_%p => d_%p",base.h_adrs, base.d_adrs);
      }else{
         dbg("Base is \033[7malready on GPU @%p", base.d_adrs);
      }
      // update our address range in device space
      mm.d_adrs = (char*)base.d_adrs + offset;
      dbg("Ranger is now on GPU %p",mm.d_adrs);
      // Continue by pushing what we are working on
      void *stream = config::Stream();
      okMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, mm.bytes, stream);
      mm.host = false; // Now this ranger is also on GPU
      return mm.d_adrs;
   }*/

   /*
   // If it hasn't been seen, alloc it in the device
   const bool is_not_device_ready = mm.d_adrs == NULL;
   //if (is_not_device_ready)
   if (not mm.d_adrs// and not mm.ranged)
   {
      dbg("[!D] Alloc + CPY, h_adrs \033[7m%p", mm.h_adrs);
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      const size_t bytes = mm.bytes;
      dbg("bytes=%ld",bytes);
      if (bytes>0) { okMemAlloc(&mm.d_adrs, bytes); }
      assert(mm.d_adrs);
      void *stream = config::Stream();
      okMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, bytes, stream);
      mm.host = false; // Now this address is GPU born
      if (mm.n_rangers!=0){
         const size_t n = mm.n_rangers;
         dbg("\033[31;7m%d ranger(s) ahead!\033[m",n);
         for(size_t k=0;k<n;k+=1){
            dbg("   Updating ranger %d/%d",k+1,n);
            assert(Known(mm.rangers[k]));
            memory_t &rng = memory->operator[](mm.rangers[k]);
            dbg("ranger %p", rng.h_adrs);
            assert(mm.h_adrs == adrs);
            assert(rng.h_adrs >= mm.h_adrs);
            assert(rng.d_adrs >= mm.d_adrs);
            assert(mm.bytes >= rng.bytes);
            const size_t offset = mm.bytes - rng.bytes;
            dbg("offset: %ld, 0x%x", offset, offset);
            // If this ranger is already on GPU
            // check that we are good
            void *n_adrs = (char*) mm.d_adrs + offset;
            if (rng.d_adrs){
               dbg("Ranger is  already on GPU @%p", rng.d_adrs);
               dbg("  Base is          on GPU @%p", mm.d_adrs);
               dbg("  mm.d_adrs + offset is   @%p", n_adrs);
               assert(rng.d_adrs == n_adrs);
            }
            rng.d_adrs = n_adrs;
            rng.host = false;
         }
         }
         return mm.d_adrs;
   }
   */
   
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
   dbg("Returning DEV_@");
   return d_adrs;
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
   const bool cuda = config::Cuda();
   if (not cuda) return;
   const bool known = Known(adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to SYNC an unknown address!");
   const memory_t &m = memories->operator[](adrs);
   if (m.host) { Push(adrs); return; }
   Pull(adrs);
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   push();
   const bool known = Known(adrs);
   const bool alias = known ? false : Alias(adrs,true);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to PUSH an unknown address!");
   if (known){
      memory_t &mem = memories->operator[](adrs);
      const bool host = mem.host;
      if (not host /*and cuda*/){
         okMemcpyHtoD(mem.d_adrs, mem.h_adrs, mem.bytes);
         mem.host = false;
      }
      return;
   }
   if (alias){
      assert(false);
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
void mm::Pull(const void *adrs)
{
   push();
   //const bool cuda = config::Cuda();
   const bool known = Known(adrs);
   const bool alias = known ? false : Alias(adrs,true);
   const bool unknown = not known and not alias;
   if (unknown) { BUILTIN_TRAP; }
   MFEM_ASSERT(not unknown, "[ERROR] Trying to PULL an unknown address!");
   if (known){
      memory_t &mem = memories->operator[](adrs);
      const bool host = mem.host;
      if (not host /*and cuda*/){
         okMemcpyDtoH(mem.h_adrs, mem.d_adrs, mem.bytes);
         mem.host = true;
      }
      return;
   }
   if (alias){
      assert(false);
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
void* mm::memcpy(void *dest, const void *src, size_t bytes)
{
   return mm::D2D(dest, src, bytes, false);
}

// ******************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kH2D(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2H(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2D(dest, src, bytes, async);
}

// *****************************************************************************
} // namespace mfem
