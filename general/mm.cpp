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
// * Looks if adrs is in one mapping's range
// * Returns base address if it is a hit, NULL otherwise
// *****************************************************************************
void *mm::Range(const void *adrs)
{
   const auto search = memory->find(adrs);
   const bool known = search != memory->end();
   assert(not known);
   for(mm_t::iterator a = memory->begin(); a != memory->end(); a++) {
      void *h_adrs = a->second.h_adrs;
      if (h_adrs > adrs) continue;
      const size_t bytes = a->second.bytes;
      const void *end = (char*)h_adrs + bytes;
      if (adrs <= end) return h_adrs;
   }
   return NULL;
}

// *****************************************************************************
// * Tests if adrs is a known address
// * If insert_if_in_range is set, it will insert this knew range
// *****************************************************************************
bool mm::Known(const void *adrs, const bool insert_if_in_range)
{
   const auto search = memory->find(adrs);
   const bool known = search != memory->end();
   if (known) return true;
   if (not insert_if_in_range) return false;
   const void *base = Range(adrs);
   if (not base) {
      dbg("\033[32m[Known] UNKNOWN");
      return false;
   }
   mm2dev_t &mm2dev_base = memory->operator[](base);
   dbg("\033[32m[Known] New ranger: %p < \033[7m%p", base, adrs);
   const size_t b_bytes = mm2dev_base.bytes;
   dbg("\033[32m[Known] Base of length %ld (0x%x)", b_bytes, b_bytes);
   assert(0 < b_bytes);
   assert(base < adrs);
   const size_t offset = (char*) adrs - (char*) base;
   assert(b_bytes >= offset);
   const size_t rng_sz = b_bytes-offset;
   dbg("\033[32m[Known] Insert at offset: %ld (0x%x), size %ld (0x%x)", offset,offset, rng_sz,rng_sz);
   Insert(adrs, rng_sz, 1, __FILE__, __LINE__, base);
   // Let's grab this new inserted adrs
   mm2dev_t &mm2dev_range = memory->operator[](adrs);
   // Double-check he's available, no insertion there
   assert(Known(adrs,false));
   // Adds to the base this new ranger
   mm2dev_base.rangers[mm2dev_base.n_rangers++] = mm2dev_range.h_adrs;
   return true;
}

// *****************************************************************************
// * Memory & Alias setup
// *****************************************************************************
void mm::Setup(){
   if (memory) return;
   memory = new mm_t();
   alias = new mm_t();
}
// *****************************************************************************
// * Adds an address 
// * Warning: size can be 0 like from mfem::GroupTopology::Create
// *****************************************************************************
void* mm::Insert(const void *h_adrs,
                 const size_t size, const size_t size_of_T,
                 const char *file, const int line,
                 const void *base)
{
   if (not memory or not alias) { Setup(); }
   const bool known = Known(h_adrs);
   if (known) {
      dbg("[Insert] Known %p",h_adrs);
      mm2dev_t &mm = memory->operator[](h_adrs);
      if (not mm.ranged){
         BUILTIN_TRAP;
         MFEM_ASSERT(false, "[ERROR] Trying to add already present address!");
      }else{
         MFEM_ASSERT(false, "[ERROR] Trying to add already RANGED address!");
      }
   }
   mm2dev_t &mm = memory->operator[](h_adrs);
   mm.host = true;
   mm.bytes = size*size_of_T;
   const bool ranger = (base != NULL);
   dbg("%s[Insert] Add %p (%ldb)", ranger?"\033[32m":"",h_adrs, mm.bytes);
   if (ranger) dbg("\033[32m[Insert] RANGER, base @ %p", base);
   mm.h_adrs = (void*) h_adrs;
   mm.d_adrs = NULL;
   mm.ranged = ranger;
   mm.b_adrs = (void*) base;
   mm.n_rangers = 0;
#warning 1024 ranger max
   mm.rangers = (void**)calloc(1024,sizeof(void*));
   return mm.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' rangers
// *****************************************************************************
void *mm::Erase(void *adrs)
{
   const bool known = Known(adrs);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to remove an unknown address!");
   mm2dev_t &m2d = memory->operator[](adrs);
   dbg("\033[31m@ %p", m2d.h_adrs);
   if (m2d.n_rangers!=0){
      const size_t n = m2d.n_rangers;
      for(size_t k=0;k<n;k+=1) {
         dbg("\t\033[31mRanger @ %p", m2d.rangers[k]);
         Erase(m2d.rangers[k]);
      }
   }
   memory->erase(adrs);
   return adrs;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   //push();
   const bool cuda = config::Cuda();
   const bool occa = config::Occa();
   const bool insert_if_in_range = true;
   const bool known = Known(adrs, insert_if_in_range);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to convert unknown address!");
   mm2dev_t &mm = memory->operator[](adrs);
   
   // Just return asked known host address if not in CUDA mode
   if (mm.host and not cuda) {
      dbg("[H,!C] Returning HOST_@");
      return mm.h_adrs;
   }
   
   if (not mm.host and not cuda and mm.d_adrs) {
      dbg("[!H,!C,D] Returning HOST_@");
      Pull(adrs);
      return mm.h_adrs;
   }
   
   if (not mm.host and cuda and mm.d_adrs) {
      dbg("[!H,C,D] Returning CUDA_@");
      return mm.d_adrs;
   }
   
   if (mm.host and cuda and mm.d_adrs) {
      dbg("[H,C,D] Push & Returning CUDA_@");
      Push(adrs);
      return mm.d_adrs;
   }
   
   // If it hasn't been seen, and we are a ranger,
   // the base should be alloc'ed and pushed!
   if (not mm.d_adrs and mm.ranged){
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
      mm2dev_t &base = memory->operator[](b_adrs);
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
      //void *stream = config::Stream();
      okMemcpyHtoD/*Async*/(mm.d_adrs, mm.h_adrs, mm.bytes);//, stream);
      mm.host = false; // Now this ranger is also on GPU
      return mm.d_adrs;
   }
   
   // If it hasn't been seen, alloc it in the device
   const bool is_not_device_ready = mm.d_adrs == NULL;
   //if (is_not_device_ready)
   if (not mm.d_adrs and not mm.ranged)
   {
      dbg("[!D] Alloc + CPY, h_adrs \033[7m%p", mm.h_adrs);
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      const size_t bytes = mm.bytes;
      dbg("bytes=%ld",bytes);
      if (bytes>0) { okMemAlloc(&mm.d_adrs, bytes); }
      assert(mm.d_adrs);
      //void *stream = config::Stream();
      okMemcpyHtoD/*Async*/(mm.d_adrs, mm.h_adrs, bytes);//, stream);
      mm.host = false; // Now this address is GPU born
      if (mm.n_rangers!=0){
         const size_t n = mm.n_rangers;
         dbg("\033[31;7m%d ranger(s) ahead!\033[m",n);
         for(size_t k=0;k<n;k+=1){
            dbg("   Updating ranger %d/%d",k+1,n);
            assert(Known(mm.rangers[k]));
            mm2dev_t &rng = memory->operator[](mm.rangers[k]);
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

   // If it hasn't been seen, alloc it in the device
   if (is_not_device_ready and occa)
   {
      assert(false);
      dbg("is_not_device_ready and OCCA");
      const size_t bytes = mm.bytes;
      if (bytes>0) { okMemAlloc(&mm.d_adrs, bytes); }
      //void *stream = config::Stream();
      okMemcpyHtoD/*Async*/(mm.d_adrs, mm.h_adrs, bytes);//, stream);
      mm.host = false; // This address is no more on the host
   }

   // Otherwise, just return known device pointer
   dbg("Returning DEV_@");
   return mm.d_adrs;
}

// *****************************************************************************
OccaMemory mm::Memory(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool occa = config::Occa();
   MFEM_ASSERT(occa, "[ERROR] Using OCCA memory without OCCA mode!");
   mm2dev_t &mm = memory->operator[](adrs);
   const bool cuda = config::Cuda();
   const size_t bytes = mm.bytes;
   OccaDevice device = config::OccaDevice();
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
   const bool known = Known(adrs,true);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to SYNC an unknown address!");
   const mm2dev_t &m = memory->operator[](adrs);
   if (m.host) { Push(adrs); return; }
   Pull(adrs);
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   push();
   const bool known = Known(adrs,true);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to PUSH an unknown address!");
   mm2dev_t &mm = memory->operator[](adrs);
   okMemcpyHtoD(mm.d_adrs, mm.h_adrs, mm.bytes);
   mm.host = false;
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
   const bool known = Known(adrs,true);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to PULL an unknown address!");
   mm2dev_t &mm = memory->operator[](adrs);
   okMemcpyDtoH(mm.h_adrs, mm.d_adrs, mm.bytes);
   mm.host = true;
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
