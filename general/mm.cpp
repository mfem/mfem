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
   const auto search = mng->find(adrs);
   const bool known = search != mng->end();
   assert(not known);
   for(mm_t::iterator address = mng->begin(); address != mng->end(); address++) {
      void *h_adrs = address->second.h_adrs;
      if (h_adrs > adrs) continue;
      const size_t bytes = address->second.bytes;
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
   const auto search = mng->find(adrs);
   const bool known = search != mng->end();
   if (known) return true;
   if (not insert_if_in_range) return false;
   // Now inserting this adrs if it is in range
   const void *base = Range(adrs);
   if (not base) return false;
   mm2dev_t &mm2dev_base = mng->operator[](base);
   const size_t bytes = mm2dev_base.bytes;
   assert(0 < bytes);
   assert(base < adrs);
   const size_t offset = (char*) adrs - (char*) base;
   //dbg("\033[32m[Known] Insert %p < %p", base, adrs);
   Insert(adrs,bytes-offset,1,__FILE__,__LINE__,base);
   // Let's grab this new inserted adrs
   mm2dev_t &mm2dev_adrs = mng->operator[](adrs);
   // Double-check he's available, no insertion there
   assert(Known(adrs,false));
   // Adds to the base this new ranger
   mm2dev_base.rangers[mm2dev_base.n_rangers++] = mm2dev_adrs.h_adrs;
   return true;
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
   if (not mng) { mng = new mm_t(); }
   const bool known = Known(h_adrs);
   if (known) {
      dbg("[Insert] Known %p",h_adrs);
      mm2dev_t &mm = mng->operator[](h_adrs);
      if (not mm.ranged){
         BUILTIN_TRAP;
         MFEM_ASSERT(false, "[ERROR] Trying to add already present address!");
      }else{
         MFEM_ASSERT(false, "[ERROR] Trying to add already RANGED address!");
      }
   }
   mm2dev_t &mm = mng->operator[](h_adrs);
   mm.host = true;
   mm.bytes = size*size_of_T;
   //dbg("[Insert] Add %p (%ldb): %s:%d",h_adrs,mm2dev.bytes,file,line);
   mm.h_adrs = (void*) h_adrs;
   mm.d_adrs = NULL;
   mm.ranged = (base != NULL);
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
   mm2dev_t &m2d = mng->operator[](adrs);
   if (m2d.n_rangers!=0){
      const size_t n = m2d.n_rangers;
      //dbg("%d ranger(s) ahead!",n);
      for(size_t k=0;k<n;k+=1){
         //dbg("\t%d",k);
         Erase(m2d.rangers[k]);
      }
   }
   mng->erase(adrs);
   return adrs;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   const bool cuda = config::Cuda();
   const bool occa = config::Occa();
   const bool insert_if_in_range = true;
   const bool known = Known(adrs, insert_if_in_range);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to convert unknown address!");
   mm2dev_t &mm = mng->operator[](adrs);
   
   // Just return asked known host address if not in CUDA mode
   if (mm.host and not cuda) { return mm.h_adrs; }
   
   // If it hasn't been seen, and we are a ranger,
   // the base should be alloc'ed!
   if (not mm.d_adrs and mm.ranged){
      dbg("\033[7mDevice Ranger: @%p < @%p", mm.b_adrs, mm.h_adrs);
      // NVCC sanity check
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      const void *b_adrs = mm.b_adrs;
      // First, make sure we have an existing base address
      assert(b_adrs!=NULL); // redondant with if's mm.ranged test
      // then, make sure base is known, without inserting it
      assert(Known(b_adrs));
      // Let's grab our base map item
      mm2dev_t &base = mng->operator[](b_adrs);
      const size_t base_bytes = base.bytes;
      const size_t range_bytes = mm.bytes;
      assert(base_bytes >= range_bytes);
      const size_t offset = base_bytes - range_bytes;
      dbg("offset=%ld", offset);
      // base should at least already be on GPU!
      //assert(base.d_adrs);
      // Treat the case base is *NOT* on GPU
      if (not base.d_adrs){
         assert(base_bytes>0);
         okMemAlloc(&base.d_adrs, base_bytes);
         base.host = false;
      }
      // update our address range in device space
      mm.d_adrs = (char*)base.d_adrs + offset;
      // Continue by pushing what we are working on
      void *stream = config::Stream();
      okMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, mm.bytes, stream);
      mm.host = false; // Now this address is GPU born
      return mm.d_adrs;
   }
   
   // If it hasn't been seen, alloc it in the device
   const bool is_not_device_ready = mm.d_adrs == NULL;
   if (is_not_device_ready)
   {
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
            dbg("\t%d",k);
            assert(Known(mm.rangers[k],false));
            mm2dev_t &rng = mng->operator[](mm.rangers[k]);
            const size_t offset = mm.bytes - rng.bytes;
            rng.d_adrs = (char*) mm.d_adrs + offset;
            rng.host = false;
         }
      }
   }

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

   // Otherwise, just return known device pointer
   return mm.d_adrs;
}

// *****************************************************************************
memory mm::Memory(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool occa = config::Occa();
   MFEM_ASSERT(occa, "[ERROR] Using OCCA memory without OCCA mode!");
   mm2dev_t &mm = mng->operator[](adrs);
   const bool cuda = config::Cuda();
   const size_t bytes = mm.bytes;
   OCCAdevice device = config::OccaDevice();
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
void mm::Push(const void *adrs)
{
   MFEM_ASSERT(Known(adrs), "[ERROR] Trying to push an unknown address!");
   const mm2dev_t &mm = mng->operator[](adrs);
   if (mm.host) { return; }
   okMemcpyHtoD(mm.d_adrs, mm.h_adrs, mm.bytes);
}

// *****************************************************************************
void mm::Pull(const void *adrs)
{
   const bool insert_if_in_range = true;
   const bool known = Known(adrs, insert_if_in_range);
   if (not known) { BUILTIN_TRAP; }
   MFEM_ASSERT(known, "[ERROR] Trying to pull an unknown address!");
   const mm2dev_t &mm = mng->operator[](adrs);
   if (mm.host) { return; }
   okMemcpyDtoH(mm.h_adrs, mm.d_adrs, mm.bytes);
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
