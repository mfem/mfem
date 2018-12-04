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

#include "../general/error.hpp"
#include "../general/okina.hpp"
#include "kernels/mm.hpp"
#include "custub.hpp"

namespace mfem
{

// *****************************************************************************
static size_t xs_shift = 0;
static bool xs_shifted = false;
#define MFEM_SIGSEGV_FOR_STACK __builtin_trap()

// *****************************************************************************
static inline void *xsShift(const void *adrs)
{
   if (!xs_shifted) { return (void*) adrs; }
   return ((size_t*) adrs) - xs_shift;
}

// *****************************************************************************
void mm::Setup(void)
{
   assert(!mng);
   // Create our mapping h_adrs => (size, h_adrs, d_adrs)
   mng = new mm_t();
   // Initialize the CUDA device to be ready to allocate memory
   config::Get().Setup();
   // Shift address accesses to trig SIGSEGV
   if ((xs_shifted=getenv("XS"))) { xs_shift = 1ull << 48; }
}

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
void* mm::Insert(const void *adrs,
                 const size_t size, const size_t size_of_T,
                 const char *file, const int line,
                 const void *base)
{
   //dbg("[Insert] %s:%d",file,line);
   if (!mm::Get().mng) { mm::Get().Setup(); }
   size_t *h_adrs = ((size_t *) adrs) + xs_shift;
   const bool known = Known(h_adrs);
   if (known) {
      dbg("[Insert] Known %p",h_adrs);
      mm2dev_t &mm2dev = mng->operator[](h_adrs);
      if (not mm2dev.ranged){
         MFEM_SIGSEGV_FOR_STACK;
         MFEM_ASSERT(false, "[ERROR] Trying to add already present address!");
      }else{
         MFEM_ASSERT(false, "[ERROR] Trying to add already RANGED address!");
      }
   }
   mm2dev_t &mm2dev = mng->operator[](h_adrs);
   mm2dev.host = true;
   mm2dev.bytes = size*size_of_T;
   //dbg("[Insert] Add %p (%ldb): %s:%d",h_adrs,mm2dev.bytes,file,line);
   mm2dev.h_adrs = h_adrs;
   mm2dev.d_adrs = NULL;
   mm2dev.ranged = (base != NULL);
   mm2dev.b_adrs = base;
   mm2dev.n_rangers = 0;
#warning 1024 ranger max
   mm2dev.rangers = (void**)calloc(1024,sizeof(void*));
   return mm2dev.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' rangers
// *****************************************************************************
void *mm::Erase(const void *adrs)
{
   const bool known = Known(adrs);
   if (not known) { MFEM_SIGSEGV_FOR_STACK; }
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
   return xsShift(adrs);
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   const bool cuda = config::Get().Cuda();
   const bool insert_if_in_range = true;
   const bool known = Known(adrs, insert_if_in_range);
   if (not known) { MFEM_SIGSEGV_FOR_STACK; }
   MFEM_ASSERT(known, "[ERROR] Trying to convert unknown address!");
   mm2dev_t &m2d = mng->operator[](adrs);
   
   // Just return asked known host address if not in CUDA mode
   if (m2d.host and not cuda) { return xsShift(m2d.h_adrs); }
   
   // If it hasn't been seen, and we are a ranger,
   // the base should be alloc'ed!
   if (not m2d.d_adrs and m2d.ranged){
      dbg("\033[7mDevice Ranger: @%p < @%p", m2d.b_adrs, m2d.h_adrs);
      // NVCC sanity check
      const bool nvcc = config::nvcc();
      if (not nvcc)
      {
         mfem_error("[ERROR] Trying to run without CUDA support!");
      }
      const void *b_adrs = m2d.b_adrs;
      // First, make sure we have an existing base address
      assert(b_adrs!=NULL); // redondant with if's m2d.ranged test
      // then, make sure base is known, without inserting it
      assert(Known(b_adrs));
      // Let's grab our base map item
      mm2dev_t &base = mng->operator[](b_adrs);
      const size_t base_bytes = base.bytes;
      const size_t range_bytes = m2d.bytes;
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
      m2d.d_adrs = (char*)base.d_adrs + offset;
      // Continue by pushing what we are working on
      void *stream = config::Get().Stream();
      okMemcpyHtoDAsync(m2d.d_adrs, m2d.h_adrs, m2d.bytes, stream);
      m2d.host = false; // Now this address is GPU born
      return m2d.d_adrs;
   }
   
   // If it hasn't been seen, alloc it in the device
   if (not m2d.d_adrs)
   {
      const bool nvcc = config::nvcc();
      if (not nvcc)
      {
         mfem_error("[ERROR] Trying to run without CUDA support!");
      }
      const size_t bytes = m2d.bytes;
      if (bytes>0) { okMemAlloc(&m2d.d_adrs, bytes); }
      void *stream = config::Get().Stream();
      okMemcpyHtoDAsync(m2d.d_adrs, m2d.h_adrs, bytes, stream);
      m2d.host = false; // Now this address is GPU born
      
      if (m2d.n_rangers!=0){
         const size_t n = m2d.n_rangers;
         dbg("\033[31;7m%d ranger(s) ahead!\033[m",n);
         for(size_t k=0;k<n;k+=1){
            dbg("\t%d",k);
            assert(Known(m2d.rangers[k],false));
            mm2dev_t &rng = mng->operator[](m2d.rangers[k]);
            const size_t offset = m2d.bytes - rng.bytes;
            rng.d_adrs = (char*) m2d.d_adrs + offset;
            rng.host = false;
         }
      }      
   }

   // Otherwise, just return known device pointer
   return m2d.d_adrs;
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   MFEM_ASSERT(Known(adrs), "[ERROR] Trying to push an unknown address!");
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host) { return; }
   okMemcpyHtoD(mm2dev.d_adrs, mm2dev.h_adrs, mm2dev.bytes);
}

// *****************************************************************************
void mm::Pull(const void *adrs)
{
   const bool insert_if_in_range = true;
   const bool known = Known(adrs, insert_if_in_range);
   if (not known) { MFEM_SIGSEGV_FOR_STACK; }
   MFEM_ASSERT(known, "[ERROR] Trying to pull an unknown address!");
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host) { return; }
   okMemcpyDtoH(mm2dev.h_adrs, mm2dev.d_adrs, mm2dev.bytes);
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
   const bool cuda = config::Get().Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kH2D(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Get().Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2H(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Get().Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2D(dest, src, bytes, async);
}

} // namespace mfem
