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

namespace mfem
{

// *****************************************************************************
static size_t xs_shift = 0;
static bool xs_shifted = false;
#define MFEM_SIGSEGV_FOR_STACK {*(size_t*)NULL = 0;}

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
// * Add an address lazily only on the host
// *****************************************************************************
void* mm::Insert(const void *adrs, const size_t size, const size_t size_of_T)
{
   if (!mm::Get().mng) { mm::Get().Setup(); }
   const size_t *h_adrs = ((size_t *) adrs) + xs_shift;
   const size_t bytes = size*size_of_T;
   const auto search = mng->find(h_adrs);
   const bool present = search != mng->end();
   if (present)
   {
      mfem_error("[ERROR] Trying to add already present address!");
   }
   mm2dev_t &mm2dev = mng->operator[](h_adrs);
   mm2dev.host = true;
   mm2dev.bytes = bytes;
   mm2dev.h_adrs = h_adrs;
   mm2dev.d_adrs = NULL;
   return (void*) mm2dev.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map
// *****************************************************************************
void *mm::Erase(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   if (not present)
   {
      MFEM_SIGSEGV_FOR_STACK;
      mfem_error("[ERROR] Trying to remove an unknown address!");
   }
   // Remove element from the map
   mng->erase(adrs);
   return xsShift(adrs);
}

// *****************************************************************************
bool mm::Known(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   return present;
}

// *****************************************************************************
// *
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) MFEM_SIGSEGV_FOR_STACK;
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool cuda = config::Get().Cuda();
   mm2dev_t &mm2dev = mng->operator[](adrs);
   // Just return asked known host address while not in CUDA mode
   if (mm2dev.host and not cuda)
   {
      return xsShift(mm2dev.h_adrs);
   }
   // Otherwise push it to the device if it hasn't been seen
   if (not mm2dev.d_adrs)
   {
#ifdef __NVCC__
      const size_t bytes = mm2dev.bytes;
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0)
      {
         cuMemAlloc(&ptr,bytes);
      }
      mm2dev.d_adrs = (void*)ptr;
      //const CUstream stream = *config::Get().Stream();
      //cuMemcpyHtoDAsync(ptr, mm2dev.h_adrs, bytes, stream);
      cuMemcpyHtoD(ptr, mm2dev.h_adrs, bytes);
      mm2dev.host = false; // Now this address is GPU born
#else
      mfem_error("[ERROR] Trying to run without CUDA support!");
#endif // __NVCC__
   }
   
   if (not cuda)
   {
#ifdef __NVCC__
      cuMemcpyDtoH((void*)mm2dev.h_adrs,
                   (CUdeviceptr)mm2dev.d_adrs, mm2dev.bytes);
#else
      assert(false);
#endif // __NVCC__
      mm2dev.host = true;
      return (void*)mm2dev.h_adrs;
   }
   return (void*)mm2dev.d_adrs;
}

// *****************************************************************************
void mm::Rsync(const void *adrs)
{
   const bool present = Known(adrs);
   MFEM_ASSERT(present, "[ERROR] Trying to rsync from an unknown address!");
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host) { return; }
#ifdef __NVCC__
   cuMemcpyDtoH((void*)mm2dev.h_adrs, (CUdeviceptr)mm2dev.d_adrs, mm2dev.bytes);
#endif // __NVCC__
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   const bool present = Known(adrs);
   MFEM_ASSERT(present, "[ERROR] Trying to push an unknown address!");
   mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host) { return; }
#ifdef __NVCC__
   const size_t bytes = mm2dev.bytes;
   if (not mm2dev.d_adrs)
   {
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0)
      {
         cuMemAlloc(&ptr,bytes);
      }
      mm2dev.d_adrs = (void*)ptr;
   }
   cuMemcpyHtoD((CUdeviceptr)mm2dev.d_adrs,
                (void*)mm2dev.h_adrs, bytes);
#endif // __NVCC__
}

// *****************************************************************************
void mm::Pull(const void *adrs) { Rsync(adrs); }

// ******************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { return memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      cuMemcpyHtoD((CUdeviceptr)dest,src,bytes);
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#endif
   return dest;
}

// *****************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { return memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes);
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#endif
   return dest;
}

// *****************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { return std::memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      if (!async)
      {
         GET_ADRS(src);
         GET_ADRS(dest);
         cuMemcpyDtoD((CUdeviceptr)d_dest,(CUdeviceptr)d_src,bytes);
      }
      else
      {
         const CUstream s = *config::Get().Stream();
         cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s);
      }
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#endif
   return dest;
}

// *****************************************************************************
void* mm::memcpy(void *dest, const void *src, size_t bytes)
{
   return D2D(dest, src, bytes, false);
}

} // namespace mfem
