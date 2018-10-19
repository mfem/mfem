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

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
static jmp_buf env;
static size_t xs_shift = 0;
static bool test_mem_xs = false;

// *****************************************************************************
void mm::Setup(void)
{
   assert(!mng);
   test_mem_xs = getenv("XS");
   // Create our mapping h_adrs => (size, h_adrs, d_adrs)
   mng = new mm_t();
   // Initialize our SIGSEGV handler
   mm::iniHandler();
   // Initialize the CUDA device to be ready to allocate memory
   config::Get().Setup();
   // We can shift address accesses to trig SIGSEGV (experimental)
   if (test_mem_xs) { xs_shift = 1ull << 48; }
}

// *****************************************************************************
static inline void *shiftBack(const void *adrs){
   if (!test_mem_xs) return (void*) adrs;
   void *shifted_address = ((size_t*) adrs) - xs_shift;
   return  shifted_address;
}

// *****************************************************************************
// * Add an address lazily only on the host
// *****************************************************************************
void* mm::add(const void *adrs, const size_t size, const size_t size_of_T)
{
   size_t *h_adrs = (size_t *) adrs;
   if (test_mem_xs) // Shift host address to force a SIGSEGV
      h_adrs += xs_shift;
   const size_t bytes = size*size_of_T;
   const auto search = mng->find(h_adrs);
   const bool present = search != mng->end();
   if (present)
      mfem_error("[ERROR] Trying to add already present address!");
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
void *mm::del(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   if (not present)
      mfem_error("[ERROR] Trying to remove an unknown address!");
   // Remove element from the map
   mng->erase(adrs);
   return shiftBack(adrs);
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
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();

   if (not present)
      mfem_error("[ERROR] Trying to convert unknown address!");
   
   const bool cuda = config::Get().Cuda();
   mm2dev_t &mm2dev = mng->operator[](adrs);
   
   // If we are asked for a known host address, just return it
   if (mm2dev.host and not cuda)
      return shiftBack(mm2dev.h_adrs);   
   
   // Otherwise push it to the device if it hasn't been seen
   if (not mm2dev.d_adrs)
   {
#ifdef __NVCC__
      const size_t bytes = mm2dev.bytes;
      dbg("\033[32;1mPushing new address to the GPU!\033[m");
      // allocate on the device
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0)
      {
         dbg(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
      mm2dev.d_adrs = (void*)ptr;
      const CUstream s = *config::Get().Stream();
      checkCudaErrors(cuMemcpyHtoDAsync(ptr,mm2dev.h_adrs,bytes,s));
      // Now we are on the GPU
      mm2dev.host = false;
#else
      mfem_error("[ERROR] Trying to run without CUDA support!");
#endif // __NVCC__
   }

   if (not cuda)
   {
      //dbg("return \033[31;1mGPU\033[m h_adrs %p",mm2dev.h_adrs);
      //dbg("return \033[31;1mGPU\033[m d_adrs %p",mm2dev.d_adrs);
#ifdef __NVCC__
      checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,(CUdeviceptr)mm2dev.d_adrs,
                                   mm2dev.bytes));
#else
      assert(false);
#endif // __NVCC__
      mm2dev.host = true;
      return (void*)mm2dev.h_adrs;
   }

   //dbg("return \033[32;1mGPU\033[m address %p",mm2dev.d_adrs);
   return (void*)mm2dev.d_adrs;
}

// *****************************************************************************
void mm::Rsync(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   assert(present);
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host)
   {
      dbg("Already on host");
      return;
   }
#ifdef __NVCC__
   dbg("From GPU");
   const size_t bytes = mm2dev.bytes;
   checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,
                                (CUdeviceptr)mm2dev.d_adrs,
                                bytes));
#endif // __NVCC__
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   assert(present);
   mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host) { return; }
#ifdef __NVCC__
   const size_t bytes = mm2dev.bytes;
   if (not mm2dev.d_adrs){
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0)
      {
         dbg(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
      mm2dev.d_adrs = (void*)ptr;
   }
   checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)mm2dev.d_adrs,
                                (void*)mm2dev.h_adrs,
                                bytes));
#endif // __NVCC__
}

// *****************************************************************************
void* mm::H2H(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   std::memcpy(dest,src,bytes);
   return dest;
}

// ******************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { return memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)dest,src,bytes));
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
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
      checkCudaErrors(cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes));
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
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
         checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)d_dest,(CUdeviceptr)d_src,bytes));
      }
      else
      {
         const CUstream s = *config::Get().Stream();
         checkCudaErrors(cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s));
      }
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
#endif
   return dest;
}

// *****************************************************************************
void* mm::memcpy(void *dest, const void *src, size_t bytes)
{
   return D2D(dest, src, bytes, false);
}

// *****************************************************************************
// *  SIGSEGV handler (not used anymore)
// *****************************************************************************
void mm::handler(int nSignum, siginfo_t* si, void* vcontext)
{
   printf("\n\033[31;7;1mSegmentation fault\033[m\n");
   fflush(0);
}

// *****************************************************************************
// *  SIGSEGV handler that longjmps
// *****************************************************************************
static void SIGSEGV_handler(int s)
{
   if (s==SIGSEGV) { longjmp(env, 1); }
   assert(false);
}

// *****************************************************************************
// *  SIGNALS actions
// *****************************************************************************
void mm::iniHandler()
{
   signal(SIGSEGV, SIGSEGV_handler);
}

// *****************************************************************************
MFEM_NAMESPACE_END
