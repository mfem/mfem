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
MFEM_NAMESPACE

// *****************************************************************************
void mm::Setup(void){
   dbg();
   assert(!mng);
   // Create our mapping host => (size, h_adrs, d_adrs)
   mng = new mm_t();
   // Initialize our SIGSEGV handler
   mm::iniHandler();
   // Initialize the CUDA device to be ready to allocate memory
   config::Get().Setup();
}

// *****************************************************************************
// * Add a host address, if we are in CUDA mode, allocate there too
// * Returns the 'instant' one
// *****************************************************************************
void* mm::add(const void *h_adrs, const size_t size, const size_t size_of_T){
   //dbg();
   //stk(true);
   const size_t bytes = size*size_of_T;
   const auto search = mng->find(h_adrs);
   const bool present = search != mng->end();
   
   if (present) { // should not happen
      printf("\n\033[31;7m[mm::add] Trying to add already present %p(%ldo)\033[m", h_adrs, bytes);
      fflush(0);
      assert(false);
   }

   //printf(" \033[31m%p(%ldo)\033[m", h_adrs, bytes);fflush(0);
   mm2dev_t &mm2dev = mng->operator[](h_adrs);
   mm2dev.host = true;
   mm2dev.bytes = bytes;
   mm2dev.h_adrs = h_adrs;
   mm2dev.d_adrs = NULL;
   
   if (config::Get().Cuda()){ // alloc also there
      CUdeviceptr ptr = (CUdeviceptr)NULL;
      const size_t bytes = mm2dev.bytes;
      if (bytes>0){
         //dbg(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }else{
         //dbg(" \033[31;1m%ldo\033[m",bytes);
      }
      mm2dev.d_adrs = (void*)ptr;
      // and say we are there
      mm2dev.host = false;
   }
   return (void*) (mm2dev.host ? mm2dev.h_adrs : mm2dev.d_adrs);
}

// *****************************************************************************
// * Remove the address from the map
// *****************************************************************************
void mm::del(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   if (!present){ // should not happen
      stk(true);
      printf("\n\033[31;7m[mm::del] %p\033[m", adrs);
      assert(false); // should not happen
      return;
   }
   //printf("\n\033[32;7m[mm::del] %p\033[m", adrs);
   // Remove element from the map
   mng->erase(adrs);
}

// *****************************************************************************
bool mm::Known(const void *adrs){
   if (!adrs){dbg("\n\033[31;7m[mm::Known] %p\033[m", adrs);} // NULL
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   return present;
}

// *****************************************************************************
// * 
// *****************************************************************************
void* mm::Adrs(const void *adrs){
   //dbg();
   const bool cuda = config::Get().Cuda();
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();

   // Should look where that comes from
   if (not present) {
      dbg();
      stk(true);
      assert(false);
      return (void*)adrs;
   }
   
   assert(present);
   /*const*/ mm2dev_t &mm2dev = mng->operator[](adrs);
   const size_t bytes = mm2dev.bytes;
   // If we are asking a known host address, just return it
   if (mm2dev.host and not cuda){
      //dbg("Returning host adrs %p\033[m", mm2dev.h_adrs);
      return (void*)mm2dev.h_adrs;
   }
   // Otherwise push it to the device if it hasn't been seen
   //assert(mm2dev.d_adrs);
   if (!mm2dev.d_adrs){
      dbg("\033[32;1mPushing new address to the GPU!\033[m");
      // allocate on the device
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0){
         dbg(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
      mm2dev.d_adrs = (void*)ptr;
      const CUstream s = *config::Get().Stream();
      checkCudaErrors(cuMemcpyHtoDAsync(ptr,mm2dev.h_adrs,bytes,s));
      // Now we are on the GPU
      mm2dev.host = false;
   }
   
   if (not cuda){
      dbg("return \033[31;1mGPU\033[m h_adrs %p",mm2dev.h_adrs);
      dbg("return \033[31;1mGPU\033[m d_adrs %p",mm2dev.d_adrs);
      checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,(CUdeviceptr)mm2dev.d_adrs,bytes));
      mm2dev.host = true;
      return (void*)mm2dev.h_adrs;
   }
   
   dbg("return \033[32;1mGPU\033[m address %p",mm2dev.d_adrs);
   return (void*)mm2dev.d_adrs;
}

// *****************************************************************************
void mm::Rsync(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   assert(present);
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host){
      dbg("Already on host");
      //assert(false);
      return;
   }
   const size_t bytes = mm2dev.bytes;
   checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,
                                (CUdeviceptr)mm2dev.d_adrs,
                                bytes));
}

// *****************************************************************************
void mm::Push(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   assert(present);
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host){
      //dbg("On host");
      return;
   }
   const size_t bytes = mm2dev.bytes;
   checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)mm2dev.d_adrs,
                                (void*)mm2dev.h_adrs,
                                bytes));
}

// **************************************************************************
void* mm::H2H(void *dest, const void *src, size_t bytes, const bool async) {
   dbg();
   if (bytes==0) return dest;
   assert(src); assert(dest);
   std::memcpy(dest,src,bytes);
   return dest;
}

// *************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async) {
   dbg();
   stk(true);
   if (bytes==0) return dest;
   assert(src); assert(dest);
   if (!config::Get().Cuda()) return memcpy(dest,src,bytes);
#ifdef __NVCC__
   if (!config::Get().Uvm()){
      checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)dest,src,bytes));
   }
   else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
   return dest;
}

// ***************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async) {
   dbg();
   if (bytes==0) return dest;
   assert(src); assert(dest);
   if (!config::Get().Cuda()) return memcpy(dest,src,bytes);
#ifdef __NVCC__
   if (!config::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes));
   else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
   return dest;
  }
  
// ***************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async) {
   dbg();//stk(true);
   if (bytes==0) return dest;
   assert(src); assert(dest);
   if (!config::Get().Cuda()) return memcpy(dest,src,bytes);
#ifdef __NVCC__
   if (!config::Get().Uvm()){
      if (!async){
         GET_ADRS(src);
         GET_ADRS(dest);
         //checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
         checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)d_dest,(CUdeviceptr)d_src,bytes));
      }else{
         const CUstream s = *config::Get().Stream();
         checkCudaErrors(cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s));
      }
   } else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
   return dest;
}

// *****************************************************************************
void mm::handler(int nSignum, siginfo_t* si, void* vcontext) {
   fflush(0);
   printf("\n\033[31;7;1mSegmentation fault\033[m\n");
   ucontext_t* context = (ucontext_t*)vcontext;
   context->uc_mcontext.gregs[REG_RIP]++;
   fflush(0);
   exit(!0);
}

// *****************************************************************************
void mm::iniHandler(){
   dbg();
   struct sigaction action;
   memset(&action, 0, sizeof(struct sigaction));
   action.sa_flags = SA_SIGINFO;
   action.sa_sigaction = handler;
   sigaction(SIGSEGV, &action, NULL);
}

// *****************************************************************************
MFEM_NAMESPACE_END
