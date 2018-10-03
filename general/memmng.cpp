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
void mm::init(void){
   dbg();
   assert(!mng);
   // Create our mapping host => (size, h_adrs, d_adrs)
   mng = new mm_t();
   // Initialize our SIGSEGV handler
   mm::iniHandler();
   // Initialize the CUDA device to be ready to allocate memory there
   config::Get().Init();
}

// *****************************************************************************
void* mm::add(const void *h_adrs, const size_t size, const size_t size_of_T){
   stk(true);
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
   
   // if config::Get().Cuda() is set, alloc also there
   if (config::Get().Cuda()){
      CUdeviceptr ptr = (CUdeviceptr)NULL;
      const size_t bytes = mm2dev.bytes;
      if (bytes>0){
         //printf(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
         
      mm2dev.d_adrs = (void*)ptr;
      // and say we are there
      mm2dev.host = false;
   }
   return (void*) (mm2dev.host ? mm2dev.h_adrs : mm2dev.d_adrs);
}

// *****************************************************************************
void mm::del(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   
   if (!present){ // should not happen
      printf("\n\033[32m[mm::del] %p\033[m", adrs);
      assert(false); // should not happen
   }
   
   //printf("\n\033[32;7m[mm::del] %p\033[m", adrs);
   // Remove element from the map
   mng->erase(adrs);
}

// *****************************************************************************
// * HOST => GPU monolithic transfer
// *****************************************************************************
void mm::Cuda(){
   assert(false);
   for(auto it = mng->begin(); it != mng->end(); ++it){
      const void *adrs = it->first;
      mm2dev_t &mm2dev = it->second;
      //dbg("\033[32;1madrs=%p",adrs);
      assert(adrs == mm2dev.h_adrs);
      
      // Now allocate on the device
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      const size_t bytes = mm2dev.bytes;
      if (bytes>0){
         //printf(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
      mm2dev.d_adrs = (void*)ptr;
      
      //dbg("\033[32;1m =>");
      //memcpy::H2D(ptr,mm2dev.h_adrs,bytes);
      //checkCudaErrors(cuMemcpyHtoD(ptr,mm2dev.h_adrs,bytes));
      
      const CUstream s = *config::Get().Stream();
      checkCudaErrors(cuMemcpyHtoDAsync(ptr,mm2dev.h_adrs,bytes,s));
      // Now we are on the GPU
      mm2dev.host = false;
   }
}

// *****************************************************************************
bool mm::Known(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   return present;
}

// *****************************************************************************
void* mm::Adrs(const void *adrs){
   dbg();
   const bool cuda = config::Get().Cuda();
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();

   // Should look where that comes from
   if (not present) return (void*)adrs;
   
   assert(present);
   /*const*/ mm2dev_t &mm2dev = mng->operator[](adrs);
   const size_t bytes = mm2dev.bytes;
   // If we are asking a known host address, just return it
   if (mm2dev.host and not cuda){
      dbg("Returning host adrs %p", mm2dev.h_adrs);
      return (void*)mm2dev.h_adrs;
   }
   // Otherwise push it to the device if it hasn't been seen
   //assert(mm2dev.d_adrs);
   if (!mm2dev.d_adrs){
      dbg("\033[32;1mPushing new address to the GPU!");
      // allocate on the device
      CUdeviceptr ptr = (CUdeviceptr) NULL;
      if (bytes>0){
         dbg(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
      }
      mm2dev.d_adrs = (void*)ptr;
      //memcpy::H2D((void*)ptr,mm2dev.h_adrs,bytes);
      const CUstream s = *config::Get().Stream();
      checkCudaErrors(cuMemcpyHtoDAsync(ptr,mm2dev.h_adrs,bytes,s));
      // Now we are on the GPU
      mm2dev.host = false;
   }
   
   if (not cuda){
      dbg("return \033[31;1mGPU\033[m h_adrs %p",mm2dev.h_adrs);
      dbg("return \033[31;1mGPU\033[m d_adrs %p",mm2dev.d_adrs);
      checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,(CUdeviceptr)mm2dev.d_adrs,bytes));
      //const CUstream s = *config::Get().Stream();
      //checkCudaErrors(cuMemcpyDtoHAsync((void*)mm2dev.h_adrs, (CUdeviceptr)mm2dev.d_adrs, bytes, s));
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
   if (mm2dev.host) return;
   const size_t bytes = mm2dev.bytes;
   checkCudaErrors(cuMemcpyDtoH((void*)mm2dev.h_adrs,
                                (CUdeviceptr)mm2dev.d_adrs,
                                bytes));
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
