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
   cfg::Get().Init();
}

// *****************************************************************************
void* mm::add(const void *adrs, const size_t size, const size_t size_of_T){
   const size_t bytes = size*size_of_T;
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   void *rtn = NULL;
   if (present) {
      printf(" \033[31;7m%p(%ldo)\033[m", adrs, bytes);fflush(0);
      assert(false); // should not happen
   }else{
      printf(" \033[31m%p(%ldo)\033[m", adrs, bytes);fflush(0);
      mm2dev_t &mm2dev = mng->operator[](adrs);
      mm2dev.host = true;
      mm2dev.bytes = bytes;
      mm2dev.h_adrs = adrs;

      // if cfg::Get().Cuda() is set, alloc also there
      if (cfg::Get().Cuda()){
         CUdeviceptr ptr;
         const size_t bytes = mm2dev.bytes;
         printf(" \033[32;1m%ldo\033[m",bytes);
         checkCudaErrors(cuMemAlloc(&ptr,bytes));
         mm2dev.d_adrs = (void*)ptr;
         // and say we are there
         mm2dev.host = false;
         // and flush host side for now to SIGSEGV
         mm2dev.h_adrs = NULL;
      }else{      
         mm2dev.d_adrs = NULL;
      }
      rtn = (void*) (mm2dev.host ? mm2dev.h_adrs : mm2dev.d_adrs);
   }
   return rtn; 
}

// *****************************************************************************
void mm::del(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   if (present){
      printf(" \033[32;7m%p\033[m", adrs);
      // Remove element from the map
      mng->erase(adrs);
   }else{
      printf(" \033[32m%p\033[m", adrs);
      assert(false); // should not happen
   }   
}

// *****************************************************************************
void mm::Cuda(){
   for(auto it = mng->begin(); it != mng->end(); ++it){
      const void *adrs = it->first;
      mm2dev_t &mm2dev = it->second;
      //dbg("\033[32;1madrs=%p",adrs);
      assert(adrs == mm2dev.h_adrs);
      
      // Now allocate on the device
      CUdeviceptr ptr;
      const size_t bytes = mm2dev.bytes;
      printf(" \033[32;1m%ldo\033[m",bytes);
      checkCudaErrors(cuMemAlloc(&ptr,bytes));
      mm2dev.d_adrs = (void*)ptr;
      
      //dbg("\033[32;1m =>");
      //memcpy::H2D(ptr,mm2dev.h_adrs,bytes);
      checkCudaErrors(cuMemcpyHtoD(ptr,mm2dev.h_adrs,bytes));
      // Now we are on the GPU
      mm2dev.host = false;
   }
}

// *****************************************************************************
void* mm::Adrs(const void *adrs){
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   assert(present);
   const mm2dev_t &mm2dev = mng->operator[](adrs);
   if (mm2dev.host)
      return (void*)mm2dev.h_adrs;
   assert(mm2dev.d_adrs);
   return (void*)mm2dev.d_adrs;
}

// *****************************************************************************
// static
void mm::handler(int nSignum, siginfo_t* si, void* vcontext) {
   fflush(0);
   printf("\n\033[31;7;1mSegmentation fault\033[m\n");
   ucontext_t* context = (ucontext_t*)vcontext;
   context->uc_mcontext.gregs[REG_RIP]++;
   stk(true);
   fflush(0);
   exit(1);
   //throw SIGSEGV;
}

// *****************************************************************************
// static
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
