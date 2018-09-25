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

#ifndef MFEM_MEMMNG_HPP
#define MFEM_MEMMNG_HPP

// *****************************************************************************
// * Memory manager
// ***************************************************************************
struct mm {
   // **************************************************************************
   template<class T>
   static inline T* malloc(size_t n, const size_t size_of_T = sizeof(T)) {
      T *ptr = nullptr;
      if (!cfg::Get().Cuda()) return ptr = ::new T[n];
#ifdef __NVCC__
      const size_t bytes = n*size_of_T;
      dbg("\033[31;7mnew NVCC (%ldo)",bytes);
      cuMemAlloc((CUdeviceptr*)&ptr,bytes);
#endif // __NVCC__
      return ptr;
   }
   
   // **************************************************************************
   template<class T>
   static inline void free(void *ptr) {
      if (!cfg::Get().Cuda()) {
         if (ptr)
            ::delete[] static_cast<T*>(ptr);
      }
#ifdef __NVCC__
      else {
         dbg("\033[31;7mdelete NVCC");
         cuMemFree((CUdeviceptr)ptr);
      }
#endif // __NVCC__
      ptr = nullptr;
   }

   // *****************************************************************************
   static void handler(int nSignum, siginfo_t* si, void* vcontext) {
      std::cout << "\n\033[31;1mSegmentation fault\033[m" << std::endl;
  
      ucontext_t* context = (ucontext_t*)vcontext;
      context->uc_mcontext.gregs[REG_RIP]++;
      stk(true);
      exit(1);
   }

   // *****************************************************************************
   static void iniHandler(){
      struct sigaction action;
      memset(&action, 0, sizeof(struct sigaction));
      action.sa_flags = SA_SIGINFO;
      action.sa_sigaction = handler;
      sigaction(SIGSEGV, &action, NULL);
   }
   
};
#endif // MFEM_MEMMNG_HPP
