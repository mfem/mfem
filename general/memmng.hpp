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

#include <unordered_map>

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
typedef struct mm2dev{
   bool host = true;
   size_t bytes = 0;
   const void *h_adrs = NULL;
   const void *d_adrs = NULL;
/*   mm2dev():
      host(true),
      size(0),
      h_adrs((void*)0x12345678ul),
      d_adrs(NULL){ }
   mm2dev(const mm2dev &m){
      host = m.host;
      size = m.size;
      h_adrs = m.h_adrs;
      d_adrs = m.d_adrs;
      }*/
} mm2dev_t;

// *****************************************************************************
typedef std::unordered_map<const void*,mm2dev_t> mm_t;

// *****************************************************************************
// * Memory manager
// ***************************************************************************
class mm {
protected:
   mm_t *mng = NULL;
private:
   mm(){}
   mm(mm const&);
   void operator=(mm const&);
public:
   static mm& Get(){
      static mm mm_singleton;
      return mm_singleton;
   }
   
   // **************************************************************************
   void init();
   void* add(const void*, const size_t, const size_t);
   void del(const void*);
   void Cuda();
   void* Adrs(const void*);
   void Rsync(const void*);

   // **************************************************************************
   template<class T>
   static inline T* malloc(size_t size, const size_t size_of_T = sizeof(T)) {
      dbg();
      stk(true);
      if (!mm::Get().mng) mm::Get().init();
      
      T *ptr = nullptr;
      /*
      if (!cfg::Get().Cuda()) ptr = ::new T[size];
#ifdef __NVCC__
      else{
         const size_t bytes = size*size_of_T;
         dbg("\033[31;1mnew NVCC (%ldo)",bytes);
         cuMemAlloc((CUdeviceptr*)&ptr,bytes);
      }
#endif // __NVCC__
      */
      // alloc on host first
      ptr = ::new T[size];
      // add to the pool of adrs
      mm::Get().add((void*)ptr,size,size_of_T);
      return ptr;
   }
   
   // **************************************************************************
   template<class T>
   static inline void free(void *ptr) {
      if (ptr){
         mm::Get().del(ptr);
         ::delete[] static_cast<T*>(ptr);
      }
      /*
      if (!cfg::Get().Cuda()) {
         if (ptr)
            ::delete[] static_cast<T*>(ptr);
      }
#ifdef __NVCC__
      else {
         dbg("\033[31;1mdelete NVCC");
         cuMemFree((CUdeviceptr)ptr);
      }
      #endif // __NVCC__*/
      ptr = nullptr;
   }

   // *****************************************************************************
   static void handler(int nSignum, siginfo_t* si, void* vcontext);

   // *****************************************************************************
   static void iniHandler();
};

// *****************************************************************************
MFEM_NAMESPACE_END

#endif // MFEM_MEMMNG_HPP
