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

#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include "config.hpp"
#include "memcpy.hpp"

// *****************************************************************************
/*
void* operator new(std::size_t size, const std::nothrow_t&) noexcept{}
void operator delete(void* ptr) noexcept{}
void operator delete(void* ptr, std::size_t size) noexcept{}
void operator delete(void* ptr, const std::nothrow_t&) noexcept{}
void* operator new[](std::size_t size){}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept{}
void operator delete[](void* ptr) noexcept{}
void operator delete[](void* ptr, std::size_t size) noexcept{}
void operator delete[](void* ptr, const std::nothrow_t&) noexcept{}
void* operator new (std::size_t size, void* ptr) noexcept{}
void* operator new[](std::size_t size, void* ptr) noexcept{}
void operator delete (void* ptr, void*) noexcept{}
void operator delete[](void* ptr, void*) noexcept{}
*/

// *****************************************************************************
// * Memory manager
// ***************************************************************************
template<class T> struct mm {
   // **************************************************************************
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
   static inline void operator delete(void *ptr) {
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
};

#endif // MFEM_MEMMNG_HPP
