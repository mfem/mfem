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

#ifndef MFEM_MM_HPP
#define MFEM_MM_HPP

#ifdef MFEM_USE_CUDA
#include <unordered_map>
#endif

namespace mfem4
{

#ifndef MFEM_USE_CUDA

/// Memory manager: default (CPU only) implementation
class MM
{
public:
   template<typename T>
   static T* Alloc(int size)
   {
      return new T[size]; // TODO: aligned
   }

   template<typename T>
   static void Free(T *ptr)
   {
      delete [] ptr;
   }

   template<typename T>
   static T* DevicePtr(T *ptr)
   {
      return ptr;
   }

   template<typename T>
   static const T* DevicePtr(const T *ptr)
   {
      return ptr;
   }

   template<typename T>
   static void Push(const T *ptr) {}

   template<typename T>
   static void Pull(const T *ptr) {}
};


#else // MFEM_USE_CUDA

class MM
{
public:
   // TODO


private:
   struct DeviceBuffer
   {
      int size;
      void *h_addr, *d_addr;
   };

   static std::unordered_map<const void*, DeviceBuffer> buffers;
};

#endif


} // namespace mfem4

#endif // MFEM_MM_HPP
