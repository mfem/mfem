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

#include "cuda.hpp"

namespace mfem
{

void* CuMemAlloc(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMalloc(dptr, bytes))
   {
      mfem_error("Error in CuMemAlloc");
   }
#endif
   return *dptr;
}

void* CuMemFree(void *dptr)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaFree(dptr))
   {
      mfem_error("Error in CuMemFree");
   }
#endif
   return dptr;
}

void* CuMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice))
   {
      mfem_error("Error in CuMemcpyHtoD");
   }
#endif
   return dst;
}

void* CuMemcpyHtoDAsync(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice))
   {
      mfem_error("Error in CuMemcpyHtoDAsync");
   }
#endif
   return dst;
}

void* CuMemcpyDtoD(void* dst, void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice))
   {
      mfem_error("Error in CuMemcpyDtoD");
   }
#endif
   return dst;
}

void* CuMemcpyDtoDAsync(void* dst, void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice))
   {
      mfem_error("Error in CuMemcpyDtoDAsync");
   }
#endif
   return dst;
}

void* CuMemcpyDtoH(void *dst, void *src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost))
   {
      mfem_error("Error in CuMemcpyDtoH");
   }
#endif
   return dst;
}

void* CuMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void *s)
{
#ifdef MFEM_USE_CUDA
   if (cudaSuccess != ::cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost))
   {
      mfem_error("Error in CuMemcpyDtoHAsync");
   }
#endif
   return dst;
}

} // namespace mfem
