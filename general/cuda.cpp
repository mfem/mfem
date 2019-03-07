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

#include "okina.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
void* cuMemAlloc(void** dptr, size_t bytes)
{
#ifdef __NVCC__
   if (bytes==0) { return *dptr; }
   if (CUDA_SUCCESS != ::cuMemAlloc((CUdeviceptr*)dptr, bytes))
   {
      mfem_error("Error in cuMemAlloc");
   }
#endif
   return *dptr;
}

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
void* cuMemFree(void *dptr)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS != ::cuMemFree((CUdeviceptr)dptr))
   {
      mfem_error("Error in cuMemFree");
   }
#endif
   return dptr;
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS != ::cuMemcpyHtoD((CUdeviceptr)dst, src, bytes))
   {
      mfem_error("Error in cuMemcpyHtoD");
   }
#endif
   return dst;
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoDAsync(void* dst, const void* src, size_t bytes, void *s)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS !=
       ::cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, (CUstream)s))
   {
      mfem_error("Error in cuMemcpyHtoDAsync");
   }
#endif
   return dst;
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoD(void* dst, void* src, size_t bytes)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS !=
       ::cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes))
   {
      mfem_error("Error in cuMemcpyDtoD");
   }
#endif
   return dst;
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void *s)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS !=
       ::cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src,
                           bytes, (CUstream)s))
   {
      mfem_error("Error in cuMemcpyDtoDAsync");
   }
#endif
   return dst;
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoH(void *dst, void *src, size_t bytes)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS != ::cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes))
   {
      mfem_error("Error in cuMemcpyDtoH");
   }
#endif
   return dst;
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void *s)
{
#ifdef __NVCC__
   if (CUDA_SUCCESS !=
       ::cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, (CUstream)s))
   {
      mfem_error("Error in cuMemcpyDtoHAsync");
   }
#endif
   return dst;
}

// *****************************************************************************
} // namespace mfem
