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

#ifndef MFEM_CUDA_HPP
#define MFEM_CUDA_HPP

#include "../config/config.hpp"
#include "error.hpp"

#ifdef MFEM_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

// CUDA block size used by MFEM.
#define MFEM_CUDA_BLOCKS 256

#ifdef MFEM_USE_CUDA
#define MFEM_ATTR_DEVICE __device__
#define MFEM_ATTR_HOST_DEVICE __host__ __device__
// Define the CUDA debug macros:
// - MFEM_CUDA_CHECK_DRV(x) where 'x' returns/is type 'CUresult'
// - MFEM_CUDA_CHECK_RT(x)  where 'x' returns/is type 'cudaError_t'
#ifdef MFEM_DEBUG
#define MFEM_CUDA_CHECK_DRV(x) \
   do \
   { \
      CUresult err = (x); \
      if (err != CUDA_SUCCESS) \
      { \
         const char *error_string; \
         cuGetErrorString(err, &error_string); \
         _MFEM_MESSAGE("CUDA error: (" << #x \
                       << ") failed with error:\n --> " \
                       << error_string, 0); \
      } \
   } \
   while (0)
#define MFEM_CUDA_CHECK_RT(x) \
   do \
   { \
      cudaError_t err = (x); \
      if (err != cudaSuccess) \
      { \
         _MFEM_MESSAGE("CUDA error: (" << #x \
                       << ") failed with error:\n --> " \
                       << cudaGetErrorString(err), 0); \
      } \
   } \
   while (0)
#else
#define MFEM_CUDA_CHECK_DRV(x) x
#define MFEM_CUDA_CHECK_RT(x) x
#endif
#else // MFEM_USE_CUDA
#define MFEM_ATTR_DEVICE
#define MFEM_ATTR_HOST_DEVICE
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
#endif // MFEM_USE_CUDA


namespace mfem
{

// Define 'atomicAdd' function.
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
static __device__ inline double atomicAdd(double* address, double val)
{
   unsigned long long int* address_as_ull = (unsigned long long int*)address;
   unsigned long long int old = *address_as_ull, assumed;
   do
   {
      assumed = old;
      old =
         atomicCAS(address_as_ull, assumed,
                   __double_as_longlong(val +
                                        __longlong_as_double(assumed)));
      // Note: uses integer comparison to avoid hang in case of NaN
      // (since NaN != NaN)
   }
   while (assumed != old);
   return __longlong_as_double(old);
}
#endif // __CUDA_ARCH__ < 600
template<typename T> MFEM_ATTR_DEVICE
inline T AtomicAdd(T volatile *address, T val)
{
   return atomicAdd((T *)address, val);
}
#else // __CUDA_ARCH__
template<typename T> inline T AtomicAdd(T volatile *address, T val)
{
#ifdef MFEM_USE_OPENMP
   #pragma omp atomic
#endif
   *address += val;
   return *address;
}
#endif // __CUDA_ARCH__


/// Allocates device memory
void* CuMemAlloc(void **d_ptr, size_t bytes);

/// Frees device memory
void* CuMemFree(void *d_ptr);

/// Copies memory from Host to Device
void* CuMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Host to Device
void* CuMemcpyHtoDAsync(void *d_dst, const void *h_src,
                        size_t bytes, void *stream);

/// Copies memory from Device to Device
void* CuMemcpyDtoD(void *d_dst, void *d_src, size_t bytes);

/// Copies memory from Device to Device
void* CuMemcpyDtoDAsync(void *d_dst, void *d_src, size_t bytes, void *stream);

/// Copies memory from Device to Host
void* CuMemcpyDtoH(void *h_dst, void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* CuMemcpyDtoHAsync(void *h_dst, void *d_src, size_t bytes, void *stream);

} // namespace mfem

#endif // MFEM_CUDA_HPP
