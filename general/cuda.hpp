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

#include <cstddef>

#if defined(__NVCC__) && defined(MFEM_USE_CUDA)
#include <cuda.h>
#endif

namespace mfem
{

#if defined (MFEM_USE_CUDA) && defined (__NVCC__)
#define MFEM_DEVICE __device__
#define MFEM_HOST_DEVICE __host__ __device__
inline void CuCheck(const unsigned int c)
{
   MFEM_ASSERT(c == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
}
#if __CUDA_ARCH__ < 600
static __device__ double atomicAdd(double* address, double val)
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
template<typename T> MFEM_HOST_DEVICE
inline T AtomicAdd(T* address, T val)
{
   return atomicAdd(address, val);
}
#else // MFEM_USE_CUDA
#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
template<typename T> inline T AtomicAdd(T* address, T val)
{
#if defined(_OPENMP)
   #pragma omp atomic
#endif
   *address += val;
   return *address;
}
#endif // MFEM_USE_CUDA

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
