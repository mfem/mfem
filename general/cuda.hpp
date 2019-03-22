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

#ifdef __NVCC__
#include <cuda.h>
#endif

namespace mfem
{

// *****************************************************************************
#ifdef __NVCC__
inline void CuCheck(const unsigned int c)
{
   MFEM_ASSERT(c == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
}
template <typename BODY> __global__ static
void CuKernel(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
template <int BLOCKS, typename DBODY>
void CuWrap(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLOCKS-1)/BLOCKS;
   CuKernel<<<GRID,BLOCKS>>>(N,d_body);
   const cudaError_t last = cudaGetLastError();
   MFEM_ASSERT(last == cudaSuccess, cudaGetErrorString(last));
}
#else // ***********************************************************************
#define __host__
#define __device__
#define __constant__
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
template <int BLOCKS, typename DBODY>
void CuWrap(const int N, DBODY &&d_body) {}
#endif // __NVCC__

#if defined(__CUDA_ARCH__)
template<typename T>
__device__ inline T AtomicAdd(T* address, T val)
{
   return atomicAdd(address, val);
}
#else
template<typename T> inline T AtomicAdd(T* address, T val)
{
#if defined(_OPENMP)
   #pragma omp atomic
#endif
   *address += val;
   return *address;
}
#endif //__CUDA_ARCH__
// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
void* CuMemAlloc(void **d_ptr, size_t bytes);

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
void* CuMemFree(void *d_ptr);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* CuMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* CuMemcpyHtoDAsync(void *d_dst, const void *h_src,
                        size_t bytes, void *stream);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* CuMemcpyDtoD(void *d_dst, void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* CuMemcpyDtoDAsync(void *d_dst, void *d_src, size_t bytes, void *stream);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* CuMemcpyDtoH(void *h_dst, void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* CuMemcpyDtoHAsync(void *h_dst, void *d_src, size_t bytes, void *stream);

} // namespace mfem

#endif // MFEM_CUDA_HPP
