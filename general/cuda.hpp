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

// *****************************************************************************
#ifdef __NVCC__
#include <cuda.h>
inline void cuCheck(const unsigned int c)
{
   MFEM_ASSERT(c == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
}
template <typename BODY> __global__ static
void cuKernel(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
template <int BLOCKS, typename DBODY>
void cuWrap(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLOCKS-1)/BLOCKS;
   cuKernel<<<GRID,BLOCKS>>>(N,d_body);
   const cudaError_t last = cudaGetLastError();
   MFEM_ASSERT(last == cudaSuccess, cudaGetErrorString(last));
}
template<typename T>
__host__ __device__ inline T AtomicAdd(T* address, T val)
{
   return atomicAdd(address, val);
}
#else // ***********************************************************************
#define __host__
#define __device__
#define __constant__
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
template <int BLOCKS, typename DBODY>
void cuWrap(const int N, DBODY &&d_body) {}
template<typename T> inline T AtomicAdd(T* address, T val)
{
   return *address += val;
}
#endif // __NVCC__

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
void* cuMemAlloc(void **d_ptr, size_t bytes);

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
void* cuMemFree(void *d_ptr);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoDAsync(void *d_dst, const void *h_src,
                        size_t bytes, void *stream);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoD(void *d_dst, void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoDAsync(void *d_dst, void *d_src, size_t bytes, void *stream);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoH(void *h_dst, void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoHAsync(void *h_dst, void *d_src, size_t bytes, void *stream);

} // namespace mfem

#endif // MFEM_CUDA_HPP
