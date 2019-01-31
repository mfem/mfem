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

// *****************************************************************************
#ifdef __NVCC__
#include <cuda.h>

inline void cuCheck(const unsigned int c)
{
   MFEM_ASSERT(not c, cudaGetErrorString(cudaGetLastError()));
}

template <typename BODY> __global__ static
void cuKernel(const size_t N, BODY body)
{
   const size_t k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void cuKernelSeq(const size_t N, BODY body)
{
   for (size_t k=0; k<N; k+=1)
   {
      body(k);
   }
}

template <size_t BLOCKS, typename DBODY>
void cuWrap(const size_t N, DBODY &&d_body)
{
   const size_t GRID = (N+BLOCKS-1)/BLOCKS;
   cuKernel<<<GRID,BLOCKS>>>(N,d_body);
   //cuKernelSeq<<<1,1>>>(N,d_body);
}

constexpr static inline bool usingNvccCompiler() { return true; }

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
template <size_t BLOCKS, typename DBODY>
void cuWrap(const size_t N, DBODY &&d_body) {}
constexpr static inline bool usingNvccCompiler() { return false; }
template<typename T> inline T AtomicAdd(T* address, T val)
{
   *address += val;
   return *address;
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
void* cuMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoHAsync(void *h_dst, void *d_src, size_t bytes, void *stream);

} // namespace mfem

#endif // MFEM_CUDA_HPP
