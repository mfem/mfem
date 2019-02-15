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
#define cuCheck(c) { MFEM_ASSERT(!c, cudaGetErrorString(cudaGetLastError())); }
extern __shared__ double gpu_mem_s[];
template <typename BODY> __global__ static
void cuKernel(const size_t N, const size_t Nspt, BODY body)
{
   const size_t tid = threadIdx.x;
   const size_t k = blockDim.x*blockIdx.x + tid;
   if (k >= N) { return; }
   body(k, gpu_mem_s+tid*Nspt);
}
template <typename DBODY>
void cuWrap(const size_t N, const size_t Nspt,
            const size_t smpb, DBODY &&d_body)
{
   // shared bytes per thread
   const size_t spt = 1 + Nspt*sizeof(double);
   // block dimension, constrainted by shared byte size
   const size_t Db = min(1024ull, 1ull<<LOG2(smpb/spt));
   // grid dimension
   const size_t Dg = (N+Db-1)/Db;
   // shared bytes per block
   const size_t Ns = spt*Db;
   MFEM_ASSERT(Ns < smpb, "Not enough shared memory per block!");
   cuKernel<<<Dg,Db,Ns>>>(N,Nspt,d_body);
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
template <typename DBODY>
void cuWrap(const size_t N, size_t Nspt,
            const size_t smpb, DBODY &&d_body) {}
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
void* cuMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoHAsync(void *h_dst, void *d_src, size_t bytes, void *stream);

} // namespace mfem

#endif // MFEM_CUDA_HPP
