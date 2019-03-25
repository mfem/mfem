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

#ifdef MFEM_USE_CUDA
#include <cuda.h>
#endif

namespace mfem
{

#ifdef MFEM_USE_CUDA
#define MFEM_HOST_DEVICE __host__ __device__
inline void CuCheck(const unsigned int c)
{
   MFEM_ASSERT(c == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
}
template<typename T> MFEM_HOST_DEVICE
inline T AtomicAdd(T* address, T val)
{
   return atomicAdd(address, val);
}
#else // MFEM_USE_CUDA
#define __host__
#define __device__
#define __constant__
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
#define MFEM_HOST_DEVICE
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
