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
template <typename BODY> __global__ static
void cuKernel(const size_t N, BODY body)
{
   const size_t k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
template <size_t BLOCK_SZ, typename DBODY>
void cuWrap(const size_t N, DBODY &&d_body)
{
   const size_t gridSize = (N+BLOCK_SZ-1)/BLOCK_SZ;
   cuKernel<<<gridSize, BLOCK_SZ>>>(N,d_body);
}
constexpr static inline bool cuNvcc() { return true; }
#else // ***********************************************************************
#define __host__
#define __device__
#define __constant__
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
template <size_t BLOCK_SIZE, typename DBODY>
void cuWrap(const size_t N, DBODY &&d_body) {}
constexpr static inline bool cuNvcc() { return false; }
#endif // __NVCC__

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
void* okMemAlloc(void**, size_t);

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
void* okMemFree(void*);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* okMemcpyHtoD(void*, const void*, size_t);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* okMemcpyHtoDAsync(void*, const void*, size_t, void*);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* okMemcpyDtoD(void*, void*, size_t);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* okMemcpyDtoDAsync(void*, void*, size_t, void*);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* okMemcpyDtoH(void*, const void*, size_t);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* okMemcpyDtoHAsync(void*, void*, size_t, void*);

} // namespace mfem

#endif // MFEM_CUDA_HPP
