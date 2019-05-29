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

#ifndef MFEM_ROCM_HPP
#define MFEM_ROCM_HPP

#include "../config/config.hpp"
#include "error.hpp"

#ifdef MFEM_USE_ROCM
#include <hip/hip_runtime.h>
#endif

// ROCM block size used by MFEM.
#define MFEM_ROCM_BLOCKS 256

#ifdef MFEM_USE_ROCM
//#define MFEM_DEVICE __device__
//#define MFEM_HOST_DEVICE __host__ __device__
// Define a ROCM error check macro, MFEM_ROCM_CHECK(x), where x returns/is of
// type 'rocmError_t'. This macro evaluates 'x' and raises an error if the
// result is not hipSuccess.
#define MFEM_HIP_CHECK(x) \
   do \
   { \
      hipError_t err = (x); \
      if (err != hipSuccess) \
      { \
         mfem_hip_error(err, #x, _MFEM_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)
#else
//#define MFEM_DEVICE
//#define MFEM_HOST_DEVICE
#endif // MFEM_USE_ROCM

// Define the MFEM inner threading macros
#if defined(MFEM_USE_ROCM) && defined(__ROCM_ARCH__)
#define MFEM_SHARED __shared__
#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_THREAD_ID(k) threadIdx.k
#define MFEM_THREAD_SIZE(k) blockDim.k
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=threadIdx.k; i<N; i+=blockDim.k)
#else
#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=0; i<N; i++)
#endif


namespace mfem
{

#ifdef MFEM_USE_ROCM
// Function used by the macro MFEM_ROCM_CHECK.
void mfem_hip_error(hipError_t err, const char *expr, const char *func,
                    const char *file, int line);
#endif

/// Allocates device memory
void* HipMemAlloc(void **d_ptr, size_t bytes);

/// Frees device memory
void* HipMemFree(void *d_ptr);

/// Copies memory from Host to Device
void* HipMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Host to Device
void* HipMemcpyHtoDAsync(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Device to Device
void* HipMemcpyDtoD(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Device
void* HipMemcpyDtoDAsync(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* HipMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* HipMemcpyDtoHAsync(void *h_dst, const void *d_src, size_t bytes);

} // namespace mfem

#endif // MFEM_ROCM_HPP
