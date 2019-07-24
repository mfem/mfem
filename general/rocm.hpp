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
// Define a ROCM error check macro, MFEM_GPU_CHECK(x), where x returns/is of
// type 'rocmError_t'. This macro evaluates 'x' and raises an error if the
// result is not hipSuccess.
#define MFEM_GPU_CHECK(x) \
   do \
   { \
      hipError_t err = (x); \
      if (err != hipSuccess) \
      { \
         mfem_rocm_error(err, #x, _MFEM_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)
#endif // MFEM_USE_ROCM

// Define the MFEM inner threading macros
#if defined(MFEM_USE_ROCM) && defined(__ROCM_ARCH__)
#define MFEM_SHARED __shared__
#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_THREAD_ID(k) hipThreadIdx_ ##k
#define MFEM_THREAD_SIZE(k) hipBlockDim_ ##k
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=hipThreadIdx_ ##k; i<N; i+=hipBlockDim_ ##k)
#endif


namespace mfem
{

#ifdef MFEM_USE_ROCM
// Function used by the macro MFEM_GPU_CHECK.
void mfem_rocm_error(hipError_t err, const char *expr, const char *func,
                     const char *file, int line);
#endif

/// Allocates device memory
void* RocMemAlloc(void **d_ptr, size_t bytes);

/// Frees device memory
void* RocMemFree(void *d_ptr);

/// Copies memory from Host to Device
void* RocMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Host to Device
void* RocMemcpyHtoDAsync(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Device to Device
void* RocMemcpyDtoD(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Device
void* RocMemcpyDtoDAsync(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* RocMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* RocMemcpyDtoHAsync(void *h_dst, const void *d_src, size_t bytes);

} // namespace mfem

#endif // MFEM_ROCM_HPP
