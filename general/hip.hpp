// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_HIP_HPP
#define MFEM_HIP_HPP

#include "../config/config.hpp"
#include "error.hpp"

// HIP block size used by MFEM.
#define MFEM_HIP_BLOCKS 256

#if defined(MFEM_USE_HIP) && defined(__HIP__)
#define MFEM_USE_CUDA_OR_HIP
#define MFEM_DEVICE __device__
#define MFEM_HOST __host__
#define MFEM_LAMBDA __host__ __device__
// #define MFEM_HOST_DEVICE __host__ __device__ // defined in config/config.hpp
#define MFEM_DEVICE_SYNC MFEM_GPU_CHECK(hipDeviceSynchronize())
#define MFEM_STREAM_SYNC MFEM_GPU_CHECK(hipStreamSynchronize(0))
// Define a HIP error check macro, MFEM_GPU_CHECK(x), where x returns/is of
// type 'hipError_t'. This macro evaluates 'x' and raises an error if the
// result is not hipSuccess.
#define MFEM_GPU_CHECK(x)                                                      \
  do {                                                                         \
    hipError_t mfem_err_internal_var_name = (x);                               \
    if (mfem_err_internal_var_name != hipSuccess) {                            \
      ::mfem::mfem_hip_error(mfem_err_internal_var_name, #x, _MFEM_FUNC_NAME,  \
                             __FILE__, __LINE__);                              \
    }                                                                          \
  } while (0)

// Define the MFEM inner threading macros
#if defined(__HIP_DEVICE_COMPILE__)
#define MFEM_SHARED __shared__
#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_BLOCK_ID(k) hipBlockIdx_ ##k
#define MFEM_THREAD_ID(k) hipThreadIdx_ ##k
#define MFEM_THREAD_SIZE(k) hipBlockDim_ ##k
#define MFEM_FOREACH_THREAD(i,k,N) \
   for(int i=hipThreadIdx_ ##k; i<N; i+=hipBlockDim_ ##k)
#define MFEM_FOREACH_THREAD_DIRECT(i,k,N) \
   if(const int i=hipThreadIdx_ ##k; i<N)
#endif // defined(__HIP_DEVICE_COMPILE__)
#endif // defined(MFEM_USE_HIP) && defined(__HIP__)

namespace mfem
{

#ifdef MFEM_USE_HIP
// Function used by the macro MFEM_GPU_CHECK.
void mfem_hip_error(hipError_t err, const char *expr, const char *func,
                    const char *file, int line);
#endif

/// Allocates device memory
void* HipMemAlloc(void **d_ptr, size_t bytes);

/// Allocates managed device memory
void* HipMallocManaged(void **d_ptr, size_t bytes);

/// Allocates page-locked (pinned) host memory
void* HipMemAllocHostPinned(void **ptr, size_t bytes);

/// Frees device memory
void* HipMemFree(void *d_ptr);

/// Frees page-locked (pinned) host memory and returns destination ptr.
void* HipMemFreeHostPinned(void *ptr);

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

/// Check the error code returned by hipGetLastError(), aborting on error.
void HipCheckLastError();

/// Get the number of HIP devices
int HipGetDeviceCount();

} // namespace mfem

#endif // MFEM_HIP_HPP
