// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SYCL_HPP
#define MFEM_SYCL_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_SYCL
/*
 * We are using:
 *  - Explicit Unified Shared Memory (USM), so that code with pointers
 *    can work naturally without buffers or accessors
 *  - SYCL 2020: C++17 Single source programming, with multiple backend options
 *    OpenCL 3.0 and SPIRV
 *
 * It has been tested on:
 *  - DPC++, which uses LLVM/Clang and is part of oneAPI
 *  - OpenSYCL, which can target host OpenMP, CUDA and HIP/ROCm
 *
 *  FP64 not fully supported on devcloud
 * */

#include <CL/sycl.hpp>
using namespace cl;

#if defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)
// https://github.com/intel/llvm/blob/sycl/sycl/doc/PreprocessorMacros.md
#define SYCL_FALLBACK_ASSERT 1

// SYCL_LANGUAGE_VERSION is composed of the general SYCL version
// followed by 2 digits representing the revision number
#if !SYCL_LANGUAGE_VERSION || SYCL_LANGUAGE_VERSION < 202001
#error MFEM requires a SYCL compiler with language version 2020
#endif // SYCL_LANGUAGE_VERSION
#endif // __INTEL_LLVM_COMPILER

#define MFEM_DEVICE
#define MFEM_LAMBDA
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC mfem::Sycl::Queue().queues_wait_and_throw()
#define MFEM_STREAM_SYNC mfem::Sycl::Queue().wait_and_throw()
#define MFEM_GPU_CHECK(...) __VA_ARGS__

#if defined(__SYCL_DEVICE_ONLY__)
#define MFEM_SHARED // the compiler does the local memory mapping if it can
#define MFEM_SYNC_THREAD sycl::detail::workGroupBarrier();
#define MFEM_BLOCK_ID(k) __spirv_BuiltInWorkgroupId.k
#define MFEM_THREAD_ID(k) __spirv_BuiltInLocalInvocationId.k
#define MFEM_THREAD_SIZE(k) __spirv_BuiltInWorkgroupSize.k
#define MFEM_FOREACH_THREAD(i,k,N) \
for(size_t i=__spirv_LocalInvocationId_##k(); i<N; i+=__spirv_WorkgroupSize_##k())
#endif

#include "debug.hpp"

namespace mfem
{

struct Sycl
{
   /// Return the queue used by MFEM.
   static sycl::queue Queue();

   /*MFEM_DEVICE inline double atomicAdd(double *add, double val)
   {
      sycl::atomic_ref<double,
           sycl::memory_order::relaxed,
           sycl::memory_scope::work_group,
           sycl::access::address_space::local_space>(*add) += (double) (val);
      return add;
   }*/
};

/**
 * @brief The SyclWrap class
 */
template <int Dim> struct SyclWrap;

/**
 * @brief The SyclWrap<1> specialized class
 */
template <> struct SyclWrap<1>
{
   template <typename DBODY>
   static void run(const int N, DBODY &&body,
                   const int /*X*/, const int /*Y*/, const int /*Z*/, const int /*G*/)
   {
      if (N == 0) { return; }
      sycl::queue Q = Sycl::Queue();
      Q.submit([&](sycl::handler &h)
      {
         h.parallel_for(sycl::range<1>(N), [=](sycl::item<1> itm)
         {
            const size_t k = itm.get_linear_id();
            body(k);
         });
      });
      Q.wait();
   }
};

/**
 * @brief The SyclWrap<2> specialized class
 */
template <> struct SyclWrap<2>
{
   template <typename T>
   static void run(const int N, T &&body,
                   const int X, const int Y, const int BZ, const int G)
   {
      assert(G == 0);
      if (N == 0) { return; }
      sycl::queue Q = Sycl::Queue();
      Q.submit([&](sycl::handler &h)
      {
#ifdef __SYCL_DEVICE_ONLY__ // SYCL-GPU:
         const int L = static_cast<int>(std::ceil(std::sqrt((N+BZ-1)/BZ)));
         const sycl::range<3> grid(L*BZ, L*Y, L*X), group(BZ, Y, X);
#else // SYCL-CPU:
         const sycl::range<3> grid(1, 1, N), group(1, 1, 1);
#endif
         h.parallel_for(sycl::nd_range<3>(grid,group), [=](sycl::nd_item<3> itm)
         {
            // const int k = itm.get_group_linear_id();
            // blockIdx.x*blockDim.z + threadIdx.z;
            const int k =
               itm.get_group(2)*itm.get_local_range().get(0) + itm.get_local_id(0);
            if (k >= N) { return; }
            body(k);
         });
      });
      Q.wait();
   }
};

/**
 * @brief The SyclWrap<3> specialized class
 */
template <> struct SyclWrap<3>
{
   template <typename T>
   static void run(const int N, T &&body,
                   const int X, const int Y, const int Z, const int G)
   {
      if (N == 0) { return; }
      sycl::queue Q = Sycl::Queue();
      Q.submit([&](sycl::handler &h)
      {
#ifdef __SYCL_DEVICE_ONLY__ // SYCL-GPU:
         const int L = static_cast<int>(std::ceil(std::cbrt(G == 0 ? N : G)));
         const sycl::range<3> grid(L*Z, L*Y, L*X), group(Z, Y, X);
#else // SYCL-CPU:
         const sycl::range<3> grid(1, 1, N), group(1, 1, 1);
#endif
         h.parallel_for(sycl::nd_range<3>(grid, group), [=](sycl::nd_item<3> itm)
         {
            const int k = itm.get_group_linear_id();
            if (k >= N) { return; }
            body(k);
         });
      });
      Q.wait();
   }
};

/// Allocates device memory and returns destination ptr.
void* SyclMemAlloc(void **d_ptr, size_t bytes);

/// Allocates managed device memory
void* SyclMallocManaged(void **d_ptr, size_t bytes);

/// Allocates page-locked (pinned) host memory
void* SyclMemAllocHostPinned(void **ptr, size_t bytes);

/// Frees device memory and returns destination ptr.
void* SyclMemFree(void *d_ptr);

/// Frees page-locked (pinned) host memory and returns destination ptr.
void* SyclMemFreeHostPinned(void *ptr);

/// Copies memory from Host to Device and returns destination ptr.
void* SyclMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Host to Device and returns destination ptr.
void* SyclMemcpyHtoDAsync(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Device to Device
void* SyclMemcpyDtoD(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Device
void* SyclMemcpyDtoDAsync(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* SyclMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* SyclMemcpyDtoHAsync(void *h_dst, const void *d_src, size_t bytes);

/// Check the error code returned by the sycl queue with throw_asynchronous().
void SyclCheckLastError();

/// Get the number of SYCL devices
int SyclGetDeviceCount();

} // namespace mfem

#endif // MFEM_USE_SYCL

#endif // MFEM_SYCL_HPP
