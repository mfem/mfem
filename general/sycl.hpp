// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include <iostream>
#include <CL/sycl.hpp>

#define MFEM_DEVICE
#define MFEM_LAMBDA
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC // SyclQueue().queues_wait_and_throw()
#define MFEM_STREAM_SYNC
#define MFEM_GPU_CHECK(...) __VA_ARGS__

/*#if defined(__SYCL_DEVICE_ONLY__)
// Define the SYCL inner threading macros
#define MFEM_SHARED // the compiler does the local memory mapping if it can
#define MFEM_SYNC_THREAD itm.barrier(cl::sycl::access::fence_space::local_space);
// sycl::group_barrier(it.get_group());
#define MFEM_THREAD_ID(k) itm.get_local_id(k);
#define MFEM_THREAD_SIZE(k) itm.get_local_range(k);
#endif*/

#define SYCL_FOREACH_THREAD(i,k,N) \
for(int i=itm.get_local_id(k); i<N; i+=itm.get_local_range(k))

#include "device.hpp"

namespace mfem
{

/// Return the default sycl::queue used by MFEM.
cl::sycl::queue& SyclQueue();

template <int Dim> struct SyclKernel;

template <> struct SyclKernel<1>
{
   template <typename DBODY>
   static void run(const int N, DBODY &&body,
                   const int /*X*/, const int /*Y*/, const int /*Z*/, const int /*G*/) noexcept
   {
      if (N == 0) { return; }
      try
      {
         cl::sycl::queue &Q = SyclQueue();
         Q.submit([&](cl::sycl::handler &h)
         {
            h.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::id<1> itm)
            {
               body(itm[0]);
            });
         });
         Q.wait();
      }
      catch (cl::sycl::exception const &e)
      {
         MFEM_ABORT("An exception is caught while multiplying matrices.");
      }
   }
};

template <> struct SyclKernel<2>
{
   template <typename T>
   static void run(const int N, T &&body,
                   const int X, const int Y, const int /*Z*/, const int /*G*/)
   {
      if (N == 0) { return; }
      try
      {
         cl::sycl::queue &Q = SyclQueue();
         Q.submit([&](cl::sycl::handler &h)
         {
            const int L = static_cast<int>(std::ceil(std::sqrt(N)));
            const cl::sycl::range<2> grid(L*X, L*Y);
            const cl::sycl::range<2> group(X, Y);
            h.parallel_for(cl::sycl::nd_range<2>(grid,group), [=](cl::sycl::nd_item<2> itm)
            {
               const int k = itm.get_group_linear_id();
               if (k >= N) { return; }
               body(k);
            });
         });
         Q.wait();
      }
      catch (cl::sycl::exception const &e)
      {
         MFEM_ABORT("An exception is caught while multiplying matrices.");
      }
   }
};

template <> struct SyclKernel<3>
{
   template <typename T>
   static void run(const int N, T &&body,
                   const int X, const int Y, const int Z, const int /*G*/)
   {
      if (N == 0) { return; }
      try
      {
         cl::sycl::queue &Q = SyclQueue();
         Q.submit([&](cl::sycl::handler &h)
         {
            const int L = static_cast<int>(std::ceil(std::cbrt(N)));
            const cl::sycl::range<3> grid(L*X, L*Y, L*Z);
            const cl::sycl::range<3> group(X, Y, Z);
            h.parallel_for(cl::sycl::nd_range<3>(grid, group), [=](cl::sycl::nd_item<3> itm)
            {
               const int k = itm.get_group_linear_id();
               if (k >= N) { return; }
               body(k);
            });
         });
         Q.wait();
      }

      catch (cl::sycl::exception const &e)
      {
         MFEM_ABORT("An exception is caught while multiplying matrices.");
      }
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
