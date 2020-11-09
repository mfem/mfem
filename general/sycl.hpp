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

#define MFEM_DEVICE
#define MFEM_LAMBDA
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC
#define MFEM_STREAM_SYNC
#define MFEM_GPU_CHECK(x)

// Define the MFEM inner threading macros
#if defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0 // itm.get_logical_local_id(k);
#define MFEM_THREAD_SIZE(k) 1 // itm.get_logical_local_range()[k];
#define MFEM_FOREACH_THREAD(i,k,N) \
    for(int i=MFEM_THREAD_ID(k); i<N; i+=MFEM_THREAD_SIZE(k))

#define SYCL_FOREACH(...)
#else
#define SYCL_FOREACH(...)
#endif

#include "mem_manager.hpp"
#include "device.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

#define SYCL_KERNEL(...) { \
    sycl::queue Q;\
    Q.submit([&](sycl::handler &h) {__VA_ARGS__}); \
}

#define SYCL_FORALL(i,N,...) \
    ForallWrap1D(N, h, [=] (int i) {__VA_ARGS__})

/// The forall kernel body wrapper
template <typename BODY>
inline void ForallWrap1D(const int N, sycl::handler &h, BODY &&body)
{
   h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> k) {body(k);});
   return;
}

////////////////////////////////////////////////////////////////////////////
// SYCL_FORALL with a 3D CUDA block, sycl::h_item<3> &itm
#define SYCL_FORALL_3D(i,N,X,Y,Z,...) \
   ForallWrap3D<X,Y,Z>(N, h, [=] (const sycl::stream &kout, int i, sycl::group<1> grp){__VA_ARGS__})

/// The forall kernel body wrapper
template <int X, int Y, int Z, typename BODY>
inline void ForallWrap3D(const int N, sycl::handler &h, BODY &&body)
{
   if (N == 0) { return; }
   const sycl::range<1> GRID(N);
   const sycl::range<1> BLCK(2*2*2); // 8 for Gen9, ? for Gen12 // X*Y*Z ?
   sycl::stream kout(N*X*Y*Z+8192, 256, h);
   h.parallel_for_work_group(GRID, BLCK, [=](sycl::group<1> grp)
   {
      SyKernel3D(kout, body, grp);
   });
   return;
}

template <typename BODY> static
void SyKernel3D(const sycl::stream &kout, BODY body, sycl::group<1> grp)
{
   const int k = grp.get_id(0);
   body(kout, k, grp);
}

/// Get the number of SYCL devices
int SyGetDeviceCount();

/** @brief Function that determines if an SYCL kernel should be used, based on
    the current mfem::Device configuration. */
inline bool DeviceCanUseSycl() { return Device::Allows(Backend::SYCL); }

} // namespace mfem

#endif // MFEM_USE_SYCL

#endif // MFEM_SYCL_HPP
