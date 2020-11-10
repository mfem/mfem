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

#include "../general/debug.hpp"
#define MFEM_DEVICE
#define MFEM_LAMBDA
#define MFEM_HOST_DEVICE
#define MFEM_DEVICE_SYNC
#define MFEM_STREAM_SYNC
#define MFEM_GPU_CHECK(x)

// Define the MFEM inner threading macros
#if defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
#define MFEM_SHARED
#define SYCL_SYNC_THREAD itm.barrier();
// item.get_sub_group().barrier();
// itm.barrier(access::fence_space::local_space); // p343
#define SYCL_THREAD_ID(k) itm.get_logical_local_id(k);
#define SYCL_THREAD_SIZE(k) itm.get_local_range(k);
#define SYCL_FOREACH_THREAD(i,k,N) for(int i=itm.get_local_id(k); i<N; i+=itm.get_local_range(k))

#define SYCL_FOREACH(...)
#else
#define SYCL_FOREACH(...)
#endif

#include "mem_manager.hpp"
#include "device.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

/// Return the default sycl::queue used by MFEM.
sycl::queue &SyclQueue();

#define SYCL_KERNEL(...) { \
    sycl::queue &Q = SyclQueue();\
    Q.submit([&](sycl::handler &h) {__VA_ARGS__}); \
    Q.wait();\
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
   ForallWrap3D<X,Y,Z>(N, h, [=] (const sycl::stream &kout,\
                                  int i, sycl::nd_item<3> itm) {__VA_ARGS__})

/// The forall kernel body wrapper
template <int X, int Y, int Z, typename BODY>
inline void ForallWrap3D(const int N, sycl::handler &h, BODY &&body)
{
   if (N == 0) { return; }
   MFEM_VERIFY(X==Y && Y==Z,"");
   constexpr int B = X*Y*Z;
   const int L = static_cast<int>(floor(cbrt(N)));
   dbg("B:%d, L:%d", B, L);
   const sycl::range<3> GRID(L*X,L*Y,L*Z);
   const sycl::range<3> BLCK(X,Y,Z);
   sycl::stream kout(L*L*L*X*Y*Z+16384, 256, h);

   h.parallel_for(sycl::nd_range<3>(GRID, BLCK), [=](sycl::nd_item<3> itm)
   {
      const int I = itm.get_global_id(0);
      const int J = itm.get_global_id(1);
      const int K = itm.get_global_id(2);
      const int i = itm.get_local_id(0);
      const int j = itm.get_local_id(1);
      const int k = itm.get_local_id(2);
      kout << "[" << itm.get_global_linear_id() << "] "
           << "g(" << I << "," << J << "," << K << ")"
           << " : "
           << "(" << ((I)/L) << "," << ((J)/L) << "," << ((K)/L) << ")"
           << " => "
           << "l(" << i << "," << j << "," << k << ")" << sycl::endl;
      SyKernel3D(kout, body, itm);
   });
   return;
}

template <typename BODY> static
void SyKernel3D(const sycl::stream &kout, BODY body, sycl::nd_item<3> itm)
{
   const int e = itm.get_global_linear_id();
   body(kout, e, itm);
}

/// Get the number of SYCL devices
int SyGetDeviceCount();

/** @brief Function that determines if an SYCL kernel should be used, based on
    the current mfem::Device configuration. */
inline bool DeviceCanUseSycl() { return Device::Allows(Backend::SYCL); }

} // namespace mfem

#endif // MFEM_USE_SYCL

#endif // MFEM_SYCL_HPP
