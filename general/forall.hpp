// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FORALL_HPP
#define MFEM_FORALL_HPP

#include "../config/config.hpp"
#include "error.hpp"
#include "backends.hpp"
#include "device.hpp"
#include "mem_manager.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

// Maximum size of dofs and quads in 1D.
const int MAX_D1D = 14;
const int MAX_Q1D = 14;

// MFEM pragma macros that can be used inside MFEM_FORALL macros.
#define MFEM_PRAGMA(X) _Pragma(#X)

// MFEM_UNROLL pragma macro that can be used inside MFEM_FORALL macros.
#if defined(MFEM_USE_CUDA)
#define MFEM_UNROLL(N) MFEM_PRAGMA(unroll(N))
#else
#define MFEM_UNROLL(N)
#endif

// Implementation of MFEM's "parallel for" (forall) device/host kernel
// interfaces supporting RAJA, CUDA, OpenMP, and sequential backends.

// The MFEM_FORALL wrapper
#define MFEM_FORALL(i,N,...)                             \
   ForallWrap<1>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__})

// MFEM_FORALL with a 2D CUDA block
#define MFEM_FORALL_2D(i,N,X,Y,BZ,...)                   \
   ForallWrap<2>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,BZ)

// MFEM_FORALL with a 3D CUDA block
#define MFEM_FORALL_3D(i,N,X,Y,Z,...)                    \
   ForallWrap<3>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,Z)

// MFEM_FORALL with a 3D CUDA block and grid
#define MFEM_FORALL_3D_GRID(i,N,X,Y,Z,G,...)             \
   ForallWrap<3>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,Z,G)

// MFEM_FORALL that uses the basic CPU backend when use_dev is false. See for
// example the functions in vector.cpp, where we don't want to use the mfem
// device for operations on small vectors.
#define MFEM_FORALL_SWITCH(use_dev,i,N,...)              \
   ForallWrap<1>(use_dev,N,                              \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__})


/// OpenMP backend
template <typename HBODY>
void OmpWrap(const int N, HBODY &&h_body)
{
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel for
   for (int k = 0; k < N; k++)
   {
      h_body(k);
   }
#else
   MFEM_CONTRACT_VAR(N);
   MFEM_CONTRACT_VAR(h_body);
   MFEM_ABORT("OpenMP requested for MFEM but OpenMP is not enabled!");
#endif
}


/// RAJA Cuda and Hip backends
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
using cuda_launch_policy =
   RAJA::expt::LaunchPolicy<RAJA::expt::null_launch_t, RAJA::expt::cuda_launch_t<true>>;
using cuda_teams_x =
   RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_block_x_direct>;
using cuda_threads_z =
   RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_thread_z_direct>;
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
using hip_launch_policy =
   RAJA::expt::LaunchPolicy<RAJA::expt::null_launch_t, RAJA::expt::hip_launch_t<true>>;
using hip_teams_x =
   RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::hip_block_x_direct>;
using hip_threads_z =
   RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::hip_thread_z_direct>;
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
template <const int BLOCKS = MFEM_CUDA_BLOCKS, typename DBODY>
void RajaCuWrap1D(const int N, DBODY &&d_body)
{
   //true denotes asynchronous kernel
   RAJA::forall<RAJA::cuda_exec<BLOCKS,true>>(RAJA::RangeSegment(0,N),d_body);
}

template <typename DBODY>
void RajaCuWrap2D(const int N, DBODY &&d_body,
                  const int X, const int Y, const int BZ)
{
   MFEM_VERIFY(N>0, "");
   MFEM_VERIFY(BZ>0, "");
   const int G = (N+BZ-1)/BZ;

   using namespace RAJA::expt;
   using RAJA::RangeSegment;

   launch<cuda_launch_policy>
   (DEVICE, Resources(Teams(G), Threads(X, Y, BZ)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<cuda_teams_x>(ctx, RangeSegment(0, G), [&] (const int n)
      {

         loop<cuda_threads_z>(ctx, RangeSegment(0, BZ), [&] (const int tz)
         {

            const int k = n*BZ + tz;
            if (k >= N) { return; }
            d_body(k);

         });

      });

   });

   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void RajaCuWrap3D(const int N, DBODY &&d_body,
                  const int X, const int Y, const int Z, const int G)
{
   MFEM_VERIFY(N>0, "");
   const int GRID = G == 0 ? N : G;
   using namespace RAJA::expt;
   using RAJA::RangeSegment;

   launch<cuda_launch_policy>
   (DEVICE, Resources(Teams(N), Threads(X, Y, Z)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<cuda_teams_x>(ctx, RangeSegment(0, N), d_body);

   });

   MFEM_GPU_CHECK(cudaGetLastError());
}

#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
template <const int BLOCKS = MFEM_HIP_BLOCKS, typename DBODY>
void RajaHipWrap1D(const int N, DBODY &&d_body)
{
   //true denotes asynchronous kernel
   RAJA::forall<RAJA::hip_exec<BLOCKS,true>>(RAJA::RangeSegment(0,N),d_body);
}

template <typename DBODY>
void RajaHipWrap2D(const int N, DBODY &&d_body,
                   const int X, const int Y, const int BZ)
{
   MFEM_VERIFY(N>0, "");
   MFEM_VERIFY(BZ>0, "");
   const int G = (N+BZ-1)/BZ;

   using namespace RAJA::expt;
   using RAJA::RangeSegment;

   launch<hip_launch_policy>
   (DEVICE, Resources(Teams(G), Threads(X, Y, BZ)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<hip_teams_x>(ctx, RangeSegment(0, G), [&] (const int n)
      {

         loop<hip_threads_z>(ctx, RangeSegment(0, BZ), [&] (const int tz)
         {

            const int k = n*BZ + tz;
            if (k >= N) { return; }
            d_body(k);

         });

      });

   });

   MFEM_GPU_CHECK(hipGetLastError());
}

template <typename DBODY>
void RajaHipWrap3D(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
{
   MFEM_VERIFY(N>0, "");
   const int GRID = G == 0 ? N : G;
   using namespace RAJA::expt;
   using RAJA::RangeSegment;

   launch<hip_launch_policy>
   (DEVICE, Resources(Teams(GRID), Threads(X, Y, Z)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<hip_teams_x>(ctx, RangeSegment(0, N), d_body);

   });

   MFEM_GPU_CHECK(hipGetLastError());
}
#endif

/// RAJA OpenMP backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)

template <typename HBODY>
void RajaOmpWrap(const int N, HBODY &&h_body)
{
   RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,N), h_body);
}

#endif


/// RAJA sequential loop backend
template <typename HBODY>
void RajaSeqWrap(const int N, HBODY &&h_body)
{
#ifdef MFEM_USE_RAJA
   RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0,N), h_body);
#else
   MFEM_CONTRACT_VAR(N);
   MFEM_CONTRACT_VAR(h_body);
   MFEM_ABORT("RAJA requested but RAJA is not enabled!");
#endif
}


/// CUDA backend
#ifdef MFEM_USE_CUDA

template <typename BODY> __global__ static
void CuKernel1D(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CuKernel2D(const int N, BODY body, const int BZ)
{
   const int k = blockIdx.x*BZ + threadIdx.z;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CuKernel3D(const int N, BODY body)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k); }
}

template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
void CuWrap1D(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   CuKernel1D<<<GRID,BLCK>>>(N, d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap2D(const int N, DBODY &&d_body,
              const int X, const int Y, const int BZ)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ>0, "");
   const int GRID = (N+BZ-1)/BZ;
   const dim3 BLCK(X,Y,BZ);
   CuKernel2D<<<GRID,BLCK>>>(N,d_body,BZ);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap3D(const int N, DBODY &&d_body,
              const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   const int GRID = G == 0 ? N : G;
   const dim3 BLCK(X,Y,Z);
   CuKernel3D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

#endif // MFEM_USE_CUDA


/// HIP backend
#ifdef MFEM_USE_HIP

template <typename BODY> __global__ static
void HipKernel1D(const int N, BODY body)
{
   const int k = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void HipKernel2D(const int N, BODY body, const int BZ)
{
   const int k = hipBlockIdx_x*BZ + hipThreadIdx_z;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void HipKernel3D(const int N, BODY body)
{
   const int k = hipBlockIdx_x;
   if (k >= N) { return; }
   body(k);
}

template <const int BLCK = MFEM_HIP_BLOCKS, typename DBODY>
void HipWrap1D(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   hipLaunchKernelGGL(HipKernel1D,GRID,BLCK,0,0,N,d_body);
   MFEM_GPU_CHECK(hipGetLastError());
}

template <typename DBODY>
void HipWrap2D(const int N, DBODY &&d_body,
               const int X, const int Y, const int BZ)
{
   if (N==0) { return; }
   const int GRID = (N+BZ-1)/BZ;
   const dim3 BLCK(X,Y,BZ);
   hipLaunchKernelGGL(HipKernel2D,GRID,BLCK,0,0,N,d_body,BZ);
   MFEM_GPU_CHECK(hipGetLastError());
}

template <typename DBODY>
void HipWrap3D(const int N, DBODY &&d_body,
               const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   const int GRID = G == 0 ? N : G;
   const dim3 BLCK(X,Y,Z);
   hipLaunchKernelGGL(HipKernel3D,GRID,BLCK,0,0,N,d_body);
   MFEM_GPU_CHECK(hipGetLastError());
}

#endif // MFEM_USE_HIP


/// The forall kernel body wrapper
template <const int DIM, typename DBODY, typename HBODY>
inline void ForallWrap(const bool use_dev, const int N,
                       DBODY &&d_body, HBODY &&h_body,
                       const int X=0, const int Y=0, const int Z=0,
                       const int G=0)
{
   MFEM_CONTRACT_VAR(X);
   MFEM_CONTRACT_VAR(Y);
   MFEM_CONTRACT_VAR(Z);
   MFEM_CONTRACT_VAR(G);
   MFEM_CONTRACT_VAR(d_body);
   if (!use_dev) { goto backend_cpu; }

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
   // If Backend::RAJA_CUDA is allowed, use it
   if (Device::Allows(Backend::RAJA_CUDA))
   {
      if (DIM == 1) { return RajaCuWrap1D(N, d_body); }
      if (DIM == 2) { return RajaCuWrap2D(N, d_body, X, Y, Z); }
      if (DIM == 3) { return RajaCuWrap3D(N, d_body, X, Y, Z, G); }
   }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
   // If Backend::RAJA_HIP is allowed, use it
   if (Device::Allows(Backend::RAJA_HIP))
   {
      if (DIM == 1) { return RajaHipWrap1D(N, d_body); }
      if (DIM == 2) { return RajaHipWrap2D(N, d_body, X, Y, Z); }
      if (DIM == 3) { return RajaHipWrap3D(N, d_body, X, Y, Z, G); }
   }
#endif

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      if (DIM == 1) { return CuWrap1D(N, d_body); }
      if (DIM == 2) { return CuWrap2D(N, d_body, X, Y, Z); }
      if (DIM == 3) { return CuWrap3D(N, d_body, X, Y, Z, G); }
   }
#endif

#ifdef MFEM_USE_HIP
   // If Backend::HIP is allowed, use it
   if (Device::Allows(Backend::HIP))
   {
      if (DIM == 1) { return HipWrap1D(N, d_body); }
      if (DIM == 2) { return HipWrap2D(N, d_body, X, Y, Z); }
      if (DIM == 3) { return HipWrap3D(N, d_body, X, Y, Z, G); }
   }
#endif

   // If Backend::DEBUG_DEVICE is allowed, use it
   if (Device::Allows(Backend::DEBUG_DEVICE)) { goto backend_cpu; }

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
   // If Backend::RAJA_OMP is allowed, use it
   if (Device::Allows(Backend::RAJA_OMP)) { return RajaOmpWrap(N, h_body); }
#endif

#ifdef MFEM_USE_OPENMP
   // If Backend::OMP is allowed, use it
   if (Device::Allows(Backend::OMP)) { return OmpWrap(N, h_body); }
#endif

#ifdef MFEM_USE_RAJA
   // If Backend::RAJA_CPU is allowed, use it
   if (Device::Allows(Backend::RAJA_CPU)) { return RajaSeqWrap(N, h_body); }
#endif

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   for (int k = 0; k < N; k++) { h_body(k); }
}

} // namespace mfem

#endif // MFEM_FORALL_HPP
