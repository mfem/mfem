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

#ifndef MFEM_FORALL_HPP
#define MFEM_FORALL_HPP

#include "../config/config.hpp"
#include "annotation.hpp"
#include "error.hpp"
#include "backends.hpp"
#include "device.hpp"
#include "mem_manager.hpp"
#include "../linalg/dtensor.hpp"
#include "mdsmem.hpp"

namespace mfem
{

// Maximum size of dofs and quads in 1D.
#ifdef MFEM_USE_HIP
const int MAX_D1D = 10;
const int MAX_Q1D = 10;
#else
const int MAX_D1D = 14;
const int MAX_Q1D = 14;
#endif

// MFEM pragma macros that can be used inside MFEM_FORALL macros.
#define MFEM_PRAGMA(X) _Pragma(#X)

// MFEM_UNROLL pragma macro that can be used inside MFEM_FORALL macros.
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
#define MFEM_UNROLL(N) MFEM_PRAGMA(unroll(N))
#define MFEM_UNROLL_DISABLED MFEM_PRAGMA(unroll(1))
#else
#define MFEM_UNROLL(N)
#define MFEM_UNROLL_DISABLED
#endif

// MFEM_GPU_FORALL: "parallel for" executed with CUDA or HIP based on the MFEM
// build-time configuration (MFEM_USE_CUDA or MFEM_USE_HIP). If neither CUDA nor
// HIP is enabled, this macro is a no-op.
#if defined(MFEM_USE_CUDA)
#define MFEM_GPU_FORALL(i, N,...) CuWrap1D(N, [=] MFEM_DEVICE      \
                                       (int i) {__VA_ARGS__})
#elif defined(MFEM_USE_HIP)
#define MFEM_GPU_FORALL(i, N,...) HipWrap1D(N, [=] MFEM_DEVICE     \
                                        (int i) {__VA_ARGS__})
#else
#define MFEM_GPU_FORALL(i, N,...) do { } while (false)
#endif

// Implementation of MFEM's "parallel for" (forall) device/host kernel
// interfaces supporting RAJA, CUDA, OpenMP, and sequential backends.

// The MFEM_FORALL wrapper
#define MFEM_FORALL(i,N,...) \
   ForallWrap<1>(true,N,[=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__})

// MFEM_FORALL with a 2D CUDA block
#define MFEM_FORALL_2D(i,N,X,Y,BZ,...) \
   ForallWrap<2>(true,N,[=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__},X,Y,BZ)

// MFEM_FORALL with a 3D CUDA block
#define MFEM_FORALL_3D(i,N,X,Y,Z,...) \
   ForallWrap<3>(true,N,[=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__},X,Y,Z)

// MFEM_FORALL with a 3D CUDA block and grid
// With G=0, this is the same as MFEM_FORALL_3D(i,N,X,Y,Z,...)
#define MFEM_FORALL_3D_GRID(i,N,X,Y,Z,G,...) \
   ForallWrap<3>(true,N,[=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__},X,Y,Z,G)

// MFEM_FORALL that uses the basic CPU backend when use_dev is false. See for
// example the functions in vector.cpp, where we don't want to use the mfem
// device for operations on small vectors.
#define MFEM_FORALL_SWITCH(use_dev,i,N,...) \
   ForallWrap<1>(use_dev,N,[=] MFEM_HOST_DEVICE (int i) {__VA_ARGS__})


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
   RAJA::LaunchPolicy<RAJA::cuda_launch_t<true>>;
using cuda_teams_x =
   RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
using cuda_threads_z =
   RAJA::LoopPolicy<RAJA::cuda_thread_z_direct>;
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
using hip_launch_policy =
   RAJA::LaunchPolicy<RAJA::hip_launch_t<true>>;
using hip_teams_x =
   RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
using hip_threads_z =
   RAJA::LoopPolicy<RAJA::hip_thread_z_direct>;
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

   using namespace RAJA;
   using RAJA::RangeSegment;

   launch<cuda_launch_policy>
   (LaunchParams(Teams(G), Threads(X, Y, BZ)),
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
   using namespace RAJA;
   using RAJA::RangeSegment;

   launch<cuda_launch_policy>
   (LaunchParams(Teams(GRID), Threads(X, Y, Z)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<cuda_teams_x>(ctx, RangeSegment(0, N), d_body);

   });

   MFEM_GPU_CHECK(cudaGetLastError());
}

template <int Dim>
struct RajaCuWrap;

template <>
struct RajaCuWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaCuWrap1D<BLCK>(N, d_body);
   }
};

template <>
struct RajaCuWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaCuWrap2D(N, d_body, X, Y, Z);
   }
};

template <>
struct RajaCuWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaCuWrap3D(N, d_body, X, Y, Z, G);
   }
};

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

   using namespace RAJA;
   using RAJA::RangeSegment;

   launch<hip_launch_policy>
   (LaunchParams(Teams(G), Threads(X, Y, BZ)),
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
   using namespace RAJA;
   using RAJA::RangeSegment;

   launch<hip_launch_policy>
   (LaunchParams(Teams(GRID), Threads(X, Y, Z)),
    [=] RAJA_DEVICE (LaunchContext ctx)
   {

      loop<hip_teams_x>(ctx, RangeSegment(0, N), d_body);

   });

   MFEM_GPU_CHECK(hipGetLastError());
}

template <int Dim>
struct RajaHipWrap;

template <>
struct RajaHipWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaHipWrap1D<BLCK>(N, d_body);
   }
};

template <>
struct RajaHipWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaHipWrap2D(N, d_body, X, Y, Z);
   }
};

template <>
struct RajaHipWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RajaHipWrap3D(N, d_body, X, Y, Z, G);
   }
};

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
void CuKernel2D(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k);
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DSmem(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   extern __shared__ Tsmem smem[];
   body(k, smem);
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k, smem + smem_size*blockIdx.x);
}

template <typename BODY> __global__ static
void CuKernel3D(const int N, BODY body)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k); }
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DSmem(const int N, BODY body)
{
   extern __shared__ Tsmem smem[];
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k, smem); }
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x)
   {
      body(k, smem + smem_size*blockIdx.x);
   }
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
   CuKernel2D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem2D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int BZ, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ > 0, "");
   MFEM_VERIFY(G == 0, "Grid not implemented!");
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   const dim3 BLCK(X,Y,BZ);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      const int GRID = (N+BZ-1)/BZ;
      CuKernel2DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = SM;
      dbg("\033[33mFolding back to GLOBAL memory!");
      static Memory<Tsmem> smem(smem_size*sizeof(Tsmem)*GRID);
      smem.UseDevice(true);
      CuKernel2DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
   }
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

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem3D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   const dim3 BLCK(X,Y,Z);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      const int NB = X*Y*Z < 16 ? 4 : 1;
      const int GRID_X = (N + NB - 1) / NB;
      const int GRID = G == 0 ? GRID_X : G;
      CuKernel3DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = G == 0 ? SM : G;
      dbg("\033[33mFolding back to GLOBAL memory (GRID:%d)!", GRID);
      Memory<Tsmem> smem(smem_size*sizeof(Tsmem)*GRID);
      smem.UseDevice(true);
      CuKernel3DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
   }
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <int Dim> struct CuWrap;
template <int Dim, typename Tsmem> struct CuWrapSmem;

template <>
struct CuWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap1D<BLCK>(N, d_body);
   }
};

template <>
struct CuWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap2D(N, d_body, X, Y, Z);
   }
};

template <typename Tsmem>
struct CuWrapSmem<2,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem2D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};

template <>
struct CuWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap3D(N, d_body, X, Y, Z, G);
   }
};

template <typename Tsmem>
struct CuWrapSmem<3,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem3D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};

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
void HipKernel2D(const int N, BODY body)
{
   const int k = hipBlockIdx_x*hipBlockDim_z + hipThreadIdx_z;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void HipKernel3D(const int N, BODY body)
{
   for (int k = hipBlockIdx_x; k < N; k += hipGridDim_x) { body(k); }
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
   hipLaunchKernelGGL(HipKernel2D,GRID,BLCK,0,0,N,d_body);
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

template <int Dim>
struct HipWrap;

template <>
struct HipWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      HipWrap1D<BLCK>(N, d_body);
   }
};

template <>
struct HipWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      HipWrap2D(N, d_body, X, Y, Z);
   }
};

template <>
struct HipWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      HipWrap3D(N, d_body, X, Y, Z, G);
   }
};

#endif // MFEM_USE_HIP


/// The forall kernel body wrapper
template <const int DIM, typename d_lambda, typename h_lambda>
inline void ForallWrap(const bool use_dev, const int N,
                       d_lambda &&d_body, h_lambda &&h_body,
                       const int X=0, const int Y=0, const int Z=0,
                       const int G=0)
{
   MFEM_CONTRACT_VAR(X);
   MFEM_CONTRACT_VAR(Y);
   MFEM_CONTRACT_VAR(Z);
   MFEM_CONTRACT_VAR(G);
   MFEM_CONTRACT_VAR(d_body);
   if (!use_dev) { goto backend_cpu; }

#ifdef MFEM_USE_SYCL
   // If Backend::SYCL_GPU or Backend::SYCL_CPU are allowed, use them
   if (Device::Allows(Backend::SYCL_GPU | Backend::SYCL_CPU))
   {
      return SyclWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
   // If Backend::RAJA_CUDA is allowed, use it
   if (Device::Allows(Backend::RAJA_CUDA))
   {
      return RajaCuWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
   // If Backend::RAJA_HIP is allowed, use it
   if (Device::Allows(Backend::RAJA_HIP))
   {
      return RajaHipWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif


#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      return CuWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#ifdef MFEM_USE_HIP
   // If Backend::HIP is allowed, use it
   if (Device::Allows(Backend::HIP))
   {
      return HipWrap<DIM>::run(N, d_body, X, Y, Z, G);
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

#ifdef MFEM_USE_SYCL
   // If Backend::SYCL_HOST is allowed, use it
   if (Device::Allows(Backend::SYCL_HOST))
   {
      return SyclWrap<DIM>::run(N, h_body, X, Y, Z, G);
   }
#endif

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   for (int k = 0; k < N; k++) { h_body(k); }
}

template <const int DIM, typename lambda>
inline void ForallWrap(const bool use_dev, const int N, lambda &&body,
                       const int X=0, const int Y=0, const int Z=0,
                       const int G=0)
{
   ForallWrap<DIM>(use_dev, N, body, body, X, Y, Z, G);
}

template<typename lambda>
inline void forall(int N, lambda &&body) { ForallWrap<1>(true, N, body); }

template<typename lambda>
inline void forall_switch(bool use_dev, int N, lambda &&body)
{
   ForallWrap<1>(use_dev, N, body);
}

template<typename lambda>
inline void forall_2D(int N, int X, int Y, lambda &&body)
{
   ForallWrap<2>(true, N, body, X, Y, 1);
}

template<typename lambda>
inline void forall_2D_batch(int N, int X, int Y, int BZ, lambda &&body)
{
   ForallWrap<2>(true, N, body, X, Y, BZ);
}

template<typename lambda>
inline void forall_3D(int N, int X, int Y, int Z, lambda &&body)
{
   ForallWrap<3>(true, N, body, X, Y, Z, 0);
}

template<typename lambda>
inline void forall_3D_grid(int N, int X, int Y, int Z, int G, lambda &&body)
{
   ForallWrap<3>(true, N, body, X, Y, Z, G);
}

/// The SMEM forall kernel body wrapper
template <const int DIM, typename Tsmem = double, typename d_lambda, typename h_lambda>
inline void ForallWrapSmem(const bool use_dev, const int N,
                           d_lambda &&d_body, h_lambda &&h_body,
                           const int smem_size,
                           const int X=0, const int Y=0, const int Z=0,
                           const int G=0)
{
   MFEM_CONTRACT_VAR(X);
   MFEM_CONTRACT_VAR(Y);
   MFEM_CONTRACT_VAR(Z);
   MFEM_CONTRACT_VAR(G);
   MFEM_CONTRACT_VAR(d_body);
   MFEM_CONTRACT_VAR(smem_size);
   if (!use_dev) { goto backend_cpu; }

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      return CuWrapSmem<DIM,Tsmem>::run(N, d_body, smem_size, X, Y, Z, G);
   }
#endif

   // If Backend::DEBUG_DEVICE is allowed, use it
   if (Device::Allows(Backend::DEBUG_DEVICE)) { goto backend_cpu; }

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   MFEM_VERIFY(smem_size > 0, "smem_size should be positive!");
   Tsmem smem[smem_size];
   for (int k = 0; k < N; k++) { h_body(k,smem); }
}

template <const int DIM, typename Tsmem = double, typename lambda>
inline void ForallWrapSmem(const bool use_dev, const int N, lambda &&body,
                           const int smem_bytes,
                           const int X=0, const int Y=0, const int Z=0,
                           const int G=0)
{
   ForallWrapSmem<DIM,Tsmem>(use_dev, N, body, body, smem_bytes, X, Y, Z, G);
}

template<typename Tsmem = double, typename lambda>
inline void forall_2D_batch_smem(int N, int X, int Y, int BZ, int smem_bytes,
                                 lambda &&body)
{
   ForallWrapSmem<2,Tsmem>(true, N, body, smem_bytes, X, Y, BZ, 0);
}

template<typename Tsmem = double, typename lambda>
inline void forall_3D_smem(int N, int X, int Y, int Z, int smem_bytes,
                           lambda &&body)
{
   ForallWrapSmem<3,Tsmem>(true, N, body, smem_bytes, X, Y, Z, 0);
}

} // namespace mfem

#endif // MFEM_FORALL_HPP
