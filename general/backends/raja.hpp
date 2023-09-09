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

#ifndef MFEM_RAJA_HPP
#define MFEM_RAJA_HPP

#include "../../config/config.hpp"
#include "../error.hpp"

namespace mfem
{

#ifdef MFEM_USE_RAJA
// The following two definitions suppress CUB and THRUST deprecation warnings
// about requiring c++14 with c++11 deprecated but still supported (to be
// removed in a future release).
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#include "RAJA/RAJA.hpp"
#if defined(RAJA_ENABLE_CUDA) && !defined(MFEM_USE_CUDA)
#error When RAJA is built with CUDA, MFEM_USE_CUDA=YES is required
#endif
#endif

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

#endif // defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)


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

#endif // defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)

/// RAJA OpenMP backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)

template <typename HBODY>
void RajaOmpWrap(const int N, HBODY &&h_body)
{
   RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,N), h_body);
}

#endif // defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)

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

} // namespace mfem

#endif // MFEM_RAJA_HPP
