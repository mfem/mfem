// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#include "../config/config.hpp"
#include "device.hpp"
#include "error.hpp"

#ifdef MFEM_USE_RAJA

// The following two definitions suppress CUB and THRUST deprecation warnings
// about requiring c++14 with c++11 deprecated but still supported (to be
// removed in a future release).
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#include "RAJA/RAJA.hpp"
using namespace RAJA::expt;

#if defined(RAJA_ENABLE_CUDA) && !defined(MFEM_USE_CUDA)
#error When RAJA is built with CUDA, MFEM_USE_CUDA=YES is required
#endif

namespace RAJA
{

/// RAJA Cuda backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
using cuda_launch_policy = LaunchPolicy<null_launch_t, cuda_launch_t<true>>;
using cuda_teams_x = LoopPolicy<loop_exec, cuda_block_x_direct>;
using cuda_threads_z = LoopPolicy<loop_exec, cuda_thread_z_direct>;

template <typename ExecutionPolicy, typename... Args,
          typename Res = typename resources::get_resource<ExecutionPolicy>::type >
RAJA_INLINE resources::EventProxy<Res> default_forall(Args&&... args)
{
   const int dev = 0; // could use Device::GetId()
   Res cuda = Res::CudaFromStream(0, dev);
   return policy_by_value_interface::forall(ExecutionPolicy(), cuda,
                                            std::forward<Args>(args)...);
}

template <bool async, typename BODY_IN> inline
void default_exec(LaunchContext const &ctx, BODY_IN &&body_in)
{
   using BODY = camp::decay<BODY_IN>;
   auto func = launch_global_fcn<BODY>;

   const int dev = 0; // could use Device::GetId()
   resources::Cuda cuda = resources::Cuda::CudaFromStream(0, dev);

   // Compute the number of blocks and threads
   cuda_dim_t gridSize{ static_cast<cuda_dim_member_t>(ctx.teams.value[0]),
                        static_cast<cuda_dim_member_t>(ctx.teams.value[1]),
                        static_cast<cuda_dim_member_t>(ctx.teams.value[2]) };

   cuda_dim_t blockSize{ static_cast<cuda_dim_member_t>(ctx.threads.value[0]),
                         static_cast<cuda_dim_member_t>(ctx.threads.value[1]),
                         static_cast<cuda_dim_member_t>(ctx.threads.value[2]) };

   // Only launch kernel if we have something to iterate over
   constexpr cuda_dim_member_t zero = 0;
   if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero )
   {
      RAJA_FT_BEGIN;
      // Setup shared memory buffers
      size_t shmem = 0;
      {
         // Privatize the loop_body, using make_launch_body to setup reductions
         BODY body = cuda::make_launch_body(gridSize, blockSize, shmem, cuda,
                                            std::forward<BODY_IN>(body_in));
         // Launch the kernel
         void *args[] = {(void*)&ctx, (void*)&body};
         cuda::launch((const void*)func, gridSize, blockSize, args, shmem,
                      cuda, async, ctx.kernel_name);
      }
      RAJA_FT_END;
   }
}

template <bool async, typename POLICY_LIST, typename BODY>
void default_launch(ExecPlace place, Grid const &grid, BODY const &body)
{
   switch (place)
   {
      case HOST:
      {
         using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
         launch_t::exec(LaunchContext(grid), body);
         break;
      }
#ifdef RAJA_DEVICE_ACTIVE
      case DEVICE:
      {
         default_exec<async>(LaunchContext(grid), body);
         break;
      }
#endif
      default:
         RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
   }
}

template <const int BLOCKS = MFEM_CUDA_BLOCKS, typename DBODY>
void CudaWrap1D(const int N, DBODY &&d_body)
{
   constexpr bool async = true;
   default_forall<cuda_exec<BLOCKS, async>>(RangeSegment(0,N),d_body);
}

template <typename DBODY>
void CudaWrap2D(const int N, DBODY &&d_body,
                const int X, const int Y, const int BZ)
{
   MFEM_VERIFY(N>0, "");
   MFEM_VERIFY(BZ>0, "");
   const int G = (N+BZ-1)/BZ;
   constexpr bool async = true;

   default_launch<async, cuda_launch_policy>(DEVICE,
                                             Grid(Teams(G), Threads(X,Y,BZ)),
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
   cudaErrchk(cudaGetLastError());
}

template <typename DBODY>
void CudaWrap3D(const int N, DBODY &&d_body,
                const int X, const int Y, const int Z, const int G)
{
   MFEM_VERIFY(N>0, "");
   const int GRID = G == 0 ? N : G;
   constexpr bool async = true;
   default_launch<async, cuda_launch_policy>(DEVICE,
                                             Grid(Teams(GRID),Threads(X, Y, Z)),
                                             [=] RAJA_DEVICE (LaunchContext ctx)
   { loop<RAJA::cuda_teams_x>(ctx, RangeSegment(0, N), d_body); });
   cudaErrchk(cudaGetLastError());
}

#endif // RAJA Cuda backend

/// RAJA Hip backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
using hip_launch_policy = LaunchPolicy<null_launch_t, hip_launch_t<true>>;
using hip_teams_x = LoopPolicy<loop_exec,hip_block_x_direct>;
using hip_threads_z = LoopPolicy<loop_exec,hip_thread_z_direct>;

template <typename ExecutionPolicy, typename... Args,
          typename Res = typename resources::get_resource<ExecutionPolicy>::type >
RAJA_INLINE resources::EventProxy<Res> default_forall(Args&&... args)
{
   const int dev = 0; // could use Device::GetId()
   Res hip = Res::HipFromStream(0, dev);
   return policy_by_value_interface::forall(ExecutionPolicy(), hip,
                                            std::forward<Args>(args)...);
}

template <bool async, typename BODY_IN> inline
void default_exec(LaunchContext const &ctx, BODY_IN &&body_in)
{
   using BODY = camp::decay<BODY_IN>;
   auto func = launch_global_fcn<BODY>;

   const int dev = 0; // could use Device::GetId()
   resources::Hip hip = resources::Hip::HipFromStream(0, dev);

   // Compute the number of blocks and threads
   hip_dim_t gridSize{ static_cast<hip_dim_member_t>(ctx.teams.value[0]),
                       static_cast<hip_dim_member_t>(ctx.teams.value[1]),
                       static_cast<hip_dim_member_t>(ctx.teams.value[2]) };

   hip_dim_t blockSize{ static_cast<hip_dim_member_t>(ctx.threads.value[0]),
                        static_cast<hip_dim_member_t>(ctx.threads.value[1]),
                        static_cast<hip_dim_member_t>(ctx.threads.value[2]) };

   // Only launch kernel if we have something to iterate over
   constexpr hip_dim_member_t zero = 0;
   if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero )
   {
      RAJA_FT_BEGIN;
      // Setup shared memory buffers
      size_t shmem = 0;
      {
         // Privatize the loop_body, using make_launch_body to setup reductions
         BODY body = hip::make_launch_body(gridSize, blockSize, shmem, hip,
                                           std::forward<BODY_IN>(body_in));

         // Launch the kernel
         void *args[] = {(void*)&ctx, (void*)&body};
         hip::launch((const void*)func, gridSize, blockSize, args, shmem,
                     hip, async, ctx.kernel_name);
      }
      RAJA_FT_END;
   }
}

template <bool async, typename POLICY_LIST, typename BODY>
void default_launch(ExecPlace place, Grid const &grid, BODY const &body)
{
   switch (place)
   {
      case HOST:
      {
         using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
         launch_t::exec(LaunchContext(grid), body);
         break;
      }
#ifdef RAJA_DEVICE_ACTIVE
      case DEVICE:
      {
         default_exec<async>(LaunchContext(grid), body);
         break;
      }
#endif
      default:
         RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
   }
}

template <const int BLOCKS = MFEM_HIP_BLOCKS, typename DBODY>
void HipWrap1D(const int N, DBODY &&d_body)
{
   constexpr bool async = true;
   default_forall<hip_exec<BLOCKS, async>>(RangeSegment(0,N),d_body);
}

template <typename DBODY>
void HipWrap2D(const int N, DBODY &&d_body,
               const int X, const int Y, const int BZ)
{
   MFEM_VERIFY(N>0, "");
   MFEM_VERIFY(BZ>0, "");
   const int G = (N+BZ-1)/BZ;
   constexpr bool async = true;

   default_launch<async, hip_launch_policy>(DEVICE,
                                            Grid(Teams(G), Threads(X,Y,BZ)),
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
   cudaErrchk(hipGetLastError());
}

template <typename DBODY>
void HipWrap3D(const int N, DBODY &&d_body,
               const int X, const int Y, const int Z, const int G)
{
   MFEM_VERIFY(N>0, "");
   const int GRID = G == 0 ? N : G;
   constexpr bool async = true;
   default_launch<async, hip_launch_policy>(DEVICE,
                                            Grid(Teams(GRID), Threads(X, Y, Z)),
                                            [=] RAJA_DEVICE (LaunchContext ctx)
   { loop<hip_teams_x>(ctx, RangeSegment(0, N), d_body); });
   cudaErrchk(hipGetLastError());
}
#endif // RAJA Hip backend

/// RAJA OpenMP backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
template <typename HBODY>
void OpenMPWrap(const int N, HBODY &&h_body)
{
   forall<omp_parallel_for_exec>(RangeSegment(0,N), h_body);
}
#endif // RAJA OpenMP backend


/// RAJA sequential loop backend
template <typename HBODY>
void SeqWrap(const int N, HBODY &&h_body)
{
#ifdef MFEM_USE_RAJA
   forall<loop_exec>(RangeSegment(0,N), h_body);
#else
   MFEM_CONTRACT_VAR(N);
   MFEM_CONTRACT_VAR(h_body);
   MFEM_ABORT("RAJA requested but RAJA is not enabled!");
#endif
}

} // RAJA namespace

#endif // MFEM_USE_RAJA

#endif // MFEM_RAJA_HPP
