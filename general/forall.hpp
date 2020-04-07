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

#ifndef MFEM_FORALL_HPP
#define MFEM_FORALL_HPP

#include "../config/config.hpp"
#include "error.hpp"
#include "cuda.hpp"
#include "hip.hpp"
#include "occa.hpp"
#include "device.hpp"
#include "mem_manager.hpp"
#include "../linalg/dtensor.hpp"

#ifdef MFEM_USE_RAJA
#include "RAJA/RAJA.hpp"
#if defined(RAJA_ENABLE_CUDA) && !defined(MFEM_USE_CUDA)
#error When RAJA is built with CUDA, MFEM_USE_CUDA=YES is required
#endif
#endif

namespace mfem
{

// Maximum size of dofs and quads in 1D.
const int MAX_D1D = 14;
const int MAX_Q1D = 14;

// Implementation of MFEM's "parallel for" (forall) device/host kernel
// interfaces supporting RAJA, CUDA, OpenMP, and sequential backends.

// The MFEM_FORALL wrapper
#define MFEM_FORALL(i,N,...)                             \
   ForallWrap<1>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&]             (int i) {__VA_ARGS__})

// MFEM_FORALL with a 2D CUDA block
#define MFEM_FORALL_2D(i,N,X,Y,BZ,...)                   \
   ForallWrap<2>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&]             (int i) {__VA_ARGS__},  \
                 X,Y,BZ)

// MFEM_FORALL with a 3D CUDA block
#define MFEM_FORALL_3D(i,N,X,Y,Z,...)                    \
   ForallWrap<3>(true,N,                                 \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&]             (int i) {__VA_ARGS__},  \
                 X,Y,Z)

// MFEM_FORALL that uses the basic CPU backend when use_dev is false. See for
// example the functions in vector.cpp, where we don't want to use the mfem
// device for operations on small vectors.
#define MFEM_FORALL_SWITCH(use_dev,i,N,...)              \
   ForallWrap<1>(use_dev,N,                              \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&]             (int i) {__VA_ARGS__})


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
   MFEM_ABORT("OpenMP requested for MFEM but OpenMP is not enabled!");
#endif
}


/// RAJA Cuda backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)

using RAJA::statement::Segs;

template <const int BLOCKS = MFEM_CUDA_BLOCKS, typename DBODY>
void RajaCudaWrap1D(const int N, DBODY &&d_body)
{
   //true denotes asynchronous kernel
   RAJA::forall<RAJA::cuda_exec<BLOCKS,true>>(RAJA::RangeSegment(0,N),d_body);
}

template <typename DBODY>
void RajaCudaWrap2D(const int N, DBODY &&d_body,
                    const int X, const int Y, const int BZ)
{
   MFEM_VERIFY(N>0, "");
   MFEM_VERIFY(BZ>0, "");
   const int G = (N+BZ-1)/BZ;
   RAJA::kernel<RAJA::KernelPolicy<
   RAJA::statement::CudaKernelAsync<
   RAJA::statement::For<0, RAJA::cuda_block_x_direct,
        RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
        RAJA::statement::For<2, RAJA::cuda_thread_y_direct,
        RAJA::statement::For<3, RAJA::cuda_thread_z_direct,
        RAJA::statement::Lambda<0, Segs<0>>>>>>>>>
        (RAJA::make_tuple(RAJA::RangeSegment(0,G), RAJA::RangeSegment(0,X),
                          RAJA::RangeSegment(0,Y), RAJA::RangeSegment(0,BZ)),
         [=] RAJA_DEVICE (const int n)
   {
      const int k = n*BZ + threadIdx.z;
      if (k >= N) { return; }
      d_body(k);
   });
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void RajaCudaWrap3D(const int N, DBODY &&d_body,
                    const int X, const int Y, const int Z)
{
   MFEM_VERIFY(N>0, "");
   RAJA::kernel<RAJA::KernelPolicy<
   RAJA::statement::CudaKernelAsync<
   RAJA::statement::For<0, RAJA::cuda_block_x_direct,
        RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
        RAJA::statement::For<2, RAJA::cuda_thread_y_direct,
        RAJA::statement::For<3, RAJA::cuda_thread_z_direct,
        RAJA::statement::Lambda<0, Segs<0>>>>>>>>>
        (RAJA::make_tuple(RAJA::RangeSegment(0,N), RAJA::RangeSegment(0,X),
                          RAJA::RangeSegment(0,Y), RAJA::RangeSegment(0,Z)),
   [=] RAJA_DEVICE (const int k) { d_body(k); });
   MFEM_GPU_CHECK(cudaGetLastError());
}

#endif


/// RAJA OpenMP backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)

using RAJA::statement::Segs;

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
   const int k = blockIdx.x;
   if (k >= N) { return; }
   body(k);
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
              const int X, const int Y, const int Z)
{
   if (N==0) { return; }
   const int GRID = N;
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
               const int X, const int Y, const int Z)
{
   if (N==0) { return; }
   const int GRID = N;
   const dim3 BLCK(X,Y,Z);
   hipLaunchKernelGGL(HipKernel3D,GRID,BLCK,0,0,N,d_body);
   MFEM_GPU_CHECK(hipGetLastError());
}

#endif // MFEM_USE_HIP


/// The forall kernel body wrapper
template <const int DIM, typename DBODY, typename HBODY>
inline void ForallWrap(const bool use_dev, const int N,
                       DBODY &&d_body, HBODY &&h_body,
                       const int X=0, const int Y=0, const int Z=0)
{
   if (!use_dev) { goto backend_cpu; }

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
   // Handle all allowed CUDA backends except Backend::CUDA
   if (DIM == 1 && Device::Allows(Backend::CUDA_MASK & ~Backend::CUDA))
   { return RajaCudaWrap1D(N, d_body); }

   if (DIM == 2 && Device::Allows(Backend::CUDA_MASK & ~Backend::CUDA))
   { return RajaCudaWrap2D(N, d_body, X, Y, Z); }

   if (DIM == 3 && Device::Allows(Backend::CUDA_MASK & ~Backend::CUDA))
   { return RajaCudaWrap3D(N, d_body, X, Y, Z); }
#endif

#ifdef MFEM_USE_CUDA
   // Handle all allowed CUDA backends
   if (DIM == 1 && Device::Allows(Backend::CUDA_MASK))
   { return CuWrap1D(N, d_body); }

   if (DIM == 2 && Device::Allows(Backend::CUDA_MASK))
   { return CuWrap2D(N, d_body, X, Y, Z); }

   if (DIM == 3 && Device::Allows(Backend::CUDA_MASK))
   { return CuWrap3D(N, d_body, X, Y, Z); }
#endif

#ifdef MFEM_USE_HIP
   // Handle all allowed HIP backends
   if (DIM == 1 && Device::Allows(Backend::HIP_MASK))
   { return HipWrap1D(N, d_body); }

   if (DIM == 2 && Device::Allows(Backend::HIP_MASK))
   { return HipWrap2D(N, d_body, X, Y, Z); }

   if (DIM == 3 && Device::Allows(Backend::HIP_MASK))
   { return HipWrap3D(N, d_body, X, Y, Z); }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
   // Handle all allowed OpenMP backends except Backend::OMP
   if (Device::Allows(Backend::OMP_MASK & ~Backend::OMP))
   { return RajaOmpWrap(N, h_body); }
#endif

#ifdef MFEM_USE_OPENMP
   // Handle all allowed OpenMP backends
   if (Device::Allows(Backend::OMP_MASK)) { return OmpWrap(N, h_body); }
#endif

#ifdef MFEM_USE_RAJA
   // Handle all allowed CPU backends except Backend::CPU
   if (Device::Allows(Backend::CPU_MASK & ~Backend::CPU))
   { return RajaSeqWrap(N, h_body); }
#endif

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   for (int k = 0; k < N; k++) { h_body(k); }
}

} // namespace mfem

#endif // MFEM_FORALL_HPP
