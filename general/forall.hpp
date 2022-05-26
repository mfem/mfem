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

#ifndef MFEM_FORALL_HPP
#define MFEM_FORALL_HPP

#include "../config/config.hpp"
#include "annotation.hpp"
#include "error.hpp"
#include "backends.hpp"
#include "device.hpp"
#include "mem_manager.hpp"
#include "../linalg/dtensor.hpp"

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
#if defined(MFEM_USE_CUDA)
#define MFEM_UNROLL(N) MFEM_PRAGMA(unroll(N))
#else
#define MFEM_UNROLL(N)
#endif

// MFEM_GPU_FORALL: "parallel for" executed with CUDA or HIP based on the MFEM
// build-time configuration (MFEM_USE_CUDA or MFEM_USE_HIP). If neither CUDA nor
// HIP is enabled, this macro is a no-op.
#if defined(MFEM_USE_CUDA)
#define MFEM_GPU_FORALL(i, N,...) CudaWrap1D(N, [=] MFEM_DEVICE    \
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
#define MFEM_FORALL(i,N,...)                             \
   ForallWrap<1>(Device::Backends(), true, N,            \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__})

// MFEM_FORALL with a 2D CUDA block
#define MFEM_FORALL_2D(i,N,X,Y,BZ,...)                   \
   ForallWrap<2>(Device::Backends(), true, N,            \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,BZ)

// MFEM_FORALL with a 3D CUDA block
#define MFEM_FORALL_3D(i,N,X,Y,Z,...)                    \
   ForallWrap<3>(Device::Backends(), true, N,            \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,Z)

// MFEM_FORALL with a 3D CUDA block and grid
// With G=0, this is the same as MFEM_FORALL_3D(i,N,X,Y,Z,...)
#define MFEM_FORALL_3D_GRID(i,N,X,Y,Z,G,...)             \
   ForallWrap<3>(Device::Backends(), true, N,            \
                 [=] MFEM_DEVICE (int i) {__VA_ARGS__},  \
                 [&] MFEM_LAMBDA (int i) {__VA_ARGS__},\
                 X,Y,Z,G)

// MFEM_FORALL that uses the basic CPU backend when use_dev is false. See for
// example the functions in vector.cpp, where we don't want to use the mfem
// device for operations on small vectors.
#define MFEM_FORALL_SWITCH(use_dev,i,N,...)              \
   ForallWrap<1>(Device::Backends(), use_dev, N,         \
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

/// RAJA Cuda backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
template <int Dim> struct RajaCudaWrap;

template <> struct RajaCudaWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::CudaWrap1D<BLCK>(N, d_body);
   }
};

template <> struct RajaCudaWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::CudaWrap2D(N, d_body, X, Y, Z);
   }
};

template <> struct RajaCudaWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::CudaWrap3D(N, d_body, X, Y, Z, G);
   }
};

#endif

/// RAJA Hip backend
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
template <int Dim> struct RajaHipWrap;

template <> struct RajaHipWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::HipWrap1D<BLCK>(N, d_body);
   }
};

template <> struct RajaHipWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::HipWrap2D(N, d_body, X, Y, Z);
   }
};

template <> struct RajaHipWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      RAJA::HipWrap3D(N, d_body, X, Y, Z, G);
   }
};
#endif

/// CUDA backend
#ifdef MFEM_USE_CUDA
template <typename BODY> __global__ static
void CudaKernel1D(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CudaKernel2D(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CudaKernel3D(const int N, BODY body)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k); }
}

template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
void CudaWrap1D(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   CudaKernel1D<<<GRID,BLCK>>>(N, d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CudaWrap2D(const int N, DBODY &&d_body,
                const int X, const int Y, const int BZ)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ>0, "");
   const int GRID = (N+BZ-1)/BZ;
   const dim3 BLCK(X,Y,BZ);
   CudaKernel2D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CudaWrap3D(const int N, DBODY &&d_body,
                const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   const int GRID = G == 0 ? N : G;
   const dim3 BLCK(X,Y,Z);
   CudaKernel3D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <int Dim> struct CudaWrap;

template <> struct CudaWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CudaWrap1D<BLCK>(N, d_body);
   }
};

template <> struct CudaWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CudaWrap2D(N, d_body, X, Y, Z);
   }
};

template <> struct CudaWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CudaWrap3D(N, d_body, X, Y, Z, G);
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

template <int Dim> struct HipWrap;

template <> struct HipWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      HipWrap1D<BLCK>(N, d_body);
   }
};

template <> struct HipWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      HipWrap2D(N, d_body, X, Y, Z);
   }
};

template <> struct HipWrap<3>
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
template <const int DIM, typename DBODY, typename HBODY>
inline void ForallWrap(const unsigned long backends, const bool use_dev,
                       const int N, DBODY &&d_body, HBODY &&h_body,
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
   if (backends & Backend::RAJA_CUDA)
   {
      return RajaCudaWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
   // If Backend::RAJA_HIP is allowed, use it
   if (backends & Backend::RAJA_HIP)
   {
      return RajaHipWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (backends & Backend::CUDA)
   {
      return CudaWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

#ifdef MFEM_USE_HIP
   // If Backend::HIP is allowed, use it
   if (backends & Backend::HIP)
   {
      return HipWrap<DIM>::run(N, d_body, X, Y, Z, G);
   }
#endif

   // If Backend::DEBUG_DEVICE is allowed, use it
   if (backends & Backend::DEBUG_DEVICE) { goto backend_cpu; }

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
   // If Backend::RAJA_OMP is allowed, use it
   if (backends & Backend::RAJA_OMP) { return RAJA::OpenMPWrap(N, h_body); }
#endif

#ifdef MFEM_USE_OPENMP
   // If Backend::OMP is allowed, use it
   if (backends & Backend::OMP) { return mfem::OpenMPWrap(N, h_body); }
#endif

#ifdef MFEM_USE_RAJA
   // If Backend::RAJA_CPU is allowed, use it
   if (backends & Backend::RAJA_CPU) { return RAJA::SeqWrap(N, h_body); }
#endif

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   for (int k = 0; k < N; k++) { h_body(k); }
}

} // namespace mfem

#endif // MFEM_FORALL_HPP
