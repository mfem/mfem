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
#include "../linalg/tensor.hpp"

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

// use DeviceVector which needs ../linalg/dtensor.hpp
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
     (defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)))
template<int M = 0>
MFEM_HOST_DEVICE auto GetSmem(double* &smem, std::size_t size)
{
   auto rtn = DeviceVector(smem, size);
   smem += size;
   return rtn;
}
#endif

#if !((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
      (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)) || \
      (defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)))
template<int M = 0>
MFEM_HOST_DEVICE auto GetSmem(double* &smem, std::size_t size)
{
   if constexpr(M == 0)
   {
      auto sm = DeviceVector(smem, size);
      smem += size;
      return sm;
   }
   if constexpr(M > 0) { return internal::tensor<double, M> {}; }
}
#endif

// MFEM pragma macros that can be used inside MFEM_FORALL macros.
#define MFEM_PRAGMA(X) _Pragma(#X)

// MFEM_UNROLL pragma macro that can be used inside MFEM_FORALL macros.
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
#define MFEM_UNROLL(N) MFEM_PRAGMA(unroll(N))
#define MFEM_UNROLL_DEV_DISABLED MFEM_PRAGMA(unroll(1))
#elif defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
#define MFEM_UNROLL(N) MFEM_PRAGMA(unroll(N))
#define MFEM_UNROLL_DEV_DISABLED
#else
#define MFEM_UNROLL(N)
#define MFEM_UNROLL_DEV_DISABLED MFEM_PRAGMA(unroll)
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

// The forall kernel body wrapper
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

// The forall kernel with dynamic shared memory body wrapper
template <const int DIM, typename Tsmem = double, typename d_lambda, typename h_lambda>
inline void ForallWrapSmem(const bool use_dev, const int N, const int smem_size,
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
      return SyclWrapSmem<DIM, Tsmem>::run(N, d_body, smem_size, X, Y, Z, G);
   }
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
   MFEM_ABORT("RAJA with dynamic shared memory not implemented");
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_HIP)
   MFEM_ABORT("RAJA with dynamic shared memory not implemented");
#endif

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      return CuWrapSmem<DIM,Tsmem>::run(N, d_body, smem_size, X, Y, Z, G);
   }
#endif

#ifdef MFEM_USE_HIP
   MFEM_ABORT("HIP with dynamic shared memory not implemented");
#endif

   // If Backend::DEBUG_DEVICE is allowed, use it
   if (Device::Allows(Backend::DEBUG_DEVICE)) { goto backend_cpu; }

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
   MFEM_ABORT("RAJA with dynamic shared memory not implemented");
#endif

#ifdef MFEM_USE_OPENMP
   MFEM_ABORT("OpenMP with dynamic shared memory not implemented");
#endif

#ifdef MFEM_USE_RAJA
   MFEM_ABORT("RAJA with dynamic shared memory not implemented");
#endif

#ifdef MFEM_USE_SYCL
   // If Backend::SYCL_HOST is allowed, use it
   if (Device::Allows(Backend::SYCL_HOST))
   {
      return SyclWrapSmem<DIM,Tsmem>::run(N, h_body, smem_size, X, Y, Z, G);
   }
#endif

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.

   Tsmem smem[smem_size];
   for (int k = 0; k < N; k++) { h_body(k, smem); }

   //alignas(alignof(Tsmem))
   //Tsmem sm00[smem_size/6], sm01[smem_size/6], sm02[smem_size/6];
   //Tsmem sm10[smem_size/6], sm11[smem_size/6], sm12[smem_size/6];
   //double *sm[6] = {sm00, sm01, sm02, sm10, sm11, sm12};
   //for (int k = 0; k < N; k++) { h_body(k, sm00, sm01, sm02, sm10, sm11, sm12); }
}

template <const int DIM, typename lambda>
inline void ForallWrap(const bool use_dev, const int N, lambda &&body,
                       const int X=0, const int Y=0, const int Z=0,
                       const int G=0)
{
   ForallWrap<DIM>(use_dev, N, body, body, X, Y, Z, G);
}

template <const int DIM, typename Tsmem = double, typename lambda>
inline void ForallWrapSmem(const bool use_dev, const int N, lambda &&body,
                           const int S,
                           const int X=0, const int Y=0, const int Z=0,
                           const int G=0)
{
   ForallWrapSmem<DIM,Tsmem>(use_dev, N, S, body, body, X, Y, Z, G);
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

template<typename Tsmem = double, typename lambda>
inline void forall_2D_batch(int N, int X, int Y, int BZ, int S, lambda &&body)
{
   ForallWrapSmem<2,Tsmem>(true, N, body, S, X, Y, BZ);
}

template<typename lambda>
inline void forall_3D(int N, int X, int Y, int Z, lambda &&body)
{
   ForallWrap<3>(true, N, body, X, Y, Z);
}

template<typename Tsmem = double, typename lambda>
inline void forall_3D(int N, int X, int Y, int Z, int sbytes, lambda &&body)
{
   ForallWrapSmem<3,Tsmem>(true, N, body, sbytes, X, Y, Z);
}

template<typename lambda>
inline void forall_3D_grid(int N, int X, int Y, int Z, int G, lambda &&body)
{
   ForallWrap<3>(true, N, body, X, Y, Z, G);
}

} // namespace mfem

#endif // MFEM_FORALL_HPP
