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

#ifndef MFEM_OKINA_HPP
#define MFEM_OKINA_HPP

#include "../config/config.hpp"
#include "../general/error.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdarg.h>

#ifdef MFEM_USE_CUDA
#include <cuda.h>
#else
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
#endif

#ifdef MFEM_USE_RAJA
#include "RAJA/RAJA.hpp"
#endif

#ifdef MFEM_USE_OCCA
#include <occa.hpp>
#else
typedef void* OccaDevice;
typedef void* OccaMemory;
#endif

#include "occa.hpp"
#include "mm.hpp"
#include "device.hpp"

namespace mfem
{

// OKINA = Okina Kernel Interface for Numerical Analysis

// Implementation of MFEM's okina device kernel interface and its
// CUDA, OpenMP, RAJA, and sequential backends.

/// The MFEM_FORALL wrapper
#define MFEM_FORALL(i,N,...)                                          \
   OkinaWrap(N, [=] MFEM_DEVICE (int i) {__VA_ARGS__},                \
                [&]             (int i) {__VA_ARGS__})

/// OpenMP backend
template <typename HBODY>
void OmpWrap(const int N, HBODY &&h_body)
{
#if defined(_OPENMP)
   #pragma omp parallel for
   for (int k=0; k<N; k+=1)
   {
      h_body(k);
   }
#else
   MFEM_ABORT("OpenMP requested for MFEM but OpenMP is not enabled!");
#endif
}

/// RAJA Cuda backend
template <int BLOCKS, typename DBODY>
void RajaCudaWrap(const int N, DBODY &&d_body)
{
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
   RAJA::forall<RAJA::cuda_exec<BLOCKS>>(RAJA::RangeSegment(0,N),d_body);
#else
   MFEM_ABORT("RAJA::Cuda requested but RAJA::Cuda is not enabled!");
#endif
}

/// RAJA OpenMP backend
template <typename HBODY>
void RajaOmpWrap(const int N, HBODY &&h_body)
{
#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_OPENMP)
   RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,N), h_body);
#else
   MFEM_ABORT("RAJA::OpenMP requested but RAJA::OpenMP is not enabled!");
#endif
}

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
#define MFEM_HOST __host__
#define MFEM_DEVICE __device__
#define MFEM_HOST_DEVICE __host__ __device__
inline void CuCheck(const unsigned int c)
{
   MFEM_ASSERT(c == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
}
template<typename T> MFEM_HOST_DEVICE
inline T AtomicAdd(T* address, T val)
{
   return atomicAdd(address, val);
}
template <typename BODY> __global__ static
void CuKernel(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
template <int BLOCKS, typename DBODY, typename HBODY>
void CuWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLOCKS-1)/BLOCKS;
   CuKernel<<<GRID,BLOCKS>>>(N,d_body);
   const cudaError_t last = cudaGetLastError();
   MFEM_ASSERT(last == cudaSuccess, cudaGetErrorString(last));
}
#else // MFEM_USE_CUDA
#define MFEM_HOST
#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
template<typename T> inline T AtomicAdd(T* address, T val)
{
#if defined(_OPENMP)
   #pragma omp atomic
#endif
   *address += val;
   return *address;
}
template <int BLOCKS, typename DBODY, typename HBODY>
void CuWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   for (int k=0; k<N; k+=1) { h_body(k); }
}
#endif // MFEM_USE_CUDA
#define MFEM_BLOCKS 256

/// The okina kernel body wrapper
template <typename DBODY, typename HBODY>
void OkinaWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = Device::UsingOmp();
   const bool gpu  = Device::UsingDevice();
   const bool raja = Device::UsingRaja();
   if (gpu && raja) { return RajaCudaWrap<MFEM_BLOCKS>(N, d_body); }
   if (gpu)         { return CuWrap<MFEM_BLOCKS>(N, d_body, h_body); }
   if (omp && raja) { return RajaOmpWrap(N, h_body); }
   if (raja)        { return RajaSeqWrap(N, h_body); }
   if (omp)         { return OmpWrap(N, h_body);  }
   for (int k=0; k<N; k+=1) { h_body(k); }
}

} // namespace mfem

#endif // MFEM_OKINA_HPP
