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
#include "error.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

#include "cuda.hpp"
#include "occa.hpp"
#include "device.hpp"
#include "mem_manager.hpp"

#ifdef MFEM_USE_RAJA
#include "RAJA/RAJA.hpp"
#endif

namespace mfem
{

// OKINA = Okina Kernel Interface for Numerical Analysis

// Implementation of MFEM's okina device kernel interface and its CUDA, OpenMP,
// RAJA, and sequential backends.

/// The MFEM_FORALL wrapper
#define MFEM_BLOCKS 256
#define MFEM_FORALL(i,N,...) MFEM_FORALL_K(i,N,MFEM_BLOCKS,__VA_ARGS__)
#define MFEM_FORALL_K(i,N,BLOCKS,...)                                   \
   OkinaWrap<BLOCKS>(N,                                                 \
                     [=] MFEM_DEVICE (int i) {__VA_ARGS__},             \
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
#if defined(MFEM_USE_CUDA)
template <typename BODY> __global__ static
void CuKernel(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
template <int BLOCKS, typename DBODY>
void CuWrap(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLOCKS-1)/BLOCKS;
   CuKernel<<<GRID,BLOCKS>>>(N,d_body);
   const cudaError_t last = cudaGetLastError();
   MFEM_ASSERT(last == cudaSuccess, cudaGetErrorString(last));
}
#else
template <int BLOCKS, typename DBODY>
void CuWrap(const int N, DBODY &&d_body) {}
#endif

/// The okina kernel body wrapper
template <int BLOCKS, typename DBODY, typename HBODY>
void OkinaWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = Device::UsingOmp();
   const bool gpu  = Device::UsingDevice();
   const bool raja = Device::UsingRaja();
   if (gpu && raja) { return RajaCudaWrap<BLOCKS>(N, d_body); }
   if (gpu)         { return CuWrap<BLOCKS>(N, d_body); }
   if (omp && raja) { return RajaOmpWrap(N, h_body); }
   if (raja)        { return RajaSeqWrap(N, h_body); }
   if (omp)         { return OmpWrap(N, h_body);  }
   for (int k=0; k<N; k+=1) { h_body(k); }
}

} // namespace mfem

#endif // MFEM_OKINA_HPP
