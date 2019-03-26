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
#else
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
#endif

/// The okina kernel body wrapper
template <typename DBODY, typename HBODY>
void OkinaWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = Device::UsingOmp();
   const bool gpu  = Device::UsingDevice();
   const bool raja = Device::UsingRaja();
   if (gpu && raja) { return RajaCudaWrap<256>(N, d_body); }
   if (gpu)         { return CuWrap<256>(N, d_body, h_body); }
   if (omp && raja) { return RajaOmpWrap(N, h_body); }
   if (raja)        { return RajaSeqWrap(N, h_body); }
   if (omp)         { return OmpWrap(N, h_body);  }
   for (int k=0; k<N; k+=1) { h_body(k); }
}

//*****************************************************************************
static inline uint8_t chk8(const char *bfr)
{
   unsigned int chk = 0;
   size_t len = strlen(bfr);
   for (; len; len--,bfr++)
   {
      chk += *bfr;
   }
   return (uint8_t) chk;
}
// *****************************************************************************
inline const char *strrnchr(const char *s, const unsigned char c, int n)
{
   size_t len = strlen(s);
   char *p = (char*)s+len-1;
   for (; n; n--,p--,len--)
   {
      for (; len; p--,len--)
         if (*p==c) { break; }
      if (!len) { return NULL; }
      if (n==1) { return p; }
   }
   return NULL;
}
// *****************************************************************************
#define MFEM_XA(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define MFEM_NA(...) MFEM_XA(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define MFEM_FILENAME ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})
#define MFEM_FLF MFEM_FILENAME,__LINE__,__FUNCTION__
// *****************************************************************************
static inline void mfem_FLF_NA_ARGS(const char *file, const int line,
                                    const char *func, const int nargs, ...)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   const uint8_t color = 17 + chk8(file)%216;
   fflush(stdout);
   fprintf(stdout,"\033[38;5;%dm",color);
   fprintf(stdout,"\n%30s\b\b\b\b:\033[2m%4d\033[22m: %s: \033[1m",
           file, line, func);
   if (nargs==0) { return; }
   va_list args;
   va_start(args,nargs);
   const char *format=va_arg(args,const char*);
   vfprintf(stdout,format,args);
   va_end(args);
   fprintf(stdout,"\033[m");
   fflush(stdout);
   fflush(0);
}
// *****************************************************************************
#define dbg(...) mfem_FLF_NA_ARGS(MFEM_FLF, MFEM_NA(__VA_ARGS__),__VA_ARGS__)

} // namespace mfem

#endif // MFEM_OKINA_HPP
