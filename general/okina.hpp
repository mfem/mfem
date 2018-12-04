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

// *****************************************************************************
#ifdef __NVCC__
#include <cuda.h>
#else
#define __host__
#define __device__
#define __constant__
#define __kernel__
#endif // __NVCC__

// *****************************************************************************
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>
#include <unordered_map>

// *****************************************************************************
#include "mm.hpp"
#include "config.hpp"

// *****************************************************************************
// * GPU kernel launcher
// *****************************************************************************
#ifdef __NVCC__
template <typename BODY> __global__
void kernel(const size_t N, BODY body)
{
   const size_t k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}
#endif // __NVCC__

// *****************************************************************************
// * GPU & HOST FOR_LOOP bodies wrapper
// *****************************************************************************
template <typename DBODY, typename HBODY>
void wrap(const size_t N, DBODY &&d_body, HBODY &&h_body)
{
#ifdef __NVCC__
   const bool gpu = mfem::config::Get().Cuda();
   if (gpu)
   {
      const size_t blockSize = 256;
      const size_t gridSize = (N+blockSize-1)/blockSize;
      kernel<<<gridSize, blockSize>>>(N,d_body);
      return;
   }
#endif // __NVCC__
   for (size_t k=0; k<N; k+=1) { h_body(k); }
}

// *****************************************************************************
// * MFEM_FORALL splitter
// *****************************************************************************
#define MFEM_FORALL(i, N, body) wrap(N,                                 \
                                     [=] __device__ (size_t i){body},   \
                                     [=]            (size_t i){body})

// *****************************************************************************
#define LOG2(X) ((unsigned) (8*sizeof(unsigned long long)-__builtin_clzll((X))))
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#define GET_CUDA const bool cuda = config::Get().Cuda();
#define TRY_ADRS(v) mfem::mm::Get().Adrs(v)
#define GET_ADRS(v) double *d_##v = (double*) mfem::mm::Get().Adrs(v)
#define GET_ADRS_T(v,T) T *d_##v = (T*) mfem::mm::Get().Adrs(v)
#define GET_CONST_ADRS(v) const double *d_##v = (const double*) mfem::mm::Get().Adrs(v)
#define GET_CONST_ADRS_T(v,T) const T *d_##v = (const T*) mfem::mm::Get().Adrs(v)

// *****************************************************************************
#define MFEM_FILE_AND_LINE __FILE__ and __LINE__
#define MFEM_CPU_CANNOT_PASS {assert(MFEM_FILE_AND_LINE and false);}
#define MFEM_GPU_CANNOT_PASS {assert(MFEM_FILE_AND_LINE and not config::Get().Cuda());}

// Offsets *********************************************************************
#define ijN(i,j,N) (i)+(N)*(j)
#define ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))

// *****************************************************************************
const char *strrnchr(const char*, const unsigned char, const int);
void dbg_F_L_F_N_A(const char*, const int, const char*, const int, ...);

// *****************************************************************************
#define X_ARGS(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define N_ARGS(...) X_ARGS(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define __FILENAME__ ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})
#define _F_L_F_ __FILENAME__,__LINE__,__FUNCTION__
#define dbg(...) dbg_F_L_F_N_A(_F_L_F_, N_ARGS(__VA_ARGS__),__VA_ARGS__)

// *****************************************************************************
#define pop(...)
#define push(...) dbg_F_L_F_N_A(_F_L_F_,0)

// *****************************************************************************
//#define stop(...) assert(false)

// *****************************************************************************
#define ok(...) assert(__FILE__ and __LINE__ and __VA_ARGS__);

#endif // MFEM_OKINA_HPP
