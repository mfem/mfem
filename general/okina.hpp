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
#include "../config/config.hpp"
#include "../general/error.hpp"

// *****************************************************************************
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>

// *****************************************************************************
#include "./cuda.hpp"
#include "./occa.hpp"
#include "./raja.hpp"

// *****************************************************************************
#include "mm.hpp"
#include "config.hpp"

// *****************************************************************************
// * Standard OpenMP wrapper
// *****************************************************************************
template <typename HBODY>
void ompWrap(const int N, HBODY &&h_body)
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

// *****************************************************************************
// * Standard sequential wrapper
// *****************************************************************************
template <typename HBODY>
void seqWrap(const int N, HBODY &&h_body)
{
   for (int k=0; k<N; k+=1)
   {
      h_body(k);
   }
}

// *****************************************************************************
// * GPU & HOST FOR_LOOP bodies wrapper
// *****************************************************************************
template <int BLOCKS, typename DBODY, typename HBODY>
void wrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = mfem::config::UsingOmp();
   const bool gpu  = mfem::config::UsingDevice();
   const bool raja = mfem::config::UsingRaja();
   if (gpu && raja) { return rajaCudaWrap<BLOCKS>(N, d_body); }
   if (gpu)         { return cuWrap<BLOCKS>(N, d_body); }
   if (omp && raja) { return rajaOmpWrap(N, h_body); }
   if (raja)        { return rajaSeqWrap(N, h_body); }
   if (omp)         { return ompWrap(N, h_body);  }
   seqWrap(N, h_body);
}

// *****************************************************************************
// * MFEM_FORALL splitter
// *****************************************************************************
#define MFEM_BLOCKS 256
#define MFEM_FORALL(i,N,...) MFEM_FORALL_K(i,N,MFEM_BLOCKS,__VA_ARGS__)
#define MFEM_FORALL_K(i,N,BLOCKS,...)                                   \
   wrap<BLOCKS>(N,                                                      \
                [=] __device__ (int i) mutable {__VA_ARGS__},           \
                [&]            (int i){__VA_ARGS__})
#define MFEM_FORALL_SEQ(...) MFEM_FORALL_K(i,1,1,__VA_ARGS__)

// *****************************************************************************
int LOG2(int);
#define ISQRT(N) static_cast<int>(sqrtf(static_cast<float>(N)))
#define ICBRT(N) static_cast<int>(cbrtf(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#ifndef __NVCC__
#define MFEM_HOST_DEVICE
#else
#define MFEM_HOST_DEVICE __host__ __device__
#endif

// *****************************************************************************
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::UsingDevice());}

// *****************************************************************************
const char *strrnchr(const char*, const unsigned char, const int);
void dbg_F_L_F_N_A(const char*, const int, const char*, const int, ...);

// *****************************************************************************
#define MFEM_XA(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define MFEM_NA(...) MFEM_XA(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define MFEM_FILENAME ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})
#define MFEM_FLF MFEM_FILENAME,__LINE__,__FUNCTION__

// *****************************************************************************
#define dbg(...) dbg_F_L_F_N_A(MFEM_FLF, MFEM_NA(__VA_ARGS__),__VA_ARGS__)

#endif // MFEM_OKINA_HPP
