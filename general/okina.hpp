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
   const bool omp  = mfem::config::usingOmp();
   const bool gpu  = mfem::config::usingGpu();
   const bool raja = mfem::config::usingRaja();
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
                [=] __device__ (int i)mutable{__VA_ARGS__},             \
                [&]            (int i){__VA_ARGS__})
#define MFEM_FORALL_SEQ(...) MFEM_FORALL_K(i,1,1,__VA_ARGS__)

// *****************************************************************************
uint32_t LOG2(uint32_t);
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#ifndef __NVCC__
#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
#else
#define MFEM_DEVICE __device__
#define MFEM_HOST_DEVICE __host__ __device__
#endif

// *****************************************************************************
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_CPU_CANNOT_PASS {assert(FILE_LINE && false);}
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::usingGpu());}

#endif // MFEM_OKINA_HPP
