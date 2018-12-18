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
#include <unordered_map>

// *****************************************************************************
#include "./cuda.hpp"
#include "./occa.hpp"

// *****************************************************************************
#include "mm.hpp"
#include "kernels/mm.hpp"
#include "config.hpp"

// *****************************************************************************
// * GPU & HOST FOR_LOOP bodies wrapper
// *****************************************************************************
template <size_t BLOCK_SZ, typename DBODY, typename HBODY>
void wrap(const size_t N, DBODY &&d_body, HBODY &&h_body)
{
   constexpr bool nvcc = usingNvccCompiler();
   const bool cuda = mfem::config::usingCuda();
   if (nvcc and cuda)
   {
      return cuWrap<BLOCK_SZ>(N,d_body);
   }
   for (size_t k=0; k<N; k+=1) { h_body(k); }
}

// *****************************************************************************
// * MFEM_FORALL splitter
// *****************************************************************************
#define MFEM_FORALL(i, N, B)                                            \
   wrap<256>(N, [=] __device__ (size_t i){B}, [=] (size_t i){B})
#define MFEM_FORALL_BLOCK(i,N,B,K)                                      \
   wrap<K>(N, [=] __device__ (size_t i){B}, [=] (size_t i){B})

// *****************************************************************************
#define LOG2(X) ((unsigned) (8*sizeof(unsigned long long)-__builtin_clzll((X))))
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#define GET_CUDA const bool cuda = config::usingCuda();
#define GET_ADRS(v) double *d_##v = (double*) mm::Get().Adrs(v)
#define GET_ADRS_T(v,T) T *d_##v = (T*) mm::Get().Adrs(v)
#define GET_CONST_ADRS(v) const double *d_##v = (const double*) mm::Get().Adrs(v)
#define GET_CONST_ADRS_T(v,T) const T *d_##v = (const T*) mm::Get().Adrs(v)

// *****************************************************************************
#define BUILTIN_TRAP __builtin_trap()
#define FILE_LINE __FILE__ and __LINE__
#define MFEM_CPU_CANNOT_PASS {assert(FILE_LINE and false);}
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE and not config::usingCuda());}

// Offsets *********************************************************************
#define ijN(i,j,N) (i)+(N)*(j)
#define ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))

// *****************************************************************************
#define dbg(...) \
 { printf("\n\033[32m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0);}

#endif // MFEM_OKINA_HPP
