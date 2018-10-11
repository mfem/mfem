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

#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#else
#define __host__
#define __device__
#define __kernel__
#define __constant__
#endif // __NVCC__

#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>

#include <stdarg.h>
#include <signal.h>

// *****************************************************************************
#define MFEM_NAMESPACE namespace mfem {
#define MFEM_NAMESPACE_END }

// *****************************************************************************
#include "dbg.hpp"
#include "memmng.hpp"
#include "config.hpp"
#include "kernels.hpp"

// *****************************************************************************
#define LOG2(X) ((unsigned) (8*sizeof(unsigned long long)-__builtin_clzll((X))))
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#define GET_CUDA const bool cuda = mfem::config::Get().Cuda();
#define GET_ADRS(v) double *d_##v = (double*) mfem::mm::Get().Adrs(v)
#define GET_ADRS_T(v,T) T *d_##v = (T*) mfem::mm::Get().Adrs(v)
#define GET_CONST_ADRS(v) const double *d_##v = (const double*) mfem::mm::Get().Adrs(v)
#define GET_CONST_ADRS_T(v,T) const T *d_##v = (const T*) mfem::mm::Get().Adrs(v)

// *****************************************************************************
#define OKINA_ASSERT_CPU {assert(__FILE__ and __LINE__ and false);}
#define OKINA_ASSERT_GPU {assert(__FILE__ and __LINE__ and not config::Get().Cuda());}

// Offsets *********************************************************************
#define   ijN(i,j,N) (i)+(N)*(j)
#define  ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define    ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define    ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define   _ijkNM(i,j,k,N,M) (j)+(N)*((k)+(M)*(i))
#define   ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define  _ijklNM(i,j,k,l,N,M) (j)+(N)*((k)+(N)*((l)+(M)*(i)))
#define   ijklmNM(i,j,k,l,m,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*(m))))
#define __ijklmNM(i,j,k,l,m,N,M) (k)+(M)*((l)+(M)*((m)+(N*N)*((i)+(N)*j)))

#define _ijklmNM(i,j,k,l,m,N,M) (j)+(N)*((k)+(N)*((l)+(N)*((m)+(M)*(i))))
#define ijklmnNM(i,j,k,l,m,n,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*((m)+(M)*(n)))))

#endif // MFEM_OKINA_HPP
