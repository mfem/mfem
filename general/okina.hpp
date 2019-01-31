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

// *****************************************************************************
#include "mm.hpp"
#include "config.hpp"

// *****************************************************************************
// * GPU & HOST FOR_LOOP bodies wrapper
// *****************************************************************************
template <size_t BLOCKS, typename DBODY, typename HBODY>
void wrap(const size_t N, DBODY &&d_body, HBODY &&h_body)
{
   const bool gpu = mfem::config::usingGpu();
   if (gpu)
   {
      return cuWrap<BLOCKS>(N,d_body);
   }
   else
   {
      for (size_t k=0; k<N; k+=1) { h_body(k); }
   }
}

// *****************************************************************************
// * MFEM_FORALL splitter
// *****************************************************************************
#define MFEM_BLOCKS 256
#define MFEM_FORALL(i,N,...) MFEM_FORALL_K(i,N,MFEM_BLOCKS,__VA_ARGS__)
#define MFEM_FORALL_K(i,N,BLOCKS,...)                                   \
   wrap<BLOCKS>(N,                                                      \
                [=] __device__ (size_t i){__VA_ARGS__},                 \
                [=]            (size_t i){__VA_ARGS__})
#define MFEM_FORALL_SEQ(...) MFEM_FORALL_K(i,1,1,__VA_ARGS__)

// *****************************************************************************
uint32_t LOG2(uint32_t);
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#define GET_GPU const bool gpu = config::usingGpu();
#define GET_PTR(v) double *d_##v = (double*) mfem::mm::ptr(v)
#define GET_PTR_T(v,T) T *d_##v = (T*) mfem::mm::ptr(v)
#define GET_CONST_PTR(v) const double *d_##v = (const double*) mfem::mm::ptr(v)
#define GET_CONST_PTR_T(v,T) const T *d_##v = (const T*) mfem::mm::ptr(v)

// *****************************************************************************
#define BUILTIN_TRAP __builtin_trap()
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_CPU_CANNOT_PASS {assert(FILE_LINE && false);}
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::usingGpu());}

// Offsets *********************************************************************
#define ijN(i,j,N) (i)+N*(j)
#define ijkN(i,j,k,N) (i)+N*((j)+N*(k))
#define ijklN(i,j,k,l,N) (i)+N*((j)+N*((k)+N*(l)))
#define ijNMt(i,j,N,M,t) (t)?((i)+N*(j)):((j)+M*(i))
#define ijkNM(i,j,k,N,M) (i)+N*((j)+M*(k))
#define ijklNM(i,j,k,l,N,M) (i)+N*((j)+N*((k)+M*(l)))
// External offsets
#define jkliNM(i,j,k,l,N,M) (j)+N*((k)+N*((l)+M*(i)))
#define jklmiNM(i,j,k,l,m,N,M) (j)+N*((k)+N*((l)+N*((m)+M*(i))))
#define xyeijDQE(i,j,x,y,e,D,Q,E) (x)+Q*((y)+Q*((e)+E*((i)+D*j)))
#define xyzeijDQE(i,j,x,y,z,e,D,Q,E) (x)+Q*((y)+Q*((z)+Q*((e)+E*((i)+D*j))))

// *****************************************************************************
const char *strrnchr(const char*, const unsigned char, const int);
void dbg_F_L_F_N_A(const char*, const int, const char*, const int, ...);

// *****************************************************************************
#define _XA_(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define _NA_(...) _XA_(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define __FILENAME__ ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})
#define _F_L_F_ __FILENAME__,__LINE__,__FUNCTION__

// *****************************************************************************
#define dbg(...)
//#define stk(...) dbg_F_L_F_N_A(_F_L_F_,0)
//#define dbg(...) dbg_F_L_F_N_A(_F_L_F_, _NA_(__VA_ARGS__),__VA_ARGS__)

#endif // MFEM_OKINA_HPP
