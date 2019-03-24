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
#include <stdarg.h>

// *****************************************************************************
#include "./cuda.hpp"
#include "./occa.hpp"
#include "./raja.hpp"
#include "./openmp.hpp"

// *****************************************************************************
#include "./mmu.hpp"
#include "./mm.hpp"
#include "./config.hpp"

namespace mfem
{

// *****************************************************************************
// * Kernel body wrapper
// *****************************************************************************
template <int BLOCKS, typename DBODY, typename HBODY>
void OkinaWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = mfem::config::UsingOmp();
   const bool gpu  = mfem::config::UsingDevice();
   const bool raja = mfem::config::UsingRaja();
   if (gpu && raja) { return mfem::RajaCudaWrap<BLOCKS>(N, d_body); }
#ifdef __NVCC__
   if (gpu)         { return CuWrap<BLOCKS>(N, d_body); }
#else
   if (gpu) {
#ifdef MFEM_DEBUG 
      return CuWrap<BLOCKS>(N, h_body);
#else
      mfem_error("gpu mode, but no CUDA support!");
#endif
   }
#endif
   if (omp && raja) { return RajaOmpWrap(N, h_body); }
   if (raja)        { return RajaSeqWrap(N, h_body); }
   if (omp)         { return OmpWrap(N, h_body);  }
   for (int k=0; k<N; k+=1) { h_body(k); }
}

// *****************************************************************************
// * MFEM_FORALL wrapper
// *****************************************************************************
#define MFEM_BLOCKS 256
#define MFEM_FORALL(i,N,...) MFEM_FORALL_K(i,N,MFEM_BLOCKS,__VA_ARGS__)
#define MFEM_FORALL_K(i,N,BLOCKS,...)                                   \
   OkinaWrap<BLOCKS>(N,                                                 \
                     [=] __device__ (int i) mutable {__VA_ARGS__},      \
                     [&]            (int i) {__VA_ARGS__})

// *****************************************************************************
#ifndef __NVCC__
#define MFEM_HOST_DEVICE
#else
#define MFEM_HOST_DEVICE __host__ __device__
#endif

// *****************************************************************************
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::UsingDevice());}

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
static inline void dbg_F_L_F_N_A(const char *file, const int line,
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
   assert(format);
   vfprintf(stdout,format,args);
   va_end(args);
   fprintf(stdout,"\033[m");
   fflush(stdout);
   fflush(0);
}
// *****************************************************************************
#define dbg(...) dbg_F_L_F_N_A(MFEM_FLF, MFEM_NA(__VA_ARGS__),__VA_ARGS__)

} // namespace mfem

#endif // MFEM_OKINA_HPP
