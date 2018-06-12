// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_BACKENDS_RAJA_DBG_HPP
#define MFEM_BACKENDS_RAJA_DBG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

// DBG *************************************************************************
inline void rdbg(const char *format,...)
{
   va_list args;
   va_start(args, format);
   fflush(stdout);
   vfprintf(stdout,format,args);
   fflush(stdout);
   va_end(args);
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
static bool env_ini = false;
static bool env_dbg = false;

// *****************************************************************************
inline void rdbge(const char *file, const int line, const char *func,
                  const bool header, const int nargs, ...)
{
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   assert(nargs>0);
   const uint8_t color = 17 + chk8(file)%216;
   fflush(stdout);
   fprintf(stdout,"\033[38;5;%dm",color);
   if (header)
   {
      fprintf(stdout,"\n%24s\b\b\b\b:\033[2m%3d\033[22m: %s: \033[1m", file, line,
              func);
   }
   else
   {
      fprintf(stdout,"\033[1m");
   }
   va_list args;
   va_start(args,nargs);
   const char *format=va_arg(args,const char*);
   assert(format);
   vfprintf(stdout,format,args);
   va_end(args);
   fprintf(stdout,"\033[m");
   fflush(stdout);
}

// *****************************************************************************
#define __NB_ARGS__(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,cnt,...) cnt
#define NB_ARGS(...) __NB_ARGS__(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define __FILENAME__ ({const char *f = strrchr(__FILE__,'/'); f?f+1:__FILE__;})
#define dbp(...) rdbge(__FILENAME__,__LINE__,__FUNCTION__,false,NB_ARGS(__VA_ARGS__),__VA_ARGS__)
#define dbg(...) rdbge(__FILENAME__,__LINE__,__FUNCTION__,true,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_DBG_HPP
