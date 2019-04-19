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


// Support out-of-source builds: if MFEM_BUILD_DIR is defined, load the config
// file MFEM_BUILD_DIR/config/_config.hpp.
//
// Otherwise, use the local file: _config.hpp.

#ifndef MFEM_CONFIG_HPP
#define MFEM_CONFIG_HPP

#ifdef MFEM_BUILD_DIR
#define MFEM_QUOTE(a) #a
#define MFEM_MAKE_PATH(x,y) MFEM_QUOTE(x/y)
#include MFEM_MAKE_PATH(MFEM_BUILD_DIR,config/_config.hpp)
#else
#include "_config.hpp"
#endif

// Common configuration macros

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)) || defined(__clang__)
#define MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#endif

// Windows specific options
#ifdef _WIN32
// Macro needed to get defines like M_PI from <cmath>. (Visual Studio C++ only?)
#define _USE_MATH_DEFINES
#endif

// Check dependencies:

// Options that require MPI
#ifndef MFEM_USE_MPI
#ifdef MFEM_USE_SUPERLU
#error Building with SuperLU_DIST (MFEM_USE_SUPERLU=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_STRUMPACK
#error Building with STRUMPACK (MFEM_USE_STRUMPACK=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PETSC
#error Building with PETSc (MFEM_USE_PETSC=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#ifdef MFEM_USE_PUMI
#error Building with PUMI (MFEM_USE_PUMI=YES) requires MPI (MFEM_USE_MPI=YES)
#endif
#endif // MFEM_USE_MPI not defined
#include <cstdarg>
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
#ifdef _WIN32
#define dbg(...)
#else
#define dbg(...) mfem_FLF_NA_ARGS(MFEM_FLF, MFEM_NA(__VA_ARGS__),__VA_ARGS__)
#endif

#endif // MFEM_CONFIG_HPP
