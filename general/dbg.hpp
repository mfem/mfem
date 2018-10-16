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

#ifndef MFEM_DBG_HPP
#define MFEM_DBG_HPP

// *****************************************************************************
__attribute__((unused))
static const char *strrnchr(const char *s, const unsigned char c, int n)
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

//*****************************************************************************
__attribute__((unused))
static uint8_t chk8(const char *bfr)
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
__attribute__((unused))
static void kdbge(const char *file, const int line, const char *func,
                  const bool header, const int nargs, ...)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   const uint8_t color = 17 + chk8(file)%216;
   fflush(stdout);
   fprintf(stdout,"\033[38;5;%dm",color);
   if (header)
   {
      fprintf(stdout,"\n%24s\b\b\b\b:\033[2m%3d\033[22m: %s: \033[1m",
              file, line, func);
   }
   else
   {
      fprintf(stdout,"\033[1m");
   }
   if (nargs==0) { return; }
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
#define __FILENAME__ ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})

// *****************************************************************************
#define _F_L_F_ __FILENAME__,__LINE__,__FUNCTION__

// *****************************************************************************
#define NX_ARGS(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define NB_ARGS(...) NX_ARGS(,##__VA_ARGS__,\
                             16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

#ifdef __NVCC__

// *****************************************************************************
#define DBG2(...) kdbge(_F_L_F_, true, NB_ARGS(__VA_ARGS__), __VA_ARGS__)
#define DBG1(...) kdbge(_F_L_F_, true, NB_ARGS(__VA_ARGS__), __VA_ARGS__)
#define DBG0() kdbge(_F_L_F_, true,0)

// *****************************************************************************
#define LPAREN (
#define COMMA_IF_PARENS(...) ,
#define EXPAND(...) __VA_ARGS__
#define DBG(a0,a1,a2,a3,a4,a5,a,...) a
#define CHOOSE(...) EXPAND(DBG LPAREN \
      __VA_ARGS__ COMMA_IF_PARENS \
      __VA_ARGS__ COMMA_IF_PARENS __VA_ARGS__ (),  \
      DBG2, impossible, DBG2, DBG1, DBG0, DBG1, ))
#define dbg(...) CHOOSE(__VA_ARGS__)(__VA_ARGS__)

#else // __NVCC__

// *****************************************************************************
#define dbp(...) kdbge(_F_L_F_,false, NB_ARGS(__VA_ARGS__),__VA_ARGS__)
#define dbg(...) kdbge(_F_L_F_, true, NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#endif // __NVCC__

// *****************************************************************************
__attribute__((unused))
static void push_flf(const char *file, const int line, const char *func)
{
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini) { env_dbg = getenv("DBG"); env_ini = true; }
   if (!env_dbg) { return; }
   const uint8_t color = 17 + chk8(file)%216;
   fflush(stdout);
   fprintf(stdout,"\033[38;5;%dm",color);
   fprintf(stdout,"\n%24s\b\b\b\b:\033[2m%3d\033[22m: %s", file, line, func);
   fprintf(stdout,"\033[m");
   fflush(stdout);
}
#define push(...) push_flf(__FILENAME__,__LINE__,__FUNCTION__)


#endif // MFEM_DBG_HPP
