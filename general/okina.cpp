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

#include <stdarg.h>
#include "../general/okina.hpp"

// *****************************************************************************
int LOG2(int v )
{
   static const int MultiplyDeBruijnBitPosition[32] =
   {
      0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
      8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
   };
   v |= v >> 1; // first round down to one less than a power of 2
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   return MultiplyDeBruijnBitPosition[(v*0x07C4ACDD)>>27];
}

//*****************************************************************************
static uint8_t chk8(const char *bfr)
{
   uint8_t chk = 0;
   for (int len = strlen(bfr); len; len--,bfr++)
   {
      chk += *bfr;
   }
   return chk;
}

// *****************************************************************************
const char *strrnchr(char const *s, const unsigned char c, int n)
{
   int len = strlen(s);
   char const *p = s+len-1;
   for (; n; n--,p--,len--)
   {
      for (; len; p--,len--)
         if (*p==c) { break; }
      if (!len) { return nullptr; }
      if (n==1) { return p; }
   }
   return nullptr;
}

// *****************************************************************************
// * file: __FILENAME__, line: __LINE__, func: __FUNCTION__
// * nargs: number of arguments that follow
// *****************************************************************************
void dbg_F_L_F_N_A(const char *file, const int line, const char *func,
                   const int nargs, ...)
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
   const char *format = va_arg(args,const char*);
   assert(format);
   vfprintf(stdout, format, args);
   va_end(args);
   fprintf(stdout,"\033[m");
   fflush(stdout);
   fflush(nullptr);
}
