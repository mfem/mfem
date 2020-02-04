#ifndef DBG_HPP
#define DBG_HPP

#ifndef _WIN32

// *****************************************************************************
#include <cstdarg>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

//#include "../config/config.hpp"
#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

//*****************************************************************************
inline uint8_t chk8(const char *bfr)
{
   unsigned int chk = 0;
   size_t len = strlen(bfr);
   for (; len; len--,bfr++)
   {
      chk += static_cast<unsigned int>(*bfr);
   }
   return (uint8_t) chk;
}

// *****************************************************************************
inline const char *strrnchr(const char *s, const unsigned char c, int n)
{
   size_t len = strlen(s);
   char *p = const_cast<char*>(s)+len-1;
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
inline void dbg_F_L_F_N_A(const char *file, const int line,
                          const char *func, const int nargs, ...)
{
   static int mpi_rank = 0;
   static int mpi_dbg = 0;
   static bool mpi = false;
   static bool env_ini = false;
   static bool env_dbg = false;
   if (!env_ini)
   {
      mpi = getenv("DBG_MPI");
#ifdef MFEM_USE_MPI
      if (mpi) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); }
#endif
      env_dbg = getenv("DBG");
      mpi_dbg = atoi(mpi?getenv("DBG_MPI"):"0");
      env_ini = true;
   }
   if (!env_dbg) { return; }
   if (mpi_rank!=mpi_dbg) { return; }
   const uint8_t color = 17 + chk8(file)%216;
   fflush(stdout);
   fprintf(stdout,"\033[38;5;%dm",color);
   fprintf(stdout,"%d%30s:\033[2m%4d\033[22m: %s: \033[1m",
           mpi_rank, file, line, func);
   if (nargs==0) { return; }
   va_list args;
   va_start(args,nargs);
   const char *format = va_arg(args, const char*);
   assert(format);
   vfprintf(stdout, format, args);
   va_end(args);
   fprintf(stdout,"\033[m\n");
   fflush(stdout);
   fflush(0);
}
// *****************************************************************************
#define dbg(...) dbg_F_L_F_N_A(MFEM_FLF, MFEM_NA(__VA_ARGS__),__VA_ARGS__)
#else
#define dbg(...)
#endif // _WIN32

#endif // DBG_HPP
