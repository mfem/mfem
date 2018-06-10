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
  //printf("\n\033[33m[chk8] %s\033[m",bfr);
  unsigned int chk = 0;
  size_t len = strlen(bfr);
  for(;len;len--,bfr++)
    chk += *bfr;
  return (uint8_t) chk;
}

// *****************************************************************************
static inline void rdbge(const char *file, const char *format, ...)
{
  if (false) return;
  const uint8_t color = 17 + chk8(file)%216;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout,"\033[38;5;%dm",color);
  vfprintf(stdout,format,args);
  fprintf(stdout,"\033[m");
  fflush(stdout);
  va_end(args);
}

// *****************************************************************************
#ifdef LAGHOS_DEBUG
//#warning LAGHOS_DEBUG
#define dbg(...) rdbg(__VA_ARGS__)
#else
//#warning LAGHOS_DEBUG else
#define dbg(...) rdbge(__FILE__,__VA_ARGS__)
#endif // LAGHOS_DEBUG

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_DBG_HPP
