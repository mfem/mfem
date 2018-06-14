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

#ifndef MFEM_BACKENDS_KERNELS_DBG_HPP
#define MFEM_BACKENDS_KERNELS_DBG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include <stdarg.h>
#include <assert.h>
#include <string.h>

// DBG *************************************************************************
void kdbg(const char *format,...);

// *****************************************************************************
void kdbge(const char*, const int, const char*, const bool, const int, ...);

// *****************************************************************************
const char * strrnchr(const char*, const unsigned char, const int);

// *****************************************************************************
uint8_t chk8(const char*);

// *****************************************************************************
#define __NB_ARGS__(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,cnt,...) cnt
#define NB_ARGS(...) __NB_ARGS__(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

// *****************************************************************************
#define __FILENAME__ ({const char *f = strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})

// *****************************************************************************
#define dbp(...) kdbge(__FILENAME__,__LINE__,__FUNCTION__,false,        \
                       NB_ARGS(__VA_ARGS__),__VA_ARGS__)
#define dbg(...) kdbge(__FILENAME__,__LINE__,__FUNCTION__,true,         \
                       NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_DBG_HPP
