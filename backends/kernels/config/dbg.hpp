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

#ifndef MFEM_BACKENDS_KERNELS_DBG_HPP
#define MFEM_BACKENDS_KERNELS_DBG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

// ****************************************************************************
void kdbg(const char *,...);

// *****************************************************************************
void kdbge(const char*, const int, const char*, const bool, const int, ...);

// *****************************************************************************
const char *strrnchr(const char*, const unsigned char, const int);

// *****************************************************************************
uint8_t chk8(const char*);

// *****************************************************************************
#define NX_ARGS(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define NB_ARGS(...) NX_ARGS(,##__VA_ARGS__,\
                             16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

// *****************************************************************************
#define __FILENAME__ ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})

// *****************************************************************************
#define _F_L_F_ __FILENAME__,__LINE__,__FUNCTION__

// *****************************************************************************
#define dbp(...) kdbge(_F_L_F_,false, NB_ARGS(__VA_ARGS__),__VA_ARGS__)
#define dbg(...) kdbge(_F_L_F_, true, NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_DBG_HPP
