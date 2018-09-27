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

#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>

#include <stdarg.h>
#include <signal.h>

#include "/home/camier1/home/stk/stk.hpp"

#define MFEM_NAMESPACE namespace mfem {
#define MFEM_NAMESPACE_END }

#include "dbg.hpp"
#include "config.hpp"
#include "memcpy.hpp"
#include "memmng.hpp"
#include "kernels.hpp"

// *****************************************************************************
#define GET_CUDA const bool cuda = cfg::Get().Cuda();
#define GET_ADRS(v) double *d_##v = (double*) (cuda?mm::Get().Adrs(v):v);
#define GET_CONST_ADRS(v) const double *d_##v = (const double*) (cuda?mm::Get().Adrs(v):v);
#define GET_ADRS_T(v,T) T *d_##v = (T*) (cuda?mm::Get().Adrs(v):v);
#define GET_CONST_ADRS_T(v,T) const T *d_##v = (const T*) (cuda?mm::Get().Adrs(v):v);

// *****************************************************************************
#define OKINA_ASSERT_CPU {dbg();stk(true);assert(__FILE__&&__LINE__&&false);}
#define OKINA_ASSERT_GPU {stk(true);assert(not cfg::Get().Cuda());}

#endif // MFEM_OKINA_HPP
