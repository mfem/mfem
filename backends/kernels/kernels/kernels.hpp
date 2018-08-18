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
#ifndef MFEM_KERNELS_KERNELS_KERNELS
#define MFEM_KERNELS_KERNELS_KERNELS

// *****************************************************************************
//#define __LAMBDA__
#define __TEMPLATES__

// *****************************************************************************
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <unordered_map>

// *****************************************************************************
#define LOG2(X) ((unsigned) (8*sizeof(unsigned long long)-__builtin_clzll((X))))
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)

// *****************************************************************************
#ifdef __NVCC__
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif // __NVCC__

// *****************************************************************************
#ifdef __RAJA__
#include <cuda.h>
#include "RAJA/RAJA.hpp"
#include "RAJA/policy/cuda.hpp"
#endif

// *****************************************************************************
#include "../config/dbg.hpp"
#include "../config/nvvp.hpp"
#include "../config/config.hpp"
#include "../general/memcpy.hpp"
#include "../general/malloc.hpp"

// *****************************************************************************
#include "include/forall.hpp"
#include "include/offsets.hpp"

#endif // MFEM_KERNELS_KERNELS_KERNELS
