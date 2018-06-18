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

#ifndef MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP
#define MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP

#include "assert.h"
#include "x86intrin.h"

// ****************************************************************************
// * ifdef switch between SCALAR, SSE, AVX, AVX2, AVX512F
// ****************************************************************************
#ifndef __SSE2__
#define __SSE2__ 0
#endif
#ifndef __AVX__
#define __AVX__ 0
#endif
#ifndef __AVX2__
#define __AVX2__ 0
#endif
#ifndef __AVX512F__
#define __AVX512F__ 0
#endif
#define __SIMD__ __SSE2__+__AVX__+__AVX2__+__AVX512F__

// *****************************************************************************
template <typename,int,int=1> struct AutoSIMD;

#include "m64.hpp"

#include "m128.hpp"

#include "m256.hpp"

#include "m512.hpp"

#endif // MFEM_TEMPLATE_CONFIG_X86INTRIN_HPP
