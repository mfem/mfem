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

#ifndef MFEM_SIMD_HPP
#define MFEM_SIMD_HPP

#include "../config/tconfig.hpp"

// --- AutoSIMD + specializations with intrinsics
#include "simd/auto.hpp"
#ifdef MFEM_USE_SIMD
#if defined(__VSX__)
#include "simd/vsx.hpp"
#elif defined (__bgq__)
#include "simd/qpx.hpp"
#elif defined(__x86_64__)
#include "simd/x86.hpp"
#else
#warning Unknown SIMD architecture
#endif
#endif

// MFEM_SIMD_SIZE is the default SIMD size used by MFEM, see e.g. class
// TBilinearForm and the default traits class AutoImplTraits.
#if defined(_WIN32)
#define MFEM_SIMD_SIZE 8
#elif defined(__AVX512F__)
#define MFEM_SIMD_SIZE 64
#elif defined(__AVX__) || defined(__VECTOR4DOUBLE__)
#define MFEM_SIMD_SIZE 32
#elif defined(__SSE2__) || defined(__VSX__)
#define MFEM_SIMD_SIZE 16
#else
#define MFEM_SIMD_SIZE 8
#endif

// derived macros
#define MFEM_ROUNDUP(val,base) ((((val)+(base)-1)/(base))*(base))
#define MFEM_ALIGN_SIZE(size,type) \
   MFEM_ROUNDUP(size,(MFEM_SIMD_SIZE)/sizeof(type))

namespace mfem
{

template<typename complex_t, typename real_t>
struct AutoImplTraits
{
   static const int block_size = MFEM_TEMPLATE_BLOCK_SIZE;

   static const int align_size = MFEM_SIMD_SIZE; // in bytes

   static const int batch_size = 1;

   static const int simd_size = MFEM_SIMD_SIZE/sizeof(complex_t);

   static const int valign_size = simd_size;

   typedef AutoSIMD<complex_t, simd_size, valign_size> vcomplex_t;
   typedef AutoSIMD<real_t, simd_size, valign_size> vreal_t;
   typedef AutoSIMD<int, simd_size, valign_size> vint_t;
};

} // mfem namespace

#endif // MFEM_SIMD_HPP
