// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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
#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#include "simd/x86.hpp"
#elif !defined(_MSC_VER)
#warning Unknown SIMD architecture
#else
#pragma message("warning: Unknown SIMD architecture")
#endif
#endif

// MFEM_SIMD_BYTES is the default SIMD size used by MFEM, see e.g. class
// TBilinearForm and the default traits class AutoSIMDTraits.
// MFEM_ALIGN_BYTES determines the padding used in TVector when its 'align'
// template parameter is set to true -- it ensues that the size of such TVector
// types is a multiple of MFEM_ALIGN_BYTES. MFEM_ALIGN_BYTES must be a multiple
// of MFEM_SIMD_BYTES.
#if !defined(MFEM_USE_SIMD)
#define MFEM_SIMD_BYTES 8
#define MFEM_ALIGN_BYTES 32
#elif defined(__AVX512F__)
#define MFEM_SIMD_BYTES 64
#define MFEM_ALIGN_BYTES 64
#elif defined(__AVX__) || defined(__VECTOR4DOUBLE__)
#define MFEM_SIMD_BYTES 32
#define MFEM_ALIGN_BYTES 32
#elif defined(__SSE2__) || defined(__VSX__)
#define MFEM_SIMD_BYTES 16
#define MFEM_ALIGN_BYTES 32
#else
#define MFEM_SIMD_BYTES 8
#define MFEM_ALIGN_BYTES 32
#endif

// derived macros
#define MFEM_ROUNDUP(val,base) ((((val)+(base)-1)/(base))*(base))
#define MFEM_ALIGN_SIZE(size,type) \
   MFEM_ROUNDUP(size,(MFEM_ALIGN_BYTES)/sizeof(type))

namespace mfem
{

template<typename complex_t, typename real_t>
struct AutoSIMDTraits
{
   static const int block_size = MFEM_TEMPLATE_BLOCK_SIZE;

   // Alignment for arrays of vcomplex_t and vreal_t
   static const int align_bytes = MFEM_SIMD_BYTES;

   static const int batch_size = 1;

   static const int simd_size = MFEM_SIMD_BYTES/sizeof(real_t);

   typedef AutoSIMD<complex_t, simd_size, MFEM_SIMD_BYTES> vcomplex_t;
   typedef AutoSIMD<real_t, simd_size, MFEM_SIMD_BYTES> vreal_t;
   typedef AutoSIMD<int, simd_size, simd_size*sizeof(int)> vint_t;
};

template<typename complex_t, typename real_t>
struct NoSIMDTraits
{
   static const int block_size = MFEM_TEMPLATE_BLOCK_SIZE;

   // Alignment for arrays of vcomplex_t and vreal_t
   static const int align_bytes = sizeof(real_t);

   static const int batch_size = 1;

   static const int simd_size = 1;

   typedef AutoSIMD<complex_t, simd_size, align_bytes> vcomplex_t;
   typedef AutoSIMD<real_t, simd_size, align_bytes> vreal_t;
   typedef AutoSIMD<int, simd_size, simd_size*sizeof(int)> vint_t;
};

} // mfem namespace

#endif // MFEM_SIMD_HPP
