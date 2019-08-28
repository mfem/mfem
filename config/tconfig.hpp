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

#ifndef MFEM_TEMPLATE_CONFIG
#define MFEM_TEMPLATE_CONFIG

// the main MFEM config header
#include "config.hpp"

// --- MFEM_STATIC_ASSERT
#if (__cplusplus >= 201103L)
#define MFEM_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#else
#define MFEM_STATIC_ASSERT(cond, msg) if (cond) { }
#endif

// --- MFEM_ALWAYS_INLINE
#if !defined(MFEM_DEBUG) && (defined(__GNUC__) || defined(__clang__))
#define MFEM_ALWAYS_INLINE __attribute__((always_inline))
#else
#define MFEM_ALWAYS_INLINE
#endif

// --- MFEM_VECTORIZE_LOOP (disabled)
#if (__cplusplus >= 201103L) && !defined(MFEM_DEBUG) && defined(__GNUC__)
//#define MFEM_VECTORIZE_LOOP _Pragma("GCC ivdep")
#define MFEM_VECTORIZE_LOOP
#else
#define MFEM_VECTORIZE_LOOP
#endif

// --- MFEM_ALIGN_AS
#if (__cplusplus >= 201103L)
#define MFEM_ALIGN_AS(bytes) alignas(bytes)
#elif !defined(MFEM_DEBUG) && (defined(__GNUC__) || defined(__clang__))
#define MFEM_ALIGN_AS(bytes) __attribute__ ((aligned (bytes)))
#else
#define MFEM_ALIGN_AS(bytes)
#endif

// --- AutoSIMD or intrinsics
#ifndef MFEM_USE_SIMD
#include "simd/auto.hpp"
#else
#ifdef __VSX__
#include "simd/vsx128.hpp"
#endif
#ifdef __bgq__
#include "simd/qpx.hpp"
#else
#include "simd/x86.hpp"
#endif
#endif

// --- SIMD Traits
#ifndef MFEM_USE_SIMD
#define MFEM_SIMD_SIZE 32
#define MFEM_TEMPLATE_BLOCK_SIZE 4
#else
#ifdef __VSX__ // 128
#define MFEM_SIMD_SIZE 16
#define MFEM_TEMPLATE_BLOCK_SIZE 2
#else // 256
#define MFEM_SIMD_SIZE 32
#define MFEM_TEMPLATE_BLOCK_SIZE 4
#endif
#endif

template<typename complex_t, typename real_t, bool simd>
struct AutoImplTraits
{
   static const int block_size = MFEM_TEMPLATE_BLOCK_SIZE;

   static const int align_size = MFEM_SIMD_SIZE; // in bytes

   static const int batch_size = 1;

   static const int simd_size = simd?(MFEM_SIMD_SIZE/sizeof(complex_t)):1;

   static const int valign_size = simd?simd_size:1;

   typedef AutoSIMD<complex_t,simd_size,valign_size> vcomplex_t;
   typedef AutoSIMD<   real_t,simd_size,valign_size> vreal_t;
#ifndef MFEM_USE_SIMD
   typedef AutoSIMD<      int,simd_size,valign_size> vint_t;
#endif // MFEM_USE_SIMD
};

#define MFEM_TEMPLATE_ENABLE_SERIALIZE

// #define MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
// #define MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
// #define MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
#define MFEM_TEMPLATE_INTRULE_COEFF_PRECOMP

// derived macros
#define MFEM_ROUNDUP(val,base) ((((val)+(base)-1)/(base))*(base))
#define MFEM_ALIGN_SIZE(size,type) \
   MFEM_ROUNDUP(size,(MFEM_SIMD_SIZE)/sizeof(type))

#ifdef MFEM_COUNT_FLOPS
namespace mfem
{
namespace internal
{
extern long long flop_count;
}
}
#define MFEM_FLOPS_RESET() (mfem::internal::flop_count = 0)
#define MFEM_FLOPS_ADD(cnt) (mfem::internal::flop_count += (cnt))
#define MFEM_FLOPS_GET() (mfem::internal::flop_count)
#else
#define MFEM_FLOPS_RESET()
#define MFEM_FLOPS_ADD(cnt)
#define MFEM_FLOPS_GET() (0)
#endif

#endif // MFEM_TEMPLATE_CONFIG
