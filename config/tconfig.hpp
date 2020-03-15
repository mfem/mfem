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

// --- POSIX MEMALIGN
#ifdef _WIN32
#define MFEM_POSIX_MEMALIGN(p,a,s) (((*(p))=_aligned_malloc((s),(a))),*(p)?0:errno)
#define MFEM_POSIX_MEMALIGN_FREE _aligned_free
#else
#define MFEM_POSIX_MEMALIGN posix_memalign
#define MFEM_POSIX_MEMALIGN_FREE free
#endif

// --- AutoSIMD or intrinsics
#ifndef MFEM_USE_SIMD
#include "simd/auto.hpp"
#else
#if defined(__VSX__)
#include "simd/vsx.hpp"
#elif defined (__bgq__)
#include "simd/qpx.hpp"
#elif defined(__x86_64__)
#include "simd/x86.hpp"
#else
#error Unknown SIMD architecture
#endif
#endif

// --- SIMD and BLOCK sizes
#if defined(_WIN32)
#define MFEM_SIMD_SIZE 8
#define MFEM_TEMPLATE_BLOCK_SIZE 1
#elif defined(__VSX__)
#define MFEM_SIMD_SIZE 16
#define MFEM_TEMPLATE_BLOCK_SIZE 2
#elif defined(__x86_64__)
#define MFEM_SIMD_SIZE 32
#define MFEM_TEMPLATE_BLOCK_SIZE 4
#else
#error Unknown SIMD architecture
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
