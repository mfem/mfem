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

// MFEM_TEMPLATE_BLOCK_SIZE is the block size used by the template matrix-matrix
// multiply, Mult_AB, defined in tmatrix.hpp. This parameter will generally
// require tuning to determine good value. It is probably highly influenced by
// the SIMD width when Mult_AB is used with a SIMD type like AutoSIMD.
#define MFEM_TEMPLATE_BLOCK_SIZE 4

#define MFEM_TEMPLATE_ENABLE_SERIALIZE

// #define MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
// #define MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
// #define MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
#define MFEM_TEMPLATE_INTRULE_COEFF_PRECOMP

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
