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
#ifndef MFEM_X86INTRIN_HPP
#define MFEM_X86INTRIN_HPP

#include <stdlib.h>
#include "x86intrin.h"

// x86 intrinsic class forward description
template <int> struct x86intrin;


// ****************************************************************************
// * ifdef switch between SCALAR, SSE, AVX, AVX2, AVX512F
// * gcc --machine-avx512f -ffreestanding -C -E general/x86intrin.hpp|more
// * strings /usr/local/cuda/bin/nvcc | grep [-]D
// * gcc -dM -E -m64 - < /dev/null|sort|grep -i x86
// * __CUDA_ARCH__ vs __x86_64__
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
//__SIMD__

// ****************************************************************************
// * AVX512 (-mavx512f)
// ****************************************************************************
#if __SIMD__==4
#include "x86/m512.hpp"
#define MFEM_SIMD_SIZE 64
#pragma message "[33;1mX86intrin::AVX512[m"
template <> struct x86intrin<4>{
public:
  const static int align = 64;
  const static int width = 8;
  typedef real vreal_t;
  typedef integer vint_t;
  static inline vreal_t set(double a){return _mm512_set1_pd(a);}
  static inline void* alloc(size_t size){return ::aligned_alloc(align,size);}
};
#endif // __AVX512__

// ****************************************************************************
// * AVX2 (-mavx2)
// ****************************************************************************
#if __SIMD__==3
#include "x86/m256.hpp"
#pragma message "[33;1mX86intrin::AVX2[m"
template <> struct x86intrin<3>{
public:
   static const int align = 32;
   static const int width = 4;
   typedef real vreal_t;
   typedef integer vint_t;
   static inline vreal_t set(double a){return _mm256_set1_pd(a);}
   static inline void* alloc(size_t size){return ::aligned_alloc(align,size);}
};
#endif // __AVX2__

// ****************************************************************************
// * AVX (-mavx -mno-avx2)
// ****************************************************************************
#if __SIMD__==2
#pragma message "[33;1mX86intrin::AVX[m"
#include "x86/m256.hpp"
template <> struct x86intrin<2>{
public:
  static const int align = 32;
  static const int width = 4;
  typedef real vreal_t;
  typedef integer vint_t;
  static inline vreal_t set(double a){return _mm256_set1_pd(a);}
  static inline void* alloc(size_t size){return ::aligned_alloc(align,size);}
};
#endif // __AVX__

// ****************************************************************************
// * SSE (-mno-avx)
// ****************************************************************************
#if __SIMD__==1
#include "x86/m128.hpp"
#pragma message "[33;1mX86intrin::SSE[m"
template <> struct x86intrin<1>{
public:
  static const int align = 16;
  static const int width = 2;
  typedef real vreal_t;
  typedef integer vint_t;
  static inline vreal_t set(double a){return _mm_set1_pd(a);}
  static inline void* alloc(size_t size){return ::aligned_alloc(align,size);}
};
#endif // __SSE__

// ****************************************************************************
// * 'SCALAR' (-mno-sse2)
// ****************************************************************************
#if __SIMD__==0
#include "x86/m64.hpp"
#pragma message "[33;1mX86intrin::STD[m"
template <> struct x86intrin<0>{
public:
  static const int align = 8;
  static const int width = 1;
  typedef real vreal_t;
  typedef integer vint_t;
  static inline vreal_t set(double a){return a;}
  static inline void* alloc(size_t size){return ::aligned_alloc(align,size);}
};
#endif // __STD__

// ****************************************************************************
// * X86 intrinsic base class
// ****************************************************************************
class x86: public x86intrin<__SIMD__>{};

#endif // MFEM_X86INTRIN_HPP
