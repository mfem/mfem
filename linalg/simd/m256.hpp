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

#ifndef MFEM_SIMD_M256_HPP
#define MFEM_SIMD_M256_HPP

#ifdef __AVX__

#include "../../config/tconfig.hpp"
#if defined(__x86_64__)
#include <x86intrin.h>
#else // assuming MSVC with _M_X64 or _M_IX86
#include <intrin.h>
#endif

namespace mfem
{

template <typename, int, int> struct AutoSIMD;

template <> struct AutoSIMD<double,4,32>
{
   typedef double scalar_type;
   static constexpr int size = 4;
   static constexpr int align_bytes = 32;

   union
   {
      __m256d m256d;
      double vec[size];
   };

   inline MFEM_ALWAYS_INLINE double &operator[](int i)
   {
      return vec[i];
   }

   inline MFEM_ALWAYS_INLINE const double &operator[](int i) const
   {
      return vec[i];
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const AutoSIMD &v)
   {
      m256d = v.m256d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      m256d = _mm256_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m256d = _mm256_add_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      m256d = _mm256_add_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m256d = _mm256_sub_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      m256d = _mm256_sub_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m256d = _mm256_mul_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      m256d = _mm256_mul_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m256d = _mm256_div_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      m256d = _mm256_div_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
      r.m256d = _mm256_xor_pd(_mm256_set1_pd(-0.0), m256d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m256d = _mm256_add_pd(m256d,v.m256d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      r.m256d = _mm256_add_pd(m256d, _mm256_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m256d = _mm256_sub_pd(m256d,v.m256d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      r.m256d = _mm256_sub_pd(m256d, _mm256_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m256d = _mm256_mul_pd(m256d,v.m256d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      r.m256d = _mm256_mul_pd(m256d, _mm256_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m256d = _mm256_div_pd(m256d,v.m256d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      r.m256d = _mm256_div_pd(m256d, _mm256_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
#ifndef __AVX2__
      m256d = _mm256_add_pd(_mm256_mul_pd(w.m256d,v.m256d),m256d);
#else
      m256d = _mm256_fmadd_pd(w.m256d,v.m256d,m256d);
#endif
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
#ifndef __AVX2__
      m256d = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(e),v.m256d),m256d);
#else
      m256d = _mm256_fmadd_pd(_mm256_set1_pd(e),v.m256d,m256d);
#endif
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
#ifndef __AVX2__
      m256d = _mm256_add_pd(_mm256_mul_pd(v.m256d,_mm256_set1_pd(e)),m256d);
#else
      m256d = _mm256_fmadd_pd(v.m256d,_mm256_set1_pd(e),m256d);
#endif
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      m256d = _mm256_mul_pd(v.m256d,w.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      m256d = _mm256_mul_pd(v.m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      m256d = _mm256_mul_pd(_mm256_set1_pd(e),v.m256d);
      return *this;
   }
};

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,4,32> operator+(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.m256d = _mm256_add_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,4,32> operator-(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.m256d = _mm256_sub_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,4,32> operator*(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.m256d = _mm256_mul_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,4,32> operator/(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.m256d = _mm256_div_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

} // namespace mfem

#endif // __AVX__

#endif // MFEM_SIMD_M256_HPP
