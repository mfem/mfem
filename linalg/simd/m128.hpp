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

#ifndef MFEM_SIMD_M128_HPP
#define MFEM_SIMD_M128_HPP

#ifdef __SSE2__

#include "../../config/tconfig.hpp"
#if defined(__x86_64__)
#include <x86intrin.h>
#else // assuming MSVC with _M_X64 or _M_IX86
#include <intrin.h>
#endif

namespace mfem
{

template <typename, int, int> struct AutoSIMD;

template <> struct AutoSIMD<double,2,16>
{
   typedef double scalar_type;
   static constexpr int size = 2;
   static constexpr int align_bytes = 16;

   union
   {
      __m128d m128d;
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
      m128d = v.m128d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      m128d = _mm_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m128d = _mm_add_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      m128d = _mm_add_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m128d = _mm_sub_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      m128d = _mm_sub_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m128d = _mm_mul_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      m128d = _mm_mul_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m128d = _mm_div_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      m128d = _mm_div_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
      r.m128d = _mm_xor_pd(_mm_set1_pd(-0.0), m128d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m128d = _mm_add_pd(m128d,v.m128d);
      return r;
   }


   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      r.m128d = _mm_add_pd(m128d, _mm_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m128d = _mm_sub_pd(m128d,v.m128d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      r.m128d = _mm_sub_pd(m128d, _mm_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m128d = _mm_mul_pd(m128d,v.m128d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      r.m128d = _mm_mul_pd(m128d, _mm_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m128d = _mm_div_pd(m128d,v.m128d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      r.m128d = _mm_div_pd(m128d, _mm_set1_pd(e));
      return r;
   }


   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      m128d = _mm_add_pd(_mm_mul_pd(w.m128d,v.m128d),m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      m128d = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(e),v.m128d),m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      m128d = _mm_add_pd(_mm_mul_pd(v.m128d,_mm_set1_pd(e)),m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      m128d = _mm_mul_pd(v.m128d,w.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      m128d = _mm_mul_pd(v.m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      m128d = _mm_mul_pd(_mm_set1_pd(e),v.m128d);
      return *this;
   }
};

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator+(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.m128d = _mm_add_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator-(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.m128d = _mm_sub_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator*(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.m128d = _mm_mul_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator/(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.m128d = _mm_div_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

} // namespace mfem

#endif // __SSE2__

#endif // MFEM_SIMD_M128_HPP

