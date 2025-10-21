// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SIMD_M512_HPP
#define MFEM_SIMD_M512_HPP

#ifdef __AVX512F__

#include "../../config/tconfig.hpp"
#if defined(__x86_64__)
#include <x86intrin.h>
#else // assuming MSVC with _M_X64 or _M_IX86
#include <intrin.h>
#endif


namespace mfem
{

template <typename, int, int> struct AutoSIMD;

template <> struct AutoSIMD<double,8,64>
{
   typedef double scalar_type;
   static constexpr int size = 8;
   static constexpr int align_bytes = 64;

   union
   {
      __m512d m512d;
      double vec[size];
   };

   AutoSIMD() = default;

   AutoSIMD(const AutoSIMD &) = default;

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
      m512d = v.m512d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      m512d = _mm512_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m512d = _mm512_add_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      m512d = _mm512_add_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m512d = _mm512_sub_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      m512d = _mm512_sub_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m512d = _mm512_mul_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      m512d = _mm512_mul_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m512d = _mm512_div_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      m512d = _mm512_div_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
#ifdef __AVX512DQ__
      r.m512d = _mm512_xor_pd(_mm512_set1_pd(-0.0), m512d);
#else
      r.m512d = _mm512_sub_pd(_mm512_set1_pd(0.0), m512d);
#endif
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+() const
   {
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m512d = _mm512_add_pd(m512d,v.m512d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      r.m512d = _mm512_add_pd(m512d, _mm512_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m512d = _mm512_sub_pd(m512d,v.m512d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      r.m512d = _mm512_sub_pd(m512d, _mm512_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m512d = _mm512_mul_pd(m512d,v.m512d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      r.m512d = _mm512_mul_pd(m512d, _mm512_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m512d = _mm512_div_pd(m512d,v.m512d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      r.m512d = _mm512_div_pd(m512d, _mm512_set1_pd(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      m512d = _mm512_fmadd_pd(w.m512d,v.m512d,m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      m512d = _mm512_fmadd_pd(_mm512_set1_pd(e),v.m512d,m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      m512d = _mm512_fmadd_pd(v.m512d,_mm512_set1_pd(e),m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      m512d = _mm512_mul_pd(v.m512d,w.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      m512d = _mm512_mul_pd(v.m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      m512d = _mm512_mul_pd(_mm512_set1_pd(e),v.m512d);
      return *this;
   }
};

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator+(const double &e,
                                const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   r.m512d = _mm512_add_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator-(const double &e,
                                const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   r.m512d = _mm512_sub_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator*(const double &e,
                                const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   r.m512d = _mm512_mul_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator/(const double &e,
                                const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   r.m512d = _mm512_div_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

} // namespace mfem

#endif // __AVX512F__

#endif // MFEM_SIMD_M512_HPP
