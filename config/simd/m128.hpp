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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_M128
#define MFEM_TEMPLATE_CONFIG_SIMD_M128

#include "../tconfig.hpp"

template <typename scalar_t> struct AutoSIMD<scalar_t,2,2>
{
   typedef scalar_t scalar_type;
   static constexpr int size = 2;
   static constexpr int align_size = 16;

   union
   {
      __m128d m128d;
      scalar_t vec[size];
   };

   inline MFEM_ALWAYS_INLINE scalar_t &operator[](int i)
   {
      return vec[i];
   }

   inline MFEM_ALWAYS_INLINE const scalar_t &operator[](int i) const
   {
      return vec[i];
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const AutoSIMD &v)
   {
      m128d = v.m128d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const scalar_t &e)
   {
      m128d = _mm_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m128d = _mm_add_pd(m128d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const scalar_t &e)
   {
      m128d = _mm_add_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m128d = _mm_sub_pd(m128d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const scalar_t &e)
   {
      m128d = _mm_sub_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m128d = _mm_mul_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const scalar_t &e)
   {
      m128d = _mm_mul_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m128d = _mm_div_pd(m128d,v.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const scalar_t &e)
   {
      m128d = _mm_div_pd(m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      return _mm_xor_pd(_mm_set1_pd(-0.0), m128d);
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m128d = _mm_add_pd(m128d,v.m128d);
      return r;
   }


   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
      m128d = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(e),v.m128d),m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
   {
      m128d = _mm_add_pd(_mm_mul_pd(v.m128d,_mm_set1_pd(e)),m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      m128d = _mm_mul_pd(v.m128d,w.m128d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      m128d = _mm_mul_pd(v.m128d,_mm_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      m128d = _mm_mul_pd(_mm_set1_pd(e),v.m128d);
      return *this;
   }
};

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,2,2> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,2,2> &v)
{
   AutoSIMD<scalar_t,2,2> r;
   r.m128d = _mm_add_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,2,2> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,2,2> &v)
{
   AutoSIMD<scalar_t,2,2> r;
   r.m128d = _mm_sub_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,2,2> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,2,2> &v)
{
   AutoSIMD<scalar_t,2,2> r;
   r.m128d = _mm_mul_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,2,2> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,2,2> &v)
{
   AutoSIMD<scalar_t,2,2> r;
   r.m128d = _mm_div_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M128

