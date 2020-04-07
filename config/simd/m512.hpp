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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_M512
#define MFEM_TEMPLATE_CONFIG_SIMD_M512

#include "../tconfig.hpp"

template <typename scalar_t> struct AutoSIMD<scalar_t,8,8>
{
   typedef scalar_t scalar_type;
   static constexpr int size = 8;
   static constexpr int align_size = 64;

   union
   {
      __m512d m512d;
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
      m512d = v.m512d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const scalar_t &e)
   {
      m512d = _mm512_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m512d = _mm512_add_pd(m512d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const scalar_t &e)
   {
      m512d = _mm512_add_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m512d = _mm512_sub_pd(m512d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const scalar_t &e)
   {
      m512d = _mm512_sub_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m512d = _mm512_mul_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const scalar_t &e)
   {
      m512d = _mm512_mul_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m512d = _mm512_div_pd(m512d,v.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const scalar_t &e)
   {
      m512d = _mm512_div_pd(m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      return _mm512_xor_pd(_mm512_set1_pd(-0.0), m512d);
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.m512d = _mm512_add_pd(m512d,v.m512d);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
      m512d = _mm512_fmadd_pd(_mm512_set1_pd(e),v.m512d,m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
   {
      m512d = _mm512_fmadd_pd(v.m512d,_mm512_set1_pd(e),m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      m512d = _mm512_mul_pd(v.m512d,w.m512d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      m512d = _mm512_mul_pd(v.m512d,_mm512_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      m512d = _mm512_mul_pd(_mm512_set1_pd(e),v.m512d);
      return *this;
   }
};

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,8,8> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,8,8> &v)
{
   AutoSIMD<scalar_t,8,8> r;
   r.m512d = _mm512_add_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,8,8> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,8,8> &v)
{
   AutoSIMD<scalar_t,8,8> r;
   r.m512d = _mm512_sub_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,8,8> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,8,8> &v)
{
   AutoSIMD<scalar_t,8,8> r;
   r.m512d = _mm512_mul_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,8,8> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,8,8> &v)
{
   AutoSIMD<scalar_t,8,8> r;
   r.m512d = _mm512_div_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M512
