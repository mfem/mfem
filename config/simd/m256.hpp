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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_M256
#define MFEM_TEMPLATE_CONFIG_SIMD_M256

#include "../tconfig.hpp"

template <typename scalar_t> struct AutoSIMD<scalar_t,4,4>
{
   typedef scalar_t scalar_type;
   static constexpr int size = 4;
   static constexpr int align_size = 32;

   union
   {
      __m256d m256d;
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
      m256d = v.m256d;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const scalar_t &e)
   {
      m256d = _mm256_set1_pd(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      m256d = _mm256_add_pd(m256d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const scalar_t &e)
   {
      m256d = _mm256_add_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      m256d = _mm256_sub_pd(m256d,v);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const scalar_t &e)
   {
      m256d = _mm256_sub_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      m256d = _mm256_mul_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const scalar_t &e)
   {
      m256d = _mm256_mul_pd(m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      m256d = _mm256_div_pd(m256d,v.m256d);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
#ifndef __AVX2__
      m256d = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(e),v.m256d),m256d);
#else
      m256d = _mm256_fmadd_pd(_mm256_set1_pd(e),v.m256d,m256d);
#endif
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      m256d = _mm256_mul_pd(v.m256d,_mm256_set1_pd(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      m256d = _mm256_mul_pd(_mm256_set1_pd(e),v.m256d);
      return *this;
   }
};

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,4,4> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.m256d = _mm256_add_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,4,4> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.m256d = _mm256_sub_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,4,4> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.m256d = _mm256_mul_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

template <typename scalar_t>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,4,4> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.m256d = _mm256_div_pd(_mm256_set1_pd(e),v.m256d);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M256
