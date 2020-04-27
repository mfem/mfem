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

template <> struct AutoSIMD<double,8,8>
{
   static constexpr int size = 8;
   static constexpr int align_size = 64;

   union
   {
      __m512d m512d;
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
      r.m512d = _mm512_xor_pd(_mm512_set1_pd(-0.0), m512d);
      return r;
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
AutoSIMD<double,8,8> operator+(const double &e,
                               const AutoSIMD<double,8,8> &v)
{
   AutoSIMD<double,8,8> r;
   r.m512d = _mm512_add_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,8> operator-(const double &e,
                               const AutoSIMD<double,8,8> &v)
{
   AutoSIMD<double,8,8> r;
   r.m512d = _mm512_sub_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,8> operator*(const double &e,
                               const AutoSIMD<double,8,8> &v)
{
   AutoSIMD<double,8,8> r;
   r.m512d = _mm512_mul_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,8> operator/(const double &e,
                               const AutoSIMD<double,8,8> &v)
{
   AutoSIMD<double,8,8> r;
   r.m512d = _mm512_div_pd(_mm512_set1_pd(e),v.m512d);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M512
