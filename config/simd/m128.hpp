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

template <> struct AutoSIMD<double,2,2>
{
   static constexpr int size = 2;
   static constexpr int align_size = 16;

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
AutoSIMD<double,2,2> operator+(const double &e,
                               const AutoSIMD<double,2,2> &v)
{
   AutoSIMD<double,2,2> r;
   r.m128d = _mm_add_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,2> operator-(const double &e,
                               const AutoSIMD<double,2,2> &v)
{
   AutoSIMD<double,2,2> r;
   r.m128d = _mm_sub_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,2> operator*(const double &e,
                               const AutoSIMD<double,2,2> &v)
{
   AutoSIMD<double,2,2> r;
   r.m128d = _mm_mul_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,2> operator/(const double &e,
                               const AutoSIMD<double,2,2> &v)
{
   AutoSIMD<double,2,2> r;
   r.m128d = _mm_div_pd(_mm_set1_pd(e),v.m128d);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_M128

