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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_AUTO
#define MFEM_TEMPLATE_CONFIG_SIMD_AUTO

#include "../tconfig.hpp"

template <int S, int align_S>
struct MFEM_ALIGN_AS(align_S*sizeof(double)) AutoSIMD
{
   static const int size = S;
   static const int align_size = align_S;

   double vec[size];

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
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] -= v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] -= e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] *= v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] *= e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] /= v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] /= e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = -vec[i]; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] + v[i]; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] + e; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] - v[i]; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] - e; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] * v[i]; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] * e; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] / v[i]; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { r[i] = vec[i] / e; }
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += v[i] * w[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += v[i] * e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += e * v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = v[i] * w[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = v[i] * e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = e * v[i]; }
      return *this;
   }
};

template <int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<double,S,A> operator+(const double &e,
                               const AutoSIMD<double,S,A> &v)
{
   AutoSIMD<double,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e + v[i]; }
   return r;
}

template <int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<double,S,A> operator-(const double &e,
                               const AutoSIMD<double,S,A> &v)
{
   AutoSIMD<double,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e - v[i]; }
   return r;
}

template <int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<double,S,A> operator*(const double &e,
                               const AutoSIMD<double,S,A> &v)
{
   AutoSIMD<double,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e * v[i]; }
   return r;
}

template <int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<double,S,A> operator/(const double &e,
                               const AutoSIMD<double,S,A> &v)
{
   AutoSIMD<double,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e / v[i]; }
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_AUTO
