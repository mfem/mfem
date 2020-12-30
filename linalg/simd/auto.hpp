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

#ifndef MFEM_SIMD_AUTO_HPP
#define MFEM_SIMD_AUTO_HPP

#include "../../config/tconfig.hpp"

namespace mfem
{

// Use this macro as a workaround for astyle formatting issue with 'alignas'
#define MFEM_AUTOSIMD_ALIGN__ alignas(align_bytes_)

template <typename scalar_t, int S, int align_bytes_>
struct MFEM_AUTOSIMD_ALIGN__ AutoSIMD
{
   typedef scalar_t scalar_type;
   static const int size = S;
   static const int align_bytes = align_bytes_;

   scalar_t vec[size];

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
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = v[i]; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const scalar_t &e)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const scalar_t &e) const
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] += v[i] * e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
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

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = v[i] * e; }
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      MFEM_VECTORIZE_LOOP
      for (int i = 0; i < size; i++) { vec[i] = e * v[i]; }
      return *this;
   }
};

template <typename scalar_t, int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,S,A> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,S,A> &v)
{
   AutoSIMD<scalar_t,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e + v[i]; }
   return r;
}

template <typename scalar_t, int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,S,A> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,S,A> &v)
{
   AutoSIMD<scalar_t,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e - v[i]; }
   return r;
}

template <typename scalar_t, int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,S,A> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,S,A> &v)
{
   AutoSIMD<scalar_t,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e * v[i]; }
   return r;
}

template <typename scalar_t, int S, int A>
inline MFEM_ALWAYS_INLINE
AutoSIMD<scalar_t,S,A> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,S,A> &v)
{
   AutoSIMD<scalar_t,S,A> r;
   MFEM_VECTORIZE_LOOP
   for (int i = 0; i < S; i++) { r[i] = e / v[i]; }
   return r;
}

} // namespace mfem

#endif // MFEM_SIMD_AUTO_HPP
