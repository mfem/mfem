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

#ifndef MFEM_SIMD_VSX128_HPP
#define MFEM_SIMD_VSX128_HPP

#ifdef __VSX__

#include "../../config/tconfig.hpp"
#include <altivec.h>

#ifdef __GNUC__
#undef bool
#endif

namespace mfem
{

template <typename,int,int> struct AutoSIMD;

template <> struct AutoSIMD<double,2,16>
{
   typedef double scalar_type;
   static constexpr int size = 2;
   static constexpr int align_bytes = 16;

   union
   {
      vector double vd;
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
      vd = v.vd;
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      vd = vec_splats(e);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      vd = vec_add(vd,v.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      vd = vec_add(vd,vec_splats(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      vd = vec_sub(vd,v.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      vd = vec_sub(vd,vec_splats(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      vd = vec_mul(vd,v.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      vd = vec_mul(vd,vec_splats(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      vd = vec_div(vd,v.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      vd = vec_div(vd,vec_splats(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
#ifndef __GNUC__
      r.vd = vec_neg(vd);
#else
      r.vd = vec_splats(0.0) - vd;
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
      r.vd = vec_add(vd,v.vd);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_add(vd, vec_splats(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_sub(vd,v.vd);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_sub(vd, vec_splats(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_mul(vd,v.vd);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_mul(vd, vec_splats(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_div(vd,v.vd);
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_div(vd, vec_splats(e));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      vd = vec_madd(w.vd,vd,v.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      vd = vec_madd(v.vd,vec_splats(e),vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      vd = vec_madd(vec_splats(e),v.vd,vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      vd = vec_mul(v.vd,w.vd);
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      vd = vec_mul(v.vd,vec_splats(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      vd = vec_mul(vec_splats(e),v.vd);
      return *this;
   }
};

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator+(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.vd = vec_add(vec_splats(e),v.vd);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator-(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.vd = vec_sub(vec_splats(e),v.vd);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator*(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.vd = vec_mul(vec_splats(e),v.vd);
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,2,16> operator/(const double &e,
                                const AutoSIMD<double,2,16> &v)
{
   AutoSIMD<double,2,16> r;
   r.vd = vec_div(vec_splats(e),v.vd);
   return r;
}

} // namespace mfem

#endif // __VSX__

#endif // MFEM_SIMD_VSX128_HPP
