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

#ifndef MFEM_SIMD_QPX_256_HPP
#define MFEM_SIMD_QPX_256_HPP

#ifdef __bgq__

#include "../../config/tconfig.hpp"
#include <builtins.h>

namespace mfem
{

template <typename,int,int> struct AutoSIMD;

template <> struct AutoSIMD<double,4,32>
{
   typedef double scalar_type;
   static constexpr int size = 4;
   static constexpr int align_bytes = 32;

   union
   {
      vector4double vd;
      double vec[size];
   };

   AutoSIMD() = default;

   AutoSIMD(const AutoSIMD &) = default;

   inline __ATTRS_ai double &operator[](int i) { return vec[i]; }

   inline __ATTRS_ai const double &operator[](int i) const { return vec[i]; }

   inline __ATTRS_ai AutoSIMD &operator=(const AutoSIMD &v)
   {
      vd = v.vd;
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator=(const double &e)
   {
      vd = vec_splats(e);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator+=(const AutoSIMD &v)
   {
      vd = vec_add(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator+=(const double &e)
   {
      vd = vec_add(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator-=(const AutoSIMD &v)
   {
      vd = vec_sub(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator-=(const double &e)
   {
      vd = vec_sub(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator*=(const AutoSIMD &v)
   {
      vd = vec_mul(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator*=(const double &e)
   {
      vd = vec_mul(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator/=(const AutoSIMD &v)
   {
      vd = vec_swdiv(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator/=(const double &e)
   {
      vd = vec_swdiv(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD operator-() const
   {
      AutoSIMD r;
      r.vd = vec_neg(vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator+() const
   {
      return *this;
   }

   inline __ATTRS_ai AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_add(vd,v.vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_add(vd, vec_splats(e));
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_sub(vd,v.vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_sub(vd, vec_splats(e));
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_mul(vd,v.vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_mul(vd, vec_splats(e));
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_swdiv(vd,v.vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      r.vd = vec_swdiv(vd, vec_splats(e));
      return r;
   }

   inline __ATTRS_ai AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      vd = vec_madd(w.vd,vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      vd = vec_madd(v.vd,vec_splats(e),vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      vd = vec_madd(vec_splats(e),v.vd,vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      vd = vec_mul(v.vd,w.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const AutoSIMD &v, const double &e)
   {
      vd = vec_mul(v.vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      vd = vec_mul(vec_splats(e),v.vd);
      return *this;
   }
};

inline __ATTRS_ai
AutoSIMD<double,4,32> operator+(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.vd = vec_add(vec_splats(e),v.vd);
   return r;
}

inline __ATTRS_ai
AutoSIMD<double,4,32> operator-(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.vd = vec_sub(vec_splats(e),v.vd);
   return r;
}

inline __ATTRS_ai
AutoSIMD<double,4,32> operator*(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.vd = vec_mul(vec_splats(e),v.vd);
   return r;
}

inline __ATTRS_ai
AutoSIMD<double,4,32> operator/(const double &e,
                                const AutoSIMD<double,4,32> &v)
{
   AutoSIMD<double,4,32> r;
   r.vd = vec_swdiv(vec_splats(e),v.vd);
   return r;
}

} // namespace mfem

#endif // __bgq__

#endif // MFEM_SIMD_QPX_256_HPP
