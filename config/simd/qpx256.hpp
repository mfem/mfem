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

#ifndef MFEM_TEMPLATE_CONFIG_SIMD_QPX_256
#define MFEM_TEMPLATE_CONFIG_SIMD_QPX_256

#include "../tconfig.hpp"

template <typename scalar_t> struct AutoSIMD<scalar_t,4,4>
{
   typedef scalar_t scalar_type;
   static constexpr int size = 4;
   static constexpr int align_size = 32;

   union
   {
      vector4double vd;
      scalar_t vec[size];
   };

   inline __ATTRS_ai scalar_t &operator[](int i) { return vec[i]; }

   inline __ATTRS_ai const scalar_t &operator[](int i) const { return vec[i]; }

   inline __ATTRS_ai AutoSIMD &operator=(const AutoSIMD &v)
   {
      vd = v.vd;
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator=(const scalar_t &e)
   {
      vd = vec_splats(e);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator+=(const AutoSIMD &v)
   {
      vd = vec_add(vd,v);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator+=(const scalar_t &e)
   {
      vd = vec_add(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator-=(const AutoSIMD &v)
   {
      vd = vec_sub(vd,v);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator-=(const scalar_t &e)
   {
      vd = vec_sub(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator*=(const AutoSIMD &v)
   {
      vd = vec_mul(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator*=(const scalar_t &e)
   {
      vd = vec_mul(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator/=(const AutoSIMD &v)
   {
      vd = vec_swdiv(vd,v.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &operator/=(const scalar_t &e)
   {
      vd = vec_swdiv(vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD operator-() const
   {
      return vec_neg(vd);
   }

   inline __ATTRS_ai AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      r.vd = vec_add(vd,v.vd);
      return r;
   }

   inline __ATTRS_ai AutoSIMD operator+(const scalar_t &e) const
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

   inline __ATTRS_ai AutoSIMD operator-(const scalar_t &e) const
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

   inline __ATTRS_ai AutoSIMD operator*(const scalar_t &e) const
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

   inline __ATTRS_ai AutoSIMD operator/(const scalar_t &e) const
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

   inline __ATTRS_ai AutoSIMD &fma(const AutoSIMD &v, const scalar_t &e)
   {
      vd = vec_madd(v.vd,vec_splats(e),vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &fma(const scalar_t &e, const AutoSIMD &v)
   {
      vd = vec_madd(vec_splats(e),v.vd,vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      vd = vec_mul(v.vd,w.vd);
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const AutoSIMD &v, const scalar_t &e)
   {
      vd = vec_mul(v.vd,vec_splats(e));
      return *this;
   }

   inline __ATTRS_ai AutoSIMD &mul(const scalar_t &e, const AutoSIMD &v)
   {
      vd = vec_mul(vec_splats(e),v.vd);
      return *this;
   }
};

template <typename scalar_t>
inline __ATTRS_ai
AutoSIMD<scalar_t,4,4> operator+(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.vd = vec_add(vec_splats(e),v.vd);
   return r;
}

template <typename scalar_t>
inline __ATTRS_ai
AutoSIMD<scalar_t,4,4> operator-(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.vd = vec_sub(vec_splats(e),v.vd);
   return r;
}

template <typename scalar_t>
inline __ATTRS_ai
AutoSIMD<scalar_t,4,4> operator*(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.vd = vec_mul(vec_splats(e),v.vd);
   return r;
}

template <typename scalar_t>
inline __ATTRS_ai
AutoSIMD<scalar_t,4,4> operator/(const scalar_t &e,
                                 const AutoSIMD<scalar_t,4,4> &v)
{
   AutoSIMD<scalar_t,4,4> r;
   r.vd = vec_swdiv(vec_splats(e),v.vd);
   return r;
}

#endif // MFEM_TEMPLATE_CONFIG_SIMD_QPX_256
