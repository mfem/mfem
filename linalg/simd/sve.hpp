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

#ifndef MFEM_SIMD_SVE_HPP
#define MFEM_SIMD_SVE_HPP

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)

#include "../../config/tconfig.hpp"
#include <arm_sve.h>

namespace mfem
{

// Use this macro as a workaround for astyle formatting issue with 'alignas'
#define MFEM_AUTOSIMD_ALIGN_SVE alignas(64)

template <typename,int,int> struct AutoSIMD;

template <> struct MFEM_AUTOSIMD_ALIGN_SVE AutoSIMD<double,8,64>
{
   typedef double scalar_type;
   static constexpr int size = 8;
   static constexpr int align_bytes = 64;

   double vec[size];

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
      svst1_f64(svptrue_b64(), vec, svld1_f64(svptrue_b64(),v.vec));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator=(const double &e)
   {
      svst1_f64(svptrue_b64(), vec, svdup_n_f64(e));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const AutoSIMD &v)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svadd_f64_z(svptrue_b64(),vd,vvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator+=(const double &e)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), vec, svadd_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const AutoSIMD &v)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svsub_f64_z(svptrue_b64(),vd,vvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator-=(const double &e)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), vec, svsub_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const AutoSIMD &v)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svmul_f64_z(svptrue_b64(),vd,vvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator*=(const double &e)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), vec, svmul_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const AutoSIMD &v)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svdiv_f64_z(svptrue_b64(),vd,vvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &operator/=(const double &e)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), vec, svdiv_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-() const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), r.vec, svneg_f64_z(svptrue_b64(),vd));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+() const
   {
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const AutoSIMD &v) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), r.vec, svadd_f64_z(svptrue_b64(),vd,vvd));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator+(const double &e) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), r.vec, svadd_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const AutoSIMD &v) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), r.vec, svsub_f64_z(svptrue_b64(),vd,vvd));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator-(const double &e) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), r.vec, svsub_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const AutoSIMD &v) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), r.vec, svmul_f64_z(svptrue_b64(),vd,vvd));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator*(const double &e) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), r.vec, svmul_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const AutoSIMD &v) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), r.vec, svdiv_f64_z(svptrue_b64(),vd,vvd));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD operator/(const double &e) const
   {
      AutoSIMD r;
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      svst1_f64(svptrue_b64(), r.vec, svdiv_f64_z(svptrue_b64(),vd,svdup_n_f64(e)));
      return r;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const AutoSIMD &w)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      const svfloat64_t wvd = svld1_f64(svptrue_b64(), w.vec);
      svst1_f64(svptrue_b64(), vec, svmad_f64_z(svptrue_b64(),wvd,vd,vvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const AutoSIMD &v, const double &e)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svmad_f64_z(svptrue_b64(),vvd,svdup_n_f64(e),vd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &fma(const double &e, const AutoSIMD &v)
   {
      const svfloat64_t vd = svld1_f64(svptrue_b64(), vec);
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svmad_f64_z(svptrue_b64(),svdup_n_f64(e),vvd,vd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v, const AutoSIMD &w)
   {
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      const svfloat64_t wvd = svld1_f64(svptrue_b64(), w.vec);
      svst1_f64(svptrue_b64(), vec, svmul_f64_z(svptrue_b64(),vvd,wvd));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const AutoSIMD &v,const double &e)
   {
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svmul_f64_z(svptrue_b64(),vvd,svdup_n_f64(e)));
      return *this;
   }

   inline MFEM_ALWAYS_INLINE AutoSIMD &mul(const double &e, const AutoSIMD &v)
   {
      const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
      svst1_f64(svptrue_b64(), vec, svmul_f64_z(svptrue_b64(),svdup_n_f64(e),vvd));
      return *this;
   }
};

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator+(const double &e, const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
   svst1_f64(svptrue_b64(), r.vec, svadd_f64_z(svptrue_b64(),svdup_n_f64(e),vvd));
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator-(const double &e, const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
   svst1_f64(svptrue_b64(), r.vec, svsub_f64_z(svptrue_b64(),svdup_n_f64(e),vvd));
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator*(const double &e, const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
   svst1_f64(svptrue_b64(), r.vec, svmul_f64_z(svptrue_b64(),svdup_n_f64(e),vvd));
   return r;
}

inline MFEM_ALWAYS_INLINE
AutoSIMD<double,8,64> operator/(const double &e, const AutoSIMD<double,8,64> &v)
{
   AutoSIMD<double,8,64> r;
   const svfloat64_t vvd = svld1_f64(svptrue_b64(), v.vec);
   svst1_f64(svptrue_b64(), r.vec, svdiv_f64_z(svptrue_b64(),svdup_n_f64(e),vvd));
   return r;
}

} // namespace mfem

#endif // __aarch64__ && __ARM_FEATURE_SVE

#endif // MFEM_SIMD_SVE_HPP
