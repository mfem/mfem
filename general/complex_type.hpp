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

#ifndef MFEM_COMPLEX_TYPE
#define MFEM_COMPLEX_TYPE

#include "../config/config.hpp"

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
#include <complex>
#include <utility>
#endif

#if defined(MFEM_USE_CUDA)
#include <cuComplex.h>
#endif

#if defined(MFEM_USE_HIP)
#include <hip/hip_complex.h>
#endif


namespace mfem
{

/// @brief Complex number type for device.
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))

#define zAbs std::abs
#define zExp std::exp
#define zNorm std::norm
using complex_t = std::complex<real_t>;

#else // CUDA or HIP

#if defined(MFEM_USE_CUDA)
using DoubleComplex_t = cuDoubleComplex;
#endif

#if defined(MFEM_USE_HIP)
using DoubleComplex_t = hipDoubleComplex;
#endif

struct Complex : public DoubleComplex_t
{
   MFEM_HOST_DEVICE Complex() = default;
   MFEM_HOST_DEVICE Complex(real_t r) { x = r, y = 0.0; }
   MFEM_HOST_DEVICE Complex(real_t r, real_t i) { x = r, y = i; }
   MFEM_HOST_DEVICE real_t real() const { return x; }
   MFEM_HOST_DEVICE void real(real_t r) { x = r; }
   MFEM_HOST_DEVICE real_t imag() const { return y; }
   MFEM_HOST_DEVICE void imag(real_t i) { y = i; }

   template <typename U>
   MFEM_HOST_DEVICE inline Complex &operator*=(const U &z)
   {
      return *this = *this * z, *this;
   }

   template <typename U>
   MFEM_HOST_DEVICE inline Complex &operator/=(const U &z)
   {
      return *this = *this / z, *this;
   }
};

MFEM_HOST_DEVICE inline Complex operator*(const Complex &x, const real_t &y)
{
   return Complex(x.real() * y, x.imag() * y);
}

MFEM_HOST_DEVICE inline Complex operator+(const Complex &a, const Complex &b)
{
   return Complex(a.real() + b.real(), a.imag() + b.imag());
}

MFEM_HOST_DEVICE inline Complex operator*(const real_t d, const Complex &z)
{
   return Complex(z.real() * d, z.imag() * d);
}

MFEM_HOST_DEVICE inline Complex operator*(const Complex &a, const Complex &b)
{
   return Complex(a.real() * b.real() - a.imag() * b.imag(),
                  a.real() * b.imag() + a.imag() * b.real());
}

MFEM_HOST_DEVICE inline Complex operator/(const Complex &z, const real_t &d)
{
   return Complex(z.real() / d, z.imag() / d);
}

MFEM_HOST_DEVICE inline real_t zAbs(const Complex &z)
{
   return std::hypot(z.real(), z.imag());
}

MFEM_HOST_DEVICE inline Complex zExp(const Complex &q)
{
   Complex z;
   real_t s, c, e = std::exp(q.real());
   sincos(q.imag(), &s, &c);
   z.real(c * e), z.imag(s * e);
   return z;
}

MFEM_HOST_DEVICE inline real_t zNorm(const Complex &z)
{
   return z.real() * z.real() + z.imag() * z.imag();
}

using complex_t = Complex;
#endif // MFEM_USE_CUDA || MFEM_USE_HIP

} // namespace mfem

#endif // MFEM_COMPLEX_TYPE
