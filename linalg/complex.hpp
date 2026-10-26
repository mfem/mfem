// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "../config/config.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace mfem
{

struct complex_t;

MFEM_HOST_DEVICE inline complex_t operator*(real_t s, const complex_t &z);

MFEM_HOST_DEVICE inline complex_t operator/(const complex_t &a,
                                            const complex_t &b);

/// Complex type: two @a real_t fields, matching GPU complex type.
struct complex_t
{
   real_t x, y;

   MFEM_HOST_DEVICE complex_t() = default;
   MFEM_HOST_DEVICE complex_t(real_t r) : x(r), y(0) {}
   MFEM_HOST_DEVICE complex_t(real_t r, real_t i) : x(r), y(i) {}

   MFEM_HOST_DEVICE real_t real() const { return x; }
   MFEM_HOST_DEVICE real_t imag() const { return y; }

   // Functions of complex_t below are only found by argument-dependent lookup

   friend MFEM_HOST_DEVICE real_t real(const complex_t &z) { return z.x; }

   friend MFEM_HOST_DEVICE real_t abs(const complex_t &z)
   {
      return std::hypot(z.x, z.y);
   }

   friend MFEM_HOST_DEVICE real_t norm(const complex_t &z)
   {
      return z.x * z.x + z.y * z.y;
   }

   friend MFEM_HOST_DEVICE complex_t conj(const complex_t &z)
   {
      return complex_t{z.x, -z.y};
   }

   friend MFEM_HOST_DEVICE complex_t exp(const complex_t &q)
   {
      // exp(inf + 0i) is (inf, 0); 0 * inf would be a NaN imaginary part.
      if (q.y == 0) { return complex_t{std::exp(q.x), q.y}; }
      const real_t c = std::cos(q.y);
      const real_t s = std::sin(q.y);
      const real_t m = std::fmax(std::fabs(c), std::fabs(s));
      const real_t e = std::exp(q.x + std::log(m));
      return complex_t{(c / m) * e, (s / m) * e};
   }

   friend MFEM_HOST_DEVICE complex_t log(const complex_t &z)
   {
      const real_t ax = std::fabs(z.x);
      const real_t ay = std::fabs(z.y);
      const real_t big = std::fmax(ax, ay);
      real_t mag;
      if (big == 0)
      {
         mag = -std::numeric_limits<real_t>::infinity();
      }
      else
      {
         const real_t r = std::fmin(ax, ay) / big;
         mag = std::log(big) + real_t{0.5} * std::log1p(r * r);
      }
      return complex_t{mag, std::atan2(z.y, z.x)};
   }

   friend MFEM_HOST_DEVICE complex_t pow(const complex_t &z, real_t s)
   {
      return exp(s * log(z));
   }

   friend MFEM_HOST_DEVICE complex_t sin(const complex_t &z)
   {
      return complex_t{std::sin(z.x) * std::cosh(z.y),
                       std::cos(z.x) * std::sinh(z.y)};
   }

   friend MFEM_HOST_DEVICE complex_t cos(const complex_t &z)
   {
      return complex_t{std::cos(z.x) * std::cosh(z.y),
                       -std::sin(z.x) * std::sinh(z.y)};
   }

   friend MFEM_HOST_DEVICE complex_t tan(const complex_t &z)
   {
      // Scale by exp(-2|y|) so a large imaginary part stays finite.
      const real_t x2 = 2 * z.x;
      const real_t y2 = 2 * z.y;
      const real_t ex = std::exp(-std::fabs(y2));
      const real_t den = 1 + ex * ex + 2 * std::cos(x2) * ex;
      const real_t re = 2 * std::sin(x2) * ex / den;
      const real_t im_num = -std::expm1(-2 * std::fabs(y2));
      return complex_t{re, std::copysign(im_num, y2) / den};
   }
};

MFEM_HOST_DEVICE inline complex_t operator-(const complex_t &z)
{
   return complex_t{-z.real(), -z.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator+(real_t a, const complex_t &b)
{
   return complex_t{a + b.real(), b.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator+(const complex_t &a, real_t b)
{
   return complex_t{a.real() + b, a.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator+(const complex_t &a,
                                            const complex_t &b)
{
   return complex_t{a.real() + b.real(), a.imag() + b.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator-(const complex_t &a,
                                            const complex_t &b)
{
   return complex_t{a.real() - b.real(), a.imag() - b.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator-(real_t a, const complex_t &b)
{
   return complex_t{a - b.real(), -b.imag()};
}

MFEM_HOST_DEVICE inline complex_t operator*(const complex_t &a,
                                            const complex_t &b)
{
   return complex_t{a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real()};
}

MFEM_HOST_DEVICE inline complex_t operator*(const complex_t &z, real_t s)
{
   return complex_t{z.real() * s, z.imag() * s};
}

MFEM_HOST_DEVICE inline complex_t operator*(real_t s, const complex_t &z)
{
   return z * s;
}

MFEM_HOST_DEVICE inline complex_t operator/(const complex_t &z, real_t s)
{
   return complex_t{z.real() / s, z.imag() / s};
}

namespace internal
{

/// (x * d / c) / den for ratio = d / c, also when a product underflows
MFEM_HOST_DEVICE inline real_t ComplexDivTerm(real_t x, real_t ratio,
                                              real_t d, real_t c,
                                              real_t den)
{
   if (ratio == 0)
   {
      return (d == 0) ? real_t{0} : (d * (x / c)) / den;
   }
   const real_t prod = x * ratio;
   if (prod != 0 || x == 0) { return prod / den; }
   return (x / den) * ratio;
}

/// True when @a q is the quotient of a non-finite input. An infinite
/// numerator over a finite denominator returns false and uses Smith's formula
MFEM_HOST_DEVICE inline bool ComplexNonfiniteQuotient(real_t ar, real_t ai,
                                                      real_t br, real_t bi,
                                                      complex_t &q)
{
   if (std::isfinite(ar) && std::isfinite(ai) &&
       std::isfinite(br) && std::isfinite(bi))
   {
      return false;
   }
   const bool a_bad = !std::isfinite(ar) || !std::isfinite(ai);
   const bool b_bad = !std::isfinite(br) || !std::isfinite(bi);
   const bool nan = std::isnan(ar) || std::isnan(ai) ||
                    std::isnan(br) || std::isnan(bi);
   if (!nan && a_bad && !b_bad) { return false; }
   if (!nan && b_bad && !a_bad)
   {
      q = complex_t{0, 0};
   }
   else
   {
      const real_t n = std::numeric_limits<real_t>::quiet_NaN();
      q = complex_t{n, n};
   }
   return true;
}

/// int when T is complex_t, no type otherwise
template <typename T>
using IfComplex = std::enable_if_t<std::is_same_v<T, complex_t>, int>;

} // namespace internal

/// Complex division: Smith's algorithm with the power-of-two scaling of
/// Baudin and Smith for components near overflow or underflow
MFEM_HOST_DEVICE inline complex_t operator/(const complex_t &a,
                                            const complex_t &b)
{
   using limits = std::numeric_limits<real_t>;
   constexpr real_t half = 0.5;
   const real_t big = half * limits::max();
   const real_t small = 2 * limits::min() / limits::epsilon();
   const real_t up = 2 / (limits::epsilon() * limits::epsilon());
   real_t ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
   complex_t special;
   if (internal::ComplexNonfiniteQuotient(ar, ai, br, bi, special))
   {
      return special;
   }
   const real_t na = std::fmax(std::fabs(ar), std::fabs(ai));
   const real_t nb = std::fmax(std::fabs(br), std::fabs(bi));
   real_t scale = 1;
   if (na >= big) { ar *= half, ai *= half, scale *= 2; }
   if (nb >= big) { br *= half, bi *= half, scale *= half; }
   if (na <= small) { ar *= up, ai *= up, scale /= up; }
   if (nb <= small) { br *= up, bi *= up, scale *= up; }
   if (std::fabs(br) >= std::fabs(bi))
   {
      const real_t ratio = bi / br, den = br + bi * ratio;
      const real_t re =
         ar / den + internal::ComplexDivTerm(ai, ratio, bi, br, den);
      const real_t im =
         ai / den - internal::ComplexDivTerm(ar, ratio, bi, br, den);
      return complex_t{re * scale, im * scale};
   }
   const real_t ratio = br / bi, den = br * ratio + bi;
   const real_t re =
      internal::ComplexDivTerm(ar, ratio, br, bi, den) + ai / den;
   const real_t im =
      internal::ComplexDivTerm(ai, ratio, br, bi, den) - ar / den;
   return complex_t{re * scale, im * scale};
}

/// Scalar over complex, as complex_t(s) / z
MFEM_HOST_DEVICE inline complex_t operator/(real_t s, const complex_t &z)
{
   return complex_t(s) / z;
}

MFEM_HOST_DEVICE inline complex_t &operator*=(complex_t &a, const complex_t &b)
{
   a = a * b;
   return a;
}

MFEM_HOST_DEVICE inline complex_t &operator+=(complex_t &a, const complex_t &b)
{
   a.x += b.x;
   a.y += b.y;
   return a;
}

} // namespace mfem

/// real, abs, norm, conj, exp, log, pow, sin, cos and tan of complex_t only,
/// calling the friend functions of complex_t
namespace mfemComplex
{

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::real_t real(const T &z) { return real(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::real_t abs(const T &z) { return abs(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::real_t norm(const T &z) { return norm(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t conj(const T &z) { return conj(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t exp(const T &z) { return exp(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t log(const T &z) { return log(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t pow(const T &z, mfem::real_t s)
{
   return pow(z, s);
}

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t sin(const T &z) { return sin(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t cos(const T &z) { return cos(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::complex_t tan(const T &z) { return tan(z); }

} // namespace mfemComplex

namespace mfem
{

// Qualified and unqualified lookup both see these names. The complex_t
// templates alone would hide pow, sin, and the rest on MSVC, whose <cmath>
// declares the real overloads only in namespace std. Keep both sets.
using std::abs;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::tan;

using mfemComplex::abs;
using mfemComplex::conj;
using mfemComplex::cos;
using mfemComplex::exp;
using mfemComplex::log;
using mfemComplex::norm;
using mfemComplex::pow;
using mfemComplex::real;
using mfemComplex::sin;
using mfemComplex::tan;

} // namespace mfem
