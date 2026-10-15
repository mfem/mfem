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

#ifndef MFEM_COMPLEX_TYPE
#define MFEM_COMPLEX_TYPE

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

/** @brief Lightweight complex scalar with two @ref real_t components.

    This trivially copyable type is suitable for host and device kernels and
    has the same two-component storage layout as the corresponding GPU complex
    types. It is intentionally distinct from `std::complex<real_t>` and does
    not implicitly convert to or from it.

    Arithmetic operators and the functions real(), imag(), abs(), norm(),
    conj(), exp(), log(), pow(), sqrt(), sin(), cos(), and tan() are available
    through qualified `mfem::` lookup or unqualified argument-dependent lookup.
    A default-initialized object has indeterminate components; use
    `complex_t{}` to obtain zero initialization. */
struct complex_t
{
   real_t x, y;

   MFEM_HOST_DEVICE complex_t() = default;
   MFEM_HOST_DEVICE complex_t(real_t r) : x(r), y(0) { }
   MFEM_HOST_DEVICE complex_t(real_t r, real_t i) : x(r), y(i) { }

   MFEM_HOST_DEVICE real_t real() const { return x; }
   MFEM_HOST_DEVICE real_t imag() const { return y; }

   // Functions of complex_t below are only found by argument-dependent lookup.

   friend MFEM_HOST_DEVICE real_t real(const complex_t &z) { return z.x; }

   friend MFEM_HOST_DEVICE real_t imag(const complex_t &z) { return z.y; }

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
      const real_t e = std::exp(q.x);
      return complex_t{std::cos(q.y) * e, std::sin(q.y) * e};
   }

   friend MFEM_HOST_DEVICE complex_t log(const complex_t &z)
   {
      return complex_t{std::log(abs(z)), std::atan2(z.y, z.x)};
   }

   friend MFEM_HOST_DEVICE complex_t pow(const complex_t &z, real_t s)
   {
      return exp(s * log(z));
   }

   friend MFEM_HOST_DEVICE complex_t sqrt(const complex_t &z)
   {
      if (z.y == 0)
      {
         if (z.x < 0)
         {
            return complex_t{0, std::copysign(std::sqrt(-z.x), z.y)};
         }
         return complex_t{std::sqrt(z.x), std::copysign(real_t{0}, z.y)};
      }

      const real_t ax = std::fabs(z.x), ay = std::fabs(z.y);
      real_t t;
      if (ax >= ay)
      {
         const real_t ratio = ay / ax;
         t = std::sqrt(ax) * std::sqrt(real_t{0.5} *
                                       (1 + std::sqrt(1 + ratio * ratio)));
      }
      else
      {
         const real_t ratio = ax / ay;
         t = std::sqrt(ay) * std::sqrt(real_t{0.5} *
                                       (ratio + std::sqrt(1 + ratio * ratio)));
      }
      if (z.x >= 0)
      {
         return complex_t{t, z.y / (2 * t)};
      }
      return complex_t{std::fabs(z.y) / (2 * t), std::copysign(t, z.y)};
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
      return sin(z) / cos(z);
   }
};

MFEM_HOST_DEVICE inline bool operator==(const complex_t &a, const complex_t &b)
{
   return a.real() == b.real() && a.imag() == b.imag();
}

MFEM_HOST_DEVICE inline bool operator!=(const complex_t &a, const complex_t &b)
{
   return !(a == b);
}

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

/// x * d / c for ratio = d / c, also when the ratio underflows to zero.
MFEM_HOST_DEVICE inline real_t ComplexDivTerm(real_t x, real_t ratio,
                                              real_t d, real_t c)
{
   return (ratio != 0 || d == 0) ? x * ratio : d * (x / c);
}

/// int when T is complex_t, no type otherwise.
template <typename T>
using IfComplex = std::enable_if_t<std::is_same_v<T, complex_t>, int>;

} // namespace internal

/// Complex division: Smith's algorithm with the power-of-two scaling of
/// Baudin and Smith for components near overflow or underflow.
MFEM_HOST_DEVICE inline complex_t operator/(const complex_t &a,
                                            const complex_t &b)
{
   using limits = std::numeric_limits<real_t>;
   constexpr real_t half = 0.5;
   const real_t big = half * limits::max();
   const real_t small = 2 * limits::min() / limits::epsilon();
   const real_t up = 2 / (limits::epsilon() * limits::epsilon());
   real_t ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
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
      const real_t re = ar + internal::ComplexDivTerm(ai, ratio, bi, br);
      const real_t im = ai - internal::ComplexDivTerm(ar, ratio, bi, br);
      return complex_t{(re / den) * scale, (im / den) * scale};
   }
   const real_t ratio = br / bi, den = br * ratio + bi;
   const real_t re = internal::ComplexDivTerm(ar, ratio, br, bi) + ai;
   const real_t im = internal::ComplexDivTerm(ai, ratio, br, bi) - ar;
   return complex_t{(re / den) * scale, (im / den) * scale};
}

/// Scalar over complex, as complex_t(s) / z.
MFEM_HOST_DEVICE inline complex_t operator/(real_t s, const complex_t &z)
{
   return complex_t(s) / z;
}

MFEM_HOST_DEVICE inline complex_t &operator*=(complex_t &a, const complex_t &b)
{
   a = a * b;
   return a;
}

MFEM_HOST_DEVICE inline complex_t &operator/=(complex_t &a, const complex_t &b)
{
   a = a / b;
   return a;
}

MFEM_HOST_DEVICE inline complex_t &operator+=(complex_t &a, const complex_t &b)
{
   a.x += b.x;
   a.y += b.y;
   return a;
}

MFEM_HOST_DEVICE inline complex_t &operator-=(complex_t &a, const complex_t &b)
{
   a.x -= b.x;
   a.y -= b.y;
   return a;
}

} // namespace mfem

/// Math functions of complex_t, forwarding to its hidden friend functions.
namespace mfemComplex
{

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::real_t real(const T &z) { return real(z); }

template <typename T, mfem::internal::IfComplex<T> = 0>
MFEM_HOST_DEVICE inline mfem::real_t imag(const T &z) { return imag(z); }

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
MFEM_HOST_DEVICE inline mfem::complex_t sqrt(const T &z) { return sqrt(z); }

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
using std::sqrt;
using std::tan;

using mfemComplex::abs;
using mfemComplex::conj;
using mfemComplex::cos;
using mfemComplex::exp;
using mfemComplex::imag;
using mfemComplex::log;
using mfemComplex::norm;
using mfemComplex::pow;
using mfemComplex::real;
using mfemComplex::sin;
using mfemComplex::sqrt;
using mfemComplex::tan;

} // namespace mfem

#endif // MFEM_COMPLEX_TYPE
