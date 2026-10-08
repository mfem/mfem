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

namespace mfem
{

/// Complex type: two @a real_t fields, matching GPU FFT's complex type.
struct complex_t
{
   real_t x, y;

   MFEM_HOST_DEVICE complex_t() = default;
   MFEM_HOST_DEVICE complex_t(real_t r) : x(r), y(0) {}
   MFEM_HOST_DEVICE complex_t(real_t r, real_t i) : x(r), y(i) {}

   MFEM_HOST_DEVICE real_t real() const { return x; }
   MFEM_HOST_DEVICE real_t imag() const { return y; }
};

MFEM_HOST_DEVICE inline real_t real(const complex_t &z) { return z.real(); }

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

MFEM_HOST_DEVICE inline complex_t operator/(real_t s, const complex_t &z)
{
   const real_t den = z.real() * z.real() + z.imag() * z.imag();
   return complex_t{s * z.real() / den, -s * z.imag() / den};
}

MFEM_HOST_DEVICE inline complex_t operator/(const complex_t &a,
                                            const complex_t &b)
{
   const real_t den = b.real() * b.real() + b.imag() * b.imag();
   return complex_t{(a.real() * b.real() + a.imag() * b.imag()) / den,
                    (a.imag() * b.real() - a.real() * b.imag()) / den};
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

MFEM_HOST_DEVICE inline real_t abs(const complex_t &z)
{
   return std::hypot(z.real(), z.imag());
}

/// Squared modulus, matching std::norm.
MFEM_HOST_DEVICE inline real_t norm(const complex_t &z)
{
   return z.real() * z.real() + z.imag() * z.imag();
}

MFEM_HOST_DEVICE inline complex_t conj(const complex_t &z)
{
   return complex_t{z.real(), -z.imag()};
}

MFEM_HOST_DEVICE inline complex_t exp(const complex_t &q)
{
   const real_t e = std::exp(q.real());
   return complex_t{std::cos(q.imag()) * e, std::sin(q.imag()) * e};
}

/// Principal logarithm.
MFEM_HOST_DEVICE inline complex_t log(const complex_t &z)
{
   return complex_t{std::log(abs(z)), std::atan2(z.imag(), z.real())};
}

MFEM_HOST_DEVICE inline complex_t pow(const complex_t &z, real_t s)
{
   return exp(s * log(z));
}

MFEM_HOST_DEVICE inline complex_t sin(const complex_t &z)
{
   return complex_t{std::sin(z.real()) * std::cosh(z.imag()),
                    std::cos(z.real()) * std::sinh(z.imag())};
}

MFEM_HOST_DEVICE inline complex_t cos(const complex_t &z)
{
   return complex_t{std::cos(z.real()) * std::cosh(z.imag()),
                    -std::sin(z.real()) * std::sinh(z.imag())};
}

MFEM_HOST_DEVICE inline complex_t tan(const complex_t &z)
{
   return sin(z) / cos(z);
}

} // namespace mfem
