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

#include "mfem.hpp"
#include "linalg/complex.hpp"
#include "catch.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

namespace mfem
{
namespace
{

// Unqualified calls inside namespace mfem: real and integer arguments keep the
// C library functions, complex_t arguments find the complex_t functions
static_assert(std::is_same<decltype(exp(1.0)), double>::value);
static_assert(std::is_same<decltype(pow(2.0, 3.0)), double>::value);
static_assert(std::is_same<decltype(sin(1.0)), double>::value);
static_assert(std::is_same<decltype(abs(-2)), int>::value);
static_assert(std::is_same<decltype(log(2.0)), double>::value);
static_assert(std::is_same<decltype(cos(1.0)), double>::value);
static_assert(std::is_same<decltype(tan(1.0)), double>::value);
static_assert(std::is_same<decltype(exp(complex_t{})), complex_t>::value);
static_assert(std::is_same<decltype(abs(complex_t{})), real_t>::value);
static_assert(std::is_same<decltype(pow(complex_t{}, 2.0)), complex_t>::value);

} // namespace
} // namespace mfem

// Qualified calls
static_assert(std::is_same<decltype(mfem::real(mfem::complex_t{})),
              mfem::real_t>::value);
static_assert(std::is_same<decltype(mfem::abs(mfem::complex_t{})),
              mfem::real_t>::value);
static_assert(std::is_same<decltype(mfem::exp(mfem::complex_t{})),
              mfem::complex_t>::value);
static_assert(std::is_same<decltype(mfem::pow(mfem::complex_t{}, 2.0)),
              mfem::complex_t>::value);

namespace
{

// The functions of z through mfem:: names
std::vector<mfem::complex_t> QualifiedValues(const mfem::complex_t &z)
{
   return {mfem::real(z), mfem::abs(z), mfem::norm(z), mfem::conj(z),
           mfem::exp(z), mfem::log(z), mfem::pow(z, 2.5), mfem::sin(z),
           mfem::cos(z), mfem::tan(z)
          };
}

} // namespace

namespace adlCheck
{

void real(int);
void abs(int);
void norm(int);
void conj(int);
void exp(int);
void log(int);
void pow(int, int);
void sin(int);
void cos(int);
void tan(int);

// The functions of z through argument-dependent lookup only
std::vector<mfem::complex_t> AdlValues(const mfem::complex_t &z)
{
   return {real(z), abs(z), norm(z), conj(z), exp(z), log(z), pow(z, 2.5),
           sin(z), cos(z), tan(z)
          };
}

} // namespace adlCheck

using namespace mfem;

namespace
{

void CheckClose(const complex_t &z, real_t re, real_t im, real_t tol)
{
   REQUIRE(std::fabs(z.real() - re) <= tol);
   REQUIRE(std::fabs(z.imag() - im) <= tol);
}

struct DivCase
{
   complex_t num, den;
   real_t re, im, tol;
};

// Quotients with components at the maximum, near the maximum,
// at the minimum normal and subnormal values of real_t
std::vector<DivCase> LimitCases()
{
   using limits = std::numeric_limits<real_t>;
   const real_t big = limits::max(), low = limits::min();
   const real_t sub = limits::denorm_min();
   const real_t eps = 4 * limits::epsilon();
   const real_t big4 = real_t{0.25} * big, big2 = real_t{0.5} * big;
   const real_t big3 = real_t{0.75} * big, invBig = real_t{0.5} / big;
   const real_t re1 = real_t{12} / 13, im1 = real_t{-5} / 13;
   const real_t re2 = real_t{0.6}, im2 = real_t{0.8};
   const real_t lowRe = real_t{0.2} / low, lowIm = real_t{-0.4} / low;
   return
   {
      {complex_t(big, big), complex_t(big, big), 1, 0, eps},
      {complex_t(big, -big), complex_t(big, big), 0, -1, eps},
      {complex_t(1, 0), complex_t(big, big), invBig, -invBig, 2 * sub},
      {complex_t(big, 0), complex_t(big, big), real_t{0.5}, real_t{-0.5}, eps},
      {complex_t(big3, big2), complex_t(big2, big3), re1, im1, eps},
      {complex_t(big4, big3), complex_t(big3, big4), re2, im2, eps},
      {complex_t(low, low), complex_t(low, low), 1, 0, eps},
      {complex_t(1, 0), complex_t(low, 2 * low), lowRe, lowIm, eps / low},
      {complex_t(16 * sub, 0), complex_t(8 * sub, 8 * sub), 1, -1, eps},
      {complex_t(4 * sub, 4 * sub), complex_t(1, 1), 4 * sub, 0, 0},
      {complex_t(3 * sub, sub), complex_t(sub, -3 * sub), 0, 1, eps},
      {complex_t(1, 0), complex_t(0.5, 0), 2, 0, 0}
   };
}

} // namespace

TEST_CASE("complex_t value initialization", "[Complex]")
{
   static_assert(std::is_trivial<complex_t>::value);
   static_assert(sizeof(complex_t) == 2 * sizeof(real_t));
   const complex_t z{};
   REQUIRE(z.real() == 0.0);
   REQUIRE(z.imag() == 0.0);
   REQUIRE(complex_t().real() == 0.0);
   REQUIRE(complex_t().imag() == 0.0);
   const complex_t arr[3] = {};
   for (const complex_t &a : arr)
   {
      REQUIRE(a.real() == 0.0);
      REQUIRE(a.imag() == 0.0);
   }
   Array<complex_t> data(3);
   data = complex_t{};
   for (const complex_t &a : data)
   {
      REQUIRE(a.real() == 0.0);
      REQUIRE(a.imag() == 0.0);
   }
   const complex_t r = 2.5;
   REQUIRE(r.real() == 2.5);
   REQUIRE(r.imag() == 0.0);
}

TEST_CASE("complex_t functions, qualified and unqualified", "[Complex]")
{
   const real_t eps = 8 * std::numeric_limits<real_t>::epsilon();
   const complex_t z(0.3, -0.7);
   const std::complex<real_t> s(0.3, -0.7);
   const std::vector<complex_t> unqualified =
   {
      real(z), abs(z), norm(z), conj(z), exp(z), log(z), pow(z, 2.5), sin(z),
      cos(z), tan(z)
   };
   const std::vector<std::complex<real_t>> reference =
   {
      std::real(s), std::abs(s), std::norm(s), std::conj(s), std::exp(s),
      std::log(s), std::pow(s, real_t{2.5}), std::sin(s), std::cos(s),
      std::tan(s)
   };
   const std::vector<complex_t> qualified = QualifiedValues(z);
   const std::vector<complex_t> adl = adlCheck::AdlValues(z);
   REQUIRE(qualified.size() == reference.size());
   for (std::size_t i = 0; i < reference.size(); ++i)
   {
      const real_t re = reference[i].real(), im = reference[i].imag();
      CheckClose(unqualified[i], re, im, eps);
      CheckClose(qualified[i], re, im, eps);
      CheckClose(adl[i], re, im, eps);
   }
}

TEST_CASE("complex_t division", "[Complex]")
{
   const real_t eps = 8 * std::numeric_limits<real_t>::epsilon();

   SECTION("functions")
   {
      CheckClose(exp(complex_t(0, M_PI)), -1.0, 0.0, eps);
      CheckClose(log(complex_t(-1, 0)), 0.0, M_PI, eps);
      CheckClose(pow(complex_t(0, 2), 2), -4.0, 0.0, 4 * eps);
      CheckClose(tan(complex_t(0.5, 0)), std::tan(0.5), 0.0, eps);
      REQUIRE(abs(complex_t(3, 4)) == 5.0);
      REQUIRE(norm(complex_t(3, 4)) == 25.0);
      REQUIRE(real(complex_t(3, 4)) == 3.0);
      CheckClose(conj(complex_t(3, 4)), 3.0, -4.0, 0.0);
   }

   SECTION("ordinary")
   {
      CheckClose(complex_t(1, 2) / complex_t(3, 4), 0.44, 0.08, eps);
      CheckClose(complex_t(1, 2) / complex_t(4, 3), 0.4, 0.2, eps);
      CheckClose(complex_t(-5, 7) / complex_t(0, 2), 3.5, 2.5, eps);
      CheckClose(complex_t(-5, 7) / complex_t(2, 0), -2.5, 3.5, eps);
      CheckClose(real_t{2} / complex_t(1, 1), 1.0, -1.0, eps);
      CheckClose(real_t{2} / complex_t(1, -1), 1.0, 1.0, eps);
      CheckClose(real_t{5} / complex_t(3, 4), 0.6, -0.8, eps);
      CheckClose(real_t{5} / complex_t(4, 3), 0.8, -0.6, eps);
      CheckClose(complex_t(3, 4) / real_t{2}, 1.5, 2.0, eps);
   }

#ifdef MFEM_USE_SINGLE
   const real_t big = 1e26f, tiny = 1e-26f;
#else
   const real_t big = 1e200, tiny = 1e-200;
#endif

   SECTION("tiny denominator")
   {
      const complex_t z = complex_t(1, 1) / complex_t(tiny, tiny);
      CheckClose(z, big, 0.0, big * eps);
      const complex_t w = real_t{1} / complex_t(tiny, tiny);
      CheckClose(w, 0.5 * big, -0.5 * big, big * eps);
      const complex_t v = complex_t(1, -1) / complex_t(tiny, -2 * tiny);
      CheckClose(v, 0.6 * big, 0.2 * big, big * eps);
   }

   SECTION("huge denominator")
   {
      const complex_t z = complex_t(1, 1) / complex_t(big, big);
      CheckClose(z, tiny, 0.0, tiny * eps);
      const complex_t w = real_t{1} / complex_t(big, big);
      CheckClose(w, 0.5 * tiny, -0.5 * tiny, tiny * eps);
      const complex_t v = real_t{1} / complex_t(2 * big, big);
      CheckClose(v, 0.4 * tiny, -0.2 * tiny, tiny * eps);
   }
}

TEST_CASE("complex_t division at the range limits", "[Complex]")
{
   for (const DivCase &c : LimitCases())
   {
      CheckClose(c.num / c.den, c.re, c.im, c.tol);
      if (c.num.imag() == 0)
      {
         CheckClose(c.num.real() / c.den, c.re, c.im, c.tol);
      }
   }
}

#ifdef MFEM_USE_MPI
TEST_CASE("complex_t range edges", "[Complex][Parallel]")
#else
TEST_CASE("complex_t range edges", "[Complex]")
#endif
{
   using limits = std::numeric_limits<real_t>;
   const real_t eps = 16 * limits::epsilon();
   const real_t mx = limits::max();
   const real_t inf = limits::infinity();

   SECTION("log of two max values")
   {
      const complex_t z = log(complex_t(mx, mx));
      const real_t re = std::log(mx) + real_t{0.5} * std::log(real_t{2});
      REQUIRE(std::isfinite(z.real()));
      CheckClose(z, re, M_PI / 4, re * eps);
   }

   SECTION("tan of a large imaginary part")
   {
#ifdef MFEM_USE_SINGLE
      const real_t y = 100;
#else
      const real_t y = 1000;
#endif
      const std::complex<real_t> s = std::tan(std::complex<real_t>(0, y));
      CheckClose(tan(complex_t(0, y)), s.real(), s.imag(), eps);
   }

   SECTION("exp near the top of the range")
   {
      const real_t x = std::log(mx) + real_t{0.25};
      const complex_t z = exp(complex_t(x, M_PI / 4));
      const real_t half_log2 = real_t{0.5} * std::log(real_t{2});
      const real_t mag = std::exp(x - half_log2);
      REQUIRE(std::isfinite(z.real()));
      REQUIRE(std::isfinite(z.imag()));
      CheckClose(z, mag, mag, mag * eps);
   }

#ifndef MFEM_USE_SINGLE
   SECTION("division with an underflowed ratio")
   {
      const complex_t z =
         complex_t(0, 1e-150) / complex_t(1e-100, 1e-300);
      REQUIRE(std::fabs(z.real() - 1e-250) <= 1e-250 * eps);
      REQUIRE(std::fabs(z.imag() - 1e-50) <= 1e-50 * eps);
   }
#endif

   SECTION("exp of a positive infinity")
   {
      const complex_t z = exp(complex_t(inf, 0));
      const std::complex<real_t> s =
         std::exp(std::complex<real_t>(inf, 0));
      REQUIRE(z.real() == s.real());
      REQUIRE(z.imag() == s.imag());
      REQUIRE(std::isinf(z.real()));
      REQUIRE(z.real() > 0);
      REQUIRE(z.imag() == 0);
   }

   SECTION("division by infinity")
   {
      const complex_t z = complex_t(1, -2) / complex_t(inf, inf);
      const std::complex<real_t> s =
         std::complex<real_t>(1, -2) / std::complex<real_t>(inf, inf);
      REQUIRE(z.real() == s.real());
      REQUIRE(z.imag() == s.imag());
      REQUIRE(z.real() == 0);
      REQUIRE(z.imag() == 0);
   }
}

TEST_CASE("complex_t division in kernels", "[Complex][GPU]")
{
#ifdef MFEM_USE_SINGLE
   const real_t big = 1e26f, tiny = 1e-26f;
#else
   const real_t big = 1e200, tiny = 1e-200;
#endif
   const real_t eps = 8 * std::numeric_limits<real_t>::epsilon();
   const int n = 4;
   Array<complex_t> num(n), den(n), quo(n), inv(n);
   num[0] = complex_t(1, 1), den[0] = complex_t(tiny, tiny);
   num[1] = complex_t(1, 1), den[1] = complex_t(big, big);
   num[2] = complex_t(1, 2), den[2] = complex_t(3, 4);
   num[3] = complex_t(-5, 7), den[3] = complex_t(0, 2);
   const complex_t *a = num.Read(), *b = den.Read();
   complex_t *q = quo.Write(), *r = inv.Write();
   forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      q[i] = a[i] / b[i];
      r[i] = real_t{1} / b[i];
   });
   const complex_t *quo_h = quo.HostRead();
   const complex_t *inv_h = inv.HostRead();
   CheckClose(quo_h[0], big, 0.0, big * eps);
   CheckClose(quo_h[1], tiny, 0.0, tiny * eps);
   CheckClose(quo_h[2], 0.44, 0.08, eps);
   CheckClose(quo_h[3], 3.5, 2.5, eps);
   CheckClose(inv_h[0], 0.5 * big, -0.5 * big, big * eps);
   CheckClose(inv_h[1], 0.5 * tiny, -0.5 * tiny, tiny * eps);
   CheckClose(inv_h[2], 0.12, -0.16, eps);
   CheckClose(inv_h[3], 0.0, -0.5, eps);

   const std::vector<DivCase> cases = LimitCases();
   const int m = static_cast<int>(cases.size());
   Array<complex_t> limNum(m), limDen(m), limQuo(m), limInv(m);
   for (int i = 0; i < m; ++i)
   {
      limNum[i] = cases[i].num;
      limDen[i] = cases[i].den;
   }
   const complex_t *ln = limNum.Read(), *ld = limDen.Read();
   complex_t *lq = limQuo.Write(), *li = limInv.Write();
   forall(m, [=] MFEM_HOST_DEVICE (int i)
   {
      lq[i] = ln[i] / ld[i];
      li[i] = ln[i].real() / ld[i];
   });

   const complex_t z(0.3, -0.7);
   const std::vector<complex_t> host = QualifiedValues(z);
   const int nf = static_cast<int>(host.size());
   Array<complex_t> qualified(nf), unqualified(nf);
   complex_t *fq = qualified.Write(), *fu = unqualified.Write();
   forall(1, [=] MFEM_HOST_DEVICE (int)
   {
      fq[0] = mfem::real(z), fq[1] = mfem::abs(z), fq[2] = mfem::norm(z);
      fq[3] = mfem::conj(z), fq[4] = mfem::exp(z), fq[5] = mfem::log(z);
      fq[6] = mfem::pow(z, 2.5), fq[7] = mfem::sin(z), fq[8] = mfem::cos(z);
      fq[9] = mfem::tan(z);
      fu[0] = real(z), fu[1] = abs(z), fu[2] = norm(z), fu[3] = conj(z);
      fu[4] = exp(z), fu[5] = log(z), fu[6] = pow(z, 2.5), fu[7] = sin(z);
      fu[8] = cos(z), fu[9] = tan(z);
   });
   const complex_t *fq_h = qualified.HostRead();
   const complex_t *fu_h = unqualified.HostRead();
   for (int i = 0; i < nf; ++i)
   {
      CheckClose(fq_h[i], host[i].real(), host[i].imag(), eps);
      CheckClose(fu_h[i], host[i].real(), host[i].imag(), eps);
   }
   const complex_t *lq_h = limQuo.HostRead();
   const complex_t *li_h = limInv.HostRead();
   for (int i = 0; i < m; ++i)
   {
      const DivCase &c = cases[i];
      CheckClose(lq_h[i], c.re, c.im, c.tol);
      if (c.num.imag() == 0) { CheckClose(li_h[i], c.re, c.im, c.tol); }
   }
}
