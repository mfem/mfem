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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#include "../../../linalg/dual.hpp"
#include "../../../linalg/tensor.hpp"
#include <cmath>
#include <limits>

using namespace mfem;
using namespace mfem::future;

using dual_t = dual<real_t, real_t>;

static const real_t tol = 1e3 * std::numeric_limits<real_t>::epsilon();

static void CheckDual(const dual_t &d, const real_t v, const real_t g)
{
   CHECK(d.value == MFEM_Approx(v, tol, tol));
   CHECK(d.gradient == MFEM_Approx(g, tol, tol));
}

TEST_CASE("Dual arithmetic", "[dFEM][Dual]")
{
   const dual_t a{1.5, 0.5}, b{-0.75, 2.0};
   const real_t r = 2.5;

   SECTION("make_dual, get_value, get_gradient")
   {
      CheckDual(make_dual(r), r, 1.0);
      CHECK(get_value(r) == r);
      CHECK(get_value(a) == a.value);
      CHECK(get_gradient(a) == a.gradient);
      const dual<real_t, tensor<real_t, 2>> t{r, {{1.0, -2.0}}};
      CHECK(get_value(t) == r);
      CHECK(get_gradient(t)[1] == -2.0);
   }

   SECTION("assignment from real")
   {
      dual_t c{1.0, 3.0};
      c = r;
      CheckDual(c, r, 0.0);
   }

   SECTION("addition")
   {
      CheckDual(a + r, 1.5 + r, 0.5);
      CheckDual(r + a, r + 1.5, 0.5);
      CheckDual(a + 1, 2.5, 0.5);
      CheckDual(a + b, 0.75, 2.5);
   }

   SECTION("subtraction and negation")
   {
      CheckDual(-a, -1.5, -0.5);
      CheckDual(a - r, 1.5 - r, 0.5);
      CheckDual(r - a, r - 1.5, -0.5);
      CheckDual(a - b, 2.25, -1.5);
   }

   SECTION("multiplication")
   {
      CheckDual(a * r, 1.5 * r, 0.5 * r);
      CheckDual(r * a, r * 1.5, r * 0.5);
      CheckDual(a * b, 1.5 * -0.75, -0.75 * 0.5 + 1.5 * 2.0);
   }

   SECTION("division")
   {
      CheckDual(a / r, 1.5 / r, 0.5 / r);
      CheckDual(r / a, r / 1.5, -r / (1.5 * 1.5) * 0.5);
      CheckDual(a / b, 1.5 / -0.75,
                0.5 / -0.75 - 1.5 * 2.0 / (0.75 * 0.75));
   }

   SECTION("compound assignment")
   {
      dual_t c = a;
      c += b;
      CheckDual(c, 0.75, 2.5);
      c -= b;
      CheckDual(c, 1.5, 0.5);
      c += r;
      CheckDual(c, 1.5 + r, 0.5);
      c -= r;
      CheckDual(c, 1.5, 0.5);
   }
}

TEST_CASE("Dual comparisons", "[dFEM][Dual]")
{
   const dual_t a{1.0, 5.0}, b{2.0, -5.0}, c{1.0, -3.0};

   SECTION("dual and dual")
   {
      CHECK(a < b);
      CHECK_FALSE(b < a);
      CHECK(a <= b);
      CHECK(a <= c);
      CHECK(a == c);
      CHECK_FALSE(a == b);
      CHECK(b >= a);
      CHECK(a >= c);
      CHECK(b > a);
      CHECK_FALSE(a > c);
   }

   SECTION("dual and real")
   {
      CHECK(a < 2.0);
      CHECK(a <= 1.0);
      CHECK(a == 1.0);
      CHECK(a >= 1.0);
      CHECK(a > 0.0);
      CHECK_FALSE(a > 1.0);
   }

   SECTION("real and dual")
   {
      CHECK(0.0 < a);
      CHECK(1.0 <= a);
      CHECK(1.0 == a);
      CHECK(2.0 >= a);
      CHECK(2.0 > a);
      CHECK_FALSE(1.0 > a);
   }
}

TEST_CASE("Dual math functions", "[dFEM][Dual]")
{
   const real_t x = 0.3, dx = 1.5;
   const dual_t a{x, dx};

   SECTION("abs and fabs")
   {
      CheckDual(abs(dual_t{-2.0, 3.0}), 2.0, -3.0);
      CheckDual(abs(dual_t{2.0, 3.0}), 2.0, 3.0);
      CheckDual(fabs(dual_t{-2.0, 3.0}), 2.0, -3.0);
      CheckDual(fabs(dual_t{2.0, 3.0}), 2.0, 3.0);
   }

   SECTION("sqrt")
   {
      CheckDual(sqrt(a), std::sqrt(x), dx / (2.0 * std::sqrt(x)));
   }

   SECTION("fmax and fmin")
   {
      const dual_t b{0.7, -2.0};
      CheckDual(fmax(a, b), 0.7, -2.0);
      CheckDual(fmax(b, a), 0.7, -2.0);
      CheckDual(fmin(a, b), x, dx);
      CheckDual(fmin(b, a), x, dx);
      CheckDual(fmax(0.5, a), 0.5, 0.0);
      CheckDual(fmax(0.1, a), x, dx);
   }

   SECTION("hypot")
   {
      const dual_t b{0.4, -2.0};
      const real_t h = std::hypot(x, 0.4);
      CheckDual(hypot(a, b), h, (x * dx + 0.4 * -2.0) / h);
   }

   SECTION("trigonometric")
   {
      CheckDual(cos(a), std::cos(x), -dx * std::sin(x));
      CheckDual(sin(a), std::sin(x), dx * std::cos(x));
      CheckDual(tan(a), std::tan(x), dx / (std::cos(x) * std::cos(x)));
      CheckDual(acos(a), std::acos(x), -dx / std::sqrt(1.0 - x * x));
      CheckDual(asin(a), std::asin(x), dx / std::sqrt(1.0 - x * x));
      CheckDual(atan(a), std::atan(x), dx / (1.0 + x * x));
   }

   SECTION("hyperbolic")
   {
      const real_t t = std::tanh(x);
      CheckDual(tanh(a), t, dx * (1.0 - t * t));
      CheckDual(sinh(a), std::sinh(x), dx * std::cosh(x));
   }

   SECTION("exp and log")
   {
      CheckDual(exp(a), std::exp(x), dx * std::exp(x));
      CheckDual(log(a), std::log(x), dx / x);
   }

   SECTION("pow")
   {
      const real_t y = 1.7, dy = -0.8;
      const dual_t b{y, dy};
      const real_t p = std::pow(x, y);
      CheckDual(pow(a, b), p, p * (dx * y / x + dy * std::log(x)));
      CheckDual(pow(x, b), p, p * dy * std::log(x));
      CheckDual(pow(a, y), p, y * std::pow(x, y - 1.0) * dx);
      CHECK(future::pow<real_t>(x, y) == MFEM_Approx(p, tol, tol));
   }
}

TEST_CASE("Dual derivatives at zero", "[dFEM][Dual]")
{
   const dual_t zero{0.0, 1.0};

   SECTION("sqrt")
   {
      CheckDual(sqrt(zero), 0.0, 0.0);
   }

   SECTION("log")
   {
      const dual_t l = log(zero);
      CHECK(std::isinf(l.value));
      CHECK(l.value < 0.0);
      CHECK(l.gradient == 0.0);
   }

   SECTION("hypot")
   {
      CheckDual(hypot(zero, zero), 0.0, 0.0);
   }

   SECTION("pow with zero base")
   {
      // b x^(b-1) is singular for b < 1 and taken as zero
      const real_t exponents[4] = {0.0, 0.5, 1.0, 2.0};
      const real_t values[4] = {1.0, 0.0, 0.0, 0.0};
      const real_t derivatives[4] = {0.0, 0.0, 1.0, 0.0};
      for (int k = 0; k < 4; k++)
      {
         const real_t e = exponents[k];
         CAPTURE(e);
         CheckDual(pow(zero, e), values[k], derivatives[k]);
         CheckDual(pow(zero, dual_t{e, 0.0}), values[k], derivatives[k]);
         // x^b log(x) vanishes at zero
         CheckDual(pow(zero, dual_t{e, 1.0}), values[k], derivatives[k]);
         CheckDual(pow(0.0, dual_t{e, 1.0}), values[k], 0.0);
      }
   }
}

TEST_CASE("Dual singular values", "[dFEM][Dual]")
{
   const tensor<real_t, 2, 2> A = {{{2.0, 1.0}, {0.5, 3.0}}};
   const tensor<real_t, 2, 2> dA = {{{0.3, -0.2}, {0.7, 0.1}}};
   tensor<dual_t, 2, 2> Ad;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++) { Ad(i, j) = dual_t{A(i, j), dA(i, j)}; }
   }

   const real_t h = 1e-6;
   for (int k = 0; k < 2; k++)
   {
      CAPTURE(k);
      const dual_t sv = calcsv(Ad, k);
      const real_t fd = (calcsv(A + h * dA, k) - calcsv(A - h * dA, k)) /
                        (2.0 * h);
      CHECK(sv.value == MFEM_Approx(calcsv(A, k), tol, tol));
      CHECK(sv.gradient == MFEM_Approx(fd, 1e-8, 1e-8));
   }
}
