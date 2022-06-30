// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_JIT

#include "catch.hpp"

#include "general/jit/jit.hpp"// for MFEM_JIT

using namespace mfem;

namespace mjit_tests
{

MFEM_JIT template <int T_A = 0>
static void A(int *r, int a) { *r = a; }

MFEM_JIT template <int T_B = 0>
static void B(int *r, int b = 0) { *r = b; }

MFEM_JIT template <int T_C=0>
static void C(int *r, int a, int c=0) { *r = a+c; }

MFEM_JIT void D(int *r, MFEM_JIT int d, const int c) { *r = d + c; }

MFEM_JIT void E(int *r, MFEM_JIT int a, MFEM_JIT int b, int c) { *r = a + b + c; }

MFEM_JIT void F(int *r, int a, MFEM_JIT int b) { *r = a + b; }

MFEM_JIT static
void bbps( const size_t q, double *result,
           const MFEM_JIT int D,
           MFEM_JIT const size_t N)
{
   const size_t b = 16, p = 8, M = N + D;
   double s = 0.0,  h = 1.0 / b;
   for (size_t i = 0, k = q; i <= N; i++, k += p)
   {
      double a = b, m = 1.0;
      for (size_t r = N - i; r > 0; r >>= 1)
      {
         auto dmod = [](double x, double y) { return x-((size_t)(x/y))*y;};
         m = (r&1) ? dmod(m*a,k) : m;
         a = dmod(a*a,k);
      }
      s += m / k;
      s -= (size_t) s;
   }
   for (size_t i = N + 1; i < M; i++, h /= b) { s += h / (p*i+q); }
   *result = s;
}

static size_t pi(size_t n, const size_t D = 100)
{
   auto p = [&](int k) { double r; bbps(k, &r, D, n-1); return r;};
   return pow(16,8)*fmod(4.0*p(1) - 2.0*p(4) - p(5) - p(6), 1.0);
}

TEST_CASE("Functions", "[JIT]")
{
   int r = 0;
   A(&r,1); REQUIRE(r == 1);
   B(&r,2); REQUIRE(r == 2);
   C(&r,4,3); REQUIRE(r == (4+3));
   D(&r,4,6); REQUIRE(r == (4+6));
   D(&r,5,6); REQUIRE(r == (5+6));
   D(&r,6,6); REQUIRE(r == (6+6));
   E(&r,1,2,3); REQUIRE(r == (1+2+3));
   F(&r,1,2); REQUIRE(r == 1+2);
   REQUIRE(pi(10) == 0x5A308D31ul);
}

} // mjit_tests

#endif // MFEM_USE_JIT
