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

#include <cmath> // pow
#include <cstddef> // size_t

#include "catch.hpp"

#include "general/jit/jit.hpp"// for MFEM_JIT

using namespace mfem;

namespace mjit_tests
{

// Bailey–Borwein–Plouffe formula to compute the nth base-16 digit of π
MFEM_JIT template<int T_DEPTH = 0, int T_N = 0>
void bbps( const size_t q, double *result, int depth = 0, int n = 0)
{
   const size_t D = T_DEPTH ? T_DEPTH : (size_t) depth;
   const size_t N = T_N ? T_N : (size_t) n;

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

size_t pi(size_t n, const size_t D = 100)
{
   auto p = [&](int k) { double r; bbps(k, &r, D, n-1); return r;};
   return pow(16,8)*fmod(4.0*p(1) - 2.0*p(4) - p(5) - p(6), 1.0);
}

TEST_CASE("Bbps", "[JIT]")
{
   SECTION("bbps")
   {
      double a = 0.0;
      bbps<64,17>(1,&a);
      const size_t ax = (size_t)(pow(16,8)*a);

      double b = 0.0;
      bbps(1,&b,64,17);
      const size_t bx = (size_t)(pow(16,8)*b);

      //printf("\033[33m[0x%lx:0x%lx]\033[m",ax,bx);
      REQUIRE(ax == bx);

      REQUIRE(pi(10) == 0x5A308D31ul);
   }
}

} // mjit_tests

#endif // MFEM_USE_JIT
