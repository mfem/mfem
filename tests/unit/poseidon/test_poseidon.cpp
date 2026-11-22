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
#include "unit_tests.hpp"

using namespace mfem;

// Without a profile of a site, the site computes what the code as written does.

namespace poseidon_test
{

MFEM_HOST_DEVICE real_t Affine(real_t a, real_t x, real_t b)
{
   return a*x + b;
}

} // namespace poseidon_test

TEST_CASE("Poseidon forall site", "[Poseidon][GPU]")
{
   constexpr int n = 100;
   Vector x(n), y(n), z(n);
   x.Randomize(1);
   const auto X = x.Read();
   auto Y = y.Write();
   auto Z = z.Write();
   forall(n, [=] MFEM_HOST_DEVICE (int i) MFEM_POSEIDON_OPTIMIZE
   {
      Y[i] = 2.0*X[i] + 1.0;
   });
   forall(n, [=] MFEM_HOST_DEVICE (int i) MFEM_POSEIDON_OPTIMIZE_TAU(1e-6)
   {
      Z[i] = 2.0*X[i] + 1.0;
   });
   const real_t *xh = x.HostRead(), *yh = y.HostRead(), *zh = z.HostRead();
   for (int i = 0; i < n; i++)
   {
      REQUIRE(yh[i] == 2.0*xh[i] + 1.0);
      REQUIRE(zh[i] == 2.0*xh[i] + 1.0);
   }
}

TEST_CASE("Poseidon call site", "[Poseidon][GPU]")
{
   constexpr int n = 100;
   Vector x(n), y(n), z(n);
   x.Randomize(1);
   const auto X = x.Read();
   auto Y = y.Write();
   auto Z = z.Write();
   forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      Y[i] = MFEM_POSEIDON_CALL(poseidon_test::Affine, 2.0, X[i], 1.0);
      Z[i] = MFEM_POSEIDON_CALL_TAU(1e-6, poseidon_test::Affine, 2.0, X[i], 1.0);
   });
   const real_t *xh = x.HostRead(), *yh = y.HostRead(), *zh = z.HostRead();
   for (int i = 0; i < n; i++)
   {
      REQUIRE(yh[i] == 2.0*xh[i] + 1.0);
      REQUIRE(zh[i] == 2.0*xh[i] + 1.0);
   }
}
