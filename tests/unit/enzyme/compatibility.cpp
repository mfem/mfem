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

#include "mfem.hpp"
#include "unit_tests.hpp"

#ifdef MFEM_USE_ENZYME

template<typename VectorT>
void square(const VectorT& v, double& y)
{
   for (int i = 0; i < 4; i++)
   {
      y += v[i]*v[i];
   }
}

template<typename VectorT>
void dsquare(const VectorT& v, double& y, VectorT& dydv)
{
   double seed = 1.0;
   __enzyme_autodiff<void>(square<VectorT>, &v, &dydv, &y, &seed);
}

template<typename VectorT>
void run_test()
{
   VectorT v(4);
   v[0] = 2.0;
   v[1] = 3.0;
   v[2] = 1.0;
   v[3] = 7.0;

   double yy = 0;
   VectorT dydv(4);
   dydv[0] = 0;
   dydv[1] = 0;
   dydv[2] = 0;
   dydv[3] = 0;
   dsquare(v, yy, dydv);

   REQUIRE(dydv[0] == MFEM_Approx(4.0));
   REQUIRE(dydv[1] == MFEM_Approx(6.0));
   REQUIRE(dydv[2] == MFEM_Approx(2.0));
   REQUIRE(dydv[3] == MFEM_Approx(14.0));
}

TEST_CASE("AD Vector implementation", "[Enzyme]")
{
   run_test<mfem::Vector>();
   run_test<std::vector<double>>();
}

namespace enzyme_test
{

template <int N>
void f(const double *x, double *y, double *a)
{
   mfem::forall<mfem::UseEnzyme>(N, [=] MFEM_HOST_DEVICE(int q)
   {
      y[q] = a[q] * x[q] * x[q];
   });
}

} // namespace enzyme_test

TEST_CASE("AD Global qfunction with GPU", "[Enzyme][GPU]")
{
   constexpr int N = 10;
   mfem::Vector x(N), xd(N), y(N), yd(N), a(N), ad(N);

   for (int i = 0; i < N; i++)
   {
      x(i) = i;
      xd(i) = 1.0;
      y(i) = 0.0;
      yd(i) = 0.0;
      a(i) = 2.0;
      ad(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto a_d = a.Read();

   __enzyme_fwddiff<void>((void *)enzyme_test::f<N>, enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d,
                          yd_d, enzyme_const, a_d, enzyme_runtime_activity);

   yd.HostRead();
   bool ok = true;
   for (int q = 0; q < N; q++)
   {
      mfem::real_t exact = 2.0 * a[q] * x[q];
      if (yd[q] != exact)
      {
         ok = false;
         break;
      }
   }
   REQUIRE(ok);
}

#endif
