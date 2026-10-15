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
#include <numeric>

using namespace mfem;

template <typename X, typename Y>
void CheckComplexVectorLinearCombination(const X &x, const Y &y,
                                         ComplexVector &z)
{
   const complex_t a(2.0, -3.0);
   const complex_t b(-1.0, 4.0);
   const auto check = [&](const complex_t x_scale, const complex_t y_scale)
   {
      for (int i = 0; i < z.Size(); i++)
      {
         const complex_t expected = x_scale * complex_t(x(i))
                                    + y_scale * complex_t(y(i));
         REQUIRE(z(i).real() == MFEM_Approx(expected.real()));
         REQUIRE(z(i).imag() == MFEM_Approx(expected.imag()));
      }
   };

   add(x, y, z);
   check(complex_t(1.0), complex_t(1.0));

   add(x, b, y, z);
   check(complex_t(1.0), b);

   add(a, x, y, z);
   check(a, a);

   add(a, x, b, y, z);
   check(a, b);

   subtract(x, y, z);
   check(complex_t(1.0), complex_t(-1.0));

   subtract(a, x, y, z);
   check(a, -a);
}

TEST_CASE("Complex Vector init-list and C-style array constructors",
          "[ComplexVector]")
{
   complex_t ContigData[6] = {complex_t(6.0,1.0),
                              complex_t(5.0,2.0),
                              complex_t(4.0,3.0),
                              complex_t(3.0,4.0),
                              complex_t(2.0,5.0),
                              complex_t(1.0,6.0)
                             };
   // Point and size constructor
   ComplexVector a(ContigData, 6);
   // Braced-list constructor
   ComplexVector b({complex_t(6.0,1.0), complex_t(5.0,2.0),
                    complex_t(4.0,3.0), complex_t(3.0,4.0),
                    complex_t(2.0,5.0), complex_t(1.0,6.0)});
   // Statically sized C-style array constructor
   ComplexVector c(ContigData);

   for (int i = 0; i < a.Size(); i++)
   {
      REQUIRE(a[i] == b[i]);
      REQUIRE(a[i] == c[i]);
   }
}

TEST_CASE("Complex Vector Move Constructor", "[ComplexVector]")
{
   constexpr int N = 6;
   complex_t ContigData[6] = {complex_t(6.0,1.0),
                              complex_t(5.0,2.0),
                              complex_t(4.0,3.0),
                              complex_t(3.0,4.0),
                              complex_t(2.0,5.0),
                              complex_t(1.0,6.0)
                             };
   ComplexVector a(ContigData, N);
   ComplexVector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = complex_t(N - i, i + 1);
   }

   complex_t *a_data = a.GetData();
   complex_t *b_data = b.GetData();

   ComplexVector move_non_owning(std::move(a));
   ComplexVector move_owning(std::move(b));

   REQUIRE(a.Size() == 0);
   REQUIRE(a.GetData() == nullptr);
   REQUIRE(b.Size() == 0);
   REQUIRE(b.GetData() == nullptr);

   // Should both be no-ops
   a.Destroy();
   b.Destroy();

   REQUIRE(move_non_owning.OwnsData() == false);
   REQUIRE(move_owning.OwnsData() == true);

   REQUIRE(move_non_owning.Size() == N);
   REQUIRE(move_owning.Size() == N);

   // Make sure that the pointers were reused
   REQUIRE(move_non_owning.GetData() == a_data);
   REQUIRE(move_owning.GetData() == b_data);

   for (int i = 0; i < N; i++)
   {
      REQUIRE(move_non_owning(i) == complex_t(N - i, i + 1));
      REQUIRE(move_owning(i) == complex_t(N - i, i + 1));
   }
}

TEST_CASE("Complex Vector Move Assignment", "[ComplexVector]")
{
   constexpr int N = 6;
   complex_t ContigData[6] = {complex_t(6.0,1.0),
                              complex_t(5.0,2.0),
                              complex_t(4.0,3.0),
                              complex_t(3.0,4.0),
                              complex_t(2.0,5.0),
                              complex_t(1.0,6.0)
                             };
   ComplexVector a(ContigData, N);
   ComplexVector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = complex_t(N - i, i + 1);
   }

   complex_t *a_data = a.GetData();
   complex_t *b_data = b.GetData();

   ComplexVector move_non_owning;
   move_non_owning = std::move(a);
   ComplexVector move_owning;
   move_owning = std::move(b);

   REQUIRE(a.Size() == 0);
   REQUIRE(a.GetData() == nullptr);
   REQUIRE(b.Size() == 0);
   REQUIRE(b.GetData() == nullptr);

   // Should both be no-ops
   a.Destroy();
   b.Destroy();

   REQUIRE(move_non_owning.OwnsData() == false);
   REQUIRE(move_owning.OwnsData() == true);

   REQUIRE(move_non_owning.Size() == N);
   REQUIRE(move_owning.Size() == N);

   // Make sure that the pointers were reused
   REQUIRE(move_non_owning.GetData() == a_data);
   REQUIRE(move_owning.GetData() == b_data);

   for (int i = 0; i < N; i++)
   {
      REQUIRE(move_non_owning(i) == complex_t(N - i, i + 1));
      REQUIRE(move_owning(i) == complex_t(N - i, i + 1));
   }
}

TEST_CASE("Complex Vector arithmetic and component extraction",
          "[ComplexVector]")
{
   Vector real_part({1.0, -2.0, 3.0});
   Vector imag_part({4.0, 5.0, -6.0});
   ComplexVector v(3);
   v.UseDevice(true);
   v.Set(real_part, imag_part);

   const Vector &real_values = v.real();
   const Vector &imag_values = v.imag();
   for (int i = 0; i < v.Size(); i++)
   {
      REQUIRE(real_values(i) == MFEM_Approx(real_part(i)));
      REQUIRE(imag_values(i) == MFEM_Approx(imag_part(i)));
   }

   ComplexVector w({complex_t(2.0, -1.0), complex_t(-3.0, 2.0),
                    complex_t(4.0, 5.0)});
   w.UseDevice(true);
   const complex_t dot = v * w;
   REQUIRE(dot.real() == MFEM_Approx(44.0));
   REQUIRE(dot.imag() == MFEM_Approx(-21.0));

   Vector weights({2.0, -3.0, 4.0});
   const complex_t mixed_dot = v * weights;
   REQUIRE(mixed_dot.real() == MFEM_Approx(20.0));
   REQUIRE(mixed_dot.imag() == MFEM_Approx(-31.0));

   ComplexVector expected(v);
   expected += w;
   expected -= w;
   for (int i = 0; i < v.Size(); i++)
   {
      REQUIRE(expected(i).real() == MFEM_Approx(v(i).real()));
      REQUIRE(expected(i).imag() == MFEM_Approx(v(i).imag()));
   }

   expected *= complex_t(2.0, -1.0);
   expected /= complex_t(2.0, -1.0);
   expected *= weights;
   expected /= weights;
   for (int i = 0; i < v.Size(); i++)
   {
      REQUIRE(expected(i).real() == MFEM_Approx(v(i).real()));
      REQUIRE(expected(i).imag() == MFEM_Approx(v(i).imag()));
   }

   expected = real_part;
   for (int i = 0; i < expected.Size(); i++)
   {
      REQUIRE(expected(i).real() == MFEM_Approx(real_part(i)));
      REQUIRE(expected(i).imag() == MFEM_Approx(0.0));
   }
}

TEST_CASE("Complex Vector add and subtract", "[ComplexVector]")
{
   Vector xr({1.0, -2.0, 3.0});
   Vector yr({-4.0, 5.0, -6.0});
   ComplexVector xc({complex_t(1.0, 2.0), complex_t(-3.0, 4.0),
                     complex_t(5.0, -6.0)});
   ComplexVector yc({complex_t(-2.0, 1.0), complex_t(3.0, -5.0),
                     complex_t(-4.0, 6.0)});
   ComplexVector z(xr.Size());

   xr.UseDevice(true);
   yr.UseDevice(true);
   xc.UseDevice(true);
   yc.UseDevice(true);
   z.UseDevice(true);

   CheckComplexVectorLinearCombination(xc, yc, z);
   CheckComplexVectorLinearCombination(xc, yr, z);
   CheckComplexVectorLinearCombination(xr, yc, z);
   CheckComplexVectorLinearCombination(xr, yr, z);

   const complex_t a(2.0, -3.0);
   const complex_t b(-1.0, 4.0);
   const ComplexVector original_xc(xc);
   add(a, xc, b, yc, xc);
   for (int i = 0; i < xc.Size(); i++)
   {
      const complex_t expected = a * original_xc(i) + b * yc(i);
      REQUIRE(xc(i).real() == MFEM_Approx(expected.real()));
      REQUIRE(xc(i).imag() == MFEM_Approx(expected.imag()));
   }
}
