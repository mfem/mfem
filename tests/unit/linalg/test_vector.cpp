// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Vector init-list and C-style array constructors", "[Vector]")
{
   real_t ContigData[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
   // Point and size constructor
   Vector a(ContigData, 6);
   // Braced-list constructor
   Vector b({6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
   // Statically sized C-style array constructor
   Vector c(ContigData);
   // Convertible type constructor
   Vector d({6, 5, 4, 3, 2, 1});

   for (int i = 0; i < a.Size(); i++)
   {
      REQUIRE(a[i] == b[i]);
      REQUIRE(a[i] == c[i]);
      REQUIRE(a[i] == d[i]);
   }
}

TEST_CASE("Vector Move Constructor", "[Vector]")
{
   constexpr int N = 6;
   real_t ContigData[N] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
   Vector a(ContigData, N);
   Vector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = N - i;
   }

   real_t* a_data = a.GetData();
   real_t* b_data = b.GetData();

   Vector move_non_owning(std::move(a));
   Vector move_owning(std::move(b));

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
      REQUIRE(move_non_owning(i) == N - i);
      REQUIRE(move_owning(i) == N - i);
   }
}

TEST_CASE("Vector Move Assignment", "[Vector]")
{
   constexpr int N = 6;
   real_t ContigData[N] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
   Vector a(ContigData, N);
   Vector b(N);
   for (int i = 0; i < N; i++)
   {
      b(i) = N - i;
   }

   real_t* a_data = a.GetData();
   real_t* b_data = b.GetData();

   Vector move_non_owning;
   move_non_owning = std::move(a);
   Vector move_owning;
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
      REQUIRE(move_non_owning(i) == N - i);
      REQUIRE(move_owning(i) == N - i);
   }
}



TEST_CASE("Vector Tests", "[Vector]")
{
   real_t tol = 1e-12;

   Vector a(3),b(3);
   a(0) = 1.0;
   a(1) = 3.0;
   a(2) = 5.0;

   b(0) = 2.0;
   b(1) = 1.0;
   b(2) = 4.0;

   real_t bp[3];
   bp[0] = b(0);
   bp[1] = b(1);
   bp[2] = b(2);

   Vector apb(3), amb(3);
   apb(0) = 3.0;
   apb(1) = 4.0;
   apb(2) = 9.0;

   amb(0) = -1.0;
   amb(1) = 2.0;
   amb(2) = 1.0;

   Vector tmp(3);
   Vector diff(3);

   SECTION("Dot product")
   {
      REQUIRE(a*b  - 25.0 < tol);
      REQUIRE(a*bp - 25.0 < tol);
   }

   SECTION("Multiply and divide")
   {
      a *= 3.0;
      b /= -4.0;

      REQUIRE(a*b  + 3.0*25.0/4.0 < tol);
      REQUIRE(a*bp - 3.0*25.0     < tol);
   }

   SECTION("Minus scalar")
   {
      a -= 3.0;
      REQUIRE(a.Norml2() - sqrt(8.0) < tol);
   }

   SECTION("Minus vector")
   {
      a -= b;
      subtract(a, amb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Subtract vector")
   {
      subtract(0.5, a, b, tmp);
      tmp*= 2.0;
      subtract(tmp, amb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Plus scalar")
   {
      a += 2.0;
      REQUIRE(a.Norml2() - sqrt(83.0) < tol);
   }

   SECTION("Plus vector")
   {
      a += b;
      add(1.0, a, -1.0, apb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Add vector 1")
   {
      a.Add(1.0, b);
      add(1.0, a, -1.0, apb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Add vector 2")
   {
      add(a, b, tmp);
      apb.Neg();
      add(tmp, apb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Add vector 3")
   {
      add(a, 1.0, b, tmp);
      apb.Neg();
      add(tmp, apb, diff);
      REQUIRE(diff.Norml2() < tol);
   }

   SECTION("Add vector 4")
   {
      add(1.0, a, b, tmp);
      subtract(tmp, apb, diff);
      REQUIRE(diff.Norml2() < tol);
   }
}

TEST_CASE("Vector Sum", "[Vector],[CUDA]")
{
   Vector x(1024);
   x.Randomize(1);
   x.UseDevice(true);

   x.HostRead();
   const real_t sum_1 = std::accumulate(x.begin(), x.end(), 0.0);
   const real_t sum_2 = x.Sum();

   REQUIRE(sum_1 == MFEM_Approx(sum_2));
}
