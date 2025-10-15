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

using namespace mfem;

TEST_CASE("Array init-list and C-style array constructors", "[Array]")
{
   int ContigData[6] = {6, 5, 4, 3, 2, 1};
   // Pointer and size constructor
   Array<int> a(ContigData, 6);
   // Braced-list constructor
   Array<int> b({6, 5, 4, 3, 2, 1});
   Array<int> c{6, 5, 4, 3, 2, 1};
   // Statically sized C-style array constructor
   Array<int> d(ContigData);
   // Convertible type constructors
   Array<int> e({6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
   Array<int> f{6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
   for (int i = 0; i < a.Size(); i++)
   {
      REQUIRE(a[i] == b[i]);
      REQUIRE(a[i] == c[i]);
      REQUIRE(a[i] == d[i]);
      REQUIRE(a[i] == e[i]);
      REQUIRE(a[i] == f[i]);
   }
}

TEST_CASE("Array entry sorting", "[Array]")
{
   int ContigData[6] = {6, 5, 4, 3, 2, 1};
   Array<int> a(ContigData, 6);
   Array<int> b{1, 2, 3, 3, 2, 1};

   a.Sort();
   b.Sort();

   for (int i = 1; i < a.Size(); i++)
   {
      REQUIRE(a[i] >= a[i-1]);
   }

   for (int i = 1; i < b.Size(); i++)
   {
      REQUIRE(b[i] >= b[i-1]);
   }
}

TEST_CASE("Array entry strict sorting", "[Array]")
{
   int ContigData[6] = {6, 1, 4, 1, 2, 1};
   Array<int> a(ContigData, 6);
   Array<int> b{1, 2, 3, 3, 2, 1};

   a.Sort();
   b.Sort();

   a.Unique();
   b.Unique();

   for (int i = 1; i < a.Size(); i++)
   {
      REQUIRE(a[i] > a[i-1]);
   }

   for (int i = 1; i < b.Size(); i++)
   {
      REQUIRE(b[i] > b[i-1]);
   }
}


TEST_CASE("Array stl-interactions", "[Array]")
{
   Array<int> x{0,1,2,3,4}, y;
   std::copy(x.begin(), x.end(), std::back_inserter(y));
   CHECK(y.Size() == 5);
   for (int i : {0,1,2,3,4})
   {
      CHECK(y[i] == i);
   }
   y.DeleteAll();

   std::copy_if(x.begin(), x.end(), std::back_inserter(y), [](int x) { return x <= 2; });
   CHECK(y.Size() == 3);
   for (int i : {0,1,2})
   {
      CHECK(y[i] == i);
   }

   std::transform(y.begin(), y.end(), y.begin(), [](int x) { return x*x; });
   CHECK(y.Size() == 3);
   for (int i : {0,1,2})
   {
      CHECK(y[i] == i*i);
   }
   y.DeleteAll();
   for (auto i : x)
   {
      y.Append(i);
   }
   for (int i : {0,1,2,3,4})
   {
      CHECK(x[i] == y[i]);
   }
   y.DeleteAll();
   for (const auto &i : x)
   {
      y.Append(i);
   }
   for (int i : {0,1,2,3,4})
   {
      CHECK(x[i] == y[i]);
   }
}
