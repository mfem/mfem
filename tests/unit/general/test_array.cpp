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

using namespace mfem;

TEST_CASE("Array init-list construction", "[Array]")
{
   int ContigData[6] = {6, 5, 4, 3, 2, 1};
   Array<int> a(ContigData, 6);
   Array<int> b{6, 5, 4, 3, 2, 1};

   for (int i = 0; i < a.Size(); i++)
   {
      REQUIRE(a[i] == b[i]);
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
