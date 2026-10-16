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

#include <algorithm>
#include <limits>

#include "mfem.hpp"
#include "unit_tests.hpp"

// must be included after mfem.hpp
#include "general/scan.hpp"

using namespace mfem;

TEST_CASE("Inclusive Scan", "[Scan],[GPU]")
{
   Array<int> a(10);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      CAPTURE(use_dev);
      a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
      }
      auto dptr = a.ReadWrite(use_dev);
      InclusiveScan(use_dev, dptr, dptr, a.Size());
      a.HostRead();
      for (int i = 0; i < a.Size(); ++i)
      {
         int expected = (i + 1) * i / 2;
         CAPTURE(i);
         REQUIRE(AsConst(a)[i] == expected);
      }
      a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i + 1;
      }
      a.ReadWrite(use_dev);
      InclusiveScan(use_dev, dptr, dptr, a.Size(), std::multiplies<> {});
      a.HostRead();
      int expected = 1;
      for (int i = 0; i < a.Size(); ++i)
      {
         expected *= i + 1;
         CAPTURE(i);
         REQUIRE(AsConst(a)[i] == expected);
      }
   }
}

TEST_CASE("Exclusive Scan", "[Scan],[GPU]")
{
   Array<int> a(10);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      CAPTURE(use_dev);
      a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
      }
      auto dptr = a.ReadWrite(use_dev);
      ExclusiveScan(use_dev, dptr, dptr, a.Size(), 5);
      a.HostRead();
      for (int i = 0; i < a.Size(); ++i)
      {
         int expected = (i + 1) * i / 2 - i + 5;
         CAPTURE(i);
         REQUIRE(AsConst(a)[i] == expected);
      }
      a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i + 1;
      }
      a.ReadWrite(use_dev);
      ExclusiveScan(use_dev, dptr, dptr, a.Size(), 5, std::multiplies<> {});
      a.HostRead();
      int expected = 5;
      for (int i = 0; i < a.Size(); ++i)
      {
         CAPTURE(i);
         REQUIRE(AsConst(a)[i] == expected);
         expected *= i + 1;
      }
   }
}
