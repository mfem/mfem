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

TEST_CASE("CopyFlagged", "[Scan],[GPU]")
{
   Array<int> a(10);
   Array<bool> flags(a.Size());
   Array<int> res(a.Size());
   Array<int> num_selected_out(1);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      CAPTURE(use_dev);
      a.HostWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
      }
      flags.HostWrite();
      // keep entries which are a multiple of 3
      for (int i = 0; i < flags.Size(); ++i)
      {
         if (i % 3)
         {
            flags[i] = false;
         }
         else
         {
            flags[i] = true;
         }
      }
      auto d_in = a.Read(use_dev);
      auto d_flags = flags.Read(use_dev);
      auto d_out = res.Write(use_dev);
      auto d_num_selected_out = num_selected_out.Write(use_dev);
      CopyFlagged(use_dev, d_in, d_flags, d_out, d_num_selected_out, a.Size());
      res.HostRead();
      num_selected_out.HostRead();
      REQUIRE(AsConst(num_selected_out)[0] == 4);
      REQUIRE(AsConst(res)[0] == 0);
      REQUIRE(AsConst(res)[1] == 3);
      REQUIRE(AsConst(res)[2] == 6);
      REQUIRE(AsConst(res)[3] == 9);
   }
}

TEST_CASE("CopyIf", "[Scan][GPU]")
{
   Array<int> a(10);
   Array<int> res(a.Size());
   Array<int> num_selected_out(1);

   for (int use_dev = 1; use_dev < 2; ++use_dev)
   {
      CAPTURE(use_dev);
      a.HostWrite();
      res.HostWrite();
      num_selected_out.HostWrite();
      num_selected_out[0] = 0;
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
         res[i] = 0;
      }
      auto d_in = a.Read(use_dev);
      auto d_out = res.Write(use_dev);
      auto d_num_selected_out = num_selected_out.Write(use_dev);
      // copy all values not divisible by 3
      // 1, 2, 4, 5, 7, 8
      CopyIf(use_dev, d_in, d_out, d_num_selected_out, a.Size(),
      [=] MFEM_HOST_DEVICE(const int &value) { return value % 3; });
      res.HostRead();
      num_selected_out.HostRead();
      REQUIRE(AsConst(num_selected_out)[0] == 6);
      REQUIRE(AsConst(res)[0] == 1);
      REQUIRE(AsConst(res)[1] == 2);
      REQUIRE(AsConst(res)[2] == 4);
      REQUIRE(AsConst(res)[3] == 5);
      REQUIRE(AsConst(res)[4] == 7);
      REQUIRE(AsConst(res)[5] == 8);
   }
}

TEST_CASE("CopyUnique", "[Scan],[GPU]")
{
   Array<int> a(10);
   Array<int> res(a.Size());
   Array<int> num_selected_out(1);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      CAPTURE(use_dev);
      a.HostWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = 2;
      }
      a[4] = 1;
      a[5] = 1;
      auto d_in = a.Read(use_dev);
      auto d_out = res.Write(use_dev);
      auto d_num_selected_out = num_selected_out.Write(use_dev);
      CopyUnique(use_dev, d_in, d_out, d_num_selected_out, a.Size());
      res.HostRead();
      num_selected_out.HostRead();
      REQUIRE(AsConst(num_selected_out)[0] == 3);
      REQUIRE(AsConst(res)[0] == 2);
      REQUIRE(AsConst(res)[1] == 1);
      REQUIRE(AsConst(res)[2] == 2);
   }
}
