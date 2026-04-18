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
#include "general/reducers.hpp"

using namespace mfem;

TEST_CASE("Reduce Sum", "[Reduction],[GPU]")
{
   Array<int> workspace;
   Array<int> a(1000);
   a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i;
   }

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      int res = 0;
      mfem::reduce(
      a.Size(), res, [=] MFEM_HOST_DEVICE(int i, int &r) { r += dptr[i]; },
      SumReducer<int> {}, use_dev, workspace);
      // correct for even-length summations
      int expected = (AsConst(a)[0] + AsConst(a)[a.Size() - 1]) * a.Size() / 2;
      CAPTURE(use_dev);
      REQUIRE(res == expected);
   }
}

TEST_CASE("Reduce Mult", "[Reduction],[GPU]")
{
   Array<long long> workspace;
   Array<long long> a(64);
   a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i % 3 + 1;
   }

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      SECTION("{ Zero Mult }")
      {
         auto dptr = a.Read(use_dev);
         // 0 * anything == 0
         long long res = 0;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, long long &r) { r *= dptr[i]; },
         MultReducer<long long> {}, use_dev, workspace);
         long long expected = 0;
         CAPTURE(use_dev);
         REQUIRE(res == expected);
      }
      SECTION("{ One Mult }")
      {
         auto dptr = a.Read(use_dev);
         long long res = 1;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, long long &r) { r *= dptr[i]; },
         MultReducer<long long> {}, use_dev, workspace);
         long long expected = 21936950640377856;
         CAPTURE(use_dev);
         REQUIRE(res == expected);
      }
   }
}

TEST_CASE("Reduce BAnd", "[Reduction],[GPU]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(10);
   SECTION("{ Bit unset }")
   {
      // pick numbers which sets all bits except for one
      a.HostReadWrite();
      constexpr unsigned unset_bit = 17;
      a[0] = ~(1u << unset_bit);
      for (int i = 1; i < a.Size(); ++i)
      {
         a[i] = (~1u) & a[0];
      }
      a[0] = ~1u;

      for (int use_dev = 0; use_dev < 2; ++use_dev)
      {
         auto dptr = a.Read(use_dev);

         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, unsigned &r) { r &= dptr[i]; },
         BAndReducer<unsigned> {}, use_dev, workspace);
         CAPTURE(use_dev);
         REQUIRE(res == ((~1u) & ~(1u << unset_bit)));
         REQUIRE((res & (1u << unset_bit)) == 0);
      }
   }
   SECTION("{ Bit set }")
   {
      // ensure one bit is set
      a.HostReadWrite();
      constexpr unsigned set_bit = 17;
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i | (1u << set_bit);
      }

      for (int use_dev = 0; use_dev < 2; ++use_dev)
      {
         auto dptr = a.Read(use_dev);

         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, unsigned &r) { r &= dptr[i]; },
         BAndReducer<unsigned> {}, use_dev, workspace);
         CAPTURE(use_dev);
         REQUIRE(res == (1u << set_bit));
      }
   }
}

TEST_CASE("Reduce BOr", "[Reduction],[GPU]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(0x210);
   a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i;
   }

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);

      unsigned res = 0u;
      mfem::reduce(
         a.Size(), res,
      [=] MFEM_HOST_DEVICE(int i, unsigned &r) { r |= dptr[i]; },
      BOrReducer<unsigned> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res == 0x3ffu);
   }
}

TEST_CASE("Reduce Min", "[Reduction],[GPU]")
{
   Array<int> workspace;
   Array<int> a(1000);
   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      int res = std::numeric_limits<int>::max();
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, int &r)
      {
         if (dptr[i] < r)
         {
            r = dptr[i];
         }
      },
      MinReducer<int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res == -10);
   }
}

TEST_CASE("Reduce Max", "[Reduction],[GPU]")
{
   Array<int> workspace;
   Array<int> a(1000);
   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      int res = std::numeric_limits<int>::min();
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, int &r)
      {
         if (dptr[i] > r)
         {
            r = dptr[i];
         }
      },
      MaxReducer<int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res == 999 - 10);
   }
}

TEST_CASE("Reduce MinMax", "[Reduction],[GPU]")
{
   Array<DevicePair<int, int>> workspace;
   Array<int> a(1000);
   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      DevicePair<int, int> res = {std::numeric_limits<int>::max(),
                                  std::numeric_limits<int>::min()
                                 };
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, DevicePair<int, int> &r)
      {
         if (dptr[i] < r.first)
         {
            r.first = dptr[i];
         }
         if (dptr[i] > r.second)
         {
            r.second = dptr[i];
         }
      },
      MinMaxReducer<int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res.first == -10);
      REQUIRE(res.second == a.Size() - 11);
   }
}

TEST_CASE("Reduce ArgMin", "[Reduction],[GPU]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);
   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      DevicePair<double, int> res = {std::numeric_limits<double>::infinity(), -1};
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, DevicePair<double, int> &r)
      {
         if (dptr[i] < r.first)
         {
            r.first = dptr[i];
            r.second = i;
         }
      },
      ArgMinReducer<double, int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res.first == -10);
      REQUIRE(res.second >= 0);
      REQUIRE(res.second < a.Size());
      REQUIRE(hptr[res.second] == res.first);
   }
}

TEST_CASE("Reduce ArgMax", "[Reduction],[GPU]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);

   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      DevicePair<double, int> res = {-std::numeric_limits<double>::infinity(),
                                     -1
                                    };
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, DevicePair<double, int> &r)
      {
         if (dptr[i] > r.first)
         {
            r.first = dptr[i];
            r.second = i;
         }
      },
      ArgMaxReducer<double, int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res.first == a.Size() - 11);
      REQUIRE(res.second >= 0);
      REQUIRE(res.second < a.Size());
      REQUIRE(AsConst(a)[res.second] == res.first);
   }
}

TEST_CASE("Reduce ArgMinMax", "[Reduction],[GPU]")
{
   Array<MinMaxLocScalar<double, int>> workspace;
   Array<double> a(1000);
   auto hptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i - 10;
   }

   std::random_device rd;
   std::mt19937 gen(rd());
   std::shuffle(hptr, hptr + a.Size(), gen);

   for (int use_dev = 0; use_dev < 2; ++use_dev)
   {
      auto dptr = a.Read(use_dev);
      MinMaxLocScalar<double, int> res =
      {
         std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(), -1, -1
      };
      mfem::reduce(
         a.Size(), res,
         [=] MFEM_HOST_DEVICE(int i, MinMaxLocScalar<double, int> &r)
      {
         if (dptr[i] < r.min_val)
         {
            r.min_val = dptr[i];
            r.min_loc = i;
         }
         if (dptr[i] > r.max_val)
         {
            r.max_val = dptr[i];
            r.max_loc = i;
         }
      },
      ArgMinMaxReducer<double, int> {}, use_dev, workspace);
      CAPTURE(use_dev);
      REQUIRE(res.min_val == -10);
      REQUIRE(res.min_loc >= 0);
      REQUIRE(res.min_loc < a.Size());
      REQUIRE(AsConst(a)[res.min_loc] == res.min_val);

      REQUIRE(res.max_val == a.Size() - 11);
      REQUIRE(res.max_loc >= 0);
      REQUIRE(res.max_loc < a.Size());
      REQUIRE(AsConst(a)[res.max_loc] == res.max_val);
   }
}
