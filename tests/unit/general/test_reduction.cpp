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

// must be included after mfem.hpp
#include "general/forall.hpp"

#include "unit_tests.hpp"

#include <algorithm>
#include <limits>

using namespace mfem;

TEST_CASE("Reduce CPU Sum", "[Reduction]")
{
   Array<int> workspace;
   Array<int> a(1000);
   auto ptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i;
   }

   {
      int res = 0;
      mfem::reduce(
      a.Size(), res, [=] MFEM_HOST_DEVICE(int i, int &r) { r += ptr[i]; },
      SumReducer<int> {}, false, workspace);
      // correct for even-length summations
      int expected = (a[0] + a[a.Size() - 1]) * a.Size() / 2;
      REQUIRE(res == expected);
   }
}

TEST_CASE("Reduce CPU Mult", "[Reduction]")
{
   Array<long long> workspace;
   Array<long long> a(100);
   auto ptr = a.HostReadWrite();
   for (long long i = 0; i < a.Size(); ++i)
   {
      a[i] = i % 3 + 1;
   }

   {
      long long res = 0;
      mfem::reduce(
         a.Size(), res,
      [=] MFEM_HOST_DEVICE(long long i, long long &r) { r *= ptr[i]; },
      MultReducer<long long> {}, false, workspace);
      long long expected = 0;
      REQUIRE(res == expected);
   }

   {
      long long res = 1;
      mfem::reduce(
         a.Size(), res,
      [=] MFEM_HOST_DEVICE(long long i, long long &r) { r *= ptr[i]; },
      MultReducer<long long> {}, false, workspace);
      long long expected = 5527454985320660992;
      REQUIRE(res == expected);
   }
}

TEST_CASE("Reduce CPU BAnd", "[Reduction]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(10);
   {
      // pick numbers which sets all bits except for one
      constexpr unsigned unset_bit = 17;

      auto ptr = a.HostReadWrite();
      a[0] = ~(1u << unset_bit);
      for (unsigned i = 1; i < a.Size(); ++i)
      {
         a[i] = (~1u) & a[0];
      }
      a[0] = ~1u;

      {
         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r &= ptr[i]; },
         BAndReducer<unsigned> {}, false, workspace);
         REQUIRE(res == ((~1u) & ~(1u << unset_bit)));
         REQUIRE((res & (1u << unset_bit)) == 0);
      }
   }
   {
      // ensure one bit is set
      constexpr unsigned set_bit = 17;

      auto ptr = a.HostReadWrite();
      for (unsigned i = 0; i < a.Size(); ++i)
      {
         a[i] = i | (1u << set_bit);
      }

      {
         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r &= ptr[i]; },
         BAndReducer<unsigned> {}, false, workspace);
         REQUIRE(res == (1u << set_bit));
      }
   }
}

TEST_CASE("Reduce CPU BOr", "[Reduction]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(10);
   {
      auto ptr = a.HostReadWrite();
      for (unsigned i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
      }

      {
         unsigned res = 0u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r |= ptr[i]; },
         BOrReducer<unsigned> {}, false, workspace);
         REQUIRE(res == 0xfu);
      }
   }
}

TEST_CASE("Reduce CPU Min", "[Reduction]")
{
   Array<int> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         int res = std::numeric_limits<int>::max();
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, int &r)
         {
            if (ptr[i] < r)
            {
               r = ptr[i];
            }
         },
         MinReducer<int> {}, false, workspace);
         REQUIRE(res == -10);
      }
   }
}

TEST_CASE("Reduce CPU Max", "[Reduction]")
{
   Array<int> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         int res = std::numeric_limits<int>::min();
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, int &r)
         {
            if (ptr[i] > r)
            {
               r = ptr[i];
            }
         },
         MaxReducer<int> {}, false, workspace);
         REQUIRE(res == 999 - 10);
      }
   }
}

TEST_CASE("Reduce CPU MinMax", "[Reduction]")
{
   Array<DevicePair<int, int>> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         DevicePair<int, int> res = {std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::min()
                                    };
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, DevicePair<int, int> &r)
         {
            if (ptr[i] < r.first)
            {
               r.first = ptr[i];
            }
            if (ptr[i] > r.second)
            {
               r.second = ptr[i];
            }
         },
         MinMaxReducer<int> {}, false, workspace);
         REQUIRE(res.first == -10);
         REQUIRE(res.second == a.Size() - 11);
      }
   }
}

TEST_CASE("Reduce CPU ArgMin", "[Reduction]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         DevicePair<double, int> res = {std::numeric_limits<double>::infinity(),
                                        -1
                                       };
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, DevicePair<double, int> &r)
         {
            if (ptr[i] < r.first)
            {
               r.first = ptr[i];
               r.second = i;
            }
         },
         ArgMinReducer<double, int> {}, false, workspace);
         REQUIRE(res.first == -10);
         REQUIRE(res.second >= 0);
         REQUIRE(res.second < a.Size());
         REQUIRE(ptr[res.second] == res.first);
      }
   }
}

TEST_CASE("Reduce CPU ArgMax", "[Reduction]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         DevicePair<double, int> res = {-std::numeric_limits<double>::infinity(),
                                        -1
                                       };
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, DevicePair<double, int> &r)
         {
            if (ptr[i] > r.first)
            {
               r.first = ptr[i];
               r.second = i;
            }
         },
         ArgMaxReducer<double, int> {}, false, workspace);
         REQUIRE(res.first == a.Size() - 11);
         REQUIRE(res.second >= 0);
         REQUIRE(res.second < a.Size());
         REQUIRE(ptr[res.second] == res.first);
      }
   }
}

TEST_CASE("Reduce CPU ArgMinMax", "[Reduction]")
{
   Array<MinMaxLocScalar<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      {
         MinMaxLocScalar<double, int> res =
         {
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(), -1, -1
         };
         mfem::reduce(
            a.Size(), res,
            [=] MFEM_HOST_DEVICE(int i, MinMaxLocScalar<double, int> &r)
         {
            if (ptr[i] < r.min_val)
            {
               r.min_val = ptr[i];
               r.min_loc = i;
            }
            if (ptr[i] > r.max_val)
            {
               r.max_val = ptr[i];
               r.max_loc = i;
            }
         },
         ArgMinMaxReducer<double, int> {}, false, workspace);
         REQUIRE(res.min_val == -10);
         REQUIRE(res.min_loc >= 0);
         REQUIRE(res.min_loc < a.Size());
         REQUIRE(ptr[res.min_loc] == res.min_val);

         REQUIRE(res.max_val == a.Size() - 11);
         REQUIRE(res.max_loc >= 0);
         REQUIRE(res.max_loc < a.Size());
         REQUIRE(ptr[res.max_loc] == res.max_val);
      }
   }
}

TEST_CASE("Reduce GPU Sum", "[Reduction],[CUDA]")
{
   Array<int> workspace;
   Array<int> a(1000);
   auto ptr = a.HostReadWrite();
   for (int i = 0; i < a.Size(); ++i)
   {
      a[i] = i;
   }

   {
      auto dptr = a.Read();
      int res = 0;
      mfem::reduce(
      a.Size(), res, [=] MFEM_HOST_DEVICE(int i, int &r) { r += dptr[i]; },
      SumReducer<int> {}, true, workspace);
      // correct for even-length summations
      int expected = (a[0] + a[a.Size() - 1]) * a.Size() / 2;
      REQUIRE(res == expected);
   }
}

TEST_CASE("Reduce GPU Mult", "[Reduction],[CUDA]")
{
   Array<long long> workspace;
   Array<long long> a(100);
   auto ptr = a.HostReadWrite();
   for (long long i = 0; i < a.Size(); ++i)
   {
      a[i] = i % 3 + 1;
   }

   {
      auto dptr = a.Read();
      // 0 * anything == 0
      long long res = 0;
      mfem::reduce(
         a.Size(), res,
      [=] MFEM_HOST_DEVICE(long long i, long long &r) { r *= dptr[i]; },
      MultReducer<long long> {}, true, workspace);
      long long expected = 0;
      REQUIRE(res == expected);
   }

   {
      auto dptr = a.Read();
      long long res = 1;
      mfem::reduce(
         a.Size(), res,
      [=] MFEM_HOST_DEVICE(long long i, long long &r) { r *= dptr[i]; },
      MultReducer<long long> {}, true, workspace);
      long long expected = 5527454985320660992;
      REQUIRE(res == expected);
   }
}

TEST_CASE("Reduce GPU BAnd", "[Reduction],[CUDA]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(10);
   {
      // pick numbers which sets all bits except for one
      a.HostReadWrite();
      constexpr unsigned unset_bit = 17;
      a[0] = ~(1u << unset_bit);
      for (unsigned i = 1; i < a.Size(); ++i)
      {
         a[i] = (~1u) & a[0];
      }
      a[0] = ~1u;

      auto dptr = a.Read();

      {
         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r &= dptr[i]; },
         BAndReducer<unsigned> {}, true, workspace);
         REQUIRE(res == ((~1u) & ~(1u << unset_bit)));
         REQUIRE((res & (1u << unset_bit)) == 0);
      }
   }
   {
      // ensure one bit is set
      a.HostReadWrite();
      constexpr unsigned set_bit = 17;
      for (unsigned i = 0; i < a.Size(); ++i)
      {
         a[i] = i | (1u << set_bit);
      }

      auto dptr = a.Read();

      {
         unsigned res = ~1u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r &= dptr[i]; },
         BAndReducer<unsigned> {}, true, workspace);
         REQUIRE(res == (1u << set_bit));
      }
   }
}

TEST_CASE("Reduce GPU BOr", "[Reduction],[CUDA]")
{
   Array<unsigned> workspace;
   Array<unsigned> a(0x210);
   {
      a.HostReadWrite();
      for (unsigned i = 0; i < a.Size(); ++i)
      {
         a[i] = i;
      }

      auto dptr = a.Read();

      {
         unsigned res = 0u;
         mfem::reduce(
            a.Size(), res,
         [=] MFEM_HOST_DEVICE(unsigned i, unsigned &r) { r |= dptr[i]; },
         BOrReducer<unsigned> {}, true, workspace);
         REQUIRE(res == 0x3ffu);
      }
   }
}

TEST_CASE("Reduce GPU Min", "[Reduction],[CUDA]")
{
   Array<int> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
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
         MinReducer<int> {}, true, workspace);
         REQUIRE(res == -10);
      }
   }
}

TEST_CASE("Reduce GPU Max", "[Reduction],[CUDA]")
{
   Array<int> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
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
         MaxReducer<int> {}, true, workspace);
         REQUIRE(res == 999 - 10);
      }
   }
}

TEST_CASE("Reduce GPU MinMax", "[Reduction],[CUDA]")
{
   Array<DevicePair<int, int>> workspace;
   Array<int> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
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
         MinMaxReducer<int> {}, true, workspace);
         REQUIRE(res.first == -10);
         REQUIRE(res.second == a.Size() - 11);
      }
   }
}

TEST_CASE("Reduce GPU ArgMin", "[Reduction],[CUDA]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
         DevicePair<double, int> res = {std::numeric_limits<double>::infinity(),
                                        -1
                                       };
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
         ArgMinReducer<double, int> {}, true, workspace);
         REQUIRE(res.first == -10);
         REQUIRE(res.second >= 0);
         REQUIRE(res.second < a.Size());
         REQUIRE(ptr[res.second] == res.first);
      }
   }
}

TEST_CASE("Reduce GPU ArgMax", "[Reduction],[CUDA]")
{
   Array<DevicePair<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
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
         ArgMaxReducer<double, int> {}, true, workspace);
         REQUIRE(res.first == a.Size() - 11);
         REQUIRE(res.second >= 0);
         REQUIRE(res.second < a.Size());
         REQUIRE(a[res.second] == res.first);
      }
   }
}

TEST_CASE("Reduce GPU ArgMinMax", "[Reduction],[CUDA]")
{
   Array<MinMaxLocScalar<double, int>> workspace;
   Array<double> a(1000);
   {
      auto ptr = a.HostReadWrite();
      for (int i = 0; i < a.Size(); ++i)
      {
         a[i] = i - 10;
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(ptr, ptr + a.Size(), gen);

      auto dptr = a.Read();

      {
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
         ArgMinMaxReducer<double, int> {}, true, workspace);
         REQUIRE(res.min_val == -10);
         REQUIRE(res.min_loc >= 0);
         REQUIRE(res.min_loc < a.Size());
         REQUIRE(a[res.min_loc] == res.min_val);

         REQUIRE(res.max_val == a.Size() - 11);
         REQUIRE(res.max_loc >= 0);
         REQUIRE(res.max_loc < a.Size());
         REQUIRE(a[res.max_loc] == res.max_val);
      }
   }
}
