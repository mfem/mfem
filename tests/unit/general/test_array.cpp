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


TEST_CASE("Array move assignment to view (MakeRef)", "[Array]")
{
   constexpr int n = 5;

   // Helper function that returns an Array by value (rvalue)
   auto make_sequence_array = [](int n)
   {
      Array<int> arr(n);
      for (int i = 0; i < n; i++)
      {
         arr[i] = i + 1;  // 1, 2, 3, ...
      }
      return arr;
   };

   // Create backing storage
   Array<int> backing1(n), backing2(n);
   backing1 = 0.0; backing2 = 0.0;

   // Create a view into the backing storage
   Array<int> view1, view2;
   view1.MakeRef(backing1);
   view2.MakeRef(backing2);

   auto seq_array = make_sequence_array(n);
   view1 = seq_array; // copy assign
   view2 = std::move(seq_array); // move assign

   CHECK(seq_array.Size() == 0); // seq_array is invalidated by the move

   // Both assignments should be semantically equivalent.
   for (int i = 0; i < n; i++)
   {
      CHECK(backing1[i] == backing2[i]);
   }

   // Create backing storage as raw array
   int backing3[n] = {0, 0, 0, 0, 0};

   // Create a view into the backing storage
   Array<int> view3;
   view3.MakeRef(backing3, n);

   // Assign from rvalue
   view3 = std::move(view2);

   for (int i = 0; i < n; i++)
   {
      CHECK(backing3[i] == backing1[i]);
   }
}

TEST_CASE("Array delete at indices", "[Array],[GPU]")
{
   for (int use_dev = 0; use_dev < 2; use_dev++)
   {
      Array<int>       test({0,1,2,3,4,5,6,7,8});
      Array<int> rm_indices({0,    3,4,  6,  8});
      Array<int>     result({  1,2,    5,  7  });

      test.GetMemory().UseDevice(use_dev);
      test.DeleteAt(rm_indices);

      REQUIRE(test.Size() == result.Size());

      test.HostReadWrite();
      for (int i = 0; i < test.Size(); i++)
      {
         CHECK(test[i] == result[i]);
      }
   }
}

TEST_CASE("BlockArray iterator equality", "[Array]")
{
   // Small blocks so the iteration crosses block boundaries.
   BlockArray<int> a(4);
   for (int i = 0; i < 10; i++) { a.Append(i); }

   int count = 0;
   for (BlockArray<int>::iterator it = a.begin(); it != a.end(); ++it)
   {
      count++;
   }
   REQUIRE(count == a.Size());

   // A compiler may compute it != end as !(end == it), so the comparison has
   // to give the same answer with the arguments swapped.
   BlockArray<int>::iterator it = a.begin();
   count = 0;
   for (; !(a.end() == it); ++it) { count++; }
   REQUIRE(count == a.Size());

   // Live iterators compare by position.
   BlockArray<int>::iterator b1 = a.begin(), b2 = a.begin();
   REQUIRE(b1 == b2);
   REQUIRE_FALSE(b1 != b2);
   ++b2;
   REQUIRE(b1 != b2);
   REQUIRE_FALSE(b1 == b2);
   ++b1;
   REQUIRE(b1 == b2);

   // The same offset in a different array is a different position.
   BlockArray<int> c(4);
   c.Append(0);
   REQUIRE(a.begin() != c.begin());
   REQUIRE(c.begin() != a.begin());

   // All end iterators are equal, and no live iterator equals an end.
   REQUIRE(a.end() == a.end());
   REQUIRE_FALSE(a.end() != a.end());
   REQUIRE(a.end() == c.end());
   REQUIRE(a.begin() != a.end());
   REQUIRE_FALSE(a.end() == a.begin());
   REQUIRE(a.end() != a.begin());

   // begin() of an empty array is an end iterator.
   BlockArray<int> empty;
   REQUIRE(empty.begin() == empty.end());
   REQUIRE(empty.begin() != a.begin());

   // The const iterators compare the same way.
   const BlockArray<int> &ca = a;
   BlockArray<int>::const_iterator cb1 = ca.begin(), cb2 = ca.begin();
   REQUIRE(cb1 == cb2);
   REQUIRE_FALSE(cb1 != cb2);
   ++cb1;
   REQUIRE(cb1 != cb2);
   REQUIRE_FALSE(cb1 == cb2);
   REQUIRE(cb1 != ca.end());
   REQUIRE_FALSE(cb1 == ca.end());
   REQUIRE_FALSE(ca.end() == cb1);
   REQUIRE(ca.end() != cb1);
   REQUIRE(ca.end() == ca.end());

   count = 0;
   for (BlockArray<int>::const_iterator it = ca.begin(); it != ca.end(); ++it)
   {
      count++;
   }
   REQUIRE(count == ca.Size());

   cb1 = ca.begin();
   count = 0;
   for (; !(ca.end() == cb1); ++cb1) { count++; }
   REQUIRE(count == ca.Size());
}

namespace
{

// Item for the HashTable test below, following the Hashed2 concept.
struct HashItem : public Hashed2 { };

} // namespace

TEST_CASE("HashTable iterator equality", "[Array]")
{
   // Small blocks so the iteration crosses block boundaries.
   HashTable<HashItem> ht(4);
   for (int i = 0; i < 10; i++) { ht.Get(i, i+1); }

   // The iterators skip deleted items.
   ht.Delete(0);

   int count = 0;
   for (HashTable<HashItem>::iterator it = ht.begin(); it != ht.end(); ++it)
   {
      count++;
   }
   // Size() counts the items that are not deleted, and the iterators skip
   // exactly those.
   REQUIRE(count == ht.Size());

   // The same iteration, written the way a compiler may rewrite it.
   HashTable<HashItem>::iterator it = ht.begin();
   count = 0;
   for (; !(ht.end() == it); ++it) { count++; }
   REQUIRE(count == ht.Size());

   // HashTable::iterator inherits its comparisons from BlockArray's.
   HashTable<HashItem>::iterator h1 = ht.begin(), h2 = ht.begin();
   REQUIRE(h1 == h2);
   REQUIRE_FALSE(h1 != h2);
   ++h2;
   REQUIRE(h1 != h2);
   REQUIRE_FALSE(h1 == h2);
   REQUIRE(h1 != ht.end());
   REQUIRE_FALSE(h1 == ht.end());
   REQUIRE_FALSE(ht.end() == h1);
   REQUIRE(ht.end() != h1);
   REQUIRE(ht.end() == ht.end());

   // And so do the const iterators.
   const HashTable<HashItem> &cht = ht;
   HashTable<HashItem>::const_iterator ch1 = cht.begin();
   REQUIRE(ch1 != cht.end());
   REQUIRE_FALSE(cht.end() == ch1);
   REQUIRE(cht.end() != ch1);
   REQUIRE(cht.end() == cht.end());

   count = 0;
   for (HashTable<HashItem>::const_iterator it = cht.begin(); it != cht.end();
        ++it)
   {
      count++;
   }
   REQUIRE(count == cht.Size());

   ch1 = cht.begin();
   count = 0;
   for (; !(cht.end() == ch1); ++ch1) { count++; }
   REQUIRE(count == cht.Size());
}
