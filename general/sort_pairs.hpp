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

#ifndef MFEM_SORT_PAIRS
#define MFEM_SORT_PAIRS

#include "../config/config.hpp"
#include <algorithm>

namespace mfem
{

/// A pair of objects
template <class A, class B>
class Pair
{
public:
   A one;
   B two;

   Pair() = default;

   Pair(const A &one, const B &two) : one(one), two(two) {}
};

/// @brief Comparison operator for class Pair, based on the first element only.
template <class A, class B>
bool operator<(const Pair<A,B> &p, const Pair<A,B> &q)
{
   return (p.one < q.one);
}

/// @brief Equality operator for class Pair, based on the first element only.
template <class A, class B>
bool operator==(const Pair<A,B> &p, const Pair<A,B> &q)
{
   return (p.one == q.one);
}

/// Sort an array of Pairs with respect to the first element
template <class A, class B>
void SortPairs (Pair<A, B> *pairs, int size)
{
   std::sort(pairs, pairs + size);
}

/// A triple of objects
template <class A, class B, class C>
class Triple
{
public:
   A one;
   B two;
   C three;

   Triple() = default;

   Triple(const A &one, const B &two, const C &three)
      : one(one), two(two), three(three) { }
};

/// @brief Lexicographic comparison operator for class Triple.
template <class A, class B, class C>
bool operator<(const Triple<A,B,C> &p, const Triple<A,B,C> &q)
{
   return (p.one < q.one ||
           (!(q.one < p.one) &&
            (p.two < q.two || (!(q.two < p.two) && p.three < q.three))));
}

/// @brief Lexicographic sort for arrays of class Triple.
template <class A, class B, class C>
void SortTriple (Triple<A, B, C> *triples, int size)
{
   std::sort(triples, triples + size);
}

}

#endif
