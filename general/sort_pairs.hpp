// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

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


template <class A, class B, class C>
class Triple
{
public:
   A one;
   B two;
   C three;

   Triple() { }

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
