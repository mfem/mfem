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
#include <cstdlib>

namespace mfem
{

/// A pair of objects
template <class A, class B>
class Pair
{
public:
   A one;
   B two;
};

/// Compare the first element of the pairs
template <class A, class B>
int ComparePairs (const void *_p, const void *_q);

/// Sort with respect to the first element
template <class A, class B>
void SortPairs (Pair<A, B> *pairs, int size);


template <class A, class B, class C>
class Triple
{
public:
   A one;
   B two;
   C three;
};

template <class A, class B, class C>
int CompareTriple (const void *_p, const void *_q)
{
   const Triple<A, B, C> *p, *q;

   p = static_cast< const Triple<A, B, C>* >(_p);
   q = static_cast< const Triple<A, B, C>* >(_q);

   if (p -> one < q -> one) { return -1; }
   if (q -> one < p -> one) { return +1; }
   if (p -> two < q -> two) { return -1; }
   if (q -> two < p -> two) { return +1; }
   if (p -> three < q -> three) { return -1; }
   if (q -> three < p -> three) { return +1; }
   return 0;
}

template <class A, class B, class C>
void SortTriple (Triple<A, B, C> *triples, int size)
{
   if (size > 0)
   {
      qsort (triples, size, sizeof(Triple<A, B, C>), CompareTriple<A, B, C>);
   }
}

}

#endif
