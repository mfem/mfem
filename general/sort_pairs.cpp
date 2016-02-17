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


#include "sort_pairs.hpp"

#ifdef MFEM_USE_MPI
#include <HYPRE_utilities.h>
#endif

namespace mfem
{

template <class A, class B>
int ComparePairs (const void *_p, const void *_q)
{
   Pair<A, B> *p, *q;

   p = (Pair<A, B> *)_p;
   q = (Pair<A, B> *)_q;

   if (p -> one < q -> one) { return -1; }
   if (q -> one < p -> one) { return +1; }
   return 0;
}

template <class A, class B>
void SortPairs (Pair<A, B> *pairs, int size)
{
   if (size > 0)
   {
      qsort (pairs, size, sizeof(Pair<A, B>), ComparePairs<A, B>);
   }
}


// Instantiate int-int, double-int, int-double pairs
template int ComparePairs<int, int> (const void *, const void *);
template int ComparePairs<double, int> (const void *, const void *);
template int ComparePairs<int, double> (const void *, const void *);
template void SortPairs<int, int> (Pair<int, int> *, int );
template void SortPairs<double, int> (Pair<double, int> *, int );
template void SortPairs<int, double> (Pair<int, double> *, int );
#ifdef HYPRE_BIGINT
template int ComparePairs<HYPRE_Int, int> (const void *, const void *);
template void SortPairs<HYPRE_Int, int> (Pair<HYPRE_Int, int> *, int );
#endif

}
