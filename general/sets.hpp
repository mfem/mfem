// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SETS
#define MFEM_SETS

#include "../config/config.hpp"
#include "array.hpp"
#include "table.hpp"

namespace mfem
{

/// A set of integers
class IntegerSet
{
private:
   Array<int> me;

public:
   /// Create an empty set.
   IntegerSet() { }

   /// Create a copy of set 's'.
   IntegerSet(IntegerSet &s);

   /// Create an integer set from C-array 'p' of 'n' integers.
   IntegerSet(const int n, const int *p) { Recreate(n, p); }

   /// Return the size of the set.
   int Size() { return me.Size(); }

   /// Return a reference to the sorted array of all the set entries.
   operator Array<int>& () { return me; }

   /// Return the value of the lowest element of the set.
   int PickElement() { return me[0]; }

   /// Return the value of a random element of the set.
   int PickRandomElement();

   /// Return 1 if the sets are equal and 0 otherwise.
   int operator==(IntegerSet &s);

   /** @brief Create an integer set from C-array 'p' of 'n' integers.
       Overwrites any existing set data. */
   void Recreate(const int n, const int *p);
};

/// List of integer sets
class ListOfIntegerSets
{
private:
   Array<IntegerSet *> TheList;

public:

   /// Return the number of integer sets in the list.
   int Size() { return TheList.Size(); }

   /// Return the value of the first element of the ith set.
   int PickElementInSet(int i) { return TheList[i]->PickElement(); }

   /// Return a random value from the ith set in the list.
   int PickRandomElementInSet(int i) { return TheList[i]->PickRandomElement(); }

   /** @brief Check to see if set 's' is in the list. If not append it to the
       end of the list. Returns the index of the list where set 's' can be
       found. */
   int Insert(IntegerSet &s);

   /** Return the index of the list where set 's' can be found. Returns -1 if
       not found. */
   int Lookup(IntegerSet &s);

   /// Write the list of sets into table 't'.
   void AsTable(Table &t);

   ~ListOfIntegerSets();
};

}

#endif
