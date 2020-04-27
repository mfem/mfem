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
   IntegerSet() { }

   IntegerSet(IntegerSet &s);

   /// Create an integer set from a block of memory containing integer values
   /// ( like an array ).
   ///
   /// n - length ( number of integers )
   /// p - pointer to block of memory containing the integer values
   IntegerSet(const int n, const int *p) { Recreate(n, p); }

   int Size() { return me.Size(); }

   operator Array<int>& () { return me; }

   int PickElement() { return me[0]; }

   int PickRandomElement();

   int operator==(IntegerSet &s);

   void Recreate(const int n, const int *p);
};

/// List of integer sets
class ListOfIntegerSets
{
private:
   Array<IntegerSet *> TheList;

public:

   int Size() { return TheList.Size(); }

   int PickElementInSet(int i) { return TheList[i]->PickElement(); }

   int PickRandomElementInSet(int i) { return TheList[i]->PickRandomElement(); }

   int Insert(IntegerSet &s);

   int Lookup(IntegerSet &s);

   void AsTable(Table &t);

   ~ListOfIntegerSets();
};

}

#endif
