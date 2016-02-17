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
