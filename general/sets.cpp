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

#include "sets.hpp"


namespace mfem
{

int IntegerSet::PickRandomElement() const
{
   int i, size = Size();
   unsigned int seed = 0;

   for (i = 0; i < size; i++)
   {
      seed += data[i];
   }

   srand(seed);

   return data[rand()/(RAND_MAX/size)];
}

void IntegerSet::Recreate(const int n, const int *p)
{
   int i, j;

   SetSize(n);

   for (i = 0; i < n; i++)
   {
      data[i] = p[i];
   }

   Sort();

   for (j = 0, i = 1; i < n; i++)
      if (data[i] != data[j])
      {
         data[++j] = data[i];
      }

   SetSize(j+1);
}


int ListOfIntegerSets::Insert(const IntegerSet &s)
{
   for (int i = 0; i < TheList.Size(); i++)
      if (*TheList[i] == s)
      {
         return i;
      }

   TheList.Append(new IntegerSet(s));

   return TheList.Size()-1;
}

int ListOfIntegerSets::Lookup(const IntegerSet &s) const
{
   for (int i = 0; i < TheList.Size(); i++)
      if (*TheList[i] == s)
      {
         return i;
      }

   mfem_error("ListOfIntegerSets::Lookup (), integer set not found.");
   return -1;
}

void ListOfIntegerSets::AsTable(Table & t) const
{
   int i;

   t.MakeI(Size());

   for (i = 0; i < Size(); i++)
   {
      t.AddColumnsInRow(i, TheList[i] -> Size());
   }

   t.MakeJ();

   for (i = 0; i < Size(); i++)
   {
      Array<int> &row = *TheList[i];
      t.AddConnections(i, row.GetData(), row.Size());
   }

   t.ShiftUpI();
}

ListOfIntegerSets::~ListOfIntegerSets()
{
   for (int i = 0; i < TheList.Size(); i++)
   {
      delete TheList[i];
   }
}

}
