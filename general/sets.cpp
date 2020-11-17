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

#include "sets.hpp"


namespace mfem
{

IntegerSet::IntegerSet(IntegerSet &s)
   : me(s.me.Size())
{
   for (int i = 0; i < me.Size(); i++)
   {
      me[i] = s.me[i];
   }
}


int IntegerSet::operator== (IntegerSet &s)
{
   if (me.Size() != s.me.Size())
   {
      return 0;
   }

   for (int i = 0; i < me.Size(); i++)
      if (me[i] != s.me[i])
      {
         return 0;
      }

   return 1;
}

int IntegerSet::PickRandomElement()
{
   int i, size = me.Size();
   unsigned int seed = 0;

   for (i = 0; i < size; i++)
   {
      seed += me[i];
   }

   srand(seed);

   return me[rand()/(RAND_MAX/size)];
}

void IntegerSet::Recreate(const int n, const int *p)
{
   int i, j;

   me.SetSize(n);

   for (i = 0; i < n; i++)
   {
      me[i] = p[i];
   }

   me.Sort();

   for (j = 0, i = 1; i < n; i++)
      if (me[i] != me[j])
      {
         me[++j] = me[i];
      }

   me.SetSize(j+1);
}


int ListOfIntegerSets::Insert(IntegerSet &s)
{
   for (int i = 0; i < TheList.Size(); i++)
      if (*TheList[i] == s)
      {
         return i;
      }

   TheList.Append(new IntegerSet(s));

   return TheList.Size()-1;
}

int ListOfIntegerSets::Lookup(IntegerSet &s)
{
   for (int i = 0; i < TheList.Size(); i++)
      if (*TheList[i] == s)
      {
         return i;
      }

   mfem_error("ListOfIntegerSets::Lookup (), integer set not found.");
   return -1;
}

void ListOfIntegerSets::AsTable(Table & t)
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
