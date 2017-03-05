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


void Set_PrintStats(std::ostream &out, int size, int idx_size, int num_bins,
                    Array<int> &bin_sz_cnt, long mem_usage)
{
   out << "Set statistics:\n"
       << "----------------\n"
       << "   number of indices:  " << idx_size << '\n'
       << "   number of elements: " << size << '\n'
       << "   number of bins:     " << num_bins << '\n'
       << "   average bin size:   " << 1.*size/num_bins << '\n'
       << "   dynamic memory use: " << mem_usage << " ("
       << mem_usage/(1024.*1024.) << " MiB)\n"
       << "   ----------+------------+----------\n"
       << "    bin size |  num bins  |  % bins\n"
       << "   ----------+------------+----------\n";
   std::ios::fmtflags old_fmt = out.flags();
   out.setf(std::ios::fixed);
   std::streamsize old_prec = out.precision(5);
   for (int sz = 0; sz < bin_sz_cnt.Size(); sz++)
   {
      out << std::setw(9) << sz << "    | "
          << std::setw(10) << bin_sz_cnt[sz] << " | "
          << std::setw(8) << 100.*bin_sz_cnt[sz]/num_bins << "\n";
   }
   out << "   ----------+------------+----------" << std::endl;
   out.precision(old_prec);
   out.flags(old_fmt);
}

}
