// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

DisjointSets::DisjointSets(int n)
   : parent(n), size(n), sizeCounters(0)
{
   finalized = false;
   for (int i = 0; i < n; ++i)
   {
      parent[i] = i;
      size[i] = 1;
   }
}

void DisjointSets::Union(int i, int j)
{
   finalized = false;

   int i_root = Find(i);
   int j_root = Find(j);

   if (i_root == j_root) { return; }

   if (size[i_root] < size[j_root]) { std::swap(i_root, j_root); }

   parent[j_root] = i_root;
   size[i_root] += size[j_root];
}

int DisjointSets::Find(int i) const
{
   if (parent[i] != i) { parent[i] = Find(parent[i]); }
   return parent[i];
}

void DisjointSets::MakeSingle(const Array<int> &to_make_single)
{
   finalized = false;

   std::unordered_map<int, std::unordered_set<int>> reps_to_sets;

   for (int i = 0; i < parent.Size(); ++i)
   {
      int rep = Find(i);

      if (!reps_to_sets.count(rep))
      {
         reps_to_sets[rep] = std::unordered_set<int>();
      }
      reps_to_sets[rep].insert(i);
   }

   for (int j = 0; j < to_make_single.Size(); ++j)
   {
      int i = to_make_single[j];
      int rep = Find(i);

      while (size[rep] != 1)
      {
         if (rep != i)
         {
            parent[i] = i;
            size[i] = 1;
            size[rep] -= 1;
            reps_to_sets[rep].erase(i);
            i = rep;
         }
         else if (size[i] != 1)
         {
            reps_to_sets[i].erase(i);
            int new_rep = *(reps_to_sets[i].begin());
            size[new_rep] = size[i] - 1;
            for (int j : reps_to_sets[i])
            {
               parent[j] = new_rep;
            }
            parent[i] = i;
            size[i] = 1;
            reps_to_sets[new_rep] = reps_to_sets[i];
            reps_to_sets[i] = std::unordered_set<int>();
            reps_to_sets[i].insert(i);

            i = new_rep;
            rep = Find(new_rep);
         }
      }
   }
}

void DisjointSets::Finalize()
{
   if (finalized) { return; }
   finalized = true;

   bounds = Array<int>();
   elems = Array<int>();
   // preallocate to 4, could be more but not usually
   sizeCounters = Array<int>(4);
   for (int i = 0; i < sizeCounters.Size(); ++i) { sizeCounters[i] = 0; }

   std::unordered_map<int, int> reps_to_groups;
   int smallest_unused = 0;
   elems.SetSize(parent.Size());

   bounds.Append(0);
   int largestGroup = 0;
   for (int i = 0; i < parent.Size(); ++i)
   {
      elems[i] = -1;

      // Assign groups a unique id starting at 0
      int rep = Find(i);
      if (reps_to_groups.count(rep)) { continue; }

      reps_to_groups[rep] = smallest_unused;
      bounds.Append(bounds.Last()+size[rep]);
      if (size[rep] >= largestGroup)
      {
         sizeCounters.SetSize(size[rep]+1,1);
         largestGroup = size[rep]+1;
      }
      else
      {
         ++sizeCounters[size[rep]];
      }
      smallest_unused++;
   }
   // since bounds is an "interior" array we need to add 1 more of each to get the true size
   for (int i = 1; i < largestGroup; ++i) { sizeCounters[i] += i; }
   // in case it was smaller than initial
   sizeCounters.SetSize(largestGroup);

   // Assemble the elems array
   for (int i = 0; i < parent.Size(); ++i)
   {
      int group = reps_to_groups[Find(i)];
      for (int j = bounds[group]; j < bounds[group+1]; ++j)
      {
         if (elems[j] == -1)
         {
            elems[j] = i;
            break;
         }
      }
   }

   elem_to_group = Array<int>(elems.Size());
   for (int group = 0; group < bounds.Size()-1; ++group)
   {
      for (int i = bounds[group]; i < bounds[group+1]; ++i)
      {
         elem_to_group[elems[i]] = group;
      }
   }
}

const Array<int> &DisjointSets::GetBounds() const
{
   MFEM_VERIFY(finalized, "DisjointSets must be finalized");
   return bounds;
}

const Array<int> &DisjointSets::GetElems() const
{
   MFEM_VERIFY(finalized, "DisjointSets must be finalized");
   return elems;
}

const Array<int> &DisjointSets::GetSizeCounter() const
{
   MFEM_VERIFY(finalized, "DisjointSets must be finalized");
   return sizeCounters;
}

void DisjointSets::Print(std::ostream& out) const
{
   MFEM_VERIFY(finalized, "DisjointSets must be finalized");

   bool first_set = true;
   for (int group = 0; group < bounds.Size()-1; ++group)
   {
      int size = bounds[group+1] - bounds[group];

      bool first_elem = true;
      out << "{";
      for (int k = bounds[group]; k < bounds[group] + size; ++k)
      {
         int i = elems[k];
         out << (first_elem ? "" : ", ") << i;
         first_elem = false;
      }
      out << "}" << std::endl;
      first_set = false;
   }
   out << std::endl;
}

void DisjointSets::RemoveLargerThan(int max)
{
   finalized = false;

   Array<int> to_make_single;
   for (int i = 0; i < parent.Size(); ++i)
   {
      int j = Find(i);
      if (size[j] > max) { to_make_single.Append(i); }
   }
   MakeSingle(to_make_single);
}

void DisjointSets::RemoveSmallerThan(int min)
{
   finalized = false;

   Array<int> to_make_single;
   for (int i = 0; i < parent.Size(); ++i)
   {
      int j = Find(i);
      if (size[j] < min) { to_make_single.Append(i); }
   }
   MakeSingle(to_make_single);
}

void DisjointSets::BreakLargerThan(int max)
{
   Finalize();
   finalized = false;

   RemoveLargerThan(max);

   for (int group = 0; group < bounds.Size()-1; ++group)
   {
      const int lower = bounds[group];
      const int size  = bounds[group+1] - lower;
      for (int l = 0; l < size; ++l)
      {
         const int i = lower+l;
         if (l % max != 0) { Union(elems[i], elems[i-l%max]); }
      }
   }
}

}
