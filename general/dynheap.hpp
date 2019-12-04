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

#ifndef MFEM_DYNHEAP
#define MFEM_DYNHEAP

#include <vector>
#include <map>

namespace mfem
{

template <
   typename T,                         // data type
   typename P,                         // priority type
   class    C = std::less<P>,          // comparator for priorities
   class    M = std::map<T, unsigned>  // maps type T to unsigned integer
   >
class DynamicHeap
{
public:
   DynamicHeap(size_t count = 0);
   ~DynamicHeap() {}

   void Insert(T data, P priority);
   void Update(T data, P priority);

   bool Top(T& data);
   bool Top(T& data, P& priority);

   bool Pop();
   bool Extract(T& data);
   bool Extract(T& data, P& priority);
   bool Erase(T data);

   bool Find(T data) const;
   bool Find(T data, P& priority) const;

   bool Empty() const { return heap.empty(); }
   size_t Size() const { return heap.size(); }

private:
   struct HeapEntry
   {
      HeapEntry(P p, T d) : priority(p), data(d) {}
      P priority;
      T data;
   };
   std::vector<HeapEntry> heap;

   M index;
   C lower;

   void Ascend(unsigned i);
   void Descend(unsigned i);
   void Swap(unsigned i, unsigned j);

   bool Ordered(unsigned i, unsigned j) const
   { return !lower(heap[i].priority, heap[j].priority); }

   unsigned Parent(unsigned i) const { return (i - 1) / 2; }
   unsigned Left(unsigned i) const { return 2 * i + 1; }
   unsigned Right(unsigned i) const { return 2 * i + 2; }
};


// inline methods

template < typename T, typename P, class C, class M >
DynamicHeap<T, P, C, M>::DynamicHeap(size_t count)
{
   heap.reserve(count);
}

template < typename T, typename P, class C, class M >
void DynamicHeap<T, P, C, M>::Insert(T data, P priority)
{
   if (index.find(data) != index.end())
   {
      Update(data, priority);
   }
   else
   {
      unsigned i = heap.size();
      heap.push_back(HeapEntry(priority, data));
      Ascend(i);
   }
}

template < typename T, typename P, class C, class M >
void DynamicHeap<T, P, C, M>::Update(T data, P priority)
{
   unsigned i = index[data];
   heap[i].priority = priority;
   Ascend(i);
   Descend(i);
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Top(T& data)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Top(T& data, P& priority)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      priority = heap[0].priority;
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Pop()
{
   if (!heap.empty())
   {
      T data = heap[0].data;
      Swap(0, heap.size() - 1);
      index.erase(data);
      heap.pop_back();
      if (!heap.empty())
      {
         Descend(0);
      }
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Extract(T& data)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      return Pop();
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Extract(T& data, P& priority)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      priority = heap[0].priority;
      return Pop();
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Erase(T data)
{
   if (index.find(data) == index.end())
   {
      return false;
   }
   unsigned i = index[data];
   Swap(i, heap.size() - 1);
   index.erase(data);
   heap.pop_back();
   if (i < heap.size())
   {
      Ascend(i);
      Descend(i);
   }
   return true;
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Find(T data) const
{
   return index.find(data) != index.end();
}

template < typename T, typename P, class C, class M >
bool DynamicHeap<T, P, C, M>::Find(T data, P& priority) const
{
   typename M::const_iterator p;
   if ((p = index.find(data)) == index.end())
   {
      return false;
   }
   unsigned i = p->second;
   priority = heap[i].priority;
   return true;
}

template < typename T, typename P, class C, class M >
void DynamicHeap<T, P, C, M>::Ascend(unsigned i)
{
   for (unsigned j; i && !Ordered(j = Parent(i), i); i = j)
   {
      Swap(i, j);
   }
   index[heap[i].data] = i;
}

template < typename T, typename P, class C, class M >
void DynamicHeap<T, P, C, M>::Descend(unsigned i)
{
   for (unsigned j, k;
        (j = ((k =  Left(i)) < heap.size() && !Ordered(i, k) ? k : i),
         j = ((k = Right(i)) < heap.size() && !Ordered(j, k) ? k : j)) != i;
        i = j)
   {
      Swap(i, j);
   }
   index[heap[i].data] = i;
}

template < typename T, typename P, class C, class M >
void DynamicHeap<T, P, C, M>::Swap(unsigned i, unsigned j)
{
   std::swap(heap[i], heap[j]);
   index[heap[i].data] = i;
}


} // namespace mfem

#endif // MFEM_DYNHEAP
