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

#include <vector>
#include <utility> // for std::pair

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


/// Default hash class for use with class Set.
/** Assuming that, for t of type T and n of type int, (t % n) is well defined
    and convertible to int. Also, it is assumed that 0 <= (t % n) < n. */
template <typename T>
class THash
{
protected:
   int num_bins, next_num_bins;
   // std::hash<T> fn;

   // Define the bin-size growth sequence
   void SetNext()
   {
      // need: next_num_bins > num_bins
      next_num_bins = 2*num_bins;

      // sequence: 2 -> 3 -> 6 -> 9 -> 15 -> 24 -> 36 -> 54 -> 81 -> 123 ...
      // next_num_bins = 3*((num_bins+1)/2);

      // sequence: 2 -> 7 -> 14 -> 28 -> 49 -> 84 -> 147 ...
      // next_num_bins = 7*((num_bins+2)/4);

      MFEM_ASSERT(next_num_bins > num_bins, "invalid sequence");
   }

public:
   THash(int number_of_bins) : num_bins(number_of_bins) { SetNext(); }

   void Reset(int number_of_bins) { num_bins = number_of_bins; SetNext(); }

   int NumBins() const { return num_bins; }

   // The binning/hashing function.
   int GetBin(const T &element) const { return element % num_bins; }
   // int GetBin(const T &element) const { return fn(element) % num_bins; }

   bool UpdateNumBins(int num_elements)
   {
      if (num_elements <= next_num_bins) { return false; }
      else { num_bins = next_num_bins; SetNext(); return true; }
   }
};


/// Adaptor class for class Array for use in class Set.
template <typename T>
class ArraySetAdaptor : public Array<T>
{
protected:
   typedef Array<T> base;

public:
   // int Size() const;

   // T &operator[](int);
   // const T &operator[](int) const;

   int Append() { int n = base::Size(); base::SetSize(n+1); return n; }

   // int Append(const T &item);

   void Pop(T &item) { item = base::Last(); base::DeleteLast(); }

   // void Reserve(int);

   // void DeleteAll();

   // long MemoryUsage() const;
};


/// Adaptor class for class std::vector for use in class Set.
template <typename T>
class StdVectorSetAdaptor : public std::vector<T>
{
protected:
   typedef std::vector<T> base;

public:
   int Size() const { return base::size(); }

   // T &operator[](int);
   // const T &operator[](int) const;

   int Append() { int n = Size(); base::resize(n+1); return n; }

   int Append(const T &item)
   { int n = Size(); base::push_back(item); return n; }

   void Pop(T &item) { item = base::back(); base::pop_back(); }

   void Reserve(int capacity) { base::reserve(capacity); }

   void DeleteAll() { base a; base::swap(a); }

   long MemoryUsage() const { return base::capacity()*sizeof(T); }
};


#if 0
/// Adaptor class for class BlockArray for use in class Set.
template <typename T>
class BlockArraySetAdaptor : public BlockArray<T>
{
protected:
   typedef BlockArray<T> base;

public:
   // int Size() const

   // T &operator[](int);
   // const T &operator[](int) const;

   int Append() { return base::New(); }

   int Append(const T &item) { return base::New(item); }

   void Pop(T &item)
   { int n = base::Size(); item = base::At(n-1); base::Delete(n-1); }

   void Reserve(int capacity) { MFEM_ABORT("TODO"); /* TODO */ }

   void DeleteAll() { base a; base::Swap(a); }

   // long MemoryUsage() const;
};
#endif


/// Auxiliary function to print Set stats.
void Set_PrintStats(std::ostream &out, int size, int idx_size, int num_bins,
                    Array<int> &bin_sz_cnt, long mem_usage);


/** A set of elements, allowing fast search, insertion and removal of
    elements. Uses a hash function to distribute the elements into bins. */
// NodeIdxArray<T> needs to implement (used with T = int):
//   * int Size() const;
//   * void Pop(T &item);
//   * void Append(const T &item);
//   * void DeleteAll();
//   * long MemoryUsage() const;
// NodeArray<T> needs to implement (used with T = Node):
//   * int Size() const;
//   * T &operator[](int)
//   * const T &operator[](int) const;
//   * int Append();
//   * void Reserve(int);
//   * void DeleteAll();
//   * long MemoryUsage() const;
template <typename T,
          template <typename> class Hash = THash,
          template <typename> class NodeArray = ArraySetAdaptor,
          template <typename> class NodeIdxArray = ArraySetAdaptor>
class Set
{
public:
   typedef T value_type;

protected:
   // bins[i] holds the first node index for bin i, or -1 if the bin is empty
   int *bins;
   Hash<T> hash;

   // The data is stored in an array of Node. It should be easy to extend this
   // class to support storing the elements and the indices in separate arrays.
   // It may be worth testing both options in different contexts.
   struct Node
   {
      value_type element;  // the stored element
      int next_node_idx;   /* an index into 'nodes', or -1 for end of list,
                              or -2 to mark the node as not used */
   };

   NodeArray<Node> nodes;
   NodeIdxArray<int> unused_nodes; // used as a stack

   void UpdateBins()
   {
      const int num_bins = NumBins();
      // std::cout << "Updating number of bins to " << num_bins << std::endl;
      delete [] bins;
      bins = new int[num_bins];
      for (int i = 0; i < num_bins; i++) { bins[i] = -1; }
      const int idx_size = IndexSize();
      for (int node_idx = 0; node_idx < idx_size; node_idx++)
      {
         Node &node = nodes[node_idx];
         // skip unused nodes
         if (node.next_node_idx == -2) { continue; }
         // insert the node at the front of the new bin list
         int bin_idx = hash.GetBin(node.element);
         node.next_node_idx = bins[bin_idx];
         bins[bin_idx] = node_idx;
      }
   }

public:
   /** @brief Create an empty set with at initial number of bins equal to
       max(2, @a number_of_bins). */
   /** If the maximal number of elements in the set is know, then setting the
       number of bins to be close to that number can speedup element insertion.
    */
   Set(int number_of_bins = 2)
      : hash(std::max(2, number_of_bins))
   {
      bins = new int[NumBins()];
      for (int i = 0; i < NumBins(); i++) { bins[i] = -1; }
   }

   /// Destructor.
   ~Set() { delete [] bins; }

   /// Return the index of @a element, or -1 if @a element is not in the set.
   int Find(const value_type &element) const
   {
      int node_idx = bins[hash.GetBin(element)];
      while (node_idx != -1)
      {
         const Node &node = nodes[node_idx];
         if (node.element == element) { return node_idx; }
         node_idx = node.next_node_idx;
      }
      return -1;
   }

   /// Insert @a element in the set.
   /** @returns An std::pair<int,bool> where the first entry is the index
       assigned to @a element and the second entry is true iff @a element was
       not in the set before this call. */
   std::pair<int,bool> Insert(const value_type &element)
   {
      const int bin_idx = hash.GetBin(element);
      int node_idx = bins[bin_idx];
      MFEM_DEBUG_DO(int bin_size = 0;)
      while (node_idx != -1)
      {
         const Node &node = nodes[node_idx];
         if (node.element == element)
         {
            return std::pair<int,bool>(node_idx, false);
         }
         node_idx = node.next_node_idx;
         MFEM_DEBUG_DO(bin_size++;)
      }
      MFEM_DEBUG_DO(
      if (bin_size == 16) { MFEM_WARNING("exceeding bin size of 16"); })
      // add the new element to the front of the bin list
      if (unused_nodes.Size())
      {
         unused_nodes.Pop(node_idx);
      }
      else
      {
         node_idx = nodes.Append();
      }
      Node &node = nodes[node_idx];
      node.element = element;
      node.next_node_idx = bins[bin_idx];
      bins[bin_idx] = node_idx;
      // check if the number of bins needs to be updated
      if (hash.UpdateNumBins(Size())) { UpdateBins(); }
      return std::pair<int,bool>(node_idx, true);
   }

   /// Remove @a element from the set.
   /** @returns The index that @a element had, or -1 if @a element was not in
       the set before this call.
       @note Indices of removed elements will be reused by new elements added to
       the set. */
   int Remove(const value_type &element)
   {
      int *node_idx_ptr = &bins[hash.GetBin(element)];
      int node_idx;
      while ((node_idx = *node_idx_ptr) != -1)
      {
         Node &node = nodes[node_idx];
         if (node.element == element)
         {
            *node_idx_ptr = node.next_node_idx;
            node.next_node_idx = -2; // mark the node as unused
            // push the node_idx to the unused node stack
            unused_nodes.Append(node_idx);
            // check if the number of bins needs to be updated
            if (hash.UpdateNumBins(Size())) { UpdateBins(); }
            return node_idx;
         }
         node_idx_ptr = &node.next_node_idx;
      }
      return -1;
   }

   /// Change the number of bins. Return true iff the actual number changed.
   bool Reset(int number_of_bins)
   {
      int num_bins = NumBins();
      hash.Reset(std::max(2, number_of_bins));
      if (num_bins != NumBins()) { UpdateBins(); return true; }
      return false;
   }

   /** @brief Optimize the set for faster access by setting the number of bins
       equal to the number of elements. */
   bool Optimize() { return Reset(Size()); }

   /// Reserve (pre-allocate) memory for at least @a capacity number of entries.
   /** Use this call right after construction to optimize the speed and memory
       usage, when the miximal size of the set is known in advance.

       This call does not affect the number of bins. */
   void Reserve(int capacity) { nodes.Reserve(capacity); }

   /// Return the number of elements in the set.
   int Size() const { return IndexSize()-unused_nodes.Size(); }

   /// Return the current number of bins.
   int NumBins() const { return hash.NumBins(); }

   /// Return the number of indices used, i.e. the maximal used set size.
   /** This method can be used together with CheckIndex() to iterate over the
       contents of the set:
       \code
       for (int idx = 0; idx < s.IndexSize(); idx++)
       {
          if (!s.CheckIndex(idx)) { continue; }
          // use s[idx] to access the elements ...
       }
       \endcode
    */
   int IndexSize() const { return nodes.Size(); }

   /// Given an index in the range [0,IndexSize), check its validity.
   /** @sa IndexSize(). */
   bool CheckIndex(int idx) const { return nodes[idx].next_node_idx != -2; }

   /// Access an entry in the set by its index.
   /** @note Modifying the returned element will, generally, invalidate its bin
       index and its list. It may also create duplicate elements. To safely
       replace an element, use Remove() followed by Insert(). */
   value_type &operator[](int idx) { return nodes[idx].element; }

   /// Constant access an entry in the set by its index.
   const value_type &operator[](int idx) const { return nodes[idx].element; }

   /** @brief Return the first element index for bin with index @a bin_idx,
       or -1 if the bin is empty. */
   /** This method can be used together with GetNextIndex() to iterate over the
       contents of the set, bin by bin:
       \code
       Set s;
       // ...
       for (int bin_idx = 0; bin_idx < s.NumBins(); bin_idx++)
       {
          for (int idx = s.GetBinStart(bin_idx); idx != -1;
               idx = s.GetNextIndex(idx))
          {
             // use s[idx] to access the elements ...
          }
       }
       \endcode
       These methods are provided as an alternative to the simpler (and probably
       faster) loop pattern, shown in IndexSize().
    */
   int GetBinStart(int bin_idx) const
   {
      MFEM_ASSERT(0 <= bin_idx && bin_idx < NumBins(),
                  "invalid bin index: " << bin_idx
                  << ", number of bins: " << NumBins());
      return bins[bin_idx];
   }

   /** @brief Get the index of the next element in the same bin as element with
       index @a idx, or -1 if there are no more elements in the same bin. */
   /** @sa GetBinStart(). */
   int GetNextIndex(int idx) const { return nodes[idx].next_node_index; }

   /// Empty the contents of the set. The number of bins is reduced to 2.
   void Clear()
   {
      nodes.DeleteAll();
      unused_nodes.DeleteAll();
      hash.Reset(2);
      UpdateBins();
   }

   /// Return the memory (in bytes) used by the set, not counting sizeof(Set).
   long MemoryUsage() const
   {
      // Note: not counting any dynamic memory used by 'hash'.
      return nodes.MemoryUsage() +
             unused_nodes.MemoryUsage() +
             NumBins()*sizeof(int);
   }

   /// Print statistics for the set.
   void PrintStats(std::ostream &out = std::cout) const
   {
      Array<int> bin_sz_cnt;
      int max_bin_size = 0;
      for (int bin_idx = 0; bin_idx < NumBins(); bin_idx++)
      {
         int bin_size = 0;
         int idx = bins[bin_idx];
         while (idx != -1)
         {
            bin_size++;
            idx = nodes[idx].next_node_idx;
         }
         max_bin_size = std::max(max_bin_size, bin_size);
         bin_sz_cnt.SetSize(max_bin_size+1, 0);
         bin_sz_cnt[bin_size]++;
      }
      Set_PrintStats(out, Size(), IndexSize(), NumBins(), bin_sz_cnt,
                     MemoryUsage());
   }
};

} // namespace mfem

#endif
