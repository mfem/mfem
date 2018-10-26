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

#ifndef MFEM_HASH
#define MFEM_HASH

#include "../config/config.hpp"
#include "array.hpp"
#include "globals.hpp"

namespace mfem
{

/** A concept for items that should be used in HashTable and be accessible by
 *  hashing two IDs.
 */
struct Hashed2
{
   int p1, p2;
   int next;
};

/** A concept for items that should be used in HashTable and be accessible by
 *  hashing four IDs.
 */
struct Hashed4
{
   int p1, p2, p3; // NOTE: p4 is neither hashed nor stored
   int next;
};


/** HashTable is a container for items that require associative access through
 *  pairs (or quadruples) of indices:
 *
 *    (p1, p2) -> item
 *    (p1, p2, p3, p4) -> item
 *
 *  An example of this are edges and faces in a mesh. Each edge is uniquely
 *  identified by two parent vertices and so can be easily accessed from
 *  different elements using this class. Similarly for faces.
 *
 *  The order of the p1, p2, ... indices is not relevant as they are sorted
 *  each time this class is invoked.
 *
 *  There are two main methods this class provides. The Get(...) methods always
 *  return an item given the two or four indices. If the item didn't previously
 *  exist, the methods creates a new one. The Find(...) methods, on the other
 *  hand, just return NULL or -1 if the item doesn't exist.
 *
 *  Each new item is automatically assigned a unique ID - the index of the item
 *  inside the BlockArray. The IDs may (but need not) be used as p1, p2, ... of
 *  other items.
 *
 *  The item type (T) needs to follow either the Hashed2 or the Hashed4
 *  concept. It is easiest to just inherit from these structs.
 *
 *  All items in the container can also be accessed sequentially using the
 *  provided iterator.
 */
template<typename T>
class HashTable : public BlockArray<T>
{
protected:
   typedef BlockArray<T> Base;

public:
   HashTable(int block_size = 16*1024, int init_hash_size = 32*1024);
   HashTable(const HashTable& other); // deep copy
   ~HashTable();

   /// Get item whose parents are p1, p2... Create it if it doesn't exist.
   T* Get(int p1, int p2);
   T* Get(int p1, int p2, int p3, int p4);

   /// Get id of item whose parents are p1, p2... Create it if it doesn't exist.
   int GetId(int p1, int p2);
   int GetId(int p1, int p2, int p3, int p4);

   /// Find item whose parents are p1, p2... Return NULL if it doesn't exist.
   T* Find(int p1, int p2);
   T* Find(int p1, int p2, int p3, int p4);

   const T* Find(int p1, int p2) const;
   const T* Find(int p1, int p2, int p3, int p4) const;

   /// Find id of item whose parents are p1, p2... Return -1 if it doesn't exist.
   int FindId(int p1, int p2) const;
   int FindId(int p1, int p2, int p3, int p4) const;

   /// Return the number of elements currently stored in the HashTable.
   int Size() const { return Base::Size() - unused.Size(); }

   /// Return the total number of ids (used and unused) in the HashTable.
   int NumIds() const { return Base::Size(); }

   /// Return the number of free/unused ids in the HashTable.
   int NumFreeIds() const { return unused.Size(); }

   /// Return true if item 'id' exists in (is used by) the container.
   /** It is assumed that 0 <= id < NumIds(). */
   bool IdExists(int id) const { return (Base::At(id).next != -2); }

   /// Remove an item from the hash table.
   /** Its id will be reused by newly added items. */
   void Delete(int id);

   /// Make an item hashed under different parent IDs.
   void Reparent(int id, int new_p1, int new_p2);
   void Reparent(int id, int new_p1, int new_p2, int new_p3, int new_p4);

   /// Return total size of allocated memory (tables plus items), in bytes.
   long MemoryUsage() const;

   void PrintMemoryDetail() const;

   class iterator : public Base::iterator
   {
   protected:
      friend class HashTable;
      typedef typename Base::iterator base;

      iterator() { }
      iterator(const base &it) : base(it)
      {
         while (base::good() && (*this)->next == -2) { base::next(); }
      }

   public:
      iterator &operator++()
      {
         while (base::next(), base::good() && (*this)->next == -2) { }
         return *this;
      }
   };

   class const_iterator : public Base::const_iterator
   {
   protected:
      friend class HashTable;
      typedef typename Base::const_iterator base;

      const_iterator() { }
      const_iterator(const base &it) : base(it)
      {
         while (base::good() && (*this)->next == -2) { base::next(); }
      }

   public:
      const_iterator &operator++()
      {
         while (base::next(), base::good() && (*this)->next == -2) { }
         return *this;
      }
   };

   iterator begin() { return iterator(Base::begin()); }
   iterator end() { return iterator(); }

   const_iterator cbegin() const { return const_iterator(Base::cbegin()); }
   const_iterator cend() const { return const_iterator(); }

protected:
   int* table;
   int mask;
   Array<int> unused;

   // hash functions (NOTE: the constants are arbitrary)
   inline int Hash(int p1, int p2) const
   { return (984120265*p1 + 125965121*p2) & mask; }

   inline int Hash(int p1, int p2, int p3) const
   { return (984120265*p1 + 125965121*p2 + 495698413*p3) & mask; }

   // Delete() and Reparent() use one of these:
   inline int Hash(const Hashed2& item) const
   { return Hash(item.p1, item.p2); }

   inline int Hash(const Hashed4& item) const
   { return Hash(item.p1, item.p2, item.p3); }

   int SearchList(int id, int p1, int p2) const;
   int SearchList(int id, int p1, int p2, int p3) const;

   inline void Insert(int idx, int id, T &item);
   void Unlink(int idx, int id);

   /// Check table load factor and resize if necessary
   inline void CheckRehash();
   void DoRehash();
};


// implementation

template<typename T>
HashTable<T>::HashTable(int block_size, int init_hash_size)
   : Base(block_size)
{
   mask = init_hash_size-1;
   MFEM_VERIFY(!(init_hash_size & mask), "init_size must be a power of two.");

   table = new int[init_hash_size];
   for (int i = 0; i < init_hash_size; i++) { table[i] = -1; }
}

template<typename T>
HashTable<T>::HashTable(const HashTable& other)
   : Base(other), mask(other.mask)
{
   int size = mask+1;
   table = new int[size];
   memcpy(table, other.table, size*sizeof(int));
   other.unused.Copy(unused);
}

template<typename T>
HashTable<T>::~HashTable()
{
   delete [] table;
}

namespace internal
{

inline void sort3(int &a, int &b, int &c)
{
   if (a > b) { std::swap(a, b); }
   if (a > c) { std::swap(a, c); }
   if (b > c) { std::swap(b, c); }
}

inline void sort4(int &a, int &b, int &c, int &d)
{
   if (a > b) { std::swap(a, b); }
   if (a > c) { std::swap(a, c); }
   if (a > d) { std::swap(a, d); }
   sort3(b, c, d);
}

} // internal

template<typename T>
inline T* HashTable<T>::Get(int p1, int p2)
{
   return &(Base::At(GetId(p1, p2)));
}

template<typename T>
inline T* HashTable<T>::Get(int p1, int p2, int p3, int p4)
{
   return &(Base::At(GetId(p1, p2, p3, p4)));
}

template<typename T>
int HashTable<T>::GetId(int p1, int p2)
{
   // search for the item in the hashtable
   if (p1 > p2) { std::swap(p1, p2); }
   int idx = Hash(p1, p2);
   int id = SearchList(table[idx], p1, p2);
   if (id >= 0) { return id; }

   // not found - use an unused item or create a new one
   int new_id;
   if (unused.Size())
   {
      new_id = unused.Last();
      unused.DeleteLast();
   }
   else
   {
      new_id = Base::Append();
   }
   T& item = Base::At(new_id);
   item.p1 = p1;
   item.p2 = p2;

   // insert into hashtable
   Insert(idx, new_id, item);
   CheckRehash();

   return new_id;
}

template<typename T>
int HashTable<T>::GetId(int p1, int p2, int p3, int p4)
{
   // search for the item in the hashtable
   internal::sort4(p1, p2, p3, p4);
   int idx = Hash(p1, p2, p3);
   int id = SearchList(table[idx], p1, p2, p3);
   if (id >= 0) { return id; }

   // not found - use an unused item or create a new one
   int new_id;
   if (unused.Size())
   {
      new_id = unused.Last();
      unused.DeleteLast();
   }
   else
   {
      new_id = Base::Append();
   }
   T& item = Base::At(new_id);
   item.p1 = p1;
   item.p2 = p2;
   item.p3 = p3;

   // insert into hashtable
   Insert(idx, new_id, item);
   CheckRehash();

   return new_id;
}

template<typename T>
inline T* HashTable<T>::Find(int p1, int p2)
{
   int id = FindId(p1, p2);
   return (id >= 0) ? &(Base::At(id)) : NULL;
}

template<typename T>
inline T* HashTable<T>::Find(int p1, int p2, int p3, int p4)
{
   int id = FindId(p1, p2, p3, p4);
   return (id >= 0) ? &(Base::At(id)) : NULL;
}

template<typename T>
inline const T* HashTable<T>::Find(int p1, int p2) const
{
   int id = FindId(p1, p2);
   return (id >= 0) ? &(Base::At(id)) : NULL;
}

template<typename T>
inline const T* HashTable<T>::Find(int p1, int p2, int p3, int p4) const
{
   int id = FindId(p1, p2, p3, p4);
   return (id >= 0) ? &(Base::At(id)) : NULL;
}

template<typename T>
int HashTable<T>::FindId(int p1, int p2) const
{
   if (p1 > p2) { std::swap(p1, p2); }
   return SearchList(table[Hash(p1, p2)], p1, p2);
}

template<typename T>
int HashTable<T>::FindId(int p1, int p2, int p3, int p4) const
{
   internal::sort4(p1, p2, p3, p4);
   return SearchList(table[Hash(p1, p2, p3)], p1, p2, p3);
}

template<typename T>
int HashTable<T>::SearchList(int id, int p1, int p2) const
{
   while (id >= 0)
   {
      const T& item = Base::At(id);
      if (item.p1 == p1 && item.p2 == p2) { return id; }
      id = item.next;
   }
   return -1;
}

template<typename T>
int HashTable<T>::SearchList(int id, int p1, int p2, int p3) const
{
   while (id >= 0)
   {
      const T& item = Base::At(id);
      if (item.p1 == p1 && item.p2 == p2 && item.p3 == p3) { return id; }
      id = item.next;
   }
   return -1;
}

template<typename T>
inline void HashTable<T>::CheckRehash()
{
   const int fill_factor = 2;

   // is the table overfull?
   if (Base::Size() > (mask+1) * fill_factor)
   {
      DoRehash();
   }
}

template<typename T>
void HashTable<T>::DoRehash()
{
   delete [] table;

   // double the table size
   int new_table_size = 2*(mask+1);
   table = new int[new_table_size];
   for (int i = 0; i < new_table_size; i++) { table[i] = -1; }
   mask = new_table_size-1;

#if defined(MFEM_DEBUG) && !defined(MFEM_USE_MPI)
   mfem::out << _MFEM_FUNC_NAME << ": rehashing to size " << new_table_size
             << std::endl;
#endif

   // reinsert all items
   for (iterator it = begin(); it != end(); ++it)
   {
      Insert(Hash(*it), it.index(), *it);
   }
}

template<typename T>
inline void HashTable<T>::Insert(int idx, int id, T &item)
{
   // add item at the beginning of the linked list
   item.next = table[idx];
   table[idx] = id;
}

template<typename T>
void HashTable<T>::Unlink(int idx, int id)
{
   // remove item from the linked list
   int* p_id = table + idx;
   while (*p_id >= 0)
   {
      T& item = Base::At(*p_id);
      if (*p_id == id)
      {
         *p_id = item.next;
         return;
      }
      p_id = &(item.next);
   }
   MFEM_ABORT("HashTable<>::Unlink: item not found!");
}

template<typename T>
void HashTable<T>::Delete(int id)
{
   T& item = Base::At(id);
   Unlink(Hash(item), id);
   item.next = -2;    // mark item as unused
   unused.Append(id); // add its id to the unused ids
}

template<typename T>
void HashTable<T>::Reparent(int id, int new_p1, int new_p2)
{
   T& item = Base::At(id);
   Unlink(Hash(item), id);

   if (new_p1 > new_p2) { std::swap(new_p1, new_p2); }
   item.p1 = new_p1;
   item.p2 = new_p2;

   // reinsert under new parent IDs
   int new_idx = Hash(new_p1, new_p2);
   Insert(new_idx, id, item);
}

template<typename T>
void HashTable<T>::Reparent(int id,
                            int new_p1, int new_p2, int new_p3, int new_p4)
{
   T& item = Base::At(id);
   Unlink(Hash(item), id);

   internal::sort4(new_p1, new_p2, new_p3, new_p4);
   item.p1 = new_p1;
   item.p2 = new_p2;
   item.p3 = new_p3;

   // reinsert under new parent IDs
   int new_idx = Hash(new_p1, new_p2, new_p3);
   Insert(new_idx, id, item);
}

template<typename T>
long HashTable<T>::MemoryUsage() const
{
   return (mask+1) * sizeof(int) + Base::MemoryUsage() + unused.MemoryUsage();
}

template<typename T>
void HashTable<T>::PrintMemoryDetail() const
{
   mfem::out << Base::MemoryUsage() << " + " << (mask+1) * sizeof(int)
             << " + " << unused.MemoryUsage();
}

} // namespace mfem

#endif
