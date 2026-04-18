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

#ifndef MFEM_HASH
#define MFEM_HASH

#include "../config/config.hpp"
#include "array.hpp"
#include "globals.hpp"
#include "hash_util.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

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
 *  return an item given the two or four indices. If the item did not previously
 *  exist, the methods creates a new one. The Find(...) methods, on the other
 *  hand, just return NULL or -1 if the item does not exist.
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
 *
 *  Notes:
 *   The data structure and implementation is based on a BlockArray<T> which
 *   provides an efficient item storage that avoids heap fragmentation, and
 *   index-based item access. The hash table implemented on top of the
 *   BlockArray provides fast associative (key -> value) access by grouping
 *   items into bins (buckets) of O(1) size.
 *   - "id" denotes the index of an item in the underlying BlockArray<T>,
 *   - "idx" denotes the index of a bin, determined by hashing a key with
 *     the function `Hash`.
 */
template<typename T>
class HashTable : public BlockArray<T>
{
protected:
   typedef BlockArray<T> Base;

public:
   /** @brief Main constructor of the HashTable class.

       @param[in] block_size The size of the storage blocks of the underlying
                             BlockArray<T>.
       @param[in] init_hash_size The initial size of the hash table. Must be
                                 a power of 2. */
   HashTable(int block_size = 16*1024, int init_hash_size = 32*1024);
   /// Deep copy
   HashTable(const HashTable& other);
   /// Copy assignment not supported
   HashTable& operator=(const HashTable&) = delete;
   ~HashTable();

   /** @brief Item accessor with key (or parents) the pair p1, p2. Default
       construct an item of type T if no value corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed2. */
   T* Get(int p1, int p2);

   /** @brief Item accessor with key (or parents) the quadruplet p1, p2, p3, p4.
       The key p4 is optional. Default construct an item of type T if no value
       corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @param[in] p4 Fourth part of the key (optional).
       @return The index "id" of the key in the BlockArray<T>.

      @warning This method should only be called if T inherits from Hashed4. */
   T* Get(int p1, int p2, int p3, int p4 = -1 /* p4 optional */);

   /** @brief Get the "id" of the item whose parents are p1, p2, this "id"
       corresponding to the index of the item in the underlying BlockArray<T>
       object. Default construct an item and "id" if no value corresponds to the
       requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed2. */
   int GetId(int p1, int p2);

   /** @brief Get the "id" of an item, this "id" corresponding to the index of
       the item in the underlying BlockArray<T> object. Default construct an item
       and "id" if no value corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @param[in] p4 Fourth part of the key (optional).
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed4. */
   int GetId(int p1, int p2, int p3, int p4 = -1);

   /** @brief Item accessor with key (or parents) the pair p1, p2. Return
       NULL if no value corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The item associated to the key (p1,p2).

       @warning This method should only be called if T inherits from Hashed2. */
   T* Find(int p1, int p2);

   /** @brief Item accessor with key (or parents) the quadruplet p1, p2, p3, p4.
       The key p4 is optional. Return NULL if no value corresponds to the
       requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @param[in] p4 Fourth part of the key (optional).
       @return The item associated to the key (p1,p2,p3,p4).

       @warning This method should only be called if T inherits from Hashed4. */
   T* Find(int p1, int p2, int p3, int p4 = -1);

   /** @brief Item const accessor with key (or parents) the pair p1, p2.
       Return NULL if no value corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The item associated to the key (p1,p2).

      @warning This method should only be called if T inherits from Hashed2. */
   const T* Find(int p1, int p2) const;

   /** @brief Item const accessor with key (or parents) the quadruplet p1, p2,
       p3, p4. The key p4 is optional. Return NULL if no value corresponds to the
       requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @param[in] p4 Fourth part of the key (optional).
       @return The item associated to the key (p1,p2,p3,p4).

       @warning This method should only be called if T inherits from Hashed4. */
   const T* Find(int p1, int p2, int p3, int p4 = -1) const;

   /** @brief Find the "id" of an item whose parents are p1, p2. Return -1 if it
       does not exist.

       This "id" corresponds to the index of the item in the underlying
       BlockArray<T> object. Default construct an item and "id" if no value
       corresponds to the requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed2. */
   int FindId(int p1, int p2) const;

   /** @brief Find the "id" of an item, this "id" corresponding to the index of
       the item in the underlying BlockArray<T> object. Return -1 if it does not
       exist. Default construct an item and "id" if no value corresponds to the
       requested key.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @param[in] p4 Fourth part of the key (optional).
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed4. */
   int FindId(int p1, int p2, int p3, int p4 = -1) const;

   /// Return the number of elements currently stored in the HashTable.
   int Size() const { return Base::Size() - unused.Size(); }

   /// Return the total number of ids (used and unused) in the HashTable.
   int NumIds() const { return Base::Size(); }

   /// Return the number of free/unused ids in the HashTable.
   int NumFreeIds() const { return unused.Size(); }

   /** @brief Return true if item @a id exists in (is used by) the container.

       @param[in] id Index of the item in the underlying BlockArray<T>.

       @warning It is assumed that 0 <= id < NumIds(). */
   bool IdExists(int id) const { return (Base::At(id).next != -2); }

   /** @brief Remove an item from the hash table.

       @param[in] id Index of the item in the underlying BlockArray<T>.

       @warning Its @a id will be reused by newly added items. */
   void Delete(int id);

   /// Remove all items.
   void DeleteAll();

   /** @brief Allocate an item at @a id. Enlarge the underlying BlockArray if
       necessary.

       @param[in] id Index of the item in the underlying BlockArray<T>.
       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.

       @warning This is a special purpose method used when loading data from a
       file. Does nothing if the slot @a id has already been allocated. */
   void Alloc(int id, int p1, int p2);

   /** @brief Reinitialize the internal list of unallocated items.

       @warning This is a special purpose method used when loading data from a file. */
   void UpdateUnused();

   /** @brief Change the key associated with an item.

       In other words, makes an item hashed under different parent IDs.

       @param[in] id Index of the item in the underlying BlockArray<T>.
       @param[in] new_p1 First part of the new key.
       @param[in] new_p2 Second part of the new key.

       @warning This method should only be called if T inherits from Hashed2. */
   void Reparent(int id, int new_p1, int new_p2);

   /** @brief Change the key associated with an item.

       In other words, makes an item hashed under different parent IDs.

       @param[in] id Index of the item in the underlying BlockArray<T>.
       @param[in] new_p1 First part of the new key.
       @param[in] new_p2 Second part of the new key.
       @param[in] new_p3 Third part of the new key.
       @param[in] new_p4 Fourth part of the new key (optional).

       @warning This method should only be called if T inherits from Hashed4. */
   void Reparent(int id, int new_p1, int new_p2, int new_p3, int new_p4 = -1);

   /// Return total size of allocated memory (tables plus items), in bytes.
   std::size_t MemoryUsage() const;

   /// Write details of the memory usage to the mfem output stream.
   void PrintMemoryDetail() const;

   /// Print a histogram of bin sizes for debugging purposes.
   void PrintStats() const;

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
   const_iterator begin() const { return const_iterator(Base::cbegin()); }
   const_iterator end() const { return const_iterator(); }

   const_iterator cbegin() const { return const_iterator(Base::cbegin()); }
   const_iterator cend() const { return const_iterator(); }

protected:
   /** The hash table: each bin is a linked list of items. For each non-empty
       bin, this arrays stores the "id" of the first item in the list, or -1
       if the bin is empty. */
   int* table;

   /** mask = table_size-1. Used for fast modulo operation in Hash(), to wrap
       the raw hashed index around the current table size (which must be a power
       of two). */
   int mask;

   /** List of deleted items in the BlockArray<T>. New items are created with
       these ids first, before they are appended to the block array. */
   Array<int> unused;

   /** @brief hash function for Hashed2 items.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The hash key "idx" identifying a bin/bucket.

       NOTE: the constants are arbitrary
       @warning This method should only be called if T inherits from Hashed2. */
   inline int Hash(size_t p1, size_t p2) const
   { return (984120265ul*p1 + 125965121ul*p2) & mask; }

   /** @brief hash function for Hashed4 items.

       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @return The hash key "idx" identifying a bin/bucket.

       NOTE: The constants are arbitrary.
       NOTE: p4 is not hashed nor stored as p1, p2, p3 identify a face uniquely.
       @warning This method should only be called if T inherits from Hashed4. */
   inline int Hash(size_t p1, size_t p2, size_t p3) const
   { return (984120265ul*p1 + 125965121ul*p2 + 495698413ul*p3) & mask; }

   // Delete() and Reparent() use one of these:
   /// Hash function for items of type T that inherit from Hashed2.
   inline int Hash(const Hashed2& item) const
   { return Hash(item.p1, item.p2); }

   /// Hash function for items of type T that inherit from Hashed4.
   inline int Hash(const Hashed4& item) const
   { return Hash(item.p1, item.p2, item.p3); }

   /** @brief Search the index of the item associated to the key (p1,p2)
       starting from the item with index @a id.

       @param[in] id Index of the item in the underlying BlockArray<T>.
       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed2. */
   int SearchList(int id, int p1, int p2) const;

   /** @brief Search the index of the item associated to the key (p1,p2,p3,(p4))
       starting from the item with index @a id.

       @param[in] id Index of the item in the underlying BlockArray<T>.
       @param[in] p1 First part of the key.
       @param[in] p2 Second part of the key.
       @param[in] p3 Third part of the key.
       @return The index "id" of the key in the BlockArray<T>.

       @warning This method should only be called if T inherits from Hashed4. */
   int SearchList(int id, int p1, int p2, int p3) const;

   /** @brief Insert the item @a id into bin @a idx.

       @param[in] idx The bin/bucket index.
       @param[in] id The index of the item in the BlockArray<T>.
       @param[in] item The item to insert at the beginning of the linked list.

       @warning The method only works with bin @a idx and does not check the
                overall fill factor of the hash table. If appropriate, use
                CheckRehash() for that. */
   inline void Insert(int idx, int id, T &item);

   /** @brief Unlink an item @a id from the linked list of bin @a idx.

       @param[in] idx The bin/bucket index.
       @param[in] id The index of the item in the BlockArray<T>.

       @warning The method aborts if the item is not found. */
   void Unlink(int idx, int id);

   /** @brief Check table fill factor and resize if necessary.

       The method checks the average size of the bins (i.e., the fill factor).
       If the fill factor is > 2, the table is enlarged (see DoRehash()). */
   inline void CheckRehash();

   /** @brief Double the size of the hash table (i.e., double the number of bins)
       and reinsert all items into the new bins.

       NOTE: Rehashing is computationally expensive (O(N) in the number of items),
       but since it is only done rarely (when the number of items doubles), the
       amortized complexity of inserting an item is still O(1). */
   void DoRehash();

   /** @brief Return the size of the bin @a idx.

       @param[in] idx The index of the bin.
       @return The size of the bin. */
   int BinSize(int idx) const;
};

/// Hash function for data sequences.
/** Depends on GnuTLS for SHA-256 hashing. */
class HashFunction
{
protected:
   void *hash_data;

   /// Add a sequence of bytes for hashing
   void HashBuffer(const void *buffer, size_t num_bytes);

   /// Integer encoding method; result is independent of endianness and type
   template <typename int_type_const_iter>
   HashFunction &EncodeAndHashInts(int_type_const_iter begin,
                                   int_type_const_iter end);

   /// Double encoding method: encode in little-endian byte-order
   template <typename double_const_iter>
   HashFunction &EncodeAndHashDoubles(double_const_iter begin,
                                      double_const_iter end);

public:
   /// Default constructor: initialize the hash function
   HashFunction();

   /// Destructor
   ~HashFunction();

   /// Add a sequence of bytes for hashing
   HashFunction &AppendBytes(const void *seq, size_t num_bytes)
   { HashBuffer(seq, num_bytes); return *this; }

   /// Add a sequence of integers for hashing, given as a c-array.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness and type: int, long, unsigned, etc. */
   template <typename int_type>
   HashFunction &AppendInts(const int_type *ints, size_t num_ints)
   { return EncodeAndHashInts(ints, ints + num_ints); }

   /// Add a sequence of integers for hashing, given as a fixed-size c-array.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness and type: int, long, unsigned, etc. */
   template <typename int_type, size_t num_ints>
   HashFunction &AppendInts(const int_type (&ints)[num_ints])
   { return EncodeAndHashInts(ints, ints + num_ints); }

   /// Add a sequence of integers for hashing, given as a container.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness and type: int, long, unsigned, etc. */
   template <typename int_type_container>
   HashFunction &AppendInts(const int_type_container &ints)
   { return EncodeAndHashInts(ints.begin(), ints.end()); }

   /// Add a sequence of doubles for hashing, given as a c-array.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness. */
   HashFunction &AppendDoubles(const real_t *doubles, size_t num_doubles)
   { return EncodeAndHashDoubles(doubles, doubles + num_doubles); }

   /// Add a sequence of doubles for hashing, given as a fixed-size c-array.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness. */
   template <size_t num_doubles>
   HashFunction &AppendDoubles(const real_t (&doubles)[num_doubles])
   { return EncodeAndHashDoubles(doubles, doubles + num_doubles); }

   /// Add a sequence of doubles for hashing, given as a container.
   /** Before hashing the sequence is encoded so that the result is independent
       of endianness. */
   template <typename double_container>
   HashFunction &AppendDoubles(const double_container &doubles)
   { return EncodeAndHashDoubles(doubles.begin(), doubles.end()); }

   /** @brief Return the hash string for the current sequence and reset (clear)
       the sequence. */
   std::string GetHash() const;
};


// implementation

template<typename T>
HashTable<T>::HashTable(int block_size, int init_hash_size)
   : Base(block_size)
{
   mask = init_hash_size-1;
   MFEM_VERIFY(!(init_hash_size & mask), "init_size must be a power of two.");

   table = new int[init_hash_size];
   for (int i = 0; i < init_hash_size; i++)
   {
      table[i] = -1;
   }
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

inline void sort4_ext(int &a, int &b, int &c, int &d)
{
   if (d < 0) // support optional last index
   {
      sort3(a, b, c);
   }
   else
   {
      sort4(a, b, c, d);
   }
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
   internal::sort4_ext(p1, p2, p3, p4);
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
   internal::sort4_ext(p1, p2, p3, p4);
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
void HashTable<T>::DeleteAll()
{
   Base::DeleteAll();
   for (int i = 0; i <= mask; i++) { table[i] = -1; }
   unused.DeleteAll();
}

template<typename T>
void HashTable<T>::Alloc(int id, int p1, int p2)
{
   // enlarge the BlockArray to hold 'id'
   while (id >= Base::Size())
   {
      Base::At(Base::Append()).next = -2; // append "unused" items
   }

   T& item = Base::At(id);
   if (item.next == -2)
   {
      item.next = -1;
      item.p1 = p1;
      item.p2 = p2;

      Insert(Hash(p1, p2), id, item);
      CheckRehash();
   }
}

template<typename T>
void HashTable<T>::UpdateUnused()
{
   unused.DeleteAll();
   for (int i = 0; i < Base::Size(); i++)
   {
      if (Base::At(i).next == -2) { unused.Append(i); }
   }
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

   internal::sort4_ext(new_p1, new_p2, new_p3, new_p4);
   item.p1 = new_p1;
   item.p2 = new_p2;
   item.p3 = new_p3;

   // reinsert under new parent IDs
   int new_idx = Hash(new_p1, new_p2, new_p3);
   Insert(new_idx, id, item);
}

template<typename T>
std::size_t HashTable<T>::MemoryUsage() const
{
   return (mask+1) * sizeof(int) + Base::MemoryUsage() + unused.MemoryUsage();
}

template<typename T>
void HashTable<T>::PrintMemoryDetail() const
{
   mfem::out << Base::MemoryUsage() << " + " << (mask+1) * sizeof(int)
             << " + " << unused.MemoryUsage();
}

template<typename T>
int HashTable<T>::BinSize(int idx) const
{
   int count = 0;
   int id = table[idx];
   while (id >= 0)
   {
      const T& item = Base::At(id);
      id = item.next;
      count++;
   }
   return count;
}

template<typename T>
void HashTable<T>::PrintStats() const
{
   int table_size = mask+1;
   mfem::out << "Hash table size: " << table_size << "\n";
   mfem::out << "Item count: " << Size() << "\n";
   mfem::out << "BlockArray size: " << Base::Size() << "\n";

   const int H = 16;
   int hist[H];

   for (int i = 0; i < H; i++) { hist[i] = 0; }

   for (int i = 0; i < table_size; i++)
   {
      int bs = BinSize(i);
      if (bs >= H) { bs = H-1; }
      hist[bs]++;
   }

   mfem::out << "Bin size histogram:\n";
   for (int i = 0; i < H; i++)
   {
      mfem::out << "  size " << i << ": "
                << hist[i] << " bins" << std::endl;
   }
}


template <typename int_type_const_iter>
HashFunction &HashFunction::EncodeAndHashInts(int_type_const_iter begin,
                                              int_type_const_iter end)
{
   // For hashing, an integer k is encoded as follows:
   // * 1 byte = sign_bit(k) + num_bytes(k), where
   //   - sign_bit(k) = (k >= 0) ? 0 : 128
   //   - num_bytes(k) = minimum number of bytes needed to represent abs(k)
   //     with the convention that num_bytes(0) = 0.
   // * num_bytes(k) bytes = the bytes of abs(k), starting with the least
   //   significant byte.

   static_assert(
      std::is_integral<
      /**/ typename std::remove_reference<decltype(*begin)>::type
      /**/ >::value,
      "invalid iterator type");

   // Skip encoding if hashing is not available:
   if (hash_data == nullptr) { return *this; }

   constexpr int max_buffer_bytes = 64*1024;
   unsigned char buffer[max_buffer_bytes];
   int buffer_counter = 0;
   while (begin != end)
   {
      int byte_counter = 0;
      auto k = *begin;
      buffer[buffer_counter] = (k >= 0) ? 0 : (k = -k, 128);
      while (k != 0)
      {
         byte_counter++;
         buffer[buffer_counter + byte_counter] = (unsigned char)(k % 256);
         k /= 256; // (k >>= 8) results in error, e.g. for 'char'
      }
      buffer[buffer_counter] |= byte_counter;
      buffer_counter += (byte_counter + 1);

      ++begin;

      if (begin == end ||
          buffer_counter + (1 + sizeof(*begin)) > max_buffer_bytes)
      {
         HashBuffer(buffer, buffer_counter);
         buffer_counter = 0;
      }
   }
   return *this;
}

template <typename double_const_iter>
HashFunction &HashFunction::EncodeAndHashDoubles(double_const_iter begin,
                                                 double_const_iter end)
{
   // For hashing, a double is encoded in little endian byte-order.

   static_assert(
      std::is_same<decltype(*begin), const real_t &>::value,
      "invalid iterator type");

   // Skip encoding if hashing is not available:
   if (hash_data == nullptr) { return *this; }

   constexpr int max_buffer_bytes = 64*1024;
   unsigned char buffer[max_buffer_bytes];
   int buffer_counter = 0;
   while (begin != end)
   {
      auto k = reinterpret_cast<const uint64_t &>(*begin);
      for (int i = 0; i != 7; i++)
      {
         buffer[buffer_counter++] = (unsigned char)(k & 255); k >>= 8;
      }
      buffer[buffer_counter++] = (unsigned char)k;

      ++begin;

      if (begin == end || buffer_counter + 8 > max_buffer_bytes)
      {
         HashBuffer(buffer, buffer_counter);
         buffer_counter = 0;
      }
   }
   return *this;
}

} // namespace mfem

#endif
