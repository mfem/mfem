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

#ifndef MFEM_ARRAY
#define MFEM_ARRAY

#include "../config/config.hpp"
#include "mem_manager.hpp"
#include "device.hpp"
#include "error.hpp"
#include "globals.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

namespace mfem
{

template <class T>
class Array;

template <class T>
void Swap(Array<T> &, Array<T> &);

/**
   Abstract data type Array.

   Array<T> is an automatically increasing array containing elements of the
   generic type T, which must be a trivial type, see `std::is_trivial`. The
   allocated size may be larger then the logical size of the array. The elements
   can be accessed by the [] operator, the range is 0 to size-1.
*/
template <class T>
class Array
{
protected:
   /// Pointer to data
   Memory<T> data;
   /// Size of the array
   int size;

   inline void GrowSize(int minsize);

   static_assert(std::is_trivial<T>::value, "type T must be trivial");

public:
   using value_type = T; ///< Type alias for stl.
   using reference = T&; ///< Type alias for stl.
   using const_reference = const T&; ///< Type alias for stl.

   friend void Swap<T>(Array<T> &, Array<T> &);

   /// Creates an empty array
   inline Array() : size(0) { }

   /// Creates an empty array with a given MemoryType
   inline Array(MemoryType mt) : data(mt), size(0) { }

   /// Creates array of @a asize elements
   explicit inline Array(int asize)
      : size(asize) { if (asize > 0) { data.New(asize); } }

   /// Creates array of @a asize elements with a given MemoryType
   inline Array(int asize, MemoryType mt)
      : data(mt), size(asize) { if (asize > 0) { data.New(asize, mt); } }

   /** @brief Creates array using an externally allocated host pointer @a data_
       to @a asize elements. If @a own_data is true, the array takes ownership
       of the pointer.

       When @a own_data is true, the pointer @a data_ must be allocated with
       MemoryType given by MemoryManager::GetHostMemoryType(). */
   inline Array(T *data_, int asize, bool own_data = false)
   { data.Wrap(data_, asize, own_data); size = asize; }

   /// Copy constructor: deep copy from @a src
   /** This method supports source arrays using any MemoryType. */
   inline Array(const Array &src);

   /// Copy constructor (deep copy) from 'src', an Array of convertible type.
   template <typename CT>
   inline Array(const Array<CT> &src);

   /// Construct an Array from a C-style array of static length
   template <typename CT, int N>
   explicit inline Array(const CT (&values)[N]);

   /// Construct an Array from a braced initializer list of convertible type
   template <typename CT, typename std::enable_if<
                std::is_convertible<CT,T>::value,bool>::type = true>
   explicit inline Array(std::initializer_list<CT> values);

   /// Move constructor ("steals" data from 'src')
   inline Array(Array<T> &&src) : Array() { Swap(src, *this); }

   /// Destructor
   inline ~Array() { data.Delete(); }

   /// Assignment operator: deep copy from 'src'.
   Array<T> &operator=(const Array<T> &src) { src.Copy(*this); return *this; }

   /// Assignment operator (deep copy) from @a src, an Array of convertible type.
   template <typename CT>
   inline Array &operator=(const Array<CT> &src);

   /// Return the data as 'T *'
   inline operator T *() { return data; }

   /// Return the data as 'const T *'
   inline operator const T *() const { return data; }

   /// Returns the data
   inline T *GetData() { return data; }
   /// Returns the data
   inline const T *GetData() const { return data; }

   /// Return a reference to the Memory object used by the Array.
   Memory<T> &GetMemory() { return data; }

   /// Return a reference to the Memory object used by the Array, const version.
   const Memory<T> &GetMemory() const { return data; }

   /// Return the device flag of the Memory object used by the Array
   bool UseDevice() const { return data.UseDevice(); }

   /// Return true if the data will be deleted by the Array
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Changes the ownership of the data
   inline void StealData(T **p) { *p = data; data.Reset(); size = 0; }

   /// NULL-ifies the data
   inline void LoseData() { data.Reset(); size = 0; }

   /// Make the Array own the data
   void MakeDataOwner() const { data.SetHostPtrOwner(true); }

   /// Return the logical size of the array.
   inline int Size() const { return size; }

   /// Change the logical size of the array, keep existing entries.
   inline void SetSize(int nsize);

   /// Same as SetSize(int) plus initialize new entries with 'initval'.
   inline void SetSize(int nsize, const T &initval);

   /** @brief Resize the array to size @a nsize using MemoryType @a mt. Note
       that unlike the other versions of SetSize(), the current content of the
       array is not preserved. */
   inline void SetSize(int nsize, MemoryType mt);

   /** Maximum number of entries the array can store without allocating more
       memory. */
   inline int Capacity() const { return data.Capacity(); }

   /// Ensures that the allocated size is at least the given size.
   inline void Reserve(int capacity)
   { if (capacity > Capacity()) { GrowSize(capacity); } }

   /// Reference access to the ith element.
   inline T & operator[](int i);

   /// Const reference access to the ith element.
   inline const T &operator[](int i) const;

   /// Append element 'el' to array, resize if necessary.
   inline int Append(const T & el);

   /// STL-like push_back. Append element 'el' to array, resize if necessary.
   void push_back(const T &el) { Append(el); }

   /// Append another array to this array, resize if necessary.
   inline int Append(const T *els, int nels);

   /// Append another array to this array, resize if necessary.
   inline int Append(const Array<T> &els) { return Append(els, els.Size()); }

   /// Prepend an 'el' to the array, resize if necessary.
   inline int Prepend(const T &el);

   /// Return the last element in the array.
   inline T &Last();

   /// Return the last element in the array.
   inline const T &Last() const;

   /// Append element when it is not yet in the array, return index.
   inline int Union(const T & el);

   /// Return the first index where 'el' is found; return -1 if not found.
   inline int Find(const T &el) const;

   /// Do bisection search for 'el' in a sorted array; return -1 if not found.
   inline int FindSorted(const T &el) const;

   /// Delete the last entry of the array.
   inline void DeleteLast() { if (size > 0) { size--; } }

   /// Delete the first entry with value == 'el'.
   inline void DeleteFirst(const T &el);

   /// Delete the whole array.
   inline void DeleteAll();

   /// Reduces the capacity of the array to exactly match the current size.
   inline void ShrinkToFit();

   ///  Create a copy of the internal array to the provided @a copy.
   inline void Copy(Array &copy) const;

   /// Make this Array a reference to a pointer.
   /** When @a own_data is true, the pointer @a data_ must be allocated with
       MemoryType given by MemoryManager::GetHostMemoryType(). */
   inline void MakeRef(T *data_, int size_, bool own_data = false);

   /// Make this Array a reference to a pointer.
   /** When @a own_data is true, the pointer @a data_ must be allocated with
       MemoryType given by @a mt. */
   inline void MakeRef(T *data_, int size, MemoryType mt, bool own_data);

   /// Make this Array a reference to 'master'.
   inline void MakeRef(const Array &master);

   /**
    * @brief Permute the array using the provided indices. Sorts the indices
    * variable in the process, thereby destroying the permutation. The rvalue
    * reference is to be used when this destruction is allowed, whilst the const
    * reference preserves at the cost of duplication.
    *
    * @param indices The indices of the ordering. data[i] = data[indices[i]].
    */
   template <typename I>
   inline void Permute(I &&indices);
   template <typename I>
   inline void Permute(const I &indices) { Permute(I(indices)); }

   /// Copy sub array starting from @a offset out to the provided @a sa.
   inline void GetSubArray(int offset, int sa_size, Array<T> &sa) const;

   /// Prints array to stream with width elements per row.
   void Print(std::ostream &out = mfem::out, int width = 4) const;

   /** @brief Save the Array to the stream @a out using the format @a fmt.
       The format @a fmt can be:

          0 - write the size followed by all entries
          1 - write only the entries
   */
   void Save(std::ostream &out, int fmt = 0) const;

   /** @brief Read an Array from the stream @a in using format @a fmt.
       The format @a fmt can be:

          0 - read the size then the entries
          1 - read Size() entries
   */
   void Load(std::istream &in, int fmt = 0);

   /** @brief Set the Array size to @a new_size and read that many entries from
       the stream @a in. */
   void Load(int new_size, std::istream &in)
   { SetSize(new_size); Load(in, 1); }

   /** @brief Find the maximal element in the array, using the comparison
       operator `<` for class T. */
   T Max() const;

   /** @brief Find the minimal element in the array, using the comparison
       operator `<` for class T. */
   T Min() const;

   /// Sorts the array in ascending order. This requires operator< to be defined for T.
   void Sort() { std::sort((T*)data, data + size); }

   /// Sorts the array in ascending order using the supplied comparison function object.
   template<class Compare>
   void Sort(Compare cmp) { std::sort((T*)data, data + size, cmp); }

   /** @brief Removes duplicities from a sorted array. This requires
       operator== to be defined for T. */
   void Unique()
   {
      T* end = std::unique((T*)data, data + size);
      SetSize((int)(end - data));
   }

   /// Return 1 if the array is sorted from lowest to highest.  Otherwise return 0.
   int IsSorted() const;

   /// Does the Array have Size zero.
   bool IsEmpty() const { return Size() == 0; }

   /// Fill the entries of the array with the cumulative sum of the entries.
   void PartialSum();

   /// Replace each entry of the array with its absolute value.
   void Abs();

   /// Return the sum of all the array entries using the '+'' operator for class 'T'.
   T Sum() const;

   /// Set all entries of the array to the provided constant.
   inline void operator=(const T &a);

   /// Copy data from a pointer. 'Size()' elements are copied.
   inline void Assign(const T *);

   /// STL-like copyTo @a dest from begin to end.
   template <typename U>
   inline void CopyTo(U *dest) { std::copy(begin(), end(), dest); }

   /** @brief Copy from @a src into this array.  Copies enough entries to
       fill the Capacity size of this array.  Careful this does not update
       the Size to match this Capacity after this.*/
   template <typename U>
   inline void CopyFrom(const U *src)
   {
      if (!begin() || size == 0) { return; }
      MFEM_ASSERT(begin() && src, "Error in Array::CopyFrom");
      std::memcpy(begin(), src, MemoryUsage());
   }

   /// STL-like begin.  Returns pointer to the first element of the array.
   inline T* begin() { return data; }

   /// STL-like end.  Returns pointer after the last element of the array.
   inline T* end() { return data + size; }

   /// STL-like begin.  Returns const pointer to the first element of the array.
   inline const T* begin() const { return data; }

   /// STL-like end.  Returns const pointer after the last element of the array.
   inline const T* end() const { return data + size; }

   /// Returns the number of bytes allocated for the array including any reserve.
   std::size_t MemoryUsage() const { return Capacity() * sizeof(T); }

   /// Shortcut for mfem::Read(a.GetMemory(), a.Size(), on_dev).
   const T *Read(bool on_dev = true) const
   { return mfem::Read(data, size, on_dev); }

   /// Shortcut for mfem::Read(a.GetMemory(), a.Size(), false).
   const T *HostRead() const
   { return mfem::Read(data, size, false); }

   /// Shortcut for mfem::Write(a.GetMemory(), a.Size(), on_dev).
   T *Write(bool on_dev = true)
   { return mfem::Write(data, size, on_dev); }

   /// Shortcut for mfem::Write(a.GetMemory(), a.Size(), false).
   T *HostWrite()
   { return mfem::Write(data, size, false); }

   /// Shortcut for mfem::ReadWrite(a.GetMemory(), a.Size(), on_dev).
   T *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, size, on_dev); }

   /// Shortcut for mfem::ReadWrite(a.GetMemory(), a.Size(), false).
   T *HostReadWrite()
   { return mfem::ReadWrite(data, size, false); }
};

template <class T>
inline bool operator==(const Array<T> &LHS, const Array<T> &RHS)
{
   if ( LHS.Size() != RHS.Size() ) { return false; }
   for (int i=0; i<LHS.Size(); i++)
   {
      if ( LHS[i] != RHS[i] ) { return false; }
   }
   return true;
}

template <class T>
inline bool operator!=(const Array<T> &LHS, const Array<T> &RHS)
{
   return !( LHS == RHS );
}


/// Utility function similar to std::as_const in c++17.
template <typename T> const T &AsConst(const T &a) { return a; }


template <class T>
class Array2D;

template <class T>
void Swap(Array2D<T> &, Array2D<T> &);

/// Dynamic 2D array using row-major layout
template <class T>
class Array2D
{
private:
   friend void Swap<T>(Array2D<T> &, Array2D<T> &);

   Array<T> array1d;
   int M, N; // number of rows and columns

public:
   Array2D() { M = N = 0; }
   Array2D(int m, int n) : array1d(m*n) { M = m; N = n; }

   Array2D(const Array2D &) = default;

   void SetSize(int m, int n) { array1d.SetSize(m*n); M = m; N = n; }

   int NumRows() const { return M; }
   int NumCols() const { return N; }

   inline const T &operator()(int i, int j) const;
   inline       T &operator()(int i, int j);

   inline const T *operator[](int i) const;
   inline T       *operator[](int i);

   const T *operator()(int i) const { return (*this)[i]; }
   T       *operator()(int i)       { return (*this)[i]; }

   const T *GetRow(int i) const { return (*this)[i]; }
   T       *GetRow(int i)       { return (*this)[i]; }

   /// Extract a copy of the @a i-th row into the Array @a sa.
   void GetRow(int i, Array<T> &sa) const
   {
      sa.SetSize(N);
      sa.Assign(GetRow(i));
   }

   /** @brief Save the Array2D to the stream @a out using the format @a fmt.
       The format @a fmt can be:

          0 - write the number of rows and columns, followed by all entries
          1 - write only the entries, using row-major layout
   */
   void Save(std::ostream &os, int fmt = 0) const
   {
      if (fmt == 0) { os << NumRows() << ' ' << NumCols() << '\n'; }
      array1d.Save(os, 1);
   }

   /** @brief Read an Array2D from the stream @a in using format @a fmt.
       The format @a fmt can be:

          0 - read the number of rows and columns, then the entries
          1 - read NumRows() x NumCols() entries, using row-major layout
   */
   void Load(std::istream &in, int fmt = 0)
   {
      if (fmt == 0) { in >> M >> N; array1d.SetSize(M*N); }
      array1d.Load(in, 1);
   }

   /// Read an Array2D from a file
   void Load(const char *filename, int fmt = 0);

   /** @brief Set the Array2D dimensions to @a new_size0 x @a new_size1 and read
       that many entries from the stream @a in. */
   void Load(int new_size0,int new_size1, std::istream &in)
   { SetSize(new_size0,new_size1); Load(in, 1); }

   void Copy(Array2D &copy) const
   { copy.M = M; copy.N = N; array1d.Copy(copy.array1d); }

   inline void operator=(const T &a)
   { array1d = a; }

   inline Array2D& operator=(const Array2D &a) = default;

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array2D &master)
   { M = master.M; N = master.N; array1d.MakeRef(master.array1d); }

   /// Delete all dynamically allocated memory, resetting all dimensions to zero.
   inline void DeleteAll() { M = 0; N = 0; array1d.DeleteAll(); }

   /// Prints array to stream with width elements per row
   void Print(std::ostream &out = mfem::out, int width = 4);
};


template <class T>
class Array3D
{
private:
   Array<T> array1d;
   int N2, N3;

public:
   Array3D() { N2 = N3 = 0; }
   Array3D(int n1, int n2, int n3)
      : array1d(n1*n2*n3) { N2 = n2; N3 = n3; }

   void SetSize(int n1, int n2, int n3)
   { array1d.SetSize(n1*n2*n3); N2 = n2; N3 = n3; }

   inline const T &operator()(int i, int j, int k) const;
   inline       T &operator()(int i, int j, int k);

   inline void operator=(const T &a)
   { array1d = a; }
};


/** A container for items of type T. Dynamically grows as items are added.
 *  Each item is accessible by its index. Items are allocated in larger chunks
 *  (blocks), so the 'Append' method is very fast on average.
 */
template<typename T>
class BlockArray
{
public:
   BlockArray(int block_size = 16*1024);
   BlockArray(const BlockArray<T> &other); // deep copy
   BlockArray& operator=(const BlockArray&) = delete; // not supported
   BlockArray(BlockArray<T> &&other) = default;
   BlockArray& operator=(BlockArray<T> &&other) = default;
   ~BlockArray() { Destroy(); }

   /// Allocate and construct a new item in the array, return its index.
   int Append();

   /// Allocate and copy-construct a new item in the array, return its index.
   int Append(const T &item);

   /// Access item of the array.
   inline T& At(int index)
   {
      CheckIndex(index);
      return blocks[index >> shift][index & mask];
   }
   inline const T& At(int index) const
   {
      CheckIndex(index);
      return blocks[index >> shift][index & mask];
   }

   /// Access item of the array.
   inline T& operator[](int index) { return At(index); }
   inline const T& operator[](int index) const { return At(index); }

   /// Return the number of items actually stored.
   int Size() const { return size; }

   /// Return the current capacity of the BlockArray.
   int Capacity() const { return blocks.Size()*(mask+1); }

   /// Destroy all items, set size to zero.
   void DeleteAll() { Destroy(); blocks.DeleteAll(); size = 0; }

   void Swap(BlockArray<T> &other);

   std::size_t MemoryUsage() const;

protected:
   template <typename cA, typename cT>
   class iterator_base
   {
   public:
      cT& operator*() const { return *ptr; }
      cT* operator->() const { return ptr; }

      bool good() const { return !stop; }
      int index() const { return (ptr - ref); }

   protected:
      cA *array;
      cT *ptr, *b_end, *ref;
      int b_end_idx;
      bool stop;

      iterator_base() { }
      iterator_base(bool stop) : stop(stop) { }
      iterator_base(cA *a)
         : array(a), ptr(a->blocks[0]), ref(ptr), stop(false)
      {
         b_end_idx = std::min(a->size, a->mask+1);
         b_end = ptr + b_end_idx;
      }

      void next()
      {
         MFEM_ASSERT(!stop, "invalid use");
         if (++ptr == b_end)
         {
            if (b_end_idx < array->size)
            {
               ptr = &array->At(b_end_idx);
               ref = ptr - b_end_idx;
               b_end_idx = std::min(array->size, (b_end_idx|array->mask) + 1);
               b_end = &array->At(b_end_idx-1) + 1;
            }
            else
            {
               MFEM_ASSERT(b_end_idx == array->size, "invalid use");
               stop = true;
            }
         }
      }
   };

public:
   class iterator : public iterator_base<BlockArray, T>
   {
   protected:
      friend class BlockArray;
      typedef iterator_base<BlockArray, T> base;

      iterator() { }
      iterator(bool stop) : base(stop) { }
      iterator(BlockArray *a) : base(a) { }

   public:
      iterator &operator++() { base::next(); return *this; }

      bool operator==(const iterator &other) const { return base::stop; }
      bool operator!=(const iterator &other) const { return !base::stop; }
   };

   class const_iterator : public iterator_base<const BlockArray, const T>
   {
   protected:
      friend class BlockArray;
      typedef iterator_base<const BlockArray, const T> base;

      const_iterator() { }
      const_iterator(bool stop) : base(stop) { }
      const_iterator(const BlockArray *a) : base(a) { }

   public:
      const_iterator &operator++() { base::next(); return *this; }

      bool operator==(const const_iterator &other) const { return base::stop; }
      bool operator!=(const const_iterator &other) const { return !base::stop; }
   };

   iterator begin() { return size ? iterator(this) : iterator(true); }
   iterator end() { return iterator(); }
   const_iterator begin() const { return cbegin(); }
   const_iterator end() const { return cend(); }

   const_iterator cbegin() const
   { return size ? const_iterator(this) : const_iterator(true); }
   const_iterator cend() const { return const_iterator(); }

protected:
   Array<T*> blocks;
   int size, shift, mask;

   int Alloc();

   inline void CheckIndex(int index) const
   {
      MFEM_ASSERT(index >= 0 && index < size,
                  "Out of bounds access: " << index << ", size = " << size);
   }

   void Destroy();
};


/// inlines ///

template <class T>
inline void Swap(T &a, T &b)
{
   T c = a;
   a = b;
   b = c;
}

template <class T>
inline void Swap(Array<T> &a, Array<T> &b)
{
   Swap(a.data, b.data);
   Swap(a.size, b.size);
}

template <class T>
inline Array<T>::Array(const Array &src)
   : size(src.Size())
{
   size > 0 ? data.New(size, src.data.GetMemoryType()) : data.Reset();
   data.CopyFrom(src.data, size);
   data.UseDevice(src.data.UseDevice());
}

template <typename T> template <typename CT>
inline Array<T>::Array(const Array<CT> &src)
   : size(src.Size())
{
   size > 0 ? data.New(size) : data.Reset();
   for (int i = 0; i < size; i++) { (*this)[i] = T(src[i]); }
}

template <typename T>
template <typename CT, typename std::enable_if<
             std::is_convertible<CT,T>::value,bool>::type>
inline Array<T>::Array(std::initializer_list<CT> values) : Array(values.size())
{
   std::copy(values.begin(), values.end(), begin());
}

template <typename T> template <typename CT, int N>
inline Array<T>::Array(const CT (&values)[N]) : Array(N)
{
   std::copy(values, values + N, begin());
}

template <class T>
inline void Array<T>::GrowSize(int minsize)
{
   const int nsize = std::max(minsize, 2 * data.Capacity());
   Memory<T> p(nsize, data.GetMemoryType());
   p.CopyFrom(data, size);
   p.UseDevice(data.UseDevice());
   data.Delete();
   data = p;
}

template <typename T>
inline void Array<T>::ShrinkToFit()
{
   if (Capacity() == size) { return; }
   Memory<T> p(size, data.GetMemoryType());
   p.CopyFrom(data, size);
   p.UseDevice(data.UseDevice());
   data.Delete();
   data = p;
}

template <typename T>
template <typename I>
inline void Array<T>::Permute(I &&indices)
{
   for (int i = 0; i < size; i++)
   {
      auto current = i;
      while (i != indices[current])
      {
         auto next = indices[current];
         std::swap(data[current], data[next]);
         indices[current] = current;
         current = next;
      }
      indices[current] = current;
   }
}

template <typename T> template <typename CT>
inline Array<T> &Array<T>::operator=(const Array<CT> &src)
{
   SetSize(src.Size());
   for (int i = 0; i < size; i++) { (*this)[i] = T(src[i]); }
   return *this;
}

template <class T>
inline void Array<T>::SetSize(int nsize)
{
   MFEM_ASSERT( nsize>=0, "Size must be non-negative.  It is " << nsize );
   if (nsize > Capacity())
   {
      GrowSize(nsize);
   }
   size = nsize;
}

template <class T>
inline void Array<T>::SetSize(int nsize, const T &initval)
{
   MFEM_ASSERT( nsize>=0, "Size must be non-negative.  It is " << nsize );
   if (nsize > size)
   {
      if (nsize > Capacity())
      {
         GrowSize(nsize);
      }
      for (int i = size; i < nsize; i++)
      {
         data[i] = initval;
      }
   }
   size = nsize;
}

template <class T>
inline void Array<T>::SetSize(int nsize, MemoryType mt)
{
   MFEM_ASSERT(nsize >= 0, "invalid new size: " << nsize);
   if (mt == data.GetMemoryType())
   {
      if (nsize <= Capacity())
      {
         size = nsize;
         return;
      }
   }
   const bool use_dev = data.UseDevice();
   data.Delete();
   if (nsize > 0)
   {
      data.New(nsize, mt);
      size = nsize;
   }
   else
   {
      data.Reset();
      size = 0;
   }
   data.UseDevice(use_dev);
}

template <class T>
inline T &Array<T>::operator[](int i)
{
   MFEM_ASSERT( i>=0 && i<size,
                "Access element " << i << " of array, size = " << size );
   return data[i];
}

template <class T>
inline const T &Array<T>::operator[](int i) const
{
   MFEM_ASSERT( i>=0 && i<size,
                "Access element " << i << " of array, size = " << size );
   return data[i];
}

template <class T>
inline int Array<T>::Append(const T &el)
{
   SetSize(size+1);
   data[size-1] = el;
   return size;
}

template <class T>
inline int Array<T>::Append(const T *els, int nels)
{
   const int old_size = size;

   SetSize(size + nels);
   for (int i = 0; i < nels; i++)
   {
      data[old_size+i] = els[i];
   }
   return size;
}

template <class T>
inline int Array<T>::Prepend(const T &el)
{
   SetSize(size+1);
   for (int i = size-1; i > 0; i--)
   {
      data[i] = data[i-1];
   }
   data[0] = el;
   return size;
}

template <class T>
inline T &Array<T>::Last()
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return data[size-1];
}

template <class T>
inline const T &Array<T>::Last() const
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return data[size-1];
}

template <class T>
inline int Array<T>::Union(const T &el)
{
   int i = 0;
   while ((i < size) && (data[i] != el)) { i++; }
   if (i == size)
   {
      Append(el);
   }
   return i;
}

template <class T>
inline int Array<T>::Find(const T &el) const
{
   for (int i = 0; i < size; i++)
   {
      if (data[i] == el) { return i; }
   }
   return -1;
}

template <class T>
inline int Array<T>::FindSorted(const T &el) const
{
   const T *begin = data, *end = begin + size;
   const T* first = std::lower_bound(begin, end, el);
   if (first == end || !(*first == el)) { return  -1; }
   return (int)(first - begin);
}

template <class T>
inline void Array<T>::DeleteFirst(const T &el)
{
   for (int i = 0; i < size; i++)
   {
      if (data[i] == el)
      {
         for (i++; i < size; i++)
         {
            data[i-1] = data[i];
         }
         size--;
         return;
      }
   }
}

template <class T>
inline void Array<T>::DeleteAll()
{
   const bool use_dev = data.UseDevice();
   data.Delete();
   data.Reset();
   size = 0;
   data.UseDevice(use_dev);
}

template <typename T>
inline void Array<T>::Copy(Array &copy) const
{
   copy.SetSize(Size(), data.GetMemoryType());
   data.CopyTo(copy.data, Size());
   copy.data.UseDevice(data.UseDevice());
}

template <class T>
inline void Array<T>::MakeRef(T *data_, int size_, bool own_data)
{
   data.Delete();
   data.Wrap(data_, size_, own_data);
   size = size_;
}

template <class T>
inline void Array<T>::MakeRef(T *data_, int size_, MemoryType mt, bool own_data)
{
   data.Delete();
   data.Wrap(data_, size_, mt, own_data);
   size = size_;
}

template <class T>
inline void Array<T>::MakeRef(const Array &master)
{
   data.Delete();
   size = master.size;
   data.MakeAlias(master.GetMemory(), 0, size);
}

template <class T>
inline void Array<T>::GetSubArray(int offset, int sa_size, Array<T> &sa) const
{
   sa.SetSize(sa_size);
   for (int i = 0; i < sa_size; i++)
   {
      sa[i] = (*this)[offset+i];
   }
}

template <class T>
inline void Array<T>::operator=(const T &a)
{
   for (int i = 0; i < size; i++)
   {
      data[i] = a;
   }
}

template <class T>
inline void Array<T>::Assign(const T *p)
{
   data.CopyFromHost(p, Size());
}


template <class T>
inline const T &Array2D<T>::operator()(int i, int j) const
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N && j>=0 && j<N,
                "Array2D: invalid access of element (" << i << ',' << j
                << ") in array of size (" << array1d.Size()/N << ',' << N
                << ")." );
   return array1d[i*N+j];
}

template <class T>
inline T &Array2D<T>::operator()(int i, int j)
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N && j>=0 && j<N,
                "Array2D: invalid access of element (" << i << ',' << j
                << ") in array of size (" << array1d.Size()/N << ',' << N
                << ")." );
   return array1d[i*N+j];
}

template <class T>
inline const T *Array2D<T>::operator[](int i) const
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N,
                "Array2D: invalid access of row " << i << " in array with "
                << array1d.Size()/N << " rows.");
   return &array1d[i*N];
}

template <class T>
inline T *Array2D<T>::operator[](int i)
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N,
                "Array2D: invalid access of row " << i << " in array with "
                << array1d.Size()/N << " rows.");
   return &array1d[i*N];
}


template <class T>
inline void Swap(Array2D<T> &a, Array2D<T> &b)
{
   Swap(a.array1d, b.array1d);
   Swap(a.N, b.N);
}


template <class T>
inline const T &Array3D<T>::operator()(int i, int j, int k) const
{
   MFEM_ASSERT(i >= 0 && i < array1d.Size() / N2 / N3 && j >= 0 && j < N2
               && k >= 0 && k < N3,
               "Array3D: invalid access of element ("
               << i << ',' << j << ',' << k << ") in array of size ("
               << array1d.Size() / N2 / N3 << ',' << N2 << ',' << N3 << ").");
   return array1d[(i*N2+j)*N3+k];
}

template <class T>
inline T &Array3D<T>::operator()(int i, int j, int k)
{
   MFEM_ASSERT(i >= 0 && i < array1d.Size() / N2 / N3 && j >= 0 && j < N2
               && k >= 0 && k < N3,
               "Array3D: invalid access of element ("
               << i << ',' << j << ',' << k << ") in array of size ("
               << array1d.Size() / N2 / N3 << ',' << N2 << ',' << N3 << ").");
   return array1d[(i*N2+j)*N3+k];
}


template<typename T>
BlockArray<T>::BlockArray(int block_size)
{
   mask = block_size-1;
   MFEM_VERIFY(!(block_size & mask), "block_size must be a power of two.");

   size = shift = 0;
   while ((1 << shift) < block_size) { shift++; }
}

template<typename T>
BlockArray<T>::BlockArray(const BlockArray<T> &other)
{
   blocks.SetSize(other.blocks.Size());

   size = other.size;
   shift = other.shift;
   mask = other.mask;

   int bsize = mask+1;
   for (int i = 0; i < blocks.Size(); i++)
   {
      blocks[i] = (T*) new char[bsize * sizeof(T)];
   }

   // copy all items
   for (int i = 0; i < size; i++)
   {
      new (&At(i)) T(other[i]);
   }
}

template<typename T>
int BlockArray<T>::Alloc()
{
   int bsize = mask+1;
   if (size >= blocks.Size() * bsize)
   {
      T* new_block = (T*) new char[bsize * sizeof(T)];
      blocks.Append(new_block);
   }
   return size++;
}

template<typename T>
int BlockArray<T>::Append()
{
   int index = Alloc();
   new (&At(index)) T();
   return index;
}

template<typename T>
int BlockArray<T>::Append(const T &item)
{
   int index = Alloc();
   new (&At(index)) T(item);
   return index;
}

template<typename T>
void BlockArray<T>::Swap(BlockArray<T> &other)
{
   mfem::Swap(blocks, other.blocks);
   std::swap(size, other.size);
   std::swap(shift, other.shift);
   std::swap(mask, other.mask);
}

template<typename T>
std::size_t BlockArray<T>::MemoryUsage() const
{
   return (mask+1)*sizeof(T)*blocks.Size() + blocks.MemoryUsage();
}

template<typename T>
void BlockArray<T>::Destroy()
{
   int bsize = size & mask;
   for (int i = blocks.Size(); i != 0; )
   {
      T *block = blocks[--i];
      for (int j = bsize; j != 0; )
      {
         block[--j].~T();
      }
      delete [] (char*) block;
      bsize = mask+1;
   }
}

} // namespace mfem

#endif
