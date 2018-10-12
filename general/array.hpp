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

#ifndef MFEM_ARRAY
#define MFEM_ARRAY

#include "../config/config.hpp"
#include "error.hpp"
#include "globals.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace mfem
{

/// Base class for array container.
class BaseArray
{
protected:
   /// Pointer to data
   void *data;
   /// Size of the array
   int size;
   /// Size of the allocated memory
   int allocsize;
   /** Increment of allocated memory on overflow,
       inc = 0 doubles the array */
   int inc;

   BaseArray() { }
   /// Creates array of asize elements of size elementsize
   BaseArray(int asize, int ainc, int elmentsize);
   /// Free the allocated memory
   ~BaseArray();
   /** Increases the allocsize of the array to be at least minsize.
       The current content of the array is copied to the newly allocated
       space. minsize must be > abs(allocsize). */
   void GrowSize(int minsize, int elementsize);
};

template <class T>
class Array;

template <class T>
void Swap(Array<T> &, Array<T> &);

/**
   Abstract data type Array.

   Array<T> is an automatically increasing array containing elements of the
   generic type T. The allocated size may be larger then the logical size
   of the array.
   The elements can be accessed by the [] operator, the range is 0 to size-1.
*/
template <class T>
class Array : public BaseArray
{
public:
   friend void Swap<T>(Array<T> &, Array<T> &);

   /// Creates array of asize elements
   explicit inline Array(int asize = 0, int ainc = 0)
      : BaseArray(asize, ainc, sizeof (T)) { }

   /** Creates array using an existing c-array of asize elements;
       allocsize is set to -asize to indicate that the data will not
       be deleted. */
   inline Array(T *_data, int asize, int ainc = 0)
   { data = _data; size = asize; allocsize = -asize; inc = ainc; }

   /// Copy constructor: deep copy
   Array(const Array<T> &src)
      : BaseArray(src.size, 0, sizeof(T))
   { std::memcpy(data, src.data, size*sizeof(T)); }

   /// Copy constructor (deep copy) from an Array of convertable type
   template <typename CT>
   Array(const Array<CT> &src)
      : BaseArray(src.Size(), 0, sizeof(T))
   { for (int i = 0; i < size; i++) { (*this)[i] = T(src[i]); } }

   /// Destructor
   inline ~Array() { }

   /// Assignment operator: deep copy
   Array<T> &operator=(const Array<T> &src) { src.Copy(*this); return *this; }

   /// Assignment operator (deep copy) from an Array of convertable type
   template <typename CT>
   Array<T> &operator=(const Array<CT> &src)
   {
      SetSize(src.Size());
      for (int i = 0; i < size; i++) { (*this)[i] = T(src[i]); }
      return *this;
   }

   /// Return the data as 'T *'
   inline operator T *() { return (T *)data; }

   /// Return the data as 'const T *'
   inline operator const T *() const { return (const T *)data; }

   /// Returns the data
   inline T *GetData() { return (T *)data; }
   /// Returns the data
   inline const T *GetData() const { return (T *)data; }

   /// Return true if the data will be deleted by the array
   inline bool OwnsData() const { return (allocsize > 0); }

   /// Changes the ownership of the data
   inline void StealData(T **p)
   { *p = (T*)data; data = 0; size = allocsize = 0; }

   /// NULL-ifies the data
   inline void LoseData() { data = 0; size = allocsize = 0; }

   /// Make the Array own the data
   void MakeDataOwner() { allocsize = abs(allocsize); }

   /// Logical size of the array
   inline int Size() const { return size; }

   /// Change logical size of the array, keep existing entries
   inline void SetSize(int nsize);

   /// Same as SetSize(int) plus initialize new entries with 'initval'
   inline void SetSize(int nsize, const T &initval);

   /** Maximum number of entries the array can store without allocating more
       memory. */
   inline int Capacity() const { return abs(allocsize); }

   /// Ensures that the allocated size is at least the given size.
   inline void Reserve(int capacity)
   { if (capacity > abs(allocsize)) { GrowSize(capacity, sizeof(T)); } }

   /// Access element
   inline T & operator[](int i);

   /// Access const element
   inline const T &operator[](int i) const;

   /// Append element to array, resize if necessary
   inline int Append(const T & el);

   /// Append another array to this array, resize if necessary
   inline int Append(const T *els, int nels);

   /// Append another array to this array, resize if necessary
   inline int Append(const Array<T> &els) { return Append(els, els.Size()); }

   /// Prepend an element to the array, resize if necessary
   inline int Prepend(const T &el);

   /// Return the last element in the array
   inline T &Last();
   inline const T &Last() const;

   /// Append element when it is not yet in the array, return index
   inline int Union(const T & el);

   /// Return the first index where 'el' is found; return -1 if not found
   inline int Find(const T &el) const;

   /// Do bisection search for 'el' in a sorted array; return -1 if not found.
   inline int FindSorted(const T &el) const;

   /// Delete the last entry
   inline void DeleteLast() { if (size > 0) { size--; } }

   /// Delete the first 'el' entry
   inline void DeleteFirst(const T &el);

   /// Delete whole array
   inline void DeleteAll();

   /// Create a copy of the current array
   inline void Copy(Array &copy) const
   {
      copy.SetSize(Size());
      std::memcpy(copy.GetData(), data, Size()*sizeof(T));
   }

   /// Make this Array a reference to a pointer
   inline void MakeRef(T *, int);

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array &master);

   inline void GetSubArray(int offset, int sa_size, Array<T> &sa);

   /// Prints array to stream with width elements per row
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

   /// Sorts the array. This requires operator< to be defined for T.
   void Sort() { std::sort((T*) data, (T*) data + size); }

   /// Sorts the array using the supplied comparison function object.
   template<class Compare>
   void Sort(Compare cmp) { std::sort((T*) data, (T*) data + size, cmp); }

   /** Removes duplicities from a sorted array. This requires operator== to be
       defined for T. */
   void Unique()
   {
      T* end = std::unique((T*) data, (T*) data + size);
      SetSize(end - (T*) data);
   }

   /// return true if the array is sorted.
   int IsSorted();

   /// Partial Sum
   void PartialSum();

   /// Sum all entries
   T Sum();

   inline void operator=(const T &a);

   /// Copy data from a pointer. Size() elements are copied.
   inline void Assign(const T *);

   template <typename U>
   inline void CopyTo(U *dest) { std::copy(begin(), end(), dest); }

   // STL-like begin/end
   inline T* begin() const { return (T*) data; }
   inline T* end() const { return (T*) data + size; }

   long MemoryUsage() const { return Capacity() * sizeof(T); }
};

template <class T>
inline bool operator==(const Array<T> &LHS, const Array<T> &RHS)
{
   if ( LHS.Size() != RHS.Size() ) { return false; }
   for (int i=0; i<LHS.Size(); i++)
      if ( LHS[i] != RHS[i] ) { return false; }
   return true;
}

template <class T>
inline bool operator!=(const Array<T> &LHS, const Array<T> &RHS)
{
   return !( LHS == RHS );
}


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
   void Save(std::ostream &out, int fmt = 0) const
   {
      if (fmt == 0) { out << NumRows() << ' ' << NumCols() << '\n'; }
      array1d.Save(out, 1);
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

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array2D &master)
   { M = master.M; N = master.N; array1d.MakeRef(master.array1d); }

   /// Delete all dynamically allocated memory, reseting all dimentions to zero.
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
   ~BlockArray();

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

   void Swap(BlockArray<T> &other);

   long MemoryUsage() const;

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
   Swap(a.allocsize, b.allocsize);
   Swap(a.inc, b.inc);
}

template <class T>
inline void Array<T>::SetSize(int nsize)
{
   MFEM_ASSERT( nsize>=0, "Size must be non-negative.  It is " << nsize );
   if (nsize > abs(allocsize))
   {
      GrowSize(nsize, sizeof(T));
   }
   size = nsize;
}

template <class T>
inline void Array<T>::SetSize(int nsize, const T &initval)
{
   MFEM_ASSERT( nsize>=0, "Size must be non-negative.  It is " << nsize );
   if (nsize > size)
   {
      if (nsize > abs(allocsize))
      {
         GrowSize(nsize, sizeof(T));
      }
      for (int i = size; i < nsize; i++)
      {
         ((T*)data)[i] = initval;
      }
   }
   size = nsize;
}

template <class T>
inline T &Array<T>::operator[](int i)
{
   MFEM_ASSERT( i>=0 && i<size,
                "Access element " << i << " of array, size = " << size );
   return ((T*)data)[i];
}

template <class T>
inline const T &Array<T>::operator[](int i) const
{
   MFEM_ASSERT( i>=0 && i<size,
                "Access element " << i << " of array, size = " << size );
   return ((T*)data)[i];
}

template <class T>
inline int Array<T>::Append(const T &el)
{
   SetSize(size+1);
   ((T*)data)[size-1] = el;
   return size;
}

template <class T>
inline int Array<T>::Append(const T *els, int nels)
{
   const int old_size = size;

   SetSize(size + nels);
   for (int i = 0; i < nels; i++)
   {
      ((T*)data)[old_size+i] = els[i];
   }
   return size;
}

template <class T>
inline int Array<T>::Prepend(const T &el)
{
   SetSize(size+1);
   for (int i = size-1; i > 0; i--)
   {
      ((T*)data)[i] = ((T*)data)[i-1];
   }
   ((T*)data)[0] = el;
   return size;
}

template <class T>
inline T &Array<T>::Last()
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return ((T*)data)[size-1];
}

template <class T>
inline const T &Array<T>::Last() const
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return ((T*)data)[size-1];
}

template <class T>
inline int Array<T>::Union(const T &el)
{
   int i = 0;
   while ((i < size) && (((T*)data)[i] != el)) { i++; }
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
      if (((T*)data)[i] == el) { return i; }
   }
   return -1;
}

template <class T>
inline int Array<T>::FindSorted(const T &el) const
{
   const T *begin = (const T*) data, *end = begin + size;
   const T* first = std::lower_bound(begin, end, el);
   if (first == end || !(*first == el)) { return  -1; }
   return first - begin;
}

template <class T>
inline void Array<T>::DeleteFirst(const T &el)
{
   for (int i = 0; i < size; i++)
   {
      if (((T*)data)[i] == el)
      {
         for (i++; i < size; i++)
         {
            ((T*)data)[i-1] = ((T*)data)[i];
         }
         size--;
         return;
      }
   }
}

template <class T>
inline void Array<T>::DeleteAll()
{
   if (allocsize > 0)
   {
      delete [] (char*)data;
   }
   data = NULL;
   size = allocsize = 0;
}

template <class T>
inline void Array<T>::MakeRef(T *p, int s)
{
   if (allocsize > 0)
   {
      delete [] (char*)data;
   }
   data = p;
   size = s;
   allocsize = -s;
}

template <class T>
inline void Array<T>::MakeRef(const Array &master)
{
   if (allocsize > 0)
   {
      delete [] (char*)data;
   }
   data = master.data;
   size = master.size;
   allocsize = -abs(master.allocsize);
   inc = master.inc;
}

template <class T>
inline void Array<T>::GetSubArray(int offset, int sa_size, Array<T> &sa)
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
      ((T*)data)[i] = a;
   }
}

template <class T>
inline void Array<T>::Assign(const T *p)
{
   memcpy(data, p, Size()*sizeof(T));
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
long BlockArray<T>::MemoryUsage() const
{
   return blocks.Size()*(mask+1)*sizeof(T) +
          blocks.MemoryUsage();
}

template<typename T>
BlockArray<T>::~BlockArray()
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
