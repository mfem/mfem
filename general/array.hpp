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

   /// Destructor
   inline ~Array() { }

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

   /// Changes the ownership of the the data
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
   inline int Append(const Array<T> &els);

   /// Prepend an element to the array, resize if necessary
   inline int Prepend(const T &el);

   /// Return the last element in the array
   inline T &Last();
   inline const T &Last() const;

   /// Append element when it is not yet in the array, return index
   inline int Union(const T & el);

   /// Return the first index where 'el' is found; return -1 if not found
   inline int Find(const T &el) const;

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
      memcpy(copy.GetData(), data, Size()*sizeof(T));
   }

   /// Make this Array a reference to a pointer
   inline void MakeRef(T *, int);

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array &master);

   inline void GetSubArray(int offset, int sa_size, Array<T> &sa);

   /// Prints array to stream with width elements per row
   void Print(std::ostream &out = std::cout, int width = 4);

   /// Prints array to stream out
   void Save(std::ostream &out);

   /** Finds the maximal element in the array.
       (uses the comparison operator '<' for class T)  */
   T Max() const;

   /** Finds the minimal element in the array.
       (uses the comparison operator '<' for class T)  */
   T Min() const;

   /// Sorts the array. This requires operator< to be defined for T.
   void Sort() { std::sort((T*) data, (T*) data + size); }

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

   long MemoryUsage() const { return Capacity() * sizeof(T); }

private:
   /// Array copy is not supported
   Array<T> &operator=(Array<T> &);
   /// Array copy is not supported
   Array(const Array<T> &);
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

template <class T>
class Array2D
{
private:
   friend void Swap<T>(Array2D<T> &, Array2D<T> &);

   Array<T> array1d;
   int N;

public:
   Array2D() { N = 0; }
   Array2D(int m, int n) : array1d(m*n) { N = n; }

   void SetSize(int m, int n) { array1d.SetSize(m*n); N = n; }

   int NumRows() const { return array1d.Size()/N; }
   int NumCols() const { return N; }

   inline const T &operator()(int i, int j) const;
   inline       T &operator()(int i, int j);

   inline const T *operator[](int i) const;
   inline T       *operator[](int i);

   const T *operator()(int i) const { return (*this)[i]; }
   T       *operator()(int i)       { return (*this)[i]; }

   const T *GetRow(int i) const { return (*this)[i]; }
   T       *GetRow(int i)       { return (*this)[i]; }

   void Copy(Array2D &copy) const
   { copy.N = N; array1d.Copy(copy.array1d); }

   inline void operator=(const T &a)
   { array1d = a; }

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array2D &master)
   { N = master.N; array1d.MakeRef(master.array1d);}
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
inline int Array<T>::Append(const Array<T> & els)
{
   int old_size = size;

   SetSize(size + els.Size());
   for (int i = 0; i < els.Size(); i++)
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
      if (((T*)data)[i] == el)
      {
         return i;
      }
   return -1;
}

template <class T>
inline void Array<T>::DeleteFirst(const T &el)
{
   for (int i = 0; i < size; i++)
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

}

#endif
