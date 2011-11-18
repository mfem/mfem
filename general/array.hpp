// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_ARRAY
#define MFEM_ARRAY

#include <iostream>
#include <cstdlib>
#include <cstring>

#include "error.hpp"

using namespace std;

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

   /// Changes the ownership of the the data
   inline void StealData(T **p)
   { *p = (T*)data; data = 0; size = allocsize = 0; }

   /// NULL-ifies the data
   inline void LoseData() { data = 0; size = allocsize = 0; }

   /// Make the Array own the data
   void MakeDataOwner() { allocsize = abs(allocsize); }

   /// Logical size of the array
   inline int Size() const { return size; };

   /// Change logical size of the array, keep existing entries
   inline void SetSize(int nsize);

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

   /// Append element when it is not yet in the array, return index
   inline int Union(const T & el);

   /// Return the first index where 'el' is found; return -1 if not found
   inline int Find(const T &el) const;

   /// Delete the last entry
   inline void DeleteLast() { size--; }

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

   /// Make this Array a reference to 'master'
   inline void MakeRef(const Array &master);

   inline void GetSubArray(int offset, int sa_size, Array<T> &sa);

   /// Prints array to stream with width elements per row
   void Print(ostream &out, int width);

   /// Prints array to stream out
   void Save(ostream &out);

   /** Finds the maximal element in the array.
       (uses the comparison operator '<' for class T)  */
   T Max() const;

   /// Sorts the array.
   void Sort();

   inline void operator=(const T &a);

private:
   /// Array copy is not supported
   Array<T> &operator=(Array<T> &);
   /// Array copy is not supported
   Array(const Array<T> &);
};

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
   int   s;
   void *data;

   data = a.data;   a.data = b.data;           b.data = data;
   s = a.size;      a.size = b.size;           b.size = s;
   s = a.allocsize; a.allocsize = b.allocsize; b.allocsize = s;
   s = a.inc;       a.inc = b.inc;             b.inc = s;
}

template <class T>
inline void Array<T>::SetSize(int nsize)
{
#ifdef MFEM_DEBUG
   if (nsize < 0)
      mfem_error("Array::SetSize : negative size!");
#endif
   if (nsize > abs(allocsize))
      GrowSize(nsize, sizeof(T));
   size = nsize;
}

template <class T>
inline T &Array<T>::operator[](int i)
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= size)
   {
      cerr << "Access element " << i << " of array, size = " << size << endl;
      mfem_error();
   }
#endif
   return ((T*)data)[i];
}

template <class T>
inline const T &Array<T>::operator[](int i) const
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= size)
   {
      cerr << "Access element " << i << " of array, size = " << size << endl;
      mfem_error();
   }
#endif
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
      ((T*)data)[old_size+i] = els[i];
   return size;
}


template <class T>
inline int Array<T>::Prepend(const T &el)
{
   SetSize(size+1);
   for (int i = size-1; i > 0; i--)
      ((T*)data)[i] = ((T*)data)[i-1];
   ((T*)data)[0] = el;
   return size;
}


template <class T>
inline T &Array<T>::Last()
{
#ifdef MFEM_DEBUG
   if (size < 1)
      mfem_error("Array<T>::Last()");
#endif
   return ((T*)data)[size-1];
}

template <class T>
inline int Array<T>::Union(const T &el)
{
   int i = 0;
   while ((i < size) && (((T*)data)[i] != el)) i++;
   if (i == size)
      Append(el);
   return i;
}

template <class T>
inline int Array<T>::Find(const T &el) const
{
   for (int i = 0; i < size; i++)
      if (((T*)data)[i] == el)
         return i;
   return -1;
}

template <class T>
inline void Array<T>::DeleteFirst(const T &el)
{
   for (int i = 0; i < size; i++)
      if (((T*)data)[i] == el)
      {
         for (i++; i < size; i++)
            ((T*)data)[i-1] = ((T*)data)[i];
         size--;
         return;
      }
}

template <class T>
inline void Array<T>::DeleteAll()
{
   if (allocsize > 0)
      delete [] (char*)data;
   data = NULL;
   size = allocsize = 0;
}

template <class T>
inline void Array<T>::MakeRef(const Array &master)
{
   if (allocsize > 0)
      delete [] (char*)data;
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
      sa[i] = (*this)[offset+i];
}

template <class T>
inline void Array<T>::operator=(const T &a)
{
   for (int i = 0; i < size; i++)
      ((T*)data)[i] = a;
}


template <class T>
inline const T &Array2D<T>::operator()(int i, int j) const
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= array1d.Size()/N || j < 0 || j >= N)
   {
      cerr << "Array2D: invalid access of element (" << i << ',' << j
           << ") in array of size (" << array1d.Size()/N << ',' << N
           << ")." << endl;
      mfem_error();
   }
#endif
   return array1d[i*N+j];
}

template <class T>
inline T &Array2D<T>::operator()(int i, int j)
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= array1d.Size()/N || j < 0 || j >= N)
   {
      cerr << "Array2D: invalid access of element (" << i << ',' << j
           << ") in array of size (" << array1d.Size()/N << ',' << N
           << ")." << endl;
      mfem_error();
   }
#endif
   return array1d[i*N+j];
}

template <class T>
inline const T *Array2D<T>::operator[](int i) const
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= array1d.Size()/N)
   {
      cerr << "Array2D: invalid access of row " << i << " in array with "
           << array1d.Size()/N << " rows." << endl;
      mfem_error();
   }
#endif
   return &array1d[i*N];
}

template <class T>
inline T *Array2D<T>::operator[](int i)
{
#ifdef MFEM_DEBUG
   if (i < 0 || i >= array1d.Size()/N)
   {
      cerr << "Array2D: invalid access of row " << i << " in array with "
           << array1d.Size()/N << " rows." << endl;
      mfem_error();
   }
#endif
   return &array1d[i*N];
}


template <class T>
inline void Swap(Array2D<T> &a, Array2D<T> &b)
{
   int s;
   Swap(a.array1d, b.array1d);
   s = a.N;  a.N = b.N;  b.N = s;
}


template <class T>
inline const T &Array3D<T>::operator()(int i, int j, int k) const
{
#ifdef MFEM_DEBUG
   int N1 = array1d.Size()/N2/N3;
   if (i < 0 || i >= N1 || j < 0 || j >= N2 || k < 0 || k >= N3)
   {
      cerr << "Array3D: invalid access of element ("
           << i << ',' << j << ',' << k << ") in array of size ("
           << N1 << ',' << N2 << ',' << N3 << ")." << endl;
      mfem_error();
   }
#endif
   return array1d[(i*N2+j)*N3+k];
}

template <class T>
inline T &Array3D<T>::operator()(int i, int j, int k)
{
#ifdef MFEM_DEBUG
   int N1 = array1d.Size()/N2/N3;
   if (i < 0 || i >= N1 || j < 0 || j >= N2 || k < 0 || k >= N3)
   {
      cerr << "Array3D: invalid access of element ("
           << i << ',' << j << ',' << k << ") in array of size ("
           << N1 << ',' << N2 << ',' << N3 << ")." << endl;
      mfem_error();
   }
#endif
   return array1d[(i*N2+j)*N3+k];
}

#endif
