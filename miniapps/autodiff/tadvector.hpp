// Copyright (c) 2020, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TADVECTOR
#define MFEM_TADVECTOR

#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
#define isfinite _finite
#endif

namespace mfem
{
/// Templated vector data type.
/** The main goal of the TADVector class is to serve as a data
  container for representing vectors in classes, methods, and
  functions utilized with automatic differentiation (AD). The
  functionality/interface is copied from the standard MFEM dense
  vector mfem::Vector.  The basic idea is to utilize the templated
  vector class in combination with AD during the development phase.
  The AD parts can be replaced with optimized code once the initial
  development of the application is complete.  The common interface
  between TADVector and Vector will ease the transition from AD to
  hand-optimized code as it does not require a change in the
  interface or the code structure. TADVector is intended to be
  utilized for dense serial vectors. */
template<typename dtype>
class TADVector
{
protected:
   dtype *data;
   int size;
   int capacity;

public:
   /// Default constructor for Vector. Sets size = 0 and data = NULL.
   TADVector()
   {
      data = nullptr;
      size = 0;
      capacity = 0;
   }

   /// Copy constructor. Allocates a new data array and copies the data.
   TADVector(const TADVector<dtype> &v)
   {
      const int s = v.Size();
      if (s > 0)
      {
         size = s;
         data = new dtype[s];
         capacity = s;
         for (int i = 0; i < s; i++)
         {
            data[i] = v[i];
         }
      }
      else
      {
         size = 0;
         capacity = 0;
         data = nullptr;
      }
   }

   TADVector(const Vector &v)
   {
      const int s = v.Size();
      if (s > 0)
      {
         size = s;
         capacity = s;
         data = new dtype[s];
         for (int i = 0; i < s; i++)
         {
            data[i] = v[i];
         }
      }
      else
      {
         size = 0;
         capacity = 0;
         data = nullptr;
      }
   }

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit TADVector(int s)
   {
      if (s > 0)
      {
         size = s;
         capacity = s;
         data = new dtype[size];
      }
      else
      {
         size = 0;
         capacity = 0;
         data = nullptr;
      }
   }

   /// Creates a vector referencing an array of doubles, owned by someone else.
   /** The pointer @a _data can be NULL. The data array can be replaced later
       with SetData(). */
   TADVector(dtype *_data, int _size)
   {
      if (capacity > 0)
      {
         delete[] data;
         capacity = 0;
      }
      size = _size;
      data = _data;
   }

   /// Reads a vector from multiple files
   void Load(std::istream **in, int np, int *dim)
   {
      int i, j, s;

      s = 0;
      for (i = 0; i < np; i++)
      {
         s += dim[i];
      }
      SetSize(s);

      int p = 0;
      double tmpd;
      for (i = 0; i < np; i++)
      {
         for (j = 0; j < dim[i]; j++)
         {
            *in[i] >> tmpd;
            data[p++] = dtype(tmpd);
         }
      }
   }

   /// Load a vector from an input stream.
   void Load(std::istream &in, int Size)
   {
      SetSize(Size);
      double tmpd;
      for (int i = 0; i < size; i++)
      {
         in >> tmpd;
         data[i] = dtype(tmpd);
      }
   }

   /// Load a vector from an input stream, reading the size from the stream.
   void Load(std::istream &in)
   {
      int s;
      in >> s;
      Load(in, s);
   }

   /// @brief Resize the vector to size @a s.
   /** If the new size is less than or equal to Capacity() then the internal
       data array remains the same. Otherwise, the old array is deleted, if
       owned, and a new array of size @a s is allocated without copying the
       previous content of the Vector.
       @warning In the second case above (new size greater than current one),
       the vector will allocate new data array, even if it did not own the
       original data! Also, new entries are not initialized! */
   void SetSize(int s)
   {
      if (s == size)
      {
         return;
      }

      if (s <= capacity)
      {
         size = s;
         return;
      }

      delete[] data;
      data = new dtype[s];
      size = s;
      capacity = s;
   }

   /// Set the Vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       @warning This method should be called only when OwnsData() is false.
       @sa NewDataAndSize(). */
   void SetDataAndSize(dtype *d, int s)
   {
      if (OwnsData())
      {
         delete[] data;
         capacity = 0;
      }
      data = d;
      size = s;
   }

   /// Set the Vector data and size, deleting the old data, if owned.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @sa SetDataAndSize(). */
   void NewDataAndSize(dtype *d, int s) { SetDataAndSize(d, s); }

   /// Reset the Vector to be a reference to a sub-vector of @a base.
   inline void MakeRef(TADVector<dtype> &base, int offset, int size_)
   {
      NewDataAndSize(base.GetData() + offset, size_);
   }

   /** @brief Reset the Vector to be a reference to a sub-vector of @a base
       without changing its current size. */
   inline void MakeRef(TADVector<dtype> &base, int offset)
   {
      int tsiz = size;
      NewDataAndSize(base.GetData() + offset, tsiz);
   }

   /// Destroy a vector
   void Destroy()
   {
      size = 0;
      capacity = 0;
      delete[] data;
   }

   /// Returns the size of the vector.
   inline int Size() const { return size; }

   /// Return the size of the currently allocated data array.
   /** It is always true that Capacity() >= Size(). */
   inline int Capacity() const { return capacity; }

   /// Return a pointer to the beginning of the Vector data.
   /** @warning This method should be used with caution as it gives write access
       to the data of const-qualified Vector%s. */
   inline dtype *GetData() const
   {
      return const_cast<dtype *>((const dtype *) data);
   }

   /// Conversion to `double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator dtype *() { return data; }

   /// Conversion to `const double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator const dtype *() const { return data; }

   /// Read the Vector data (host pointer) ownership flag.
   inline bool OwnsData() const { return (capacity > 0); }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline void StealData(dtype **p)
   {
      *p = data;
      delete[] data;
      size = 0;
      capacity = 0;
   }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline dtype *StealData()
   {
      dtype *p;
      StealData(&p);
      return p;
   }

   /// Access Vector entries. Index i = 0 .. size-1.
   dtype &Elem(int i) { return operator()(i); }
   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const dtype &Elem(int i) const { return operator()(i); }

   /// Access Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline dtype &operator()(int i)
   {
      MFEM_ASSERT(data && i >= 0 && i < size,
                  "index [" << i << "] is out of range [0," << size << ")");

      return data[i];
   }

   /// Read only access to Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline const dtype &operator()(int i) const
   {
      MFEM_ASSERT(data && i >= 0 && i < size,
                  "index [" << i << "] is out of range [0," << size << ")");

      return data[i];
   }

   /// Dot product with a `dtype *` array.
   dtype operator*(const dtype *v) const
   {
      dtype dot = 0.0;
      for (int i = 0; i < size; i++)
      {
         dot += data[i] * v[i];
      }
      return dot;
   }

   /// Return the inner-product.
   dtype operator*(const TADVector<dtype> &v) const
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      dtype dot = 0.0;
      for (int i = 0; i < size; i++)
      {
         dot += data[i] * v[i];
      }
      return dot;
   }

   dtype operator*(const Vector &v) const
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      dtype dot = 0.0;
      for (int i = 0; i < size; i++)
      {
         dot += data[i] * v[i];
      }
      return dot;
   }

   /// Copy Size() entries from @a v.
   TADVector<dtype> &operator=(const dtype *v)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = v[i];
      }
      return *this;
   }

   /// Copy assignment.
   /** @note Defining this method overwrites the implicitly defined copy
       assignemnt operator. */
   TADVector<dtype> &operator=(const TADVector<dtype> &v)
   {
      SetSize(v.Size());
      for (int i = 0; i < size; i++)
      {
         data[i] = v[i];
      }
      return *this;
   }

   TADVector<dtype> &operator=(const Vector &v)
   {
      SetSize(v.Size());
      for (int i = 0; i < size; i++)
      {
         data[i] = v[i];
      }
      return *this;
   }

   /// Redefine '=' for vector = constant.
   template<typename ivtype>
   TADVector &operator=(ivtype value)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = (dtype) value;
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator*=(ivtype c)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] * c;
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator/=(ivtype c)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] / c;
      }
      return *this;
   }

   TADVector &operator-=(const TADVector<dtype> &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] - v[i];
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator-=(ivtype v)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] - v;
      }
      return *this;
   }

   TADVector &operator+=(const TADVector<dtype> &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] + v[i];
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator+=(ivtype v)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] + v;
      }
      return *this;
   }

   /// (*this) += a * Va
   template<typename ivtype, typename vtype>
   TADVector &Add(const ivtype a, const vtype &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < size; i++)
      {
         data[i] = data[i] + a * v[i];
      }
      return *this;
   }

   /// (*this) = a * x
   template<typename ivtype, typename vtype>
   TADVector &Set(const ivtype a, const vtype &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < size; i++)
      {
         data[i] = a * v[i];
      }
      return *this;
   }

   template<typename vtype>
   void SetVector(const vtype &v, int offset)
   {
      MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");
      for (int i = 0; i < size; i++)
      {
         data[i + offset] = v[i];
      }
   }

   /// (*this) = -(*this)
   void Neg()
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = -data[i];
      }
   }

   /// Swap the contents of two Vectors
   inline void Swap(TADVector<dtype> &other)
   {
      Swap(data, other.data);
      Swap(size, other.size);
      Swap(capacity, other.capacity);
   }

   /// Set v = v1 + v2.
   template<typename vtype1, typename vtype2>
   friend void add(const vtype1 &v1, const vtype2 &v2, TADVector<dtype> &v)
   {
      MFEM_ASSERT(v1.Size() == v.Size(), "incompatible Vectors!");
      MFEM_ASSERT(v2.Size() == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < v.Size(); i++)
      {
         v[i] = v1[i] + v2[i];
      }
   }

   /// Set v = v1 + alpha * v2.
   template<typename vtype1, typename vtype2>
   friend void add(const vtype1 &v1,
                   dtype alpha,
                   const vtype2 &v2,
                   TADVector<dtype> &v)
   {
      MFEM_ASSERT(v1.Size() == v.Size(), "incompatible Vectors!");
      MFEM_ASSERT(v2.Size() == v.Size(), "incompatible Vectors!");
      for (int i = 0; i < v.Size(); i++)
      {
         v[i] = v1[i] + alpha * v2[i];
      }
   }

   template<typename vtype1, typename vtype2>
   friend void add(const dtype a,
                   const vtype1 &x,
                   const dtype b,
                   const vtype2 &y,
                   TADVector<dtype> &z)
   {
      MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
                  "incompatible Vectors!");

      for (int i = 0; i < z.Size(); i++)
      {
         z[i] = a * x[i] + b * y[i];
      }
   }

   template<typename vtype1, typename vtype2>
   friend void add(const dtype a,
                   const vtype1 &x,
                   const vtype2 &y,
                   TADVector<dtype> &z)
   {
      MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
                  "incompatible Vectors!");

      for (int i = 0; i < z.Size(); i++)
      {
         z[i] = a * x[i] + y[i];
      }
   }

   template<typename vtype1, typename vtype2>
   friend void subtract(const vtype1 &x, const vtype2 &y, TADVector<dtype> &z)
   {
      MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
                  "incompatible Vectors!");
      for (int i = 0; i < z.Size(); i++)
      {
         z[i] = x[i] - y[i];
      }
   }

   template<typename ivtype, typename vtype1, typename vtype2>
   friend void subtract(const ivtype a,
                        const vtype1 &x,
                        const vtype2 &y,
                        TADVector<dtype> &z)
   {
      MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
                  "incompatible Vectors!");
      for (int i = 0; i < z.Size(); i++)
      {
         z[i] = a * (x[i] - y[i]);
      }
   }

   /// Destroys vector.
   ~TADVector() { delete[] data; }

   /// Prints vector to stream out.
   void Print(std::ostream &out = mfem::out, int width = 8) const
   {
      if (!size)
      {
         return;
      }
      for (int i = 0; 1;)
      {
         out << data[i];
         i++;
         if (i == size)
         {
            break;
         }
         if (i % width == 0)
         {
            out << '\n';
         }
         else
         {
            out << ' ';
         }
      }
      out << '\n';
   }

   /// Set random values in the vector.
   void Randomize(int seed = 0)
   {
      // static unsigned int seed = time(0);
      const double max = (double) (RAND_MAX) + 1.;

      if (seed == 0)
      {
         seed = (int) time(0);
      }

      // srand(seed++);
      srand((unsigned) seed);

      for (int i = 0; i < size; i++)
      {
         data[i] = std::abs(rand() / max);
      }
   }
   /// Returns the l2 norm of the vector.
   dtype Norml2() const
   {
      // Scale entries of Vector on the fly, using algorithms from
      // std::hypot() and LAPACK's drm2. This scaling ensures that the
      // argument of each call to std::pow is <= 1 to avoid overflow.
      if (0 == size)
      {
         return 0.0;
      } // end if 0 == size

      if (1 == size)
      {
         return abs(data[0]);
      } // end if 1 == size

      dtype scale = 0.0;
      dtype sum = 0.0;

      for (int i = 0; i < size; i++)
      {
         if (data[i] != 0.0)
         {
            const dtype absdata = abs(data[i]);
            if (scale <= absdata)
            {
               const dtype sqr_arg = scale / absdata;
               sum = 1.0 + sum * (sqr_arg * sqr_arg);
               scale = absdata;
               continue;
            } // end if scale <= absdata
            const dtype sqr_arg = absdata / scale;
            sum += (sqr_arg * sqr_arg); // else scale > absdata
         }                              // end if data[i] != 0
      }
      return scale * sqrt(sum);
   }

   /// Returns the l_infinity norm of the vector.
   dtype Normlinf() const
   {
      dtype max = 0.0;
      for (int i = 0; i < size; i++)
      {
         max = max(abs(data[i]), max);
      }
      return max;
   }
   /// Returns the l_1 norm of the vector.
   dtype Norml1() const
   {
      dtype sum = 0.0;
      for (int i = 0; i < size; i++)
      {
         sum += abs(data[i]);
      }
      return sum;
   }
};

} // namespace mfem

#endif
