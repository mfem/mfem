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

#include "../general/mem_manager.hpp"
#include "vector.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
#define isfinite _finite
#endif


namespace mfem
{


/// Vector data type.
template<typename dtype>
class TADVector
{
protected:

   Memory<dtype> data;
   int size;

public:

   /// Default constructor for Vector. Sets size = 0 and data = NULL.
   TADVector() { data.Reset(); size = 0; }

   /// Copy constructor. Allocates a new data array and copies the data.
   TADVector(const TADVector<dtype> &v)
   {
      const int s = v.Size();
      if (s > 0)
      {
         size = s;
         data.New(s);
         for (int i=0; i<s; i++)
         {
            data[i]=v[i];
         }
      }
      else
      {
         size = 0;
         data.Reset();
      }
   }

   TADVector(const Vector &v)
   {
      const int s = v.Size();
      if (s > 0)
      {
         size = s;
         data.New(s);
         for (int i=0; i<s; i++)
         {
            data[i]=v[i];
         }
      }
      else
      {
         size = 0;
         data.Reset();
      }
   }

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit TADVector(int s)
   {
      if (s > 0)
      {
         size = s;
         data.New(size);
      }
      else
      {
         size = 0;
         data.Reset();
      }
   }

   /// Creates a vector referencing an array of doubles, owned by someone else.
   /** The pointer @a _data can be NULL. The data array can be replaced later
       with SetData(). */
   TADVector(dtype *_data, int _size)
   { data.Wrap(_data, _size, false); size = _size; }

   /// Create a Vector of size @a size_ using MemoryType @a mt.
   TADVector(int size_, MemoryType mt)
      : data(size_, mt), size(size_) { }

   /// Enable execution of Vector operations using the mfem::Device.
   /** The default is to use Backend::CPU (serial execution on each MPI rank),
       regardless of the mfem::Device configuration.

       When appropriate, MFEM functions and class methods will enable the use
       of the mfem::Device for their Vector parameters.

       Some derived classes, e.g. GridFunction, enable the use of the
       mfem::Device by default. */
   void UseDevice(bool use_dev) const { data.UseDevice(use_dev); }

   /// Return the device flag of the Memory object used by the Vector
   bool UseDevice() const { return data.UseDevice(); }

   /// Reads a vector from multiple files
   void Load(std::istream ** in, int np, int * dim)
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
            data[p++]=dtype(tmpd);
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
         data[i]=dtype(tmpd);
      }
   }

   /// Load a vector from an input stream, reading the size from the stream.
   void Load(std::istream &in) { int s; in >> s; Load(in, s); }

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
      if (s <= data.Capacity())
      {
         size = s;
         return;
      }
      // preserve a valid MemoryType and device flag
      const MemoryType mt = data.GetMemoryType();
      const bool use_dev = data.UseDevice();
      data.Delete();
      size = s;
      data.New(s, mt);
      data.UseDevice(use_dev);
   }

   /// Resize the vector to size @a s using MemoryType @a mt.
   void SetSize(int s, MemoryType mt)
   {
      if (mt == data.GetMemoryType())
      {
         if (s == size)
         {
            return;
         }
         if (s <= data.Capacity())
         {
            size = s;
            return;
         }
      }
      const bool use_dev = data.UseDevice();
      data.Delete();
      if (s > 0)
      {
         data.New(s, mt);
         size = s;
      }
      else
      {
         data.Reset();
         size = 0;
      }
      data.UseDevice(use_dev);
   }

   /// Set the Vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(dtype *d) { data.Wrap(d, data.Capacity(), false); }

   /// Set the Vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @warning This method should be called only when OwnsData() is false.
       @sa NewDataAndSize(). */
   void SetDataAndSize(dtype *d, int s)
   { data.Wrap(d, s, false); size = s; }

   /// Set the Vector data and size, deleting the old data, if owned.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @sa SetDataAndSize(). */
   void NewDataAndSize(dtype *d, int s)
   {
      data.Delete();
      SetDataAndSize(d, s);
   }

   /// Reset the Vector to use the given external Memory @a mem and size @a s.
   /** If @a own_mem is false, the Vector will not own any of the pointers of
       @a mem.
       @sa NewDataAndSize(). */
   void NewMemoryAndSize(const Memory<dtype> &mem, int s, bool own_mem)
   {
      data.Delete();
      size = s;
      data = mem;
      if (!own_mem) { data.ClearOwnerFlags(); }

   }

   /// Reset the Vector to be a reference to a sub-vector of @a base.
   inline void MakeRef(TADVector<dtype> &base, int offset, int size_)
   {
      data.Delete();
      size = size_;
      data.MakeAlias(base.GetMemory(), offset, size_);
   }

   /** @brief Reset the Vector to be a reference to a sub-vector of @a base
       without changing its current size. */
   inline void MakeRef(TADVector<dtype> &base, int offset)
   {
      data.Delete();
      data.MakeAlias(base.GetMemory(), offset, size);
   }

   /// Set the Vector data (host pointer) ownership flag.
   inline void MakeDataOwner() const { data.SetHostPtrOwner(true); }

   /// Destroy a vector
   void Destroy()
   {
      data.Delete();
      size = 0;
      data.Reset();
   }

   /// Returns the size of the vector.
   inline int Size() const { return size; }

   /// Return the size of the currently allocated data array.
   /** It is always true that Capacity() >= Size(). */
   inline int Capacity() const { return data.Capacity(); }

   /// Return a pointer to the beginning of the Vector data.
   /** @warning This method should be used with caution as it gives write access
       to the data of const-qualified Vector%s. */
   inline dtype *GetData() const
   { return const_cast<dtype*>((const dtype*)data); }

   /// Conversion to `double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator dtype *() { return data; }

   /// Conversion to `const double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator const dtype *() const { return data; }

   /// Return a reference to the Memory object used by the Vector.
   Memory<dtype> &GetMemory() { return data; }

   /** @brief Return a reference to the Memory object used by the Vector, const
       version. */
   const Memory<dtype> &GetMemory() const { return data; }

   /// Update the memory location of the vector to match @a v.
   void SyncMemory(const TADVector<dtype> &v) { GetMemory().Sync(v.GetMemory()); }

   /// Update the alias memory location of the vector to match @a v.
   void SyncAliasMemory(const TADVector<dtype> &v)
   { GetMemory().SyncAlias(v.GetMemory(),Size()); }

   /// Read the Vector data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline void StealData(dtype **p)
   { *p = data; data.Reset(); size = 0; }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline dtype *StealData() { dtype *p; StealData(&p); return p; }

   /// Access Vector entries. Index i = 0 .. size-1.
   dtype &Elem(int i)
   {
      return operator()(i);
   }
   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const double &Elem(int i) const
   {
      return operator()(i);
   }

   /// Access Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline double &operator()(int i)
   {
      MFEM_ASSERT(data && i >= 0 && i < size,
                  "index [" << i << "] is out of range [0," << size << ")");

      return data[i];
   }

   /// Read only access to Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline const double &operator()(int i) const
   {
      MFEM_ASSERT(data && i >= 0 && i < size,
                  "index [" << i << "] is out of range [0," << size << ")");

      return data[i];
   }

   /// Dot product with a `dtype *` array.
   dtype operator*(const dtype *v) const
   {
      dtype dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
      #pragma omp parallel for reduction(+:dot)
#endif
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
      for (int i=0; i<size; i++)
      {
         data[i]=v[i];
      }
      return *this;
   }

   /// Copy assignment.
   /** @note Defining this method overwrites the implicitly defined copy
       assignemnt operator. */
   TADVector<dtype> &operator=(const TADVector<dtype> &v)
   {
      SetSize(v.Size());
      for (int i=0; i<size; i++)
      {
         data[i]=v[i];
      }
      return *this;
   }

   TADVector<dtype> &operator=(const Vector &v)
   {
      SetSize(v.Size());
      for (int i=0; i<size; i++)
      {
         data[i]=v[i];
      }
      return *this;
   }

   /// Redefine '=' for vector = constant.
   template<typename ivtype>
   TADVector &operator=(ivtype value)
   {
      for (int i=0; i<size; i++)
      {
         data[i]=value;
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator*=(ivtype c)
   {
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]*c;
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator/=(ivtype c)
   {
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]/c;
      }
      return *this;
   }

   template<typename ivtype>
   TADVector &operator-=(ivtype c)
   {
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]-c;
      }
      return *this;
   }

   TADVector &operator-=(const TADVector<dtype> &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]-v[i];
      }
      return *this;
   }

   TADVector &operator+=(const TADVector<dtype> &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]+v[i];
      }
      return *this;
   }

   /// (*this) += a * Va
   template<typename ivtype, typename vtype>
   TADVector &Add(const ivtype a, const vtype &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i=0; i<size; i++)
      {
         data[i]=data[i]+a*v[i];
      }
      return *this;
   }

   /// (*this) = a * x
   template<typename ivtype, typename vtype>
   TADVector &Set(const ivtype a, const vtype &v)
   {
      MFEM_ASSERT(size == v.Size(), "incompatible Vectors!");
      for (int i=0; i<size; i++)
      {
         data[i]=a*v[i];
      }
      return *this;
   }

   template<typename vtype>
   void SetVector(const vtype &v, int offset)
   {
      MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");
      for (int i = 0; i < size; i++)
      {
         data[i+offset] = v[i];
      }
   }

   /// (*this) = -(*this)
   void Neg()
   {
      for (int i = 0; i < size; i++)
      {
         data[i]=-data[i];
      }
   }

   /// Swap the contents of two Vectors
   inline void Swap(TADVector &other)
   {
      Swap(data, other.data);
      Swap(size, other.size);
   }

   /// Set v = v1 + v2.
   template<typename vtype1, typename vtype2>
   friend void add(const vtype1 &v1, const vtype2 &v2, TADVector<dtype> &v)
   {
      MFEM_ASSERT(v1.Size() == v.Size(), "incompatible Vectors!");
      MFEM_ASSERT(v2.Size() == v.Size(), "incompatible Vectors!");
      for (int i=0; i<v.Size(); i++)
      {
         v[i]=v1[i]+v2[i];
      }
   }

   /// Set v = v1 + alpha * v2.
   template<typename vtype1, typename ivtype, typename vtype2>
   friend void add(const vtype1 &v1, ivtype alpha, const vtype2 &v2,
                   TADVector<dtype> &v)
   {
      MFEM_ASSERT(v1.Size() == v.Size(), "incompatible Vectors!");
      MFEM_ASSERT(v2.Size() == v.Size(), "incompatible Vectors!");
      for (int i=0; i<v.Size(); i++)
      {
         v[i]=v1[i]+alpha*v2[i];
      }
   }


   /// Destroys vector.
   ~TADVector()
   {
      data.Delete();
   }


   /// Prints vector to stream out.
   void Print(std::ostream &out = mfem::out, int width = 8) const
   {
      if (!size) { return; }
      data.Read(MemoryClass::HOST, size);
      for (int i = 0; 1; )
      {
         out << data[i];
         i++;
         if (i == size)
         {
            break;
         }
         if ( i % width == 0 )
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
      const double max = (double)(RAND_MAX) + 1.;

      if (seed == 0)
      {
         seed = (int)time(0);
      }

      // srand(seed++);
      srand((unsigned)seed);

      for (int i = 0; i < size; i++)
      {
         data[i] = std::abs(rand()/max);
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
         return std::abs(data[0]);
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
         } // end if data[i] != 0
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

