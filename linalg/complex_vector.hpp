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

#ifndef MFEM_COMPLEX_VECTOR
#define MFEM_COMPLEX_VECTOR

#include "vector.hpp"
#include "../general/complex_type.hpp"

namespace mfem
{

class ComplexVector
{
private:

   Memory<complex_t > data;
   int size;

   mutable Vector re_part;
   mutable Vector im_part;

public:

   /// Default constructor for ComplexVector. Sets size = 0
   ComplexVector() : size(0) { }

   /// Copy constructor. Allocates a new data array and copies the data.
   ComplexVector(const ComplexVector &);

   /// Copy constructor. Allocates a new data array and copies the
   /// data into real part of this vector.
   ComplexVector(const Vector &);

   /// Move constructor. "Steals" data from its argument.
   ComplexVector(ComplexVector&& v);

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit ComplexVector(int s);

   /// Creates a vector referencing an array of complex<doubles>,
   /// owned by someone else.
   /// The pointer @a data_ can be NULL. The data array can be replaced later
   /// with SetData().
   ComplexVector(complex_t *data_, int size_)
   { data.Wrap(data_, size_, false); size = size_; }

   /// @brief Create a ComplexVector referencing a sub-vector of the
   //  ComplexVector @a base starting at the given offset, @a
   //  base_offset, and size @a size_.
   ComplexVector(ComplexVector &base, int base_offset, int size_)
      : data(base.data, base_offset, size_), size(size_) { }

   /// Create a ComplexVector of size @a size_ using MemoryType @a mt.
   ComplexVector(int size_, MemoryType mt)
      : data(size_, mt), size(size_) { }

   /// @brief Create a ComplexVector of size @a size_ using host
   /// MemoryType @a h_mt and device MemoryType @a d_mt.
   ComplexVector(int size_, MemoryType h_mt, MemoryType d_mt)
      : data(size_, h_mt, d_mt), size(size_) { }

   /// Create a vector from a statically sized C-style array of convertible type
   template <typename CT, int N>
   explicit ComplexVector(const CT (&values)[N]) : ComplexVector(N)
   { std::copy(values, values + N, begin()); }

   /// Create a vector using a braced initializer list
   template <typename CT, typename std::enable_if<
                std::is_convertible<CT,complex_t >::value,bool>::type = true>
   explicit ComplexVector(std::initializer_list<CT> values) : ComplexVector(
         values.size())
   { std::copy(values.begin(), values.end(), begin()); }

   /// Enable execution of Vector operations using the mfem::Device.
   /// The default is to use Backend::CPU (serial execution on each MPI rank),
   /// regardless of the mfem::Device configuration.
   ///
   /// When appropriate, MFEM functions and class methods will enable the use
   /// of the mfem::Device for their Vector parameters.
   ///
   /// Some derived classes, e.g. GridFunction, enable the use of the
   /// mfem::Device by default.
   virtual void UseDevice(bool use_dev) const { data.UseDevice(use_dev); }

   /// Return the device flag of the Memory object used by the Vector
   virtual bool UseDevice() const { return data.UseDevice(); }

   /// @brief Resize the vector to size @a s.
   /// If the new size is less than or equal to Capacity() then the internal
   /// data array remains the same. Otherwise, the old array is deleted, if
   /// owned, and a new array of size @a s is allocated without copying the
   /// previous content of the ComplexVector.
   /// @warning In the second case above (new size greater than current one),
   /// the vector will allocate new data array, even if it did not own the
   /// original data! Also, new entries are not initialized!
   void SetSize(int s);

   /// Resize the vector to size @a s using MemoryType @a mt.
   void SetSize(int s, MemoryType mt);

   /// Resize the vector to size @a s using the MemoryType of @a v.
   void SetSize(int s, const ComplexVector &v)
   { SetSize(s, v.GetMemory().GetMemoryType()); }

   /// Resize the vector to size @a s using the MemoryType of @a v.
   void SetSize(int s, const Vector &v)
   { SetSize(s, v.GetMemory().GetMemoryType()); }

   /// Set the Vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(complex_t *d)
   { data.Wrap(d, data.Capacity(), false); }

   /// Set the Vector data and size.
   /// The Vector does not assume ownership of the new data. The new size is
   /// also used as the new Capacity().
   /// @warning This method should be called only when OwnsData() is false.
   /// @sa NewDataAndSize().
   void SetDataAndSize(complex_t *d, int s)
   { data.Wrap(d, s, false); size = s; }

   /// Set the Vector data and size, deleting the old data, if owned.
   /// The Vector does not assume ownership of the new data. The new size is
   /// also used as the new Capacity().
   /// @sa SetDataAndSize().
   void NewDataAndSize(complex_t *d, int s)
   {
      data.Delete();
      SetDataAndSize(d, s);
   }

   /// Reset the Vector to use the given external Memory @a mem and size @a s.
   /// If @a own_mem is false, the Vector will not own any of the pointers of
   /// @a mem.
   ///
   /// Note that when @a own_mem is true, the @a mem object can be destroyed
   /// immediately by the caller but `mem.Delete()` should NOT be called since
   /// the Vector object takes ownership of all pointers owned by @a mem.
   ///
   /// @sa NewDataAndSize().
   inline void NewMemoryAndSize(const Memory<complex_t > &mem,
                                int s, bool own_mem);

   /// Reset the Vector to be a reference to a sub-vector of @a base.
   inline void MakeRef(ComplexVector &base, int offset, int size);

   /// @brief Reset the Vector to be a reference to a sub-vector of @a base
   /// without changing its current size.
   inline void MakeRef(ComplexVector &base, int offset);

   /// Set the Vector data (host pointer) ownership flag.
   void MakeDataOwner() const { data.SetHostPtrOwner(true); }

   /// Destroy a vector
   void Destroy();

   /// @brief Delete the device pointer, if owned. If @a copy_to_host is true
   /// and the data is valid only on device, move it to host before deleting.
   /// Invalidates the device memory.
   void DeleteDevice(bool copy_to_host = true)
   { data.DeleteDevice(copy_to_host); }

   /// Returns the size of the vector.
   inline int Size() const { return size; }

   /// Return the size of the currently allocated data array.
   /// It is always true that Capacity() >= Size().
   inline int Capacity() const { return data.Capacity(); }

   /// Return a pointer to the beginning of the ComplexVector data.
   /// @warning This method should be used with caution as it gives write access
   /// to the data of const-qualified ComplexVector%s.
   inline complex_t *GetData() const
   { return const_cast<complex_t*>((const complex_t*)data); }

   /// STL-like begin.
   inline complex_t *begin() { return data; }

   /// STL-like end.
   inline complex_t *end() { return data + size; }

   /// STL-like begin (const version).
   inline const complex_t *begin() const { return data; }

   /// STL-like end (const version).
   inline const complex_t *end() const { return data + size; }

   /// Return a reference to the Memory object used by the Vector.
   Memory<complex_t > &GetMemory() { return data; }

   /// @brief Return a reference to the Memory object used by the
   /// ComplexVector, const version.
   const Memory<complex_t > &GetMemory() const { return data; }

   /// Update the memory location of the vector to match @a v.
   void SyncMemory(const ComplexVector &v) const
   { GetMemory().Sync(v.GetMemory()); }

   /// Update the alias memory location of the vector to match @a v.
   void SyncAliasMemory(const ComplexVector &v) const
   { GetMemory().SyncAlias(v.GetMemory(),Size()); }

   /// Read the Vector data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline void StealData(complex_t **p)
   { *p = data; data.Reset(); size = 0; }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline complex_t *StealData()
   { complex_t *p; StealData(&p); return p; }

   /// Access Vector entries. Index i = 0 .. size-1.
   complex_t &Elem(int i);

   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const complex_t &Elem(int i) const;

   /// Access Vector entries using () for 0-based indexing.
   /// @note If MFEM_DEBUG is enabled, bounds checking is performed.
   inline complex_t &operator()(int i);

   /// Read only access to Vector entries using () for 0-based indexing.
   /// @note If MFEM_DEBUG is enabled, bounds checking is performed.
   inline const complex_t &operator()(int i) const;

   /// Access Vector entries using [] for 0-based indexing.
   /// @note If MFEM_DEBUG is enabled, bounds checking is performed.
   inline complex_t &operator[](int i) { return (*this)(i); }

   /// Read only access to Vector entries using [] for 0-based indexing.
   /// @note If MFEM_DEBUG is enabled, bounds checking is performed.
   inline const complex_t &operator[](int i) const
   { return (*this)(i); }

   /// Dot product with a `complex<double> *` array.
   /// @note No complex conjugate is performed
   complex_t operator*(const complex_t *v) const;
   complex_t operator*(const real_t *v) const;

   /// Return the inner-product.
   /// @note No complex conjugate is performed
   complex_t operator*(const ComplexVector &v) const;
   complex_t operator*(const Vector &v) const;

   /// Copy Size() entries from @a v.
   ComplexVector &operator=(const complex_t *v);
   ComplexVector &operator=(const real_t *v);

   /// Copy assignment.
   /// @note Defining this method overwrites the implicitly defined copy
   /// assignment operator.
   ComplexVector &operator=(const ComplexVector &v);
   ComplexVector &operator=(const Vector &v);

   /// Move assignment
   ComplexVector &operator=(ComplexVector&& v);

   /// Redefine '=' for vector = constant.
   ComplexVector &operator=(complex_t value);
   ComplexVector &operator=(real_t value);

   /// Scale vector by a constant
   ComplexVector &operator*=(complex_t c);
   ComplexVector &operator*=(real_t c);

   /// Component-wise scaling: (*this)(i) *= v(i)
   ComplexVector &operator*=(const ComplexVector &v);
   ComplexVector &operator*=(const Vector &v);

   /// Divide vector by a consant
   ComplexVector &operator/=(complex_t c);
   ComplexVector &operator/=(real_t c);

   /// Component-wise division: (*this)(i) /= v(i)
   ComplexVector &operator/=(const ComplexVector &v);
   ComplexVector &operator/=(const Vector &v);

   /// Subtract a constant from this vector
   ComplexVector &operator-=(complex_t c);
   ComplexVector &operator-=(real_t c);

   /// Subtract a vector from this vector
   ComplexVector &operator-=(const ComplexVector &v);
   ComplexVector &operator-=(const Vector &v);

   /// Add a constant to this vector
   ComplexVector &operator+=(complex_t c);
   ComplexVector &operator+=(real_t c);

   /// Add a vector to this vector
   ComplexVector &operator+=(const ComplexVector &v);
   ComplexVector &operator+=(const Vector &v);

   /// (*this) = x + i * y
   ComplexVector &Set(const Vector &x, const Vector &y);

   /// Swap the contents of two Vectors
   inline void Swap(ComplexVector &other);

   /// Return a reference to the real part of this vector
   const Vector &real() const;

   /// Return a reference to the imaginary part of this vector
   const Vector &imag() const;

   /// Destroys vector.
   virtual ~ComplexVector();

   /// Shortcut for mfem::Read(vec.GetMemory(), vec.Size(), on_dev).
   virtual const complex_t *Read(bool on_dev = true) const
   { return mfem::Read(data, size, on_dev); }

   /// Shortcut for mfem::Read(vec.GetMemory(), vec.Size(), false).
   virtual const complex_t *HostRead() const
   { return mfem::Read(data, size, false); }

   /// Shortcut for mfem::Write(vec.GetMemory(), vec.Size(), on_dev).
   virtual complex_t *Write(bool on_dev = true)
   { return mfem::Write(data, size, on_dev); }

   /// Shortcut for mfem::Write(vec.GetMemory(), vec.Size(), false).
   virtual complex_t *HostWrite()
   { return mfem::Write(data, size, false); }

   /// Shortcut for mfem::ReadWrite(vec.GetMemory(), vec.Size(), on_dev).
   virtual complex_t *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, size, on_dev); }

   /// Shortcut for mfem::ReadWrite(vec.GetMemory(), vec.Size(), false).
   virtual complex_t *HostReadWrite()
   { return mfem::ReadWrite(data, size, false); }
};

inline ComplexVector::ComplexVector(int s)
{
   MFEM_ASSERT(s>=0,"Unexpected negative size.");
   size = s;
   if (s > 0)
   {
      data.New(s);
   }
}

inline void ComplexVector::SetSize(int s)
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

inline void ComplexVector::SetSize(int s, MemoryType mt)
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

inline void ComplexVector::NewMemoryAndSize(
   const Memory<complex_t > &mem,
   int s,
   bool own_mem)
{
   data.Delete();
   size = s;
   if (own_mem)
   {
      data = mem;
   }
   else
   {
      data.MakeAlias(mem, 0, s);
   }
}

inline void ComplexVector::MakeRef(ComplexVector &base, int offset, int s)
{
   data.Delete();
   size = s;
   data.MakeAlias(base.GetMemory(), offset, s);
}

inline void ComplexVector::MakeRef(ComplexVector &base, int offset)
{
   data.Delete();
   data.MakeAlias(base.GetMemory(), offset, size);
}

inline void ComplexVector::Destroy()
{
   const bool use_dev = data.UseDevice();
   data.Delete();
   size = 0;
   data.Reset();
   data.UseDevice(use_dev);
}

inline complex_t &ComplexVector::operator()(int i)
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline const complex_t &ComplexVector::operator()(int i) const
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline void ComplexVector::Swap(ComplexVector &other)
{
   mfem::Swap(data, other.data);
   mfem::Swap(size, other.size);
}

/// Specialization of the template function Swap<> for class ComplexVector
template<> inline void Swap<ComplexVector>(ComplexVector &a, ComplexVector &b)
{
   a.Swap(b);
}

inline ComplexVector::~ComplexVector()
{
   data.Delete();
}

} // namespace mfem

#endif
