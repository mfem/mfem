// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_VECTOR
#define MFEM_VECTOR

#include "../general/array.hpp"
#ifdef MFEM_USE_ADIOS2
#include "../general/adios2stream.hpp"
#endif
#include "../general/globals.hpp"
#include "../general/mem_manager.hpp"
#include "../general/device.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <initializer_list>
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
#define isfinite _finite
#endif

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

/** Count the number of entries in an array of doubles for which isfinite
    is false, i.e. the entry is a NaN or +/-Inf. */
inline int CheckFinite(const real_t *v, const int n);

/// Define a shortcut for std::numeric_limits<double>::infinity()
#ifndef __CYGWIN__
inline real_t infinity()
{
   return std::numeric_limits<real_t>::infinity();
}
#else
// On Cygwin math.h defines a function 'infinity()' which will conflict with the
// above definition if we have 'using namespace mfem;' and try to use something
// like 'double a = infinity();'. This 'infinity()' function is non-standard and
// is defined by the Newlib C standard library implementation used by Cygwin,
// see https://en.wikipedia.org/wiki/Newlib, http://www.sourceware.org/newlib.
using ::infinity;
#endif

/// Generate a random `real_t` number in the interval [0,1) using rand().
inline real_t rand_real()
{
   constexpr real_t max = (real_t)(RAND_MAX) + 1_r;
#if defined(MFEM_USE_SINGLE)
   // Note: For RAND_MAX = 2^31-1, float(RAND_MAX) = 2^31, and
   // max = float(RAND_MAX)+1.0f is equal to 2^31 too; this is the actual value
   // that we want for max. However, this rounding behavior means that for a
   // range of values of rand() close to RAND_MAX, the expression
   // float(rand())/max will give 1.0f.
   //
   // Therefore, to ensure we return a number less than 1, we take the minimum
   // of float(rand())/max and (1 - 2^(-24)), where the latter number is the
   // largest float less than 1.
   return std::fmin(real_t(rand())/max, 0.9999999403953552_r);
#else
   return real_t(rand())/max;
#endif
}

/// Vector data type.
class Vector
{
protected:

   Memory<real_t> data;
   int size;

public:

   /** Default constructor for Vector. Sets size = 0, and calls Memory::Reset on
       data through Memory<double>'s default constructor. */
   Vector(): size(0) { }

   /// Copy constructor. Allocates a new data array and copies the data.
   Vector(const Vector &);

   /// Move constructor. "Steals" data from its argument.
   Vector(Vector&& v);

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit Vector(int s);

   /// Creates a vector referencing an array of doubles, owned by someone else.
   /** The pointer @a data_ can be NULL. The data array can be replaced later
       with SetData(). */
   Vector(real_t *data_, int size_)
   { data.Wrap(data_, size_, false); size = size_; }

   /** @brief Create a Vector referencing a sub-vector of the Vector @a base
       starting at the given offset, @a base_offset, and size @a size_. */
   Vector(Vector &base, int base_offset, int size_)
      : data(base.data, base_offset, size_), size(size_) { }

   /// Create a Vector of size @a size_ using MemoryType @a mt.
   Vector(int size_, MemoryType mt)
      : data(size_, mt), size(size_) { }

   /** @brief Create a Vector of size @a size_ using host MemoryType @a h_mt and
       device MemoryType @a d_mt. */
   Vector(int size_, MemoryType h_mt, MemoryType d_mt)
      : data(size_, h_mt, d_mt), size(size_) { }

   /// Create a vector from a statically sized C-style array of convertible type
   template <typename CT, int N>
   explicit Vector(const CT (&values)[N]) : Vector(N)
   { std::copy(values, values + N, begin()); }

   /// Create a vector using a braced initializer list
   template <typename CT, typename std::enable_if<
                std::is_convertible<CT,real_t>::value,bool>::type = true>
   explicit Vector(std::initializer_list<CT> values) : Vector(values.size())
   { std::copy(values.begin(), values.end(), begin()); }

   /// Enable execution of Vector operations using the mfem::Device.
   /** The default is to use Backend::CPU (serial execution on each MPI rank),
       regardless of the mfem::Device configuration.

       When appropriate, MFEM functions and class methods will enable the use
       of the mfem::Device for their Vector parameters.

       Some derived classes, e.g. GridFunction, enable the use of the
       mfem::Device by default. */
   virtual void UseDevice(bool use_dev) const { data.UseDevice(use_dev); }

   /// Return the device flag of the Memory object used by the Vector
   virtual bool UseDevice() const { return data.UseDevice(); }

   /// Reads a vector from multiple files
   void Load(std::istream ** in, int np, int * dim);

   /// Load a vector from an input stream.
   void Load(std::istream &in, int Size);

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
   void SetSize(int s);

   /// Resize the vector to size @a s using MemoryType @a mt.
   void SetSize(int s, MemoryType mt);

   /// Resize the vector to size @a s using the MemoryType of @a v.
   void SetSize(int s, const Vector &v) { SetSize(s, v.GetMemory().GetMemoryType()); }

   /// Set the Vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(real_t *d) { data.Wrap(d, data.Capacity(), false); }

   /// Set the Vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @warning This method should be called only when OwnsData() is false.
       @sa NewDataAndSize(). */
   void SetDataAndSize(real_t *d, int s) { data.Wrap(d, s, false); size = s; }

   /// Set the Vector data and size, deleting the old data, if owned.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @sa SetDataAndSize(). */
   void NewDataAndSize(real_t *d, int s)
   {
      data.Delete();
      SetDataAndSize(d, s);
   }

   /// Reset the Vector to use the given external Memory @a mem and size @a s.
   /** If @a own_mem is false, the Vector will not own any of the pointers of
       @a mem.

       Note that when @a own_mem is true, the @a mem object can be destroyed
       immediately by the caller but `mem.Delete()` should NOT be called since
       the Vector object takes ownership of all pointers owned by @a mem.

       @sa NewDataAndSize(). */
   inline void NewMemoryAndSize(const Memory<real_t> &mem, int s, bool own_mem);

   /// Reset the Vector to be a reference to a sub-vector of @a base.
   inline void MakeRef(Vector &base, int offset, int size);

   /** @brief Reset the Vector to be a reference to a sub-vector of @a base
       without changing its current size. */
   inline void MakeRef(Vector &base, int offset);

   /// Set the Vector data (host pointer) ownership flag.
   void MakeDataOwner() const { data.SetHostPtrOwner(true); }

   /// Destroy a vector
   void Destroy();

   /** @brief Delete the device pointer, if owned. If @a copy_to_host is true
       and the data is valid only on device, move it to host before deleting.
       Invalidates the device memory. */
   void DeleteDevice(bool copy_to_host = true)
   { data.DeleteDevice(copy_to_host); }

   /// Returns the size of the vector.
   inline int Size() const { return size; }

   /// Return the size of the currently allocated data array.
   /** It is always true that Capacity() >= Size(). */
   inline int Capacity() const { return data.Capacity(); }

   /// Return a pointer to the beginning of the Vector data.
   /** @warning This method should be used with caution as it gives write access
       to the data of const-qualified Vector%s. */
   inline real_t *GetData() const
   { return const_cast<real_t*>((const real_t*)data); }

   /// Conversion to `double *`. Deprecated.
   MFEM_DEPRECATED inline operator real_t *() { return data; }

   /// Conversion to `const double *`. Deprecated.
   MFEM_DEPRECATED inline operator const real_t *() const { return data; }

   /// STL-like begin.
   inline real_t *begin() { return data; }

   /// STL-like end.
   inline real_t *end() { return data + size; }

   /// STL-like begin (const version).
   inline const real_t *begin() const { return data; }

   /// STL-like end (const version).
   inline const real_t *end() const { return data + size; }

   /// Return a reference to the Memory object used by the Vector.
   Memory<real_t> &GetMemory() { return data; }

   /** @brief Return a reference to the Memory object used by the Vector, const
       version. */
   const Memory<real_t> &GetMemory() const { return data; }

   /// Update the memory location of the vector to match @a v.
   void SyncMemory(const Vector &v) const { GetMemory().Sync(v.GetMemory()); }

   /// Update the alias memory location of the vector to match @a v.
   void SyncAliasMemory(const Vector &v) const
   { GetMemory().SyncAlias(v.GetMemory(),Size()); }

   /// Read the Vector data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline void StealData(real_t **p)
   { *p = data; data.Reset(); size = 0; }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline real_t *StealData() { real_t *p; StealData(&p); return p; }

   /// Access Vector entries. Index i = 0 .. size-1.
   real_t &Elem(int i);

   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const real_t &Elem(int i) const;

   /// Access Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline real_t &operator()(int i);

   /// Read only access to Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline const real_t &operator()(int i) const;

   /// Access Vector entries using [] for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline real_t &operator[](int i) { return (*this)(i); }

   /// Read only access to Vector entries using [] for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline const real_t &operator[](int i) const { return (*this)(i); }

   /// Dot product with a `double *` array.
   real_t operator*(const real_t *) const;

   /// Return the inner-product.
   real_t operator*(const Vector &v) const;

   /// Copy Size() entries from @a v.
   Vector &operator=(const real_t *v);

   /// Copy assignment.
   /** @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   Vector &operator=(const Vector &v);

   /// Move assignment
   Vector &operator=(Vector&& v);

   /// Redefine '=' for vector = constant.
   Vector &operator=(real_t value);

   Vector &operator*=(real_t c);

   /// Component-wise scaling: (*this)(i) *= v(i)
   Vector &operator*=(const Vector &v);

   Vector &operator/=(real_t c);

   /// Component-wise division: (*this)(i) /= v(i)
   Vector &operator/=(const Vector &v);

   Vector &operator-=(real_t c);

   Vector &operator-=(const Vector &v);

   Vector &operator+=(real_t c);

   Vector &operator+=(const Vector &v);

   /// (*this) += a * Va
   Vector &Add(const real_t a, const Vector &Va);

   /// (*this) = a * x
   Vector &Set(const real_t a, const Vector &x);

   void SetVector(const Vector &v, int offset);

   void AddSubVector(const Vector &v, int offset);

   /// (*this) = -(*this)
   void Neg();

   /// (*this)(i) = 1.0 / (*this)(i)
   void Reciprocal();

   /// Swap the contents of two Vectors
   inline void Swap(Vector &other);

   /// Set v = v1 + v2.
   friend void add(const Vector &v1, const Vector &v2, Vector &v);

   /// Set v = v1 + alpha * v2.
   friend void add(const Vector &v1, real_t alpha, const Vector &v2, Vector &v);

   /// z = a * (x + y)
   friend void add(const real_t a, const Vector &x, const Vector &y, Vector &z);

   /// z = a * x + b * y
   friend void add(const real_t a, const Vector &x,
                   const real_t b, const Vector &y, Vector &z);

   /// Set v = v1 - v2.
   friend void subtract(const Vector &v1, const Vector &v2, Vector &v);

   /// z = a * (x - y)
   friend void subtract(const real_t a, const Vector &x,
                        const Vector &y, Vector &z);

   /// Computes cross product of this vector with another 3D vector.
   /// vout = this x vin.
   void cross3D(const Vector &vin, Vector &vout) const;

   /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
   void median(const Vector &lo, const Vector &hi);

   /// Extract entries listed in @a dofs to the output Vector @a elemvect.
   /** Negative dof values cause the -dof-1 position in @a elemvect to receive
       the -val in from this Vector. */
   void GetSubVector(const Array<int> &dofs, Vector &elemvect) const;

   /// Extract entries listed in @a dofs to the output array @a elem_data.
   /** Negative dof values cause the -dof-1 position in @a elem_data to receive
       the -val in from this Vector. */
   void GetSubVector(const Array<int> &dofs, real_t *elem_data) const;

   /// Set the entries listed in @a dofs to the given @a value.
   /** Negative dof values cause the -dof-1 position in this Vector to receive
       the -value. */
   void SetSubVector(const Array<int> &dofs, const real_t value);

   /** @brief Set the entries listed in @a dofs to the values given in the @a
       elemvect Vector. Negative dof values cause the -dof-1 position in this
       Vector to receive the -val from @a elemvect. */
   void SetSubVector(const Array<int> &dofs, const Vector &elemvect);

   /** @brief Set the entries listed in @a dofs to the values given the @a ,
       elem_data array. Negative dof values cause the -dof-1 position in this
       Vector to receive the -val from @a elem_data. */
   void SetSubVector(const Array<int> &dofs, real_t *elem_data);

   /** @brief Add elements of the @a elemvect Vector to the entries listed in @a
       dofs. Negative dof values cause the -dof-1 position in this Vector to add
       the -val from @a elemvect. */
   void AddElementVector(const Array<int> & dofs, const Vector & elemvect);

   /** @brief Add elements of the @a elem_data array to the entries listed in @a
       dofs. Negative dof values cause the -dof-1 position in this Vector to add
       the -val from @a elem_data. */
   void AddElementVector(const Array<int> & dofs, real_t *elem_data);

   /** @brief Add @a times the elements of the @a elemvect Vector to the entries
       listed in @a dofs. Negative dof values cause the -dof-1 position in this
       Vector to add the -a*val from @a elemvect. */
   void AddElementVector(const Array<int> & dofs, const real_t a,
                         const Vector & elemvect);

   /// Set all vector entries NOT in the @a dofs Array to the given @a val.
   void SetSubVectorComplement(const Array<int> &dofs, const real_t val);

   /// Prints vector to stream out.
   void Print(std::ostream &out = mfem::out, int width = 8) const;

#ifdef MFEM_USE_ADIOS2
   /// Prints vector to stream out.
   /// @param out adios2stream output
   /// @param variable_name variable name associated with current Vector
   void Print(adios2stream & out, const std::string& variable_name) const;
#endif

   /// Prints vector to stream out in HYPRE_Vector format.
   void Print_HYPRE(std::ostream &out) const;

   /// Prints vector as a List for importing into Mathematica.
   /** The resulting file can be read into Mathematica using an expression such
       as: myVec = Get["output_file_name"]
       The Mathematica variable "myVec" will then be assigned to a new
       List object containing the data from this MFEM Vector. */
   void PrintMathematica(std::ostream &out = mfem::out) const;

   /// Print the Vector size and hash of its data.
   /** This is a compact text representation of the Vector contents that can be
       used to compare vectors from different runs without the need to save the
       whole vector. */
   void PrintHash(std::ostream &out) const;

   /// Set random values in the vector.
   void Randomize(int seed = 0);
   /// Returns the l2 norm of the vector.
   real_t Norml2() const;
   /// Returns the l_infinity norm of the vector.
   real_t Normlinf() const;
   /// Returns the l_1 norm of the vector.
   real_t Norml1() const;
   /// Returns the l_p norm of the vector.
   real_t Normlp(real_t p) const;
   /// Returns the maximal element of the vector.
   real_t Max() const;
   /// Returns the minimal element of the vector.
   real_t Min() const;
   /// Return the sum of the vector entries
   real_t Sum() const;
   /// Compute the square of the Euclidean distance to another vector.
   inline real_t DistanceSquaredTo(const real_t *p) const;
   /// Compute the square of the Euclidean distance to another vector.
   inline real_t DistanceSquaredTo(const Vector &p) const;
   /// Compute the Euclidean distance to another vector.
   inline real_t DistanceTo(const real_t *p) const;
   /// Compute the Euclidean distance to another vector.
   inline real_t DistanceTo(const Vector &p) const;

   /** @brief Count the number of entries in the Vector for which isfinite
       is false, i.e. the entry is a NaN or +/-Inf. */
   int CheckFinite() const { return mfem::CheckFinite(HostRead(), size); }

   /// Destroys vector.
   virtual ~Vector();

   /// Shortcut for mfem::Read(vec.GetMemory(), vec.Size(), on_dev).
   virtual const real_t *Read(bool on_dev = true) const
   { return mfem::Read(data, size, on_dev); }

   /// Shortcut for mfem::Read(vec.GetMemory(), vec.Size(), false).
   virtual const real_t *HostRead() const
   { return mfem::Read(data, size, false); }

   /// Shortcut for mfem::Write(vec.GetMemory(), vec.Size(), on_dev).
   virtual real_t *Write(bool on_dev = true)
   { return mfem::Write(data, size, on_dev); }

   /// Shortcut for mfem::Write(vec.GetMemory(), vec.Size(), false).
   virtual real_t *HostWrite()
   { return mfem::Write(data, size, false); }

   /// Shortcut for mfem::ReadWrite(vec.GetMemory(), vec.Size(), on_dev).
   virtual real_t *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, size, on_dev); }

   /// Shortcut for mfem::ReadWrite(vec.GetMemory(), vec.Size(), false).
   virtual real_t *HostReadWrite()
   { return mfem::ReadWrite(data, size, false); }

};

// Inline methods

template <typename T>
inline T ZeroSubnormal(T val)
{
   return (std::fpclassify(val) == FP_SUBNORMAL) ? 0.0 : val;
}

inline bool IsFinite(const real_t &val)
{
   // isfinite didn't appear in a standard until C99, and later C++11. It wasn't
   // standard in C89 or C++98. PGI as of 14.7 still defines it as a macro.
#ifdef isfinite
   return isfinite(val);
#else
   return std::isfinite(val);
#endif
}

inline int CheckFinite(const real_t *v, const int n)
{
   int bad = 0;
   for (int i = 0; i < n; i++)
   {
      if (!IsFinite(v[i])) { bad++; }
   }
   return bad;
}

inline Vector::Vector(int s)
{
   MFEM_ASSERT(s>=0,"Unexpected negative size.");
   size = s;
   if (s > 0)
   {
      data.New(s);
   }
}

inline void Vector::SetSize(int s)
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

inline void Vector::SetSize(int s, MemoryType mt)
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

inline void Vector::NewMemoryAndSize(const Memory<real_t> &mem, int s,
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

inline void Vector::MakeRef(Vector &base, int offset, int s)
{
   data.Delete();
   size = s;
   data.MakeAlias(base.GetMemory(), offset, s);
}

inline void Vector::MakeRef(Vector &base, int offset)
{
   data.Delete();
   data.MakeAlias(base.GetMemory(), offset, size);
}

inline void Vector::Destroy()
{
   const bool use_dev = data.UseDevice();
   data.Delete();
   size = 0;
   data.Reset();
   data.UseDevice(use_dev);
}

inline real_t &Vector::operator()(int i)
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline const real_t &Vector::operator()(int i) const
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline void Vector::Swap(Vector &other)
{
   mfem::Swap(data, other.data);
   mfem::Swap(size, other.size);
}

/// Specialization of the template function Swap<> for class Vector
template<> inline void Swap<Vector>(Vector &a, Vector &b)
{
   a.Swap(b);
}

inline Vector::~Vector()
{
   data.Delete();
}

inline real_t DistanceSquared(const real_t *x, const real_t *y, const int n)
{
   real_t d = 0.0;

   for (int i = 0; i < n; i++)
   {
      d += (x[i]-y[i])*(x[i]-y[i]);
   }

   return d;
}

inline real_t Distance(const real_t *x, const real_t *y, const int n)
{
   return std::sqrt(DistanceSquared(x, y, n));
}

inline real_t Distance(const Vector &x, const Vector &y)
{
   return x.DistanceTo(y);
}

inline real_t Vector::DistanceSquaredTo(const real_t *p) const
{
   return DistanceSquared(data, p, size);
}

inline real_t Vector::DistanceSquaredTo(const Vector &p) const
{
   MFEM_ASSERT(p.Size() == Size(), "Incompatible vector sizes.");
   return DistanceSquared(data, p.data, size);
}

inline real_t Vector::DistanceTo(const real_t *p) const
{
   return Distance(data, p, size);
}

inline real_t Vector::DistanceTo(const Vector &p) const
{
   MFEM_ASSERT(p.Size() == Size(), "Incompatible vector sizes.");
   return Distance(data, p.data, size);
}

/// Returns the inner product of x and y
/** In parallel this computes the inner product of the local vectors,
    producing different results on each MPI rank.
*/
inline real_t InnerProduct(const Vector &x, const Vector &y)
{
   return x * y;
}

#ifdef MFEM_USE_MPI
/// Returns the inner product of x and y in parallel
/** In parallel this computes the inner product of the global vectors,
    producing identical results on each MPI rank.
*/
inline real_t InnerProduct(MPI_Comm comm, const Vector &x, const Vector &y)
{
   real_t loc_prod = x * y;
   real_t glb_prod;
   MPI_Allreduce(&loc_prod, &glb_prod, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
   return glb_prod;
}
#endif

} // namespace mfem

#endif
