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

#ifndef MFEM_VECTOR
#define MFEM_VECTOR

// Data type vector

#include "../general/array.hpp"
#include "../general/globals.hpp"
#ifdef MFEM_USE_SUNDIALS
#include <nvector/nvector_serial.h>
#endif
#include <cmath>
#include <iostream>
#include <limits>
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
inline int CheckFinite(const double *v, const int n);

/// Define a shortcut for std::numeric_limits<double>::infinity()
inline double infinity()
{
   return std::numeric_limits<double>::infinity();
}

/// Vector data type.
class Vector
{
protected:

   int size, allocsize;
   double * data;

public:

   /// Default constructor for Vector. Sets size = 0 and data = NULL.
   Vector () { allocsize = size = 0; data = 0; }

   /// Copy constructor. Allocates a new data array and copies the data.
   Vector(const Vector &);

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit Vector (int s);

   /// Creates a vector referencing an array of doubles, owned by someone else.
   /** The pointer @a _data can be NULL. The data array can be replaced later
       with SetData(). */
   Vector (double *_data, int _size)
   { data = _data; size = _size; allocsize = -size; }

   /// Reads a vector from multiple files
   void Load (std::istream ** in, int np, int * dim);

   /// Load a vector from an input stream.
   void Load(std::istream &in, int Size);

   /// Load a vector from an input stream, reading the size from the stream.
   void Load(std::istream &in) { int s; in >> s; Load (in, s); }

   /// @brief Resize the vector to size @a s.
   /** If the new size is less than or equal to Capacity() then the internal
       data array remains the same. Otherwise, the old array is deleted, if
       owned, and a new array of size @a s is allocated without copying the
       previous content of the Vector.
       @warning In the second case above (new size greater than current one),
       the vector will allocate new data array, even if it did not own the
       original data! Also, new entries are not initialized! */
   void SetSize(int s);

   /// Set the Vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(double *d) { data = d; }

   /// Set the Vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @warning This method should be called only when OwnsData() is false.
       @sa NewDataAndSize(). */
   void SetDataAndSize(double *d, int s)
   { data = d; size = s; allocsize = -s; }

   /// Set the Vector data and size, deleting the old data, if owned.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @sa SetDataAndSize(). */
   void NewDataAndSize(double *d, int s)
   {
      if (allocsize > 0) { delete [] data; }
      SetDataAndSize(d, s);
   }

   void MakeDataOwner() { allocsize = abs(allocsize); }

   /// Destroy a vector
   void Destroy();

   /// Returns the size of the vector.
   inline int Size() const { return size; }

   /// Return the size of the currently allocated data array.
   /** It is always true that Capacity() >= Size(). */
   inline int Capacity() const { return abs(allocsize); }

   /// Return a pointer to the beginning of the Vector data.
   /** @warning This method should be used with caution as it gives write access
       to the data of const-qualified Vector%s. */
   inline double *GetData() const { return data; }

   /// Conversion to `double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator double *() { return data; }

   /// Conversion to `const double *`.
   /** @note This conversion function makes it possible to use [] for indexing
       in addition to the overloaded operator()(int). */
   inline operator const double *() const { return data; }

   inline bool OwnsData() const { return (allocsize > 0); }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline void StealData(double **p)
   { *p = data; data = 0; size = allocsize = 0; }

   /// Changes the ownership of the data; after the call the Vector is empty
   inline double *StealData() { double *p; StealData(&p); return p; }

   /// Access Vector entries. Index i = 0 .. size-1.
   double & Elem (int i);

   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const double & Elem (int i) const;

   /// Access Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline double & operator() (int i);

   /// Read only access to Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   inline const double & operator() (int i) const;

   /// Dot product with a `double *` array.
   double operator*(const double *) const;

   /// Return the inner-product.
   double operator*(const Vector &v) const;

   /// Copy Size() entries from @a v.
   Vector & operator=(const double *v);

   /// Copy assignment.
   /** @note Defining this method overwrites the implicitly defined copy
       assignemnt operator. */
   Vector & operator=(const Vector &v);

   /// Redefine '=' for vector = constant.
   Vector & operator=(double value);

   Vector & operator*=(double c);

   Vector & operator/=(double c);

   Vector & operator-=(double c);

   Vector & operator-=(const Vector &v);

   Vector & operator+=(const Vector &v);

   /// (*this) += a * Va
   Vector & Add(const double a, const Vector &Va);

   /// (*this) = a * x
   Vector & Set(const double a, const Vector &x);

   void SetVector (const Vector &v, int offset);

   /// (*this) = -(*this)
   void Neg();

   /// Swap the contents of two Vectors
   inline void Swap(Vector &other);

   /// Set v = v1 + v2.
   friend void add(const Vector &v1, const Vector &v2, Vector &v);

   /// Set v = v1 + alpha * v2.
   friend void add(const Vector &v1, double alpha, const Vector &v2, Vector &v);

   /// z = a * (x + y)
   friend void add(const double a, const Vector &x, const Vector &y, Vector &z);

   /// z = a * x + b * y
   friend void add (const double a, const Vector &x,
                    const double b, const Vector &y, Vector &z);

   /// Set v = v1 - v2.
   friend void subtract(const Vector &v1, const Vector &v2, Vector &v);

   /// z = a * (x - y)
   friend void subtract(const double a, const Vector &x,
                        const Vector &y, Vector &z);

   /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
   void median(const Vector &lo, const Vector &hi);

   void GetSubVector(const Array<int> &dofs, Vector &elemvect) const;
   void GetSubVector(const Array<int> &dofs, double *elem_data) const;

   /// Set the entries listed in `dofs` to the given `value`.
   void SetSubVector(const Array<int> &dofs, const double value);
   void SetSubVector(const Array<int> &dofs, const Vector &elemvect);
   void SetSubVector(const Array<int> &dofs, double *elem_data);

   /// Add (element) subvector to the vector.
   void AddElementVector(const Array<int> & dofs, const Vector & elemvect);
   void AddElementVector(const Array<int> & dofs, double *elem_data);
   void AddElementVector(const Array<int> & dofs, const double a,
                         const Vector & elemvect);

   /// Set all vector entries NOT in the 'dofs' array to the given 'val'.
   void SetSubVectorComplement(const Array<int> &dofs, const double val);

   /// Prints vector to stream out.
   void Print(std::ostream & out = mfem::out, int width = 8) const;

   /// Prints vector to stream out in HYPRE_Vector format.
   void Print_HYPRE(std::ostream &out) const;

   /// Set random values in the vector.
   void Randomize(int seed = 0);
   /// Returns the l2 norm of the vector.
   double Norml2() const;
   /// Returns the l_infinity norm of the vector.
   double Normlinf() const;
   /// Returns the l_1 norm of the vector.
   double Norml1() const;
   /// Returns the l_p norm of the vector.
   double Normlp(double p) const;
   /// Returns the maximal element of the vector.
   double Max() const;
   /// Returns the minimal element of the vector.
   double Min() const;
   /// Return the sum of the vector entries
   double Sum() const;
   /// Compute the square of the Euclidean distance to another vector.
   inline double DistanceSquaredTo(const double *p) const;
   /// Compute the Euclidean distance to another vector.
   inline double DistanceTo(const double *p) const;

   /** Count the number of entries in the Vector for which isfinite
       is false, i.e. the entry is a NaN or +/-Inf. */
   int CheckFinite() const { return mfem::CheckFinite(data, size); }

   /// Destroys vector.
   virtual ~Vector ();

#ifdef MFEM_USE_SUNDIALS
   /// Construct a wrapper Vector from SUNDIALS N_Vector.
   explicit Vector(N_Vector nv);

   /// Return a new wrapper SUNDIALS N_Vector of type SUNDIALS_NVEC_SERIAL.
   /** The returned N_Vector must be destroyed by the caller. */
   virtual N_Vector ToNVector() { return N_VMake_Serial(Size(), GetData()); }

   /** @brief Update an existing wrapper SUNDIALS N_Vector to point to this
       Vector. */
   virtual void ToNVector(N_Vector &nv);
#endif
};

// Inline methods

inline bool IsFinite(const double &val)
{
   // isfinite didn't appear in a standard until C99, and later C++11. It wasn't
   // standard in C89 or C++98. PGI as of 14.7 still defines it as a macro.
#ifdef isfinite
   return isfinite(val);
#else
   return std::isfinite(val);
#endif
}

inline int CheckFinite(const double *v, const int n)
{
   int bad = 0;
   for (int i = 0; i < n; i++)
   {
      if (!IsFinite(v[i])) { bad++; }
   }
   return bad;
}

inline Vector::Vector (int s)
{
   if (s > 0)
   {
      allocsize = size = s;
      data = new double[s];
   }
   else
   {
      allocsize = size = 0;
      data = NULL;
   }
}

inline void Vector::SetSize(int s)
{
   if (s == size)
   {
      return;
   }
   if (s <= abs(allocsize))
   {
      size = s;
      return;
   }
   if (allocsize > 0)
   {
      delete [] data;
   }
   allocsize = size = s;
   data = new double[s];
}

inline void Vector::Destroy()
{
   if (allocsize > 0)
   {
      delete [] data;
   }
   allocsize = size = 0;
   data = NULL;
}

inline double & Vector::operator() (int i)
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline const double & Vector::operator() (int i) const
{
   MFEM_ASSERT(data && i >= 0 && i < size,
               "index [" << i << "] is out of range [0," << size << ")");

   return data[i];
}

inline void Vector::Swap(Vector &other)
{
   mfem::Swap(size, other.size);
   mfem::Swap(allocsize, other.allocsize);
   mfem::Swap(data, other.data);
}

/// Specialization of the template function Swap<> for class Vector
template<> inline void Swap<Vector>(Vector &a, Vector &b)
{
   a.Swap(b);
}

inline Vector::~Vector()
{
   if (allocsize > 0)
   {
      delete [] data;
   }
}

inline double DistanceSquared(const double *x, const double *y, const int n)
{
   double d = 0.0;

   for (int i = 0; i < n; i++)
   {
      d += (x[i]-y[i])*(x[i]-y[i]);
   }

   return d;
}

inline double Distance(const double *x, const double *y, const int n)
{
   return std::sqrt(DistanceSquared(x, y, n));
}

inline double Vector::DistanceSquaredTo(const double *p) const
{
   return DistanceSquared(data, p, size);
}

inline double Vector::DistanceTo(const double *p) const
{
   return Distance(data, p, size);
}

/// Returns the inner product of x and y
/** In parallel this computes the inner product of the local vectors,
    producing different results on each MPI rank.
*/
inline double InnerProduct(const Vector &x, const Vector &y)
{
   return x * y;
}

#ifdef MFEM_USE_MPI
/// Returns the inner product of x and y in parallel
/** In parallel this computes the inner product of the global vectors,
    producing identical results on each MPI rank.
*/
inline double InnerProduct(MPI_Comm comm, const Vector &x, const Vector &y)
{
   double loc_prod = x * y;
   double glb_prod;
   MPI_Allreduce(&loc_prod, &glb_prod, 1, MPI_DOUBLE, MPI_SUM, comm);
   return glb_prod;
}
#endif

}

#endif
