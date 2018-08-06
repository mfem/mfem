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
#include "../general/scalars.hpp"
#ifdef MFEM_USE_BACKENDS
#include "../backends/base/backend.hpp"
#endif
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


/// TODO: doxygen
template <typename scalar_t, typename idx_t>
class BVector : public BArray<scalar_t,idx_t>
{
protected:
   typedef BArray<scalar_t,idx_t> base_class;

public:
   typedef scalar_t scalar_type;

   /// @brief Resize the vector to size @a new_size.
   /** If the new size is less than or equal to Capacity() then the internal
       data array remains the same. Otherwise, the old array is deleted, if
       owned, and a new array of size @a s is allocated without copying the
       previous content of the vector.
       @warning In the second case above (new size greater than current one),
       the vector will allocate new data array, even if it did not own the
       original data! Also, new entries are not initialized! */
   inline void SetSize(idx_t new_size);

   BVector &operator=(const scalar_t &value)
   { base_class::operator=(value); return *this; }

   BVector &operator=(const scalar_t *src)
   { base_class::operator=(src); return *this; }
};


#ifndef MFEM_USE_BACKENDS
struct DVector { };
#endif


/// Vector class template using an arbitrary entry type.
template <typename scalar_t>
class Vector_ : public DevExtension<BVector<scalar_t,int>,DVector>
{
protected:
   typedef DevExtension<BVector<scalar_t,int>,DVector> dev_class;
   typedef BArray<scalar_t,int> base_class;

#ifdef MFEM_USE_BACKENDS
   using dev_class::dev_ext;
#endif

   using base_class::data;
   using base_class::size;
   using base_class::allocsize;

public:
   Vector_() { this->SetEmpty(); }

   /// Create a vector of size s.
   /** @warning Entries are not initialized to zero! */
   explicit Vector_(int s) { this->InitSize(s); }

#ifdef MFEM_USE_BACKENDS
   /// TODO
   /** @warning Problem: if @a layout has engine and it is not allocated itself
       with ::operator::new().

       Should we just remove this ctor?
   */
   explicit Vector_(PLayout &layout) { this->InitLayout(layout); }

   /// TODO
   explicit Vector_(const DLayout &layout) { this->InitLayout(layout); }

   /// TODO: doxygen; wraps a device vector.
   /** @warning If the object @a dev_vector is not itself dynamically allocated
       with ::operator::new(), e.g. if it is allocated in a block (on the stack)
       or as a sub-object of another object, it's inherited method
       RefCounted::DontDelete() must be called before the destruction of the
       wrapper object created by this constructor.

       Should we just remove this ctor?
   */
   explicit Vector_(PVector &dev_vector)
      : dev_class(dev_vector) { }
#endif

   /** @brief Copy constructor. Create an Vector_ of the same size/layout as
       @a orig and if @a copy_data == true, copy the contents from @a orig. */
   Vector_(const Vector_ &orig, bool copy_data = true)
   { this->InitClone(orig, copy_data); }

   /// Creates a vector referencing an array of scalar_t, owned by someone else.
   Vector_(scalar_t *data, int size)
   { this->InitDataAndSize(data, size, false); }

   /// Destroys vector.
   virtual ~Vector_() { }

#ifdef MFEM_USE_BACKENDS
   /// TODO: doxygen
   PVector *Get_PVector() { return dev_ext.Get(); }

   /// TODO: doxygen
   const PVector *Get_PVector() const { return dev_ext.Get(); }
#endif

   /// Access Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   scalar_t &operator()(int i) { return this->operator[](i); }

   /// Read only access to Vector entries using () for 0-based indexing.
   /** @note If MFEM_DEBUG is enabled, bounds checking is performed. */
   const scalar_t &operator()(int i) const { return this->operator[](i); }

   /// TODO
   /// Note: return dot(*this, x) where dot(a, b) = sum_i a_i * conj(b_i)
   inline scalar_t DotProduct(const Vector_ &x) const;

   /// TODO
   inline void Axpby(const scalar_t &a, const Vector_ &x,
                     const scalar_t &b, const Vector_ &y);

   /// Destroy (clear) a vector, deleting any owned data.
   inline void Destroy() { this->Clear(); MFEM_IF_BACKENDS(dev_ext.Reset();,;) }

   /// Swap the contents of two Vectors
   inline void Swap(Vector_ &other);


   // The remaining methods only work when the Vector_(s) are on the host.

   /// Redefine '=' for Vector_ = constant.
   inline Vector_ &operator=(const scalar_t &value)
   { this->base_class::operator=(value); return *this; }

   /// Redefine '=' for Vector_ = scalar_t *.
   inline Vector_ &operator=(const scalar_t *v)
   { this->base_class::operator=(v); return *this; }

   /// Redefine '=' for Vector_ = Vector_.
   inline Vector_ &operator=(const Vector_ &rhs)
   { this->base_class::operator=(rhs); return *this; }

   /// Dot product with another Vector_.
   inline scalar_t operator*(const Vector_ &x) const;

   /// Perform *this = a*x + b*y.
   inline void Sum(const scalar_t &a, const Vector_ &x,
                   const scalar_t &b, const Vector_ &y);
};


/// Vector data type.
class Vector : public Vector_<double>
{
public:
   /// Default constructor for Vector. Sets size = 0 and data = NULL.
   Vector() : Vector_<double>() { }

   /** @brief Copy constructor. Create a Vector of the same size/layout as
       @a orig and if @a copy_data == true, copy the contents from @a orig. */
   Vector(const Vector &orig, bool copy_data = true)
      : Vector_<double>(orig, copy_data) { }

   /// @brief Creates vector of size s.
   /// @warning Entries are not initialized to zero!
   explicit Vector(int s) : Vector_<double>(s) { }

   /// Creates a vector referencing an array of doubles, owned by someone else.
   /** The pointer @a data can be NULL. The data array can be replaced later
       with SetData(). */
   Vector(double *data, int size) : Vector_<double>(data, size) { }

#ifdef MFEM_USE_BACKENDS
   /// TODO
   /** @warning Problem: if @a layout has engine and it is not allocated itself
       with ::operator::new().

       Should we just remove this ctor?
   */
   explicit Vector(PLayout &layout) : Vector_<double>(layout) { }

   /// TODO
   explicit Vector(const DLayout &layout) : Vector_<double>(layout) { }

   /// TODO: doxygen; wraps a device vector.
   /** @warning If the object @a dev_vector is not itself dynamically allocated
       with ::operator::new(), e.g. if it is allocated in a block (on the stack)
       or as a sub-object of another object, it's inherited method
       RefCounted::DontDelete() must be called before the destruction of the
       wrapper object created by this constructor.

       Should we just remove this ctor?
   */
   explicit Vector(PVector &dev_vector) : Vector_<double>(dev_vector) { }
#endif

   /// Reads a vector from multiple files
   void Load(std::istream ** in, int np, int * dim);

   /// Load a vector from an input stream.
   void Load(std::istream &in, int Size);

   /// Load a vector from an input stream, reading the size from the stream.
   void Load(std::istream &in) { int s; in >> s; Load (in, s); }

   /// Set the Vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(double *d) { data = d; }

   /// Set the Vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @warning This method should be called only when OwnsData() is false.
       @sa NewDataAndSize(). */
   void SetDataAndSize(double *d, int s) { this->InitAll(d, s, -s); }

   /// Set the Vector data and size, deleting the old data, if owned.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @sa SetDataAndSize(). */
   void NewDataAndSize(double *d, int s) { this->MakeRef(d, s); }

   /// Return a pointer to the beginning of the Vector data.
   /** @warning This method should be used with caution as it gives write access
       to the data of const-qualified Vector%s. */
   inline double *GetData() const { return data; }

   /// Conversion to `double *`.
   inline operator double *() { return data; }

   /// Conversion to `const double *`.
   inline operator const double *() const { return data; }

   /// Access Vector entries. Index i = 0 .. size-1.
   double &Elem(int i);

   /// Read only access to Vector entries. Index i = 0 .. size-1.
   const double &Elem(int i) const;

   /// Dot product with a `double *` array.
   double operator*(const double *) const;

   /// Return the inner-product.
   double operator*(const Vector &v) const;

   /// Copy Size() entries from @a src.
   Vector & operator=(const double *src)
   { base_class::operator=(src); return *this; }

   /// Redefine '=' for vector = vector.
   Vector & operator=(const Vector &rhs)
   { base_class::operator=(rhs); return *this; }

   /// Redefine '=' for vector = constant.
   Vector & operator=(double value)
   { base_class::operator=(value); return *this; }

   Vector & operator*=(double c);

   Vector & operator/=(double c);

   Vector & operator-=(double c);

   Vector & operator-=(const Vector &v);

   Vector & operator+=(const Vector &v);

   /// (*this) += a * Va
   Vector &Add(const double a, const Vector &Va);

   /// (*this) = a * x
   Vector &Set(const double a, const Vector &x);

   void SetVector(const Vector &v, int offset);

   /// (*this) = -(*this)
   void Neg();

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

   /** @brief Count the number of entries in the Vector for which isfinite
       is false, i.e. the entry is a NaN or +/-Inf. */
   int CheckFinite() const { return mfem::CheckFinite(data, size); }

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


// Inline methods: class BVector

template <typename scalar_t, typename idx_t>
inline void BVector<scalar_t,idx_t>::SetSize(idx_t new_size)
{
   if (new_size > this->Capacity())
   {
      this->template GrowSizeNoCopy<1,0>(new_size);
   }
   this->size = new_size;
}


// Inline methods: class Vector_

template <typename scalar_t>
inline scalar_t Vector_<scalar_t>::DotProduct(const Vector_ &x) const
{
#ifdef MFEM_USE_BACKENDS
   MFEM_ASSERT(bool(dev_ext) == bool(x.dev_ext), "");
   return (dev_ext ? dev_ext->template DotProduct<scalar_t>(*x.dev_ext) :
           operator*(x));
#else
   return operator*(x);
#endif
}

template <typename scalar_t>
inline void Vector_<scalar_t>::Axpby(const scalar_t &a, const Vector_ &x,
                                     const scalar_t &b, const Vector_ &y)
{
#ifdef MFEM_USE_BACKENDS
   /*const bool t_ok = dev_ext!=NULL;
   const bool x_ok = x.dev_ext!=NULL;
   const bool y_ok = y.dev_ext!=NULL;
   const bool all_ok = t_ok && x_ok && y_ok;
   printf("\n\033[32;7m[Axpby] ext:%s, x:%s & y%s\033[m",
          t_ok?"ok":"\033[31mX",
          x_ok?"ok":"\033[31mX",
          y_ok?"ok":"\033[31mX");
   fflush(0);
   all_ok ? dev_ext->Axpby(a, *x.dev_ext, b, *y.dev_ext) : Sum(a, x, b, y);*/
   dev_ext ? dev_ext->Axpby(a, *x.dev_ext, b, *y.dev_ext) : Sum(a, x, b, y);
#else
   Sum(a, x, b, y);
#endif
}

template <typename scalar_t>
inline scalar_t Vector_<scalar_t>::operator*(const Vector_ &x) const
{
   scalar_t result = scalar_t(0);
   for (int i = 0; i < size; i++)
   {
      result += data[i] * mfem::ScalarOps<scalar_t>::conj(x[i]);
   }
   return result;
}

template <typename scalar_t>
inline void Vector_<scalar_t>::Sum(const scalar_t &a, const Vector_ &x,
                                   const scalar_t &b, const Vector_ &y)
{
   MFEM_ASSERT(size == x.size, "");
   MFEM_ASSERT(size == y.size, "");
   if (b == scalar_t(0))
   {
      if (a == scalar_t(0))
      {
         operator=(scalar_t(0));
      }
      else
      {
         for (int i = 0; i < size; i++)
         {
            data[i] = a*x.data[i];
         }
      }
   }
   else
   {
      if (a == scalar_t(0))
      {
         for (int i = 0; i < size; i++)
         {
            data[i] = b*y.data[i];
         }
      }
      else
      {
         for (int i = 0; i < size; i++)
         {
            data[i] = a*x.data[i] + b*y.data[i];
         }
      }
   }
}

template <typename scalar_t>
inline void Vector_<scalar_t>::Swap(Vector_ &other)
{
   base_class::Swap(other);
#ifdef MFEM_USE_BACKENDS
   mfem::Swap(dev_ext, other.dev_ext);
#endif
}

/// Specialization of the template function Swap<> for class Vector_
template <typename scalar_t>
inline void Swap(Vector_<scalar_t> &a, Vector_<scalar_t> &b)
{
   a.Swap(b);
}


// Inline methods: class Vector

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
