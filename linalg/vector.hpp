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

#ifndef MFEM_VECTOR
#define MFEM_VECTOR

// Data type vector

#include <math.h>
#include "../general/array.hpp"

/// Vector data type.
class Vector
{
protected:

   int size, allocsize;
   double * data;

public:

   /// Default constructor for Vector. Sets size = 0 and data = NULL
   Vector () { allocsize = size = 0; data = 0; };

   /// Copy constructor
   Vector(const Vector &);

   /// Creates vector of size s.
   explicit Vector (int s);

   Vector (double *_data, int _size)
   { data = _data; size = _size; allocsize = -size; }

   /// Reads a vector from multpile files
   void Load (istream ** in, int np, int * dim);

   /// Load a vector from an input stream.
   void Load(istream &in, int Size);

   /// Load a vector from an input stream.
   void Load(istream &in) { int s; in >> s; Load (in, s); };

   /// Resizes the vector if the new size is different
   void SetSize(int s);

   void SetData(double *d) { data = d; }

   void SetDataAndSize(double *d, int s)
   { data = d; size = s; allocsize = -s; }

   void MakeDataOwner() { allocsize = abs(allocsize); }

   /// Destroy a vector
   void Destroy();

   /// Returns the size of the vector.
   inline int Size() const {return size;};

   // double *GetData() { return data; }

   inline double *GetData() const { return data; }

   inline operator double *() { return data; }

   inline operator const double *() const { return data; }

   /// Changes the ownership of the the data
   inline void StealData(double **p) { *p = data; data = 0; size = 0; }

   /// Sets value in vector. Index i = 0 .. size-1
   double & Elem (int i);

   /// Sets value in vector. Index i = 0 .. size-1
   const double & Elem (int i) const;

   /// Sets value in vector. Index i = 0 .. size-1
   inline double & operator() (int i);

   /// Sets value in vector. Index i = 0 .. size-1
   inline const double & operator() (int i) const;

   double operator*(const double *) const;

   /// Return the inner-product.
   double operator*(const Vector &v) const;

   /// Redefine '=' for vector = vector.
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

   /// Swap v1 and v2.
   friend void swap(Vector *v1, Vector *v2);

   /// Do v = v1 + v2.
   friend void add(const Vector &v1, const Vector &v2, Vector &v);

   /// Do v = v1 + alpha * v2.
   friend void add(const Vector &v1, double alpha, const Vector &v2, Vector &v);

   /// z = a * (x + y)
   friend void add(const double a, const Vector &x, const Vector &y, Vector &z);

   /// z = a * x + b * y
   friend void add (const double a, const Vector &x,
                    const double b, const Vector &y, Vector &z);

   /// Do v = v1 - v2.
   friend void subtract(const Vector &v1, const Vector &v2, Vector &v);

   /// z = a * (x - y)
   friend void subtract(const double a, const Vector &x,
                        const Vector &y, Vector &z);

   void GetSubVector(const Array<int> &dofs, Vector &elemvect) const;
   void GetSubVector(const Array<int> &dofs, double *elem_data) const;

   void SetSubVector(const Array<int> &dofs, const Vector &elemvect);
   void SetSubVector(const Array<int> &dofs, double *elem_data);

   /// Add (element) subvector to the vector.
   void AddElementVector(const Array<int> & dofs, const Vector & elemvect);
   void AddElementVector(const Array<int> & dofs, double *elem_data);
   void AddElementVector(const Array<int> & dofs, const double a,
                         const Vector & elemvect);

   /// Prints vector to stream out.
   void Print(ostream & out = cout, int width = 8) const;

   /// Prints vector to stream out in HYPRE_Vector format.
   void Print_HYPRE(ostream &out) const;

   /// Set random values in the vector.
   void Randomize(int seed = 0);
   /// Returns the l2 norm of the vector.
   double Norml2();
   /// Returns the l_infinity norm of the vector.
   double Normlinf();
   /// Returns the l_1 norm of the vector.
   double Norml1();
   /// Returns the maximal element of the vector.
   double Max();
   /// Returns the minimal element of the vector.
   double Min();
   /// Compute the Euclidian distance to another vector.
   double DistanceTo (const double *p) const;

   /// Destroys vector.
   ~Vector ();
};

// Inline methods

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
      return;
   if (s <= abs(allocsize))
   {
      size = s;
      return;
   }
   if (allocsize > 0)
      delete [] data;
   allocsize = size = s;
   data = new double[s];
}

inline void Vector::Destroy()
{
   if (allocsize > 0)
      delete [] data;
   allocsize = size = 0;
   data = NULL;
}

inline double & Vector::operator() (int i)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= size)
      mfem_error ("Vector::operator()");
#endif

   return data[i];
}

inline const double & Vector::operator() (int i) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= size)
      mfem_error ("Vector::operator() const");
#endif

   return data[i];
}

inline Vector::~Vector()
{
   if (allocsize > 0)
      delete [] data;
}

inline double Distance(const double *x, const double *y, const int n)
{
   double d = 0.0;

   for (int i = 0; i < n; i++)
      d += (x[i]-y[i])*(x[i]-y[i]);

   return sqrt(d);
}

#endif
