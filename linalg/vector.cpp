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

// Implementation of data type vector

#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "vector.hpp"

Vector::Vector(const Vector &v)
{
   int s = v.Size();
   if (s > 0)
   {
      allocsize = size = s;
      data = new double[s];
      for (int i = 0; i < s; i++)
         data[i] = v(i);
   }
   else
   {
      allocsize = size = 0;
      data = NULL;
   }
}

void Vector::Load(istream **in, int np, int *dim)
{
   int i, j, s;

   s = 0;
   for (i = 0; i < np; i++)
      s += dim[i];

   SetSize(s);

   int p = 0;
   for (i = 0; i < np; i++)
      for (j = 0; j < dim[i]; j++)
         *in[i] >> data[p++];
}

void Vector::Load(istream &in, int Size)
{
   SetSize(Size);

   for (int i = 0; i < size; i++)
      in >> data[i];
}

double &Vector::Elem(int i)
{
   return operator()(i);
}

const double &Vector::Elem(int i) const
{
   return operator()(i);
}

double Vector::operator*(const double *v) const
{
   int s = size;
   const double *d = data;
   double prod = 0.0;
#ifdef MFEM_USE_OPENMP
#pragma omp parallel for reduction(+:prod)
#endif
   for (int i = 0; i < s; i++)
      prod += d[i] * v[i];
   return prod;
}

double Vector::operator*(const Vector &v) const
{
#ifdef MFEM_DEBUG
   if (v.size != size)
      mfem_error("Vector::operator*(const Vector &) const");
#endif

   return operator*(v.data);
}

Vector &Vector::operator=(const Vector &v)
{
   SetSize(v.Size());
   for (int i = 0; i < size; i++)
      data[i] = v.data[i];
   return *this;
}

Vector &Vector::operator=(double value)
{
   register int i, s = size;
   register double *p = data, v = value;
   for (i = 0; i < s; i++)
      *(p++) = v;
   return *this;
}

Vector &Vector::operator*=(double c)
{
   for (int i = 0; i < size; i++)
      data[i] *= c;
   return *this;
}

Vector &Vector::operator/=(double c)
{
   double m = 1.0/c;
   for (int i = 0; i < size; i++)
      data[i] *= m;
   return *this;
}

Vector &Vector::operator-=(double c)
{
   for (int i = 0; i < size; i++)
      data[i] -= c;
   return *this;
}

Vector &Vector::operator-=(const Vector &v)
{
#ifdef MFEM_DEBUG
   if (size != v.size)
      mfem_error("Vector::operator-=(const Vector &)");
#endif
   for (int i = 0; i < size; i++)
      data[i] -= v(i);
   return *this;
}

Vector &Vector::operator+=(const Vector &v)
{
#ifdef MFEM_DEBUG
   if (size != v.size)
      mfem_error("Vector::operator+=(const Vector &)");
#endif
   for (int i = 0; i < size; i++)
      data[i] += v(i);
   return *this;
}

Vector &Vector::Add(const double a, const Vector &Va)
{
#ifdef MFEM_DEBUG
   if (size != Va.size)
      mfem_error("Vector::Add(const double, const Vector &)");
#endif
   if (a != 0.0)
   {
      for (int i = 0; i < size; i++)
         data[i] += a * Va(i);
   }
   return *this;
}

Vector &Vector::Set(const double a, const Vector &Va)
{
#ifdef MFEM_DEBUG
   if (size != Va.size)
      mfem_error("Vector::Set(const double, const Vector &)");
#endif
   for (int i = 0; i < size; i++)
      data[i] = a * Va(i);
   return *this;
}

void Vector::SetVector(const Vector &v, int offset)
{
   int vs = v.Size();
   double *vp = v.data, *p = data + offset;

#ifdef MFEM_DEBUG
   if (offset+vs > size)
      mfem_error("Vector::SetVector(const Vector &, int)");
#endif

   for (int i = 0; i < vs; i++)
      p[i] = vp[i];
}

void Vector::Neg()
{
   for (int i = 0; i < size; i++)
      data[i] = -data[i];
}

void swap(Vector *v1, Vector *v2)
{
   int size = v1->size, allocsize = v1->allocsize;
   double *data = v1->data;

   v1->size      = v2->size;
   v1->allocsize = v2->allocsize;
   v1->data      = v2->data;

   v2->size      = size;
   v2->allocsize = allocsize;
   v2->data      = data;
}

void add(const Vector &v1, const Vector &v2, Vector &v)
{
#ifdef MFEM_DEBUG
   if (v.size != v1.size || v.size != v2.size)
      mfem_error("add(Vector &v1, Vector &v2, Vector &v)");
#endif

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < v.size; i++)
      v.data[i] = v1.data[i] + v2.data[i];
}

void add(const Vector &v1, double alpha, const Vector &v2, Vector &v)
{
#ifdef MFEM_DEBUG
   if (v.size != v1.size || v.size != v2.size)
      mfem_error ("add(Vector &v1, double alpha, Vector &v2, Vector &v)");
#endif
   if (alpha == 0.0)
   {
      v = v1;
   }
   else if (alpha == 1.0)
   {
      add(v1, v2, v);
   }
   else
   {
      const double *v1p = v1.data, *v2p = v2.data;
      double *vp = v.data;
      int s = v.size;
#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < s; i++)
         vp[i] = v1p[i] + alpha*v2p[i];
   }
}

void add(const double a, const Vector &x, const Vector &y, Vector &z)
{
#ifdef MFEM_DEBUG
   if (x.size != y.size || x.size != z.size)
      mfem_error ("add(const double a, const Vector &x, const Vector &y,"
                  " Vector &z)");
#endif
   if (a == 0.0)
   {
      z = 0.0;
   }
   else if (a == 1.0)
   {
      add(x, y, z);
   }
   else
   {
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
      int            s = x.size;

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < s; i++)
         zp[i] = a * (xp[i] + yp[i]);
   }
}

void add(const double a, const Vector &x,
         const double b, const Vector &y, Vector &z)
{
#ifdef MFEM_DEBUG
   if (x.size != y.size || x.size != z.size)
      mfem_error("add(const double a, const Vector &x,\n"
                 "    const double b, const Vector &y, Vector &z)");
#endif
   if (a == 0.0)
   {
      z.Set(b, y);
   }
   else if (b == 0.0)
   {
      z.Set(a, x);
   }
   else if (a == 1.0)
   {
      add(x, b, y, z);
   }
   else if (b == 1.0)
   {
      add(y, a, x, z);
   }
   else if (a == b)
   {
      add(a, x, y, z);
   }
   else
   {
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
      int            s = x.size;

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < s; i++)
         zp[i] = a * xp[i] + b * yp[i];
   }
}

void subtract(const Vector &x, const Vector &y, Vector &z)
{
#ifdef MFEM_DEBUG
   if (x.size != y.size || x.size != z.size)
      mfem_error ("subtract(const Vector &, const Vector &, Vector &)");
#endif
   const double *xp = x.data;
   const double *yp = y.data;
   double       *zp = z.data;
   int            s = x.size;

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < s; i++)
      zp[i] = xp[i] - yp[i];
}

void subtract(const double a, const Vector &x, const Vector &y, Vector &z)
{
#ifdef MFEM_DEBUG
   if (x.size != y.size || x.size != z.size)
      mfem_error("subtract(const double a, const Vector &x,"
                 " const Vector &y, Vector &z)");
#endif

   if (a == 0.)
   {
      z = 0.;
   }
   else if (a == 1.)
   {
      subtract(x, y, z);
   }
   else
   {
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
      int            s = x.size;

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < s; i++)
         zp[i] = a * (xp[i] - yp[i]);
   }
}

void Vector::GetSubVector(const Array<int> &dofs, Vector &elemvect) const
{
   int i, j, n = dofs.Size();

   elemvect.SetSize (n);

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         elemvect(i) = data[j];
      else
         elemvect(i) = -data[-1-j];
}

void Vector::GetSubVector(const Array<int> &dofs, double *elem_data) const
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         elem_data[i] = data[j];
      else
         elem_data[i] = -data[-1-j];
}

void Vector::SetSubVector(const Array<int> &dofs, const Vector &elemvect)
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         data[j] = elemvect(i);
      else
         data[-1-j] = -elemvect(i);
}

void Vector::SetSubVector(const Array<int> &dofs, double *elem_data)
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         data[j] = elem_data[i];
      else
         data[-1-j] = -elem_data[i];
}

void Vector::AddElementVector(const Array<int> &dofs, const Vector &elemvect)
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         data[j] += elemvect(i);
      else
         data[-1-j] -= elemvect(i);
}

void Vector::AddElementVector(const Array<int> &dofs, double *elem_data)
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         data[j] += elem_data[i];
      else
         data[-1-j] -= elem_data[i];
}

void Vector::AddElementVector(const Array<int> &dofs, const double a,
                              const Vector &elemvect)
{
   int i, j, n = dofs.Size();

   for (i = 0; i < n; i++)
      if ((j=dofs[i]) >= 0)
         data[j] += a * elemvect(i);
      else
         data[-1-j] -= a * elemvect(i);
}

void Vector::Print(ostream &out, int width) const
{
   for (int i = 0; 1; )
   {
      out << data[i];
      i++;
      if (i == size)
         break;
      if ( i % width == 0 )
         out << '\n';
      else
         out << ' ';
   }
   out << '\n';
}

void Vector::Print_HYPRE(ostream &out) const
{
   int i;
   ios::fmtflags old_fmt = out.setf(ios::scientific);
   int old_prec = out.precision(14);

   out << size << '\n';  // number of rows

   for (i = 0; i < size; i++)
      out << data[i] << '\n';

   out.precision(old_prec);
   out.setf(old_fmt);
}

void Vector::Randomize(int seed)
{
   // static unsigned int seed = time(0);
   const double max = (double)(RAND_MAX) + 1.;

   if (seed == 0)
      seed = time(0);

   // srand(seed++);
   srand(seed);

   for (int i = 0; i < size; i++)
      data[i] = fabs(rand()/max);
}

double Vector::Norml2()
{
   return sqrt((*this)*(*this));
}

double Vector::Normlinf()
{
   double max = fabs(data[0]);

   for (int i = 1; i < size; i++)
      if (fabs(data[i]) > max)
         max = fabs(data[i]);

   return max;
}

double Vector::Norml1()
{
   double sum = 0.0;

   for (int i = 0; i < size; i++)
      sum += fabs (data[i]);

   return sum;
}

double Vector::Max()
{
   double max = data[0];

   for (int i = 1; i < size; i++)
      if (data[i] > max)
         max = data[i];

   return max;
}

double Vector::Min()
{
   double min = data[0];

   for (int i = 1; i < size; i++)
      if (data[i] < min)
         min = data[i];

   return min;
}

double Vector::DistanceTo(const double *p) const
{
   return Distance(data, p, size);
}
