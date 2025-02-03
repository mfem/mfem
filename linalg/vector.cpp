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

// Implementation of data type vector

#include "kernels.hpp"
#include "vector.hpp"
#include "../general/forall.hpp"

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <limits>

namespace mfem
{

#if defined(MFEM_USE_CUDA) or defined(MFEM_USE_HIP)
/**
 * Reducer for helping to compute L2-norms. Given two partial results:
 * a0 = sum_i (|v_i|/a1)^2
 * b0 = sum_j (|v_j|/b1)^2 (j disjoint from i for vector v)
 * computes:
 * a1 = max(a1, b1)
 * a0 = (a1 == 0 ? 0 : sum_{k in union(i,j)} (|v_k|/a1)^2)
 *
 * This form is resiliant against overflow/underflow, similar to std::hypot
 */
struct L2Reducer
{
   using value_type = DevicePair<real_t, real_t>;
   MFEM_HOST_DEVICE void join(value_type& a, const value_type &b) const
   {
      real_t scale = fmax(a.second, b.second);
      if (scale > 0)
      {
         real_t s = a.second / scale;
         a.first *= s * s;
         s = b.second / scale;
         a.first += b.first * s * s;
         a.second = scale;
      }
   }

   MFEM_HOST_DEVICE void init_val(value_type &a) const
   {
      a.first = 0;
      a.second = 0;
   }
};

/**
 * Reducer for helping to compute Lp-norms. Given two partial results:
 * a0 = sum_i (|v_i|/a1)^p
 * b0 = sum_j (|v_j|/b1)^p (j disjoint from i for vector v)
 * computes:
 * a1 = max(a1, b1)
 * a0 = (a1 == 0 ? 0 : sum_{k in union(i,j)} (|v_k|/a1)^p)
 *
 * This form is resiliant against overflow/underflow, similar to std::hypot
 */
struct LpReducer
{
   real_t p;
   using value_type = DevicePair<real_t, real_t>;
   MFEM_HOST_DEVICE void join(value_type& a, const value_type &b) const
   {
      real_t scale = fmax(a.second, b.second);
      if (scale > 0)
      {
         a.first = a.first * pow(a.second / scale, p) +
                   b.first * pow(b.second / scale, p);
         a.second = scale;
      }
   }

   MFEM_HOST_DEVICE void init_val(value_type &a) const
   {
      a.first = 0;
      a.second = 0;
   }
};

static Array<real_t>& vector_workspace()
{
   static Array<real_t> instance;
   return instance;
}

static Array<DevicePair<real_t, real_t>> &Lpvector_workspace()
{
   static Array<DevicePair<real_t, real_t>> instance;
   return instance;
}

static real_t devVectorMin(int size, const real_t *m_data)
{
   real_t res = infinity();
   reduce(
      size, res,
   [=] MFEM_HOST_DEVICE(int i, real_t &r) { r = fmin(r, m_data[i]); },
   MinReducer<real_t> {}, true, vector_workspace());
   return res;
}

static real_t devVectorMax(int size, const real_t *m_data)
{
   real_t res = -infinity();
   reduce(
      size, res,
   [=] MFEM_HOST_DEVICE(int i, real_t &r) { r = fmax(r, m_data[i]); },
   MaxReducer<real_t> {}, true, vector_workspace());
   return res;
}

static real_t devVectorLinf(int size, const real_t *m_data)
{
   real_t res = 0;
   reduce(
      size, res,
   [=] MFEM_HOST_DEVICE(int i, real_t &r) { r = fmax(r, fabs(m_data[i])); },
   MaxReducer<real_t> {}, true, vector_workspace());
   return res;
}

static real_t devVectorL1(int size, const real_t *m_data)
{
   real_t res = 0;
   reduce(
      size, res,
   [=] MFEM_HOST_DEVICE(int i, real_t &r) { r += fabs(m_data[i]); },
   SumReducer<real_t> {}, true, vector_workspace());
   return res;
}

static real_t devVectorL2(int size, const real_t *m_data)
{
   using value_type = DevicePair<real_t, real_t>;
   value_type res;
   res.first = 0;
   res.second = 0;
   // first compute sum (|m_data|/scale)^2
   reduce(
      size, res,
      [=] MFEM_HOST_DEVICE(int i, value_type &r)
   {
      real_t n = fabs(m_data[i]);
      if (n > 0)
      {
         if (r.second <= n)
         {
            real_t arg = r.second / n;
            r.first = r.first * (arg * arg) + 1;
            r.second = n;
         }
         else
         {
            real_t arg = n / r.second;
            r.first += arg * arg;
         }
      }
   },
   L2Reducer{}, true, Lpvector_workspace());
   // final answer
   return res.second * sqrt(res.first);
}

static real_t devVectorLp(int size, real_t p, const real_t *m_data)
{
   using value_type = DevicePair<real_t, real_t>;
   value_type res;
   res.first = 0;
   res.second = 0;
   // first compute sum (|m_data|/scale)^p
   reduce(
      size, res,
      [=] MFEM_HOST_DEVICE(int i, value_type &r)
   {
      real_t n = fabs(m_data[i]);
      if (n > 0)
      {
         if (r.second <= n)
         {
            real_t arg = r.second / n;
            r.first = r.first * pow(arg, p) + 1;
            r.second = n;
         }
         else
         {
            real_t arg = n / r.second;
            r.first += pow(arg, p);
         }
      }
   },
   LpReducer{p}, true, Lpvector_workspace());
   // final answer
   return res.second * pow(res.first, 1.0 / p);
}

static real_t devVectorSum(int size, const real_t *m_data)
{
   real_t res = 0;
   reduce(
   size, res, [=] MFEM_HOST_DEVICE(int i, real_t &r) { r += m_data[i]; },
   SumReducer<real_t> {}, true, vector_workspace());
   return res;
}

static real_t devVectorDot(int size, const real_t *m_data,
                           const real_t *v_data)
{
   real_t res = 0;
   reduce(
      size, res,
   [=] MFEM_HOST_DEVICE(int i, real_t &r) { r += m_data[i] * v_data[i]; },
   SumReducer<real_t> {}, true, vector_workspace());
   return res;
}
#endif

Vector::Vector(const Vector &v)
{
   const int s = v.Size();
   size = s;
   if (s > 0)
   {
      MFEM_ASSERT(!v.data.Empty(), "invalid source vector");
      data.New(s, v.data.GetMemoryType());
      data.CopyFrom(v.data, s);
   }
   UseDevice(v.UseDevice());
}

Vector::Vector(Vector &&v)
{
   *this = std::move(v);
}

void Vector::Load(std::istream **in, int np, int *dim)
{
   int i, j, s;

   s = 0;
   for (i = 0; i < np; i++)
   {
      s += dim[i];
   }

   SetSize(s);
   HostWrite();

   int p = 0;
   for (i = 0; i < np; i++)
   {
      for (j = 0; j < dim[i]; j++)
      {
         *in[i] >> data[p++];
         // Clang's libc++ sets the failbit when (correctly) parsing subnormals,
         // so we reset the failbit here.
         if (!*in[i] && errno == ERANGE)
         {
            in[i]->clear();
         }
      }
   }
}

void Vector::Load(std::istream &in, int Size)
{
   SetSize(Size);
   HostWrite();

   for (int i = 0; i < size; i++)
   {
      in >> data[i];
      // Clang's libc++ sets the failbit when (correctly) parsing subnormals,
      // so we reset the failbit here.
      if (!in && errno == ERANGE)
      {
         in.clear();
      }
   }
}

real_t &Vector::Elem(int i)
{
   return operator()(i);
}

const real_t &Vector::Elem(int i) const
{
   return operator()(i);
}

real_t Vector::operator*(const real_t *v) const
{
   real_t dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < size; i++)
   {
      dot += data[i] * v[i];
   }
   return dot;
}

Vector &Vector::operator=(const real_t *v)
{
   data.CopyFromHost(v, size);
   return *this;
}

Vector &Vector::operator=(const Vector &v)
{
#if 0
   SetSize(v.Size(), v.data.GetMemoryType());
   data.CopyFrom(v.data, v.Size());
   UseDevice(v.UseDevice());
#else
   SetSize(v.Size());
   bool vuse = v.UseDevice();
   const bool use_dev = UseDevice() || vuse;
   v.UseDevice(use_dev);
   // keep 'data' where it is, unless 'use_dev' is true
   if (use_dev) { Write(); }
   data.CopyFrom(v.data, v.Size());
   v.UseDevice(vuse);
#endif
   return *this;
}

Vector &Vector::operator=(Vector &&v)
{
   v.Swap(*this);
   if (this != &v) { v.Destroy(); }
   return *this;
}

Vector &Vector::operator=(real_t value)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = value; });
   return *this;
}

Vector &Vector::operator*=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= c; });
   return *this;
}

Vector &Vector::operator*=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   auto x = v.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= x[i]; });
   return *this;
}

Vector &Vector::operator/=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   const real_t m = 1.0/c;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= m; });
   return *this;
}

Vector &Vector::operator/=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   auto x = v.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] /= x[i]; });
   return *this;
}

Vector &Vector::operator-=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] -= c; });
   return *this;
}

Vector &Vector::operator-=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   auto x = v.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] -= x[i]; });
   return *this;
}

Vector &Vector::operator+=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += c; });
   return *this;
}

Vector &Vector::operator+=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   auto x = v.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += x[i]; });
   return *this;
}

Vector &Vector::Add(const real_t a, const Vector &Va)
{
   MFEM_ASSERT(size == Va.size, "incompatible Vectors!");

   if (a != 0.0)
   {
      const int N = size;
      const bool use_dev = UseDevice() || Va.UseDevice();
      auto y = ReadWrite(use_dev);
      auto x = Va.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += a * x[i]; });
   }
   return *this;
}

Vector &Vector::Set(const real_t a, const Vector &Va)
{
   MFEM_ASSERT(size == Va.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || Va.UseDevice();
   const int N = size;
   auto x = Va.Read(use_dev);
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = a * x[i]; });
   return *this;
}

void Vector::SetVector(const Vector &v, int offset)
{
   MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");

   const int vs = v.Size();
   const real_t *vp = v.data;
   real_t *p = data + offset;
   for (int i = 0; i < vs; i++)
   {
      p[i] = vp[i];
   }
}

void Vector::AddSubVector(const Vector &v, int offset)
{
   MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");

   const int vs = v.Size();
   const real_t *vp = v.data;
   real_t *p = data + offset;
   for (int i = 0; i < vs; i++)
   {
      p[i] += vp[i];
   }
}

void Vector::Neg()
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = -y[i]; });
}

void Vector::Reciprocal()
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = 1.0/y[i]; });
}

void add(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v.size == v1.size && v.size == v2.size,
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = v1.UseDevice() || v2.UseDevice() || v.UseDevice();
   const int N = v.size;
   // Note: get read access first, in case v is the same as v1/v2.
   auto x1 = v1.Read(use_dev);
   auto x2 = v2.Read(use_dev);
   auto y = v.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = x1[i] + x2[i]; });
#else
   #pragma omp parallel for
   for (int i = 0; i < v.size; i++)
   {
      v.data[i] = v1.data[i] + v2.data[i];
   }
#endif
}

void add(const Vector &v1, real_t alpha, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v.size == v1.size && v.size == v2.size,
               "incompatible Vectors!");

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
#if !defined(MFEM_USE_LEGACY_OPENMP)
      const bool use_dev = v1.UseDevice() || v2.UseDevice() || v.UseDevice();
      const int N = v.size;
      // Note: get read access first, in case v is the same as v1/v2.
      auto d_x = v1.Read(use_dev);
      auto d_y = v2.Read(use_dev);
      auto d_z = v.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         d_z[i] = d_x[i] + alpha * d_y[i];
      });
#else
      const real_t *v1p = v1.data, *v2p = v2.data;
      real_t *vp = v.data;
      const int s = v.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         vp[i] = v1p[i] + alpha*v2p[i];
      }
#endif
   }
}

void add(const real_t a, const Vector &x, const Vector &y, Vector &z)
{
   MFEM_ASSERT(x.size == y.size && x.size == z.size,
               "incompatible Vectors!");

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
#if !defined(MFEM_USE_LEGACY_OPENMP)
      const bool use_dev = x.UseDevice() || y.UseDevice() || z.UseDevice();
      const int N = x.size;
      // Note: get read access first, in case z is the same as x/y.
      auto xd = x.Read(use_dev);
      auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * (xd[i] + yd[i]);
      });
#else
      const real_t *xp = x.data;
      const real_t *yp = y.data;
      real_t       *zp = z.data;
      const int      s = x.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] + yp[i]);
      }
#endif
   }
}

void add(const real_t a, const Vector &x,
         const real_t b, const Vector &y, Vector &z)
{
   MFEM_ASSERT(x.size == y.size && x.size == z.size,
               "incompatible Vectors!");

   if (a == 0.0)
   {
      z.Set(b, y);
   }
   else if (b == 0.0)
   {
      z.Set(a, x);
   }
#if 0
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
#endif
   else
   {
#if !defined(MFEM_USE_LEGACY_OPENMP)
      const bool use_dev = x.UseDevice() || y.UseDevice() || z.UseDevice();
      const int N = x.size;
      // Note: get read access first, in case z is the same as x/y.
      auto xd = x.Read(use_dev);
      auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * xd[i] + b * yd[i];
      });
#else
      const real_t *xp = x.data;
      const real_t *yp = y.data;
      real_t       *zp = z.data;
      const int      s = x.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * xp[i] + b * yp[i];
      }
#endif
   }
}

void subtract(const Vector &x, const Vector &y, Vector &z)
{
   MFEM_ASSERT(x.size == y.size && x.size == z.size,
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = x.UseDevice() || y.UseDevice() || z.UseDevice();
   const int N = x.size;
   // Note: get read access first, in case z is the same as x/y.
   auto xd = x.Read(use_dev);
   auto yd = y.Read(use_dev);
   auto zd = z.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      zd[i] = xd[i] - yd[i];
   });
#else
   const real_t *xp = x.data;
   const real_t *yp = y.data;
   real_t       *zp = z.data;
   const int     s = x.size;
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      zp[i] = xp[i] - yp[i];
   }
#endif
}

void subtract(const real_t a, const Vector &x, const Vector &y, Vector &z)
{
   MFEM_ASSERT(x.size == y.size && x.size == z.size,
               "incompatible Vectors!");

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
#if !defined(MFEM_USE_LEGACY_OPENMP)
      const bool use_dev = x.UseDevice() || y.UseDevice() || z.UseDevice();
      const int N = x.size;
      // Note: get read access first, in case z is the same as x/y.
      auto xd = x.Read(use_dev);
      auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * (xd[i] - yd[i]);
      });
#else
      const real_t *xp = x.data;
      const real_t *yp = y.data;
      real_t       *zp = z.data;
      const int      s = x.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] - yp[i]);
      }
#endif
   }
}

void Vector::cross3D(const Vector &vin, Vector &vout) const
{
   HostRead();
   vin.HostRead();
   vout.HostWrite();
   MFEM_VERIFY(size == 3, "Only 3D vectors supported in cross.");
   MFEM_VERIFY(vin.Size() == 3, "Only 3D vectors supported in cross.");
   vout.SetSize(3);
   vout(0) = data[1]*vin(2)-data[2]*vin(1);
   vout(1) = data[2]*vin(0)-data[0]*vin(2);
   vout(2) = data[0]*vin(1)-data[1]*vin(0);
}

void Vector::median(const Vector &lo, const Vector &hi)
{
   MFEM_ASSERT(size == lo.size && size == hi.size,
               "incompatible Vectors!");

   const bool use_dev = UseDevice() || lo.UseDevice() || hi.UseDevice();
   const int N = size;
   // Note: get read access first, in case *this is the same as lo/hi.
   auto l = lo.Read(use_dev);
   auto h = hi.Read(use_dev);
   auto m = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      if (m[i] < l[i])
      {
         m[i] = l[i];
      }
      else if (m[i] > h[i])
      {
         m[i] = h[i];
      }
   });
}

void Vector::GetSubVector(const Array<int> &dofs, Vector &elemvect) const
{
   const int n = dofs.Size();
   elemvect.SetSize(n);
   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   auto d_y = elemvect.Write(use_dev);
   auto d_X = Read(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof_i = d_dofs[i];
      d_y[i] = dof_i >= 0 ? d_X[dof_i] : -d_X[-dof_i-1];
   });
}

void Vector::GetSubVector(const Array<int> &dofs, real_t *elem_data) const
{
   data.Read(MemoryClass::HOST, size);
   const int n = dofs.Size();
   for (int i = 0; i < n; i++)
   {
      const int j = dofs[i];
      elem_data[i] = (j >= 0) ? data[j] : -data[-1-j];
   }
}

void Vector::SetSubVector(const Array<int> &dofs, const real_t value)
{
   const bool use_dev = dofs.UseDevice();
   const int n = dofs.Size();
   // Use read+write access for *this - we only modify some of its entries
   auto d_X = ReadWrite(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j = d_dofs[i];
      if (j >= 0)
      {
         d_X[j] = value;
      }
      else
      {
         d_X[-1-j] = -value;
      }
   });
}

void Vector::SetSubVector(const Array<int> &dofs, const Vector &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(),
               "Size mismatch: length of dofs is " << dofs.Size()
               << ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   // Use read+write access for X - we only modify some of its entries
   auto d_X = ReadWrite(use_dev);
   auto d_y = elemvect.Read(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof_i = d_dofs[i];
      if (dof_i >= 0)
      {
         d_X[dof_i] = d_y[i];
      }
      else
      {
         d_X[-1-dof_i] = -d_y[i];
      }
   });
}

void Vector::SetSubVector(const Array<int> &dofs, real_t *elem_data)
{
   // Use read+write access because we overwrite only part of the data.
   data.ReadWrite(MemoryClass::HOST, size);
   const int n = dofs.Size();
   for (int i = 0; i < n; i++)
   {
      const int j= dofs[i];
      if (j >= 0)
      {
         operator()(j) = elem_data[i];
      }
      else
      {
         operator()(-1-j) = -elem_data[i];
      }
   }
}

void Vector::AddElementVector(const Array<int> &dofs, const Vector &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(), "Size mismatch: "
               "length of dofs is " << dofs.Size() <<
               ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   auto d_y = elemvect.Read(use_dev);
   auto d_X = ReadWrite(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j = d_dofs[i];
      if (j >= 0)
      {
         d_X[j] += d_y[i];
      }
      else
      {
         d_X[-1-j] -= d_y[i];
      }
   });
}

void Vector::AddElementVector(const Array<int> &dofs, real_t *elem_data)
{
   data.ReadWrite(MemoryClass::HOST, size);
   const int n = dofs.Size();
   for (int i = 0; i < n; i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         operator()(j) += elem_data[i];
      }
      else
      {
         operator()(-1-j) -= elem_data[i];
      }
   }
}

void Vector::AddElementVector(const Array<int> &dofs, const real_t a,
                              const Vector &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(), "Size mismatch: "
               "length of dofs is " << dofs.Size() <<
               ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   auto d_y = ReadWrite(use_dev);
   auto d_x = elemvect.Read(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j = d_dofs[i];
      if (j >= 0)
      {
         d_y[j] += a * d_x[i];
      }
      else
      {
         d_y[-1-j] -= a * d_x[i];
      }
   });
}

void Vector::SetSubVectorComplement(const Array<int> &dofs, const real_t val)
{
   const bool use_dev = UseDevice() || dofs.UseDevice();
   const int n = dofs.Size();
   const int N = size;
   Vector dofs_vals(n, use_dev ?
                    Device::GetDeviceMemoryType() :
                    Device::GetHostMemoryType());
   auto d_data = ReadWrite(use_dev);
   auto d_dofs_vals = dofs_vals.Write(use_dev);
   auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { d_dofs_vals[i] = d_data[d_dofs[i]]; });
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { d_data[i] = val; });
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { d_data[d_dofs[i]] = d_dofs_vals[i]; });
}

void Vector::Print(std::ostream &os, int width) const
{
   if (!size) { return; }
   data.Read(MemoryClass::HOST, size);
   for (int i = 0; 1; )
   {
      os << ZeroSubnormal(data[i]);
      i++;
      if (i == size)
      {
         break;
      }
      if ( i % width == 0 )
      {
         os << '\n';
      }
      else
      {
         os << ' ';
      }
   }
   os << '\n';
}

#ifdef MFEM_USE_ADIOS2
void Vector::Print(adios2stream &os,
                   const std::string& variable_name) const
{
   if (!size) { return; }
   data.Read(MemoryClass::HOST, size);
   os.engine.Put(variable_name, &data[0] );
}
#endif

void Vector::Print_HYPRE(std::ostream &os) const
{
   int i;
   std::ios::fmtflags old_fmt = os.flags();
   os.setf(std::ios::scientific);
   std::streamsize old_prec = os.precision(14);

   os << size << '\n';  // number of rows

   data.Read(MemoryClass::HOST, size);
   for (i = 0; i < size; i++)
   {
      os << ZeroSubnormal(data[i]) << '\n';
   }

   os.precision(old_prec);
   os.flags(old_fmt);
}

void Vector::PrintMathematica(std::ostream & os) const
{
   std::ios::fmtflags old_fmt = os.flags();
   os.setf(std::ios::scientific);
   std::streamsize old_prec = os.precision(14);

   os << "(* Read file into Mathematica using: "
      << "myVec = Get[\"this_file_name\"] *)\n";
   os << "{\n";

   data.Read(MemoryClass::HOST, size);
   for (int i = 0; i < size; i++)
   {
      os << "Internal`StringToMReal[\"" << ZeroSubnormal(data[i]) << "\"]";
      if (i < size - 1) { os << ','; }
      os << '\n';
   }

   os << "}\n";

   os.precision(old_prec);
   os.flags(old_fmt);
}

void Vector::PrintHash(std::ostream &os) const
{
   os << "size: " << size << '\n';
   HashFunction hf;
   hf.AppendDoubles(HostRead(), size);
   os << "hash: " << hf.GetHash() << '\n';
}

void Vector::Randomize(int seed)
{
   if (seed == 0)
   {
      seed = (int)time(0);
   }

   srand((unsigned)seed);

   HostWrite();
   for (int i = 0; i < size; i++)
   {
      data[i] = rand_real();
   }
}

real_t Vector::Norml2() const
{
   // Scale entries of Vector on the fly, using algorithms from
   // std::hypot() and LAPACK's drm2. This scaling ensures that the
   // argument of each call to std::pow is <= 1 to avoid overflow.
   if (size == 0)
   {
      return 0.0;
   }

   if (UseDevice())
   {
#ifdef MFEM_USE_CUDA
      if (Device::Allows(Backend::CUDA_MASK))
      {
         return devVectorL2(size, Read());
      }
#endif
#ifdef MFEM_USE_HIP
      if (Device::Allows(Backend::HIP_MASK))
      {
         return devVectorL2(size, Read());
      }
#endif
   }

   auto ptr = data.Read(MemoryClass::HOST, size);
   if (1 == size)
   {
      return std::abs(ptr[0]);
   } // end if 1 == size
   return kernels::Norml2(size, ptr);
}

real_t Vector::Normlinf() const
{
   if (size == 0) { return 0; }

   const bool use_dev = UseDevice();
   auto m_data = Read(use_dev);

   if (!use_dev) { goto vector_linf_cpu; }

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return devVectorLinf(size, m_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return devVectorLinf(size, m_data);
   }
#endif

   if (Device::Allows(Backend::DEBUG_DEVICE))
   {
      const int N = size;
      auto m_data_ = Read();
      Vector max(1);
      max = 0.;
      max.UseDevice(true);
      auto d_max = max.ReadWrite();
      mfem::forall(N, [=] MFEM_HOST_DEVICE(int i)
      {
         d_max[0] = fmax(d_max[0], fabs(m_data_[i]));
      });
      max.HostReadWrite();
      return max[0];
   }

vector_linf_cpu:
   real_t maximum = fabs(data[0]);
   for (int i = 1; i < size; i++)
   {
      maximum = fmax(maximum, fabs(m_data[i]));
   }
   return maximum;
}

real_t Vector::Norml1() const
{
   if (size == 0) { return 0.0; }

   if (UseDevice())
   {
#ifdef MFEM_USE_CUDA
      if (Device::Allows(Backend::CUDA_MASK))
      {
         return devVectorL1(size, Read());
      }
#endif
#ifdef MFEM_USE_HIP
      if (Device::Allows(Backend::HIP_MASK))
      {
         return devVectorL1(size, Read());
      }
#endif
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         const int N = size;
         auto d_data = Read();
         Vector sum(1);
         sum.UseDevice(true);
         auto d_sum = sum.Write();
         d_sum[0] = 0.0;
         mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
         {
            d_sum[0] += fabs(d_data[i]);
         });
         sum.HostReadWrite();
         return sum[0];
      }
   }

   // CPU fallback
   const real_t *h_data = HostRead();
   real_t sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      sum += fabs(h_data[i]);
   }
   return sum;
}

real_t Vector::Normlp(real_t p) const
{
   MFEM_ASSERT(p > 0.0, "Vector::Normlp");

   if (p == 1.0)
   {
      return Norml1();
   }
   if (p == 2.0)
   {
      return Norml2();
   }
   if (p < infinity())
   {
      // Scale entries of Vector on the fly, using algorithms from
      // std::hypot() and LAPACK's drm2. This scaling ensures that the
      // argument of each call to std::pow is <= 1 to avoid overflow.
      if (size == 0)
      {
         return 0.0;
      }

      if (UseDevice())
      {
#ifdef MFEM_USE_CUDA
         if (Device::Allows(Backend::CUDA_MASK))
         {
            return devVectorLp(size, p, Read());
         }
#endif
#ifdef MFEM_USE_HIP
         if (Device::Allows(Backend::HIP_MASK))
         {
            return devVectorLp(size, p, Read());
         }
#endif
      }

      auto ptr = data.Read(MemoryClass::HOST, size);
      if (1 == size)
      {
         return std::abs(ptr[0]);
      } // end if 1 == size

      real_t scale = 0.0;
      real_t sum = 0.0;

      for (int i = 0; i < size; i++)
      {
         if (ptr[i] != 0.0)
         {
            const real_t absdata = std::abs(ptr[i]);
            if (scale <= absdata)
            {
               sum = 1.0 + sum * std::pow(scale / absdata, p);
               scale = absdata;
               continue;
            } // end if scale <= absdata
            sum += std::pow(absdata / scale, p); // else scale > absdata
         } // end if ptr[i] != 0
      }
      return scale * std::pow(sum, 1.0/p);
   } // end if p < infinity()

   return Normlinf(); // else p >= infinity()
}

real_t Vector::operator*(const Vector &v) const
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");
   if (size == 0) { return 0.0; }

   const bool use_dev = UseDevice() || v.UseDevice();
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP) || defined(MFEM_USE_OPENMP)
   auto m_data = Read(use_dev);
#else
   Read(use_dev);
#endif
   auto v_data = v.Read(use_dev);

   if (!use_dev) { goto vector_dot_cpu; }

#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      return occa::linalg::dot<real_t,real_t,real_t>(
                OccaMemoryRead(data, size), OccaMemoryRead(v.data, size));
   }
#endif

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return devVectorDot(size, m_data, v_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return devVectorDot(size, m_data, v_data);
   }
#endif

#ifdef MFEM_USE_OPENMP
   if (Device::Allows(Backend::OMP_MASK))
   {
#define MFEM_USE_OPENMP_DETERMINISTIC_DOT
#ifdef MFEM_USE_OPENMP_DETERMINISTIC_DOT
      // By default, use a deterministic way of computing the dot product
      static Vector th_dot;
      #pragma omp parallel
      {
         const int nt = omp_get_num_threads();
         #pragma omp master
         th_dot.SetSize(nt);
         const int tid    = omp_get_thread_num();
         const int stride = (size + nt - 1)/nt;
         const int start  = tid*stride;
         const int stop   = std::min(start + stride, size);
         real_t my_dot = 0.0;
         for (int i = start; i < stop; i++)
         {
            my_dot += m_data[i] * v_data[i];
         }
         #pragma omp barrier
         th_dot(tid) = my_dot;
      }
      return th_dot.Sum();
#else
      // The standard way of computing the dot product is non-deterministic
      real_t prod = 0.0;
      #pragma omp parallel for reduction(+:prod)
      for (int i = 0; i < size; i++)
      {
         prod += m_data[i] * v_data[i];
      }
      return prod;
#endif // MFEM_USE_OPENMP_DETERMINISTIC_DOT
   }
#endif // MFEM_USE_OPENMP
   if (Device::Allows(Backend::DEBUG_DEVICE))
   {
      const int N = size;
      auto v_data_ = v.Read();
      auto m_data_ = Read();
      Vector dot(1);
      dot.UseDevice(true);
      auto d_dot = dot.Write();
      dot = 0.0;
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
      {
         d_dot[0] += m_data_[i] * v_data_[i];
      });
      dot.HostReadWrite();
      return dot[0];
   }
vector_dot_cpu:
   return operator*(v_data);
}

real_t Vector::Min() const
{
   if (size == 0) { return infinity(); }

   const bool use_dev = UseDevice();
   auto m_data = Read(use_dev);

   if (!use_dev) { goto vector_min_cpu; }

#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      return occa::linalg::min<real_t,real_t>(OccaMemoryRead(data, size));
   }
#endif

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return devVectorMin(size, m_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return devVectorMin(size, m_data);
   }
#endif

#ifdef MFEM_USE_OPENMP
   if (Device::Allows(Backend::OMP_MASK))
   {
      real_t minimum = m_data[0];
      #pragma omp parallel for reduction(min:minimum)
      for (int i = 0; i < size; i++)
      {
         minimum = std::min(minimum, m_data[i]);
      }
      return minimum;
   }
#endif

   if (Device::Allows(Backend::DEBUG_DEVICE))
   {
      const int N = size;
      auto m_data_ = Read();
      Vector min(1);
      min = infinity();
      min.UseDevice(true);
      auto d_min = min.ReadWrite();
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
      {
         d_min[0] = fmin(d_min[0], m_data_[i]);
      });
      min.HostReadWrite();
      return min[0];
   }

vector_min_cpu:
   real_t minimum = data[0];
   for (int i = 1; i < size; i++)
   {
      minimum = fmin(minimum, m_data[i]);
   }
   return minimum;
}

real_t Vector::Max() const
{
   if (size == 0) { return -infinity(); }

   const bool use_dev = UseDevice();
   auto m_data = Read(use_dev);

   if (!use_dev) { goto vector_max_cpu; }

#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      return occa::linalg::max<real_t,real_t>(OccaMemoryRead(data, size));
   }
#endif

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return devVectorMax(size, m_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return devVectorMax(size, m_data);
   }
#endif

#ifdef MFEM_USE_OPENMP
   if (Device::Allows(Backend::OMP_MASK))
   {
      real_t maximum = m_data[0];
      #pragma omp parallel for reduction(max:maximum)
      for (int i = 0; i < size; i++)
      {
         maximum = fmax(maximum, m_data[i]);
      }
      return maximum;
   }
#endif

   if (Device::Allows(Backend::DEBUG_DEVICE))
   {
      const int N = size;
      auto m_data_ = Read();
      Vector max(1);
      max = -infinity();
      max.UseDevice(true);
      auto d_max = max.ReadWrite();
      mfem::forall(N, [=] MFEM_HOST_DEVICE(int i)
      {
         d_max[0] = fmax(d_max[0], m_data_[i]);
      });
      max.HostReadWrite();
      return max[0];
   }

vector_max_cpu:
   real_t maximum = data[0];
   for (int i = 1; i < size; i++)
   {
      maximum = fmax(maximum, m_data[i]);
   }
   return maximum;
}

real_t Vector::Sum() const
{
   if (size == 0) { return 0.0; }

   if (UseDevice())
   {
#ifdef MFEM_USE_CUDA
      if (Device::Allows(Backend::CUDA_MASK))
      {
         return devVectorSum(size, Read());
      }
#endif
#ifdef MFEM_USE_HIP
      if (Device::Allows(Backend::HIP_MASK))
      {
         return devVectorSum(size, Read());
      }
#endif
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         const int N = size;
         auto d_data = Read();
         Vector sum(1);
         sum.UseDevice(true);
         auto d_sum = sum.Write();
         d_sum[0] = 0.0;
         mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
         {
            d_sum[0] += d_data[i];
         });
         sum.HostReadWrite();
         return sum[0];
      }
   }

   // CPU fallback
   const real_t *h_data = HostRead();
   real_t sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      sum += h_data[i];
   }
   return sum;
}

}
