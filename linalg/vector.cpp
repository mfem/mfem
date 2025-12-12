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

// Implementation of data type vector

#include "../general/forall.hpp"
#include "../general/reducers.hpp"
#include "../general/hash.hpp"
#include "../general/scan.hpp"
#include "vector.hpp"

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <cmath>
#include <ctime>

namespace mfem
{

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
   static MFEM_HOST_DEVICE void Join(value_type& a, const value_type &b)
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

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
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
   MFEM_HOST_DEVICE void Join(value_type& a, const value_type &b) const
   {
      real_t scale = fmax(a.second, b.second);
      if (scale > 0)
      {
         a.first = a.first * pow(a.second / scale, p) +
                   b.first * pow(b.second / scale, p);
         a.second = scale;
      }
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a.first = 0;
      a.second = 0;
   }
};

template <class T>
static Array<T>& vector_workspace()
{
   static Array<T> instance;
   return instance;
}

template <class T>
static Array<DevicePair<T, T>> &Lpvector_workspace()
{
   static Array<DevicePair<T, T>> instance;
   return instance;
}

template <class T>
VectorMP<T>::VectorMP(const VectorMP<T> &v)
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

template <class T>
VectorMP<T>::VectorMP(VectorMP<T> &&v)
   : data(std::move(v.data)), size(v.size)
{
   v.size = 0;
}

template <class T>
void VectorMP<T>::Load(std::istream **in, int np, int *dim)
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

template <class T>
void VectorMP<T>::Load(std::istream &in, int Size)
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

template <class T>
T &VectorMP<T>::Elem(int i)
{
   return operator()(i);
}

template <class T>
const T &VectorMP<T>::Elem(int i) const
{
   return operator()(i);
}

template <class T>
T VectorMP<T>::operator*(const T *v) const
{
   HostRead();
   T dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < size; i++)
   {
      dot += data[i] * v[i];
   }
   return dot;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator=(const T *v)
{
   data.CopyFromHost(v, size);
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator=(const VectorMP<T> &v)
{
#if 0
   SetSize(v.Size(), v.data.GetMemoryType());
   data.CopyFrom(v.data, v.Size());
   UseDevice(v.UseDevice());
#else
   SetSize(v.Size());
   const bool vuse = v.UseDevice();
   const bool use_dev = UseDevice() || vuse;
   if (use_dev != vuse) { v.UseDevice(use_dev); }
   // keep 'data' where it is, unless 'use_dev' is true
   if (use_dev) { Write(); }
   data.CopyFrom(v.data, v.Size());
   if (use_dev != vuse) { v.UseDevice(vuse); }
#endif
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator=(VectorMP<T> &&v)
{
   if (this != &v)
   {
      v.Swap(*this);
      v.Destroy();
   }
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator=(T value)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = value; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator*=(T c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= c; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator*=(const VectorMP<T> &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= x[i]; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator/=(T c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   const T m = 1.0/c;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] *= m; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator/=(const VectorMP<T> &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] /= x[i]; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator-=(T c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] -= c; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator-=(const VectorMP<T> &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] -= x[i]; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator+=(T c)
{
   const bool use_dev = UseDevice();
   const int N = size;

   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += c; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::operator+=(const VectorMP<T> &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += x[i]; });
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::Add(const T a, const VectorMP<T> &Va)
{
   MFEM_ASSERT(size == Va.size, "incompatible Vectors!");

   if (a != 0.0)
   {
      const int N = size;
      const bool use_dev = UseDevice() || Va.UseDevice();
      const auto x = Va.Read(use_dev);
      auto y = ReadWrite(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] += a * x[i]; });
   }
   return *this;
}

template <class T>
VectorMP<T> &VectorMP<T>::Set(const T a, const VectorMP<T> &Va)
{
   MFEM_ASSERT(size == Va.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || Va.UseDevice();
   const int N = size;
   const auto x = Va.Read(use_dev);
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = a * x[i]; });
   return *this;
}

template <class T>
void VectorMP<T>::SetVector(const VectorMP<T> &v, int offset)
{
   MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int vs = v.Size();
   const auto vp = v.Read(use_dev);
   // Use read+write access for *this - we only modify some of its entries
   auto p = ReadWrite(use_dev) + offset;
   mfem::forall_switch(use_dev, vs, [=] MFEM_HOST_DEVICE (int i) { p[i] = vp[i]; });
}

template <class T>
void VectorMP<T>::AddSubVector(const VectorMP<T> &v, int offset)
{
   MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int vs = v.Size();
   const auto vp = v.Read(use_dev);
   auto p = ReadWrite(use_dev) + offset;
   mfem::forall_switch(use_dev, vs, [=] MFEM_HOST_DEVICE (int i) { p[i] += vp[i]; });
}

template <class T>
void VectorMP<T>::Neg()
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = -y[i]; });
}

template <class T>
void VectorMP<T>::Reciprocal()
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = 1.0/y[i]; });
}

template <class T>
void VectorMP<T>::Abs()
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      y[i] = std::abs(y[i]);
   });
}

template <class T>
void VectorMP<T>::Pow(const T p)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      y[i] = std::pow(y[i], p);
   });
}

template <class T>
void add(const VectorMP<T> &v1, const VectorMP<T> &v2, VectorMP<T> &v)
{
   MFEM_ASSERT(v.Size() == v1.Size() && v.Size() == v2.Size(),
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = v1.UseDevice() || v2.UseDevice() || v.UseDevice();
   const int N = v.size;
   // Note: get read access first, in case v is the same as v1/v2.
   const auto x1 = v1.Read(use_dev);
   const auto x2 = v2.Read(use_dev);
   auto y = v.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = x1[i] + x2[i]; });
#else
   #pragma omp parallel for
   for (int i = 0; i < v.Size(); i++)
   {
      v.data[i] = v1.data[i] + v2.data[i];
   }
#endif
}

template <class T, class U>
void add(const VectorMP<T> &v1, U alpha, const VectorMP<T> &v2, VectorMP<T> &v)
{
   MFEM_ASSERT(v.Size() == v1.Size() && v.Size() == v2.Size(),
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
      const int N = v.Size();
      // Note: get read access first, in case v is the same as v1/v2.
      const auto d_x = v1.Read(use_dev);
      const auto d_y = v2.Read(use_dev);
      auto d_z = v.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         d_z[i] = d_x[i] + alpha * d_y[i];
      });
#else
      const T *v1p = v1.data, *v2p = v2.data;
      T *vp = v.data;
      const int s = v.Size();
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         vp[i] = v1p[i] + alpha*v2p[i];
      }
#endif
   }
}

template <class T, class U>
void add(const U a, const VectorMP<T> &x, const VectorMP<T> &y, VectorMP<T> &z)
{
   MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
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
      const int N = x.Size();
      // Note: get read access first, in case z is the same as x/y.
      const auto xd = x.Read(use_dev);
      const auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * (xd[i] + yd[i]);
      });
#else
      const T *xp = x.data;
      const T *yp = y.data;
      T       *zp = z.data;
      const int      s = x.Size();
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] + yp[i]);
      }
#endif
   }
}

template <class T, class U>
void add(const U a, const VectorMP<T> &x,
         const U b, const VectorMP<T> &y, VectorMP<T> &z)
{
   MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
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
      const int N = x.Size();
      // Note: get read access first, in case z is the same as x/y.
      const auto xd = x.Read(use_dev);
      const auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * xd[i] + b * yd[i];
      });
#else
      const T *xp = x.data;
      const T *yp = y.data;
      T       *zp = z.data;
      const int      s = x.Size();
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * xp[i] + b * yp[i];
      }
#endif
   }
}

template <class T>
void subtract(const VectorMP<T> &x, const VectorMP<T> &y, VectorMP<T> &z)
{
   MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = x.UseDevice() || y.UseDevice() || z.UseDevice();
   const int N = x.Size();
   // Note: get read access first, in case z is the same as x/y.
   const auto xd = x.Read(use_dev);
   const auto yd = y.Read(use_dev);
   auto zd = z.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      zd[i] = xd[i] - yd[i];
   });
#else
   const T *xp = x.data;
   const T *yp = y.data;
   T       *zp = z.data;
   const int     s = x.Size();
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      zp[i] = xp[i] - yp[i];
   }
#endif
}

template <class T, class U>
void subtract(const U a, const VectorMP<T> &x, const VectorMP<T> &y,
              VectorMP<T> &z)
{
   MFEM_ASSERT(x.Size() == y.Size() && x.Size() == z.Size(),
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
      const int N = x.Size();
      // Note: get read access first, in case z is the same as x/y.
      const auto xd = x.Read(use_dev);
      const auto yd = y.Read(use_dev);
      auto zd = z.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         zd[i] = a * (xd[i] - yd[i]);
      });
#else
      const T *xp = x.data;
      const T *yp = y.data;
      T       *zp = z.data;
      const int      s = x.Size();
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] - yp[i]);
      }
#endif
   }
}

template <class T>
void VectorMP<T>::cross3D(const VectorMP<T> &vin, VectorMP<T> &vout) const
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

template <class T>
void VectorMP<T>::median(const VectorMP<T> &lo, const VectorMP<T> &hi)
{
   MFEM_ASSERT(size == lo.size && size == hi.size,
               "incompatible Vectors!");

   const bool use_dev = UseDevice() || lo.UseDevice() || hi.UseDevice();
   const int N = size;
   // Note: get read access first, in case *this is the same as lo/hi.
   const auto l = lo.Read(use_dev);
   const auto h = hi.Read(use_dev);
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

template <class T>
void VectorMP<T>::GetSubVector(const Array<int> &dofs,
                               VectorMP<T> &elemvect) const
{
   const int n = dofs.Size();
   elemvect.SetSize(n);
   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const auto d_X = Read(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
   auto d_y = elemvect.Write(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof_i = d_dofs[i];
      d_y[i] = dof_i >= 0 ? d_X[dof_i] : -d_X[-dof_i-1];
   });
}

template <class T>
void VectorMP<T>::GetSubVector(const Array<int> &dofs, T *elem_data) const
{
   HostRead();
   const int n = dofs.Size();
   for (int i = 0; i < n; i++)
   {
      const int j = dofs[i];
      elem_data[i] = (j >= 0) ? data[j] : -data[-1-j];
   }
}

template <class T>
void VectorMP<T>::SetSubVector(const Array<int> &dofs, const T value)
{
   const bool use_dev = UseDevice() || dofs.UseDevice();
   const int n = dofs.Size();
   // Use read+write access for *this - we only modify some of its entries
   auto d_X = ReadWrite(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
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

template <class T>
void VectorMP<T>::SetSubVectorHost(const Array<int> &dofs, const T value)
{
   HostReadWrite();
   for (int i = 0; i < dofs.Size(); ++i)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         (*this)[j] = value;
      }
      else
      {
         (*this)[-1-j] = -value;
      }
   }
}

template <class T>
void VectorMP<T>::SetSubVector(const Array<int> &dofs,
                               const VectorMP<T> &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(),
               "Size mismatch: length of dofs is " << dofs.Size()
               << ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   // Use read+write access for X - we only modify some of its entries
   auto d_X = ReadWrite(use_dev);
   const auto d_y = elemvect.Read(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
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

template <class T>
void VectorMP<T>::SetSubVector(const Array<int> &dofs, T *elem_data)
{
   // Use read+write access because we overwrite only part of the data.
   HostReadWrite();
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

template <class T>
void VectorMP<T>::AddElementVector(const Array<int> &dofs,
                                   const VectorMP<T> &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(), "Size mismatch: "
               "length of dofs is " << dofs.Size() <<
               ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   const auto d_y = elemvect.Read(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
   auto d_X = ReadWrite(use_dev);
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

template <class T>
void VectorMP<T>::AddElementVector(const Array<int> &dofs, T *elem_data)
{
   HostReadWrite();
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

template <class T>
void VectorMP<T>::AddElementVector(const Array<int> &dofs, const T a,
                                   const VectorMP<T> &elemvect)
{
   MFEM_ASSERT(dofs.Size() <= elemvect.Size(), "Size mismatch: "
               "length of dofs is " << dofs.Size() <<
               ", length of elemvect is " << elemvect.Size());

   const bool use_dev = dofs.UseDevice() || elemvect.UseDevice();
   const int n = dofs.Size();
   const auto d_x = elemvect.Read(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
   auto d_y = ReadWrite(use_dev);
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

template <class T>
void VectorMP<T>::SetSubVectorComplement(const Array<int> &dofs, const T val)
{
   const bool use_dev = UseDevice() || dofs.UseDevice();
   const int n = dofs.Size();
   const int N = size;
   VectorMP<T> dofs_vals(n, use_dev ?
                         Device::GetDeviceMemoryType() :
                         Device::GetHostMemoryType());
   auto d_data = ReadWrite(use_dev);
   auto d_dofs_vals = dofs_vals.Write(use_dev);
   const auto d_dofs = dofs.Read(use_dev);
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { d_dofs_vals[i] = d_data[d_dofs[i]]; });
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { d_data[i] = val; });
   mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { d_data[d_dofs[i]] = d_dofs_vals[i]; });
}

template <class T>
void VectorMP<T>::Print(std::ostream &os, int width) const
{
   if (!size) { return; }
   HostRead();
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
template <class T>
void VectorMP<T>::Print(adios2stream &os,
                        const std::string& variable_name) const
{
   if (!size) { return; }
   HostRead();
   os.engine.Put(variable_name, &data[0] );
}
#endif

template <class T>
void VectorMP<T>::Print_HYPRE(std::ostream &os) const
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

template <class T>
void VectorMP<T>::PrintMathematica(std::ostream & os) const
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

template <class T>
void VectorMP<T>::PrintHash(std::ostream &os) const
{
   os << "size: " << size << '\n';
   HashFunction hf;
   // TODO: eliminate this hack by generalizing HashFunction.
   Vector d(this->Size());
   for (int i=0; i<this->Size(); ++i)
   {
      d[i] = (*this)[i];
   }

   hf.AppendDoubles(d.HostRead(), size);
   os << "hash: " << hf.GetHash() << '\n';
}

template <class T>
void VectorMP<T>::Randomize(int seed)
{
   if (seed == 0) { seed = (int)time(0); }

   srand((unsigned)seed);

   HostWrite();
   for (int i = 0; i < size; i++)
   {
      data[i] = rand_real();
   }
}

template <class T>
T VectorMP<T>::Norml2() const
{
   // Scale entries of Vector on the fly, using algorithms from
   // std::hypot() and LAPACK's drm2. This scaling ensures that the
   // argument of each call to std::pow is <= 1 to avoid overflow.
   if (size == 0) { return 0.0; }

   const auto m_data = Read(UseDevice());
   using value_type = DevicePair<T, T>;
   value_type res;
   res.first = 0;
   res.second = 0;
   // first compute sum (|m_data|/scale)^2
   reduce(size, res, [=] MFEM_HOST_DEVICE(int i, value_type &r)
   {
      T n = fabs(m_data[i]);
      if (n > 0)
      {
         if (r.second <= n)
         {
            T arg = r.second / n;
            r.first = r.first * (arg * arg) + 1;
            r.second = n;
         }
         else
         {
            T arg = n / r.second;
            r.first += arg * arg;
         }
      }
   },
   L2Reducer{}, UseDevice(), Lpvector_workspace<T>());
   // final answer
   return res.second * sqrt(res.first);
}

template <class T>
T VectorMP<T>::Normlinf() const
{
   if (size == 0) { return 0; }

   T res = 0;
   const auto m_data = Read(UseDevice());
   reduce(size, res, [=] MFEM_HOST_DEVICE(int i, T &r)
   {
      r = fmax(r, fabs(m_data[i]));
   },
   MaxReducer<T> {}, UseDevice(), vector_workspace<T>());
   return res;
}

template <class T>
T VectorMP<T>::Norml1() const
{
   if (size == 0) { return 0.0; }

   T res = 0;
   const auto m_data = Read(UseDevice());
   reduce(size, res, [=] MFEM_HOST_DEVICE(int i, T &r)
   {
      r += fabs(m_data[i]);
   },
   SumReducer<T> {}, UseDevice(), vector_workspace<T>());
   return res;
}

template <class T>
T VectorMP<T>::Normlp(T p) const
{
   MFEM_ASSERT(p > 0.0, "Vector::Normlp");

   if (p == 1.0) { return Norml1(); }

   if (p == 2.0) { return Norml2(); }

   if (p < infinity())
   {
      // Scale entries of Vector on the fly, using algorithms from
      // std::hypot() and LAPACK's drm2. This scaling ensures that the
      // argument of each call to std::pow is <= 1 to avoid overflow.
      if (size == 0) { return 0.0; }

      using value_type = DevicePair<T, T>;
      value_type res;
      res.first = 0;
      res.second = 0;
      const auto m_data = Read(UseDevice());
      // first compute sum (|m_data|/scale)^p
      reduce(size, res, [=] MFEM_HOST_DEVICE(int i, value_type &r)
      {
         T n = fabs(m_data[i]);
         if (n > 0)
         {
            if (r.second <= n)
            {
               T arg = r.second / n;
               r.first = r.first * pow(arg, p) + 1;
               r.second = n;
            }
            else
            {
               T arg = n / r.second;
               r.first += pow(arg, p);
            }
         }
      },
      LpReducer{p}, UseDevice(), Lpvector_workspace<T>());
      // final answer
      return res.second * pow(res.first, 1.0 / p);
   } // end if p < infinity()

   return Normlinf(); // else p >= infinity()
}

template <class T>
T VectorMP<T>::operator*(const VectorMP<T> &v) const
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   if (size == 0) { return 0.0; }

   const bool use_dev = UseDevice() || v.UseDevice();
   const auto m_data = Read(use_dev), v_data = v.Read(use_dev);

   // If OCCA is enabled, it handles all selected backends
#ifdef MFEM_USE_OCCA
   if (use_dev && DeviceCanUseOcca())
   {
      return occa::linalg::dot<T, T, T>(
                OccaMemoryRead(data, size), OccaMemoryRead(v.data, size));
   }
#endif

   const auto compute_dot = [&]()
   {
      T res = 0;
      reduce(size, res, [=] MFEM_HOST_DEVICE (int i, T &r)
      {
         r += m_data[i] * v_data[i];
      },
      SumReducer<T> {}, use_dev, vector_workspace<T>());
      return res;
   };

   // Device backends have top priority
   if (Device::Allows(Backend::DEVICE_MASK)) { return compute_dot(); }

   // Special path for OpenMP
#ifdef MFEM_USE_OPENMP
   if (use_dev && Device::Allows(Backend::OMP_MASK))
   {
      // By default, use a deterministic way of computing the dot product
#define MFEM_USE_OPENMP_DETERMINISTIC_DOT
#ifdef MFEM_USE_OPENMP_DETERMINISTIC_DOT
      static Vector th_dot;
      #pragma omp parallel
      {
         const int nt = omp_get_num_threads();
         #pragma omp master
         th_dot.SetSize(nt);
         const int tid = omp_get_thread_num();
         const int stride = (size + nt - 1) / nt;
         const int start = tid * stride;
         const int stop = std::min(start + stride, size);
         T my_dot = 0.0;
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
      T prod = 0.0;
      #pragma omp parallel for reduction(+ : prod)
      for (int i = 0; i < size; i++)
      {
         prod += m_data[i] * v_data[i];
      }
      return prod;
#endif // MFEM_USE_OPENMP_DETERMINISTIC_DOT
   }
#endif // MFEM_USE_OPENMP

   // All other CPU backends
   return compute_dot();
}

template <class T>
T VectorMP<T>::Min() const
{
   if (size == 0) { return infinity(); }

   const auto use_dev = UseDevice();
   const auto m_data = Read(use_dev);

#ifdef MFEM_USE_OCCA
   if (use_dev && DeviceCanUseOcca())
   {
      return occa::linalg::min<real_t,real_t>(OccaMemoryRead(data, size));
   }
#endif

   const auto compute_min = [&]()
   {
      T res = infinity();
      reduce(size, res, [=] MFEM_HOST_DEVICE(int i, T &r)
      {
         r = fmin(r, m_data[i]);
      },
      MinReducer<T> {}, use_dev, vector_workspace<T>());
      return res;
   };

   // Device backends have top priority
   if (Device::Allows(Backend::DEVICE_MASK)) { return compute_min(); }

   // Special path for OpenMP
#ifdef MFEM_USE_OPENMP
   if (use_dev && Device::Allows(Backend::OMP_MASK))
   {
      T minimum = m_data[0];
      #pragma omp parallel for reduction(min:minimum)
      for (int i = 0; i < size; i++)
      {
         minimum = std::min(minimum, m_data[i]);
      }
      return minimum;
   }
#endif

   // All other CPU backends
   return compute_min();
}

template <class T>
T VectorMP<T>::Max() const
{
   if (size == 0) { return -infinity(); }

   const auto use_dev = UseDevice();
   const auto m_data = Read(use_dev);

#ifdef MFEM_USE_OCCA
   if (use_dev && DeviceCanUseOcca())
   {
      return occa::linalg::max<real_t, real_t>(OccaMemoryRead(data, size));
   }
#endif

   const auto compute_max = [&]()
   {
      T res = -infinity();
      reduce(size, res, [=] MFEM_HOST_DEVICE(int i, T &r)
      {
         r = fmax(r, m_data[i]);
      },
      MaxReducer<T> {}, use_dev, vector_workspace<T>());
      return res;
   };

   // Device backends have top priority
   if (Device::Allows(Backend::DEVICE_MASK)) { return compute_max(); }

   // Special path for OpenMP
#ifdef MFEM_USE_OPENMP
   if (use_dev && Device::Allows(Backend::OMP_MASK))
   {
      real_t maximum = m_data[0];
      #pragma omp parallel for reduction(max : maximum)
      for (int i = 0; i < size; i++)
      {
         maximum = fmax(maximum, m_data[i]);
      }
      return maximum;
   }
#endif

   // All other CPU backends
   return compute_max();
}

template <class T>
T VectorMP<T>::Sum() const
{
   if (size == 0) { return 0.0; }

   T res = 0;
   const auto m_data = Read(UseDevice());
   reduce(size, res, [=] MFEM_HOST_DEVICE(int i, T &r)
   {
      r += m_data[i];
   },
   SumReducer<T> {}, UseDevice(), vector_workspace<T>());
   return res;
}

template <class T>
void VectorMP<T>::DeleteAt(const Array<int> &indices)
{
   if (indices.Size())
   {
      const bool use_dev = UseDevice();

      // extra entry for number of selected out
      Array<int> workspace(size + 1);
      const auto d_flag = workspace.Write(use_dev);
      mfem::forall_switch(use_dev, size,
      [=] MFEM_HOST_DEVICE(int i) { d_flag[i] = true; });
      const auto d_indices = indices.Read(use_dev);
      mfem::forall_switch(use_dev, indices.Size(), [=] MFEM_HOST_DEVICE(int i)
      {
         // fine as long as indices are unique; to support non-unique indices
         // assignment to d_flag must be atomic
         d_flag[d_indices[i]] = false;
      });

      VectorMP<T> copy(*this);
      auto d_in = copy.Read(use_dev);
      auto d_out = Write(use_dev);
      CopyFlagged(use_dev, d_in, d_flag, d_out, d_flag + size, size);

      // assumes indices are unique
      size -= indices.Size();
   }
}

template class VectorMP<float>;
template class VectorMP<double>;

template
void add<real_t>(const VectorMP<real_t> &v1, const VectorMP<real_t> &v2,
                 VectorMP<real_t> &v);

template
void add<float,float>(const VectorMP<float> &v1, float alpha,
                      const VectorMP<float> &v2, VectorMP<float> &v);

template
void add<float,double>(const VectorMP<float> &v1, double alpha,
                       const VectorMP<float> &v2, VectorMP<float> &v);

template
void add<double,double>(const VectorMP<double> &v1, double alpha,
                        const VectorMP<double> &v2, VectorMP<double> &v);

template
void add<double,float>(const VectorMP<double> &v1, float alpha,
                       const VectorMP<double> &v2, VectorMP<double> &v);

template
void add<float,float>(const float a, const VectorMP<float> &x,
                      const VectorMP<float> &y, VectorMP<float> &z);

template
void add<float,double>(const double a, const VectorMP<float> &x,
                       const VectorMP<float> &y, VectorMP<float> &z);

template
void add<double,double>(const double a, const VectorMP<double> &x,
                        const VectorMP<double> &y, VectorMP<double> &z);

template
void add<double,float>(const float a, const VectorMP<double> &x,
                       const VectorMP<double> &y, VectorMP<double> &z);

template
void add<float,float>(const float a, const VectorMP<float> &x,
                      const float b, const VectorMP<float> &y, VectorMP<float> &z);

template
void add<float,double>(const double a, const VectorMP<float> &x,
                       const double b, const VectorMP<float> &y, VectorMP<float> &z);

template
void add<double,double>(const double a, const VectorMP<double> &x,
                        const double b, const VectorMP<double> &y, VectorMP<double> &z);

template
void add<double,float>(const float a, const VectorMP<double> &x,
                       const float b, const VectorMP<double> &y, VectorMP<double> &z);

template
void subtract<float>(const VectorMP<float> &x, const VectorMP<float> &y,
                     VectorMP<float> &z);

template
void subtract<double>(const VectorMP<double> &x, const VectorMP<double> &y,
                      VectorMP<double> &z);

template
void subtract<float,float>(const float a, const VectorMP<float> &x,
                           const VectorMP<float> &y,
                           VectorMP<float> &z);

template
void subtract<float,double>(const double a, const VectorMP<float> &x,
                            const VectorMP<float> &y,
                            VectorMP<float> &z);

template
void subtract<double,double>(const double a, const VectorMP<double> &x,
                             const VectorMP<double> &y,
                             VectorMP<double> &z);

template
void subtract<double,float>(const float a, const VectorMP<double> &x,
                            const VectorMP<double> &y,
                            VectorMP<double> &z);
} // namespace mfem
