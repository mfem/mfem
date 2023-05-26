// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#if defined(MFEM_USE_SUNDIALS)
#include "sundials.hpp"
#if defined(MFEM_USE_MPI)
#include <nvector/nvector_parallel.h>
#endif
#endif

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace mfem
{

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
   double dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < size; i++)
   {
      dot += data[i] * v[i];
   }
   return dot;
}

Vector &Vector::operator=(const double *v)
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
   data = std::move(v.data);
   // Self-assignment-safe way to move v.size to size:
   const auto size_tmp = v.size;
   v.size = 0;
   size = size_tmp;
   return *this;
}

Vector &Vector::operator=(double value)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { y[i] = value; });
   return *this;
}

Vector &Vector::operator*=(double c)
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

Vector &Vector::operator/=(double c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   const double m = 1.0/c;
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

Vector &Vector::operator-=(double c)
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

Vector &Vector::operator+=(double c)
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

Vector &Vector::Add(const double a, const Vector &Va)
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

Vector &Vector::Set(const double a, const Vector &Va)
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
   const double *vp = v.data;
   double *p = data + offset;
   for (int i = 0; i < vs; i++)
   {
      p[i] = vp[i];
   }
}

void Vector::AddSubVector(const Vector &v, int offset)
{
   MFEM_ASSERT(v.Size() + offset <= size, "invalid sub-vector");

   const int vs = v.Size();
   const double *vp = v.data;
   double *p = data + offset;
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

void add(const Vector &v1, double alpha, const Vector &v2, Vector &v)
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
      const double *v1p = v1.data, *v2p = v2.data;
      double *vp = v.data;
      const int s = v.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         vp[i] = v1p[i] + alpha*v2p[i];
      }
#endif
   }
}

void add(const double a, const Vector &x, const Vector &y, Vector &z)
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
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
      const int      s = x.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] + yp[i]);
      }
#endif
   }
}

void add(const double a, const Vector &x,
         const double b, const Vector &y, Vector &z)
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
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
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
   const double *xp = x.data;
   const double *yp = y.data;
   double       *zp = z.data;
   const int     s = x.size;
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      zp[i] = xp[i] - yp[i];
   }
#endif
}

void subtract(const double a, const Vector &x, const Vector &y, Vector &z)
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
      const double *xp = x.data;
      const double *yp = y.data;
      double       *zp = z.data;
      const int      s = x.size;
      #pragma omp parallel for
      for (int i = 0; i < s; i++)
      {
         zp[i] = a * (xp[i] - yp[i]);
      }
#endif
   }
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

void Vector::GetSubVector(const Array<int> &dofs, double *elem_data) const
{
   data.Read(MemoryClass::HOST, size);
   const int n = dofs.Size();
   for (int i = 0; i < n; i++)
   {
      const int j = dofs[i];
      elem_data[i] = (j >= 0) ? data[j] : -data[-1-j];
   }
}

void Vector::SetSubVector(const Array<int> &dofs, const double value)
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

void Vector::SetSubVector(const Array<int> &dofs, double *elem_data)
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

void Vector::AddElementVector(const Array<int> &dofs, double *elem_data)
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

void Vector::AddElementVector(const Array<int> &dofs, const double a,
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

void Vector::SetSubVectorComplement(const Array<int> &dofs, const double val)
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

void Vector::PrintHash(std::ostream &os) const
{
   os << "size: " << size << '\n';
   HashFunction hf;
   hf.AppendDoubles(HostRead(), size);
   os << "hash: " << hf.GetHash() << '\n';
}

void Vector::Randomize(int seed)
{
   const double max = (double)(RAND_MAX) + 1.;

   if (seed == 0)
   {
      seed = (int)time(0);
   }

   srand((unsigned)seed);

   HostWrite();
   for (int i = 0; i < size; i++)
   {
      data[i] = std::abs(rand()/max);
   }
}

double Vector::Norml2() const
{
   // Scale entries of Vector on the fly, using algorithms from
   // std::hypot() and LAPACK's drm2. This scaling ensures that the
   // argument of each call to std::pow is <= 1 to avoid overflow.
   if (0 == size)
   {
      return 0.0;
   } // end if 0 == size

   data.Read(MemoryClass::HOST, size);
   if (1 == size)
   {
      return std::abs(data[0]);
   } // end if 1 == size
   return kernels::Norml2(size, (const double*) data);
}

double Vector::Normlinf() const
{
   HostRead();
   double max = 0.0;
   for (int i = 0; i < size; i++)
   {
      max = std::max(std::abs(data[i]), max);
   }
   return max;
}

double Vector::Norml1() const
{
   HostRead();
   double sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      sum += std::abs(data[i]);
   }
   return sum;
}

double Vector::Normlp(double p) const
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
      if (0 == size)
      {
         return 0.0;
      } // end if 0 == size

      if (1 == size)
      {
         return std::abs(data[0]);
      } // end if 1 == size

      double scale = 0.0;
      double sum = 0.0;

      for (int i = 0; i < size; i++)
      {
         if (data[i] != 0.0)
         {
            const double absdata = std::abs(data[i]);
            if (scale <= absdata)
            {
               sum = 1.0 + sum * std::pow(scale / absdata, p);
               scale = absdata;
               continue;
            } // end if scale <= absdata
            sum += std::pow(absdata / scale, p); // else scale > absdata
         } // end if data[i] != 0
      }
      return scale * std::pow(sum, 1.0/p);
   } // end if p < infinity()

   return Normlinf(); // else p >= infinity()
}

double Vector::Max() const
{
   if (size == 0) { return -infinity(); }

   HostRead();
   double max = data[0];

   for (int i = 1; i < size; i++)
   {
      if (data[i] > max)
      {
         max = data[i];
      }
   }

   return max;
}

double Vector::Sum() const
{
   double sum = 0.0;

   const double *h_data = this->HostRead();
   for (int i = 0; i < size; i++)
   {
      sum += h_data[i];
   }

   return sum;
}

#ifdef MFEM_USE_CUDA
static __global__ void cuKernelMin(const int N, double *gdsr, const double *x)
{
   __shared__ double s_min[MFEM_CUDA_BLOCKS];
   const int n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const int bid = blockIdx.x;
   const int tid = threadIdx.x;
   const int bbd = bid*blockDim.x;
   const int rid = bbd+tid;
   s_min[tid] = x[n];
   for (int workers=blockDim.x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const int dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const int rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= blockDim.x) { continue; }
      s_min[tid] = fmin(s_min[tid], s_min[dualTid]);
   }
   if (tid==0) { gdsr[bid] = s_min[0]; }
}

static Array<double> cuda_reduce_buf;

static double cuVectorMin(const int N, const double *X)
{
   const int tpb = MFEM_CUDA_BLOCKS;
   const int blockSize = MFEM_CUDA_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int min_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   cuda_reduce_buf.SetSize(min_sz);
   Memory<double> &buf = cuda_reduce_buf.GetMemory();
   double *d_min = buf.Write(MemoryClass::DEVICE, min_sz);
   cuKernelMin<<<gridSize,blockSize>>>(N, d_min, X);
   MFEM_GPU_CHECK(cudaGetLastError());
   const double *h_min = buf.Read(MemoryClass::HOST, min_sz);
   double min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < min_sz; i++) { min = fmin(min, h_min[i]); }
   return min;
}

static __global__ void cuKernelDot(const int N, double *gdsr,
                                   const double *x, const double *y)
{
   __shared__ double s_dot[MFEM_CUDA_BLOCKS];
   const int n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const int bid = blockIdx.x;
   const int tid = threadIdx.x;
   const int bbd = bid*blockDim.x;
   const int rid = bbd+tid;
   s_dot[tid] = x[n] * y[n];
   for (int workers=blockDim.x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const int dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const int rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= blockDim.x) { continue; }
      s_dot[tid] += s_dot[dualTid];
   }
   if (tid==0) { gdsr[bid] = s_dot[0]; }
}

static double cuVectorDot(const int N, const double *X, const double *Y)
{
   const int tpb = MFEM_CUDA_BLOCKS;
   const int blockSize = MFEM_CUDA_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int dot_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   cuda_reduce_buf.SetSize(dot_sz, Device::GetDeviceMemoryType());
   Memory<double> &buf = cuda_reduce_buf.GetMemory();
   double *d_dot = buf.Write(MemoryClass::DEVICE, dot_sz);
   cuKernelDot<<<gridSize,blockSize>>>(N, d_dot, X, Y);
   MFEM_GPU_CHECK(cudaGetLastError());
   const double *h_dot = buf.Read(MemoryClass::HOST, dot_sz);
   double dot = 0.0;
   for (int i = 0; i < dot_sz; i++) { dot += h_dot[i]; }
   return dot;
}
#endif // MFEM_USE_CUDA

#ifdef MFEM_USE_HIP
static __global__ void hipKernelMin(const int N, double *gdsr, const double *x)
{
   __shared__ double s_min[MFEM_HIP_BLOCKS];
   const int n = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
   if (n>=N) { return; }
   const int bid = hipBlockIdx_x;
   const int tid = hipThreadIdx_x;
   const int bbd = bid*hipBlockDim_x;
   const int rid = bbd+tid;
   s_min[tid] = x[n];
   for (int workers=hipBlockDim_x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const int dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const int rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= hipBlockDim_x) { continue; }
      s_min[tid] = fmin(s_min[tid], s_min[dualTid]);
   }
   if (tid==0) { gdsr[bid] = s_min[0]; }
}

static Array<double> cuda_reduce_buf;

static double hipVectorMin(const int N, const double *X)
{
   const int tpb = MFEM_HIP_BLOCKS;
   const int blockSize = MFEM_HIP_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int min_sz = (N%tpb)==0 ? (N/tpb) : (1+N/tpb);
   cuda_reduce_buf.SetSize(min_sz);
   Memory<double> &buf = cuda_reduce_buf.GetMemory();
   double *d_min = buf.Write(MemoryClass::DEVICE, min_sz);
   hipLaunchKernelGGL(hipKernelMin,gridSize,blockSize,0,0,N,d_min,X);
   MFEM_GPU_CHECK(hipGetLastError());
   const double *h_min = buf.Read(MemoryClass::HOST, min_sz);
   double min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < min_sz; i++) { min = fmin(min, h_min[i]); }
   return min;
}

static __global__ void hipKernelDot(const int N, double *gdsr,
                                    const double *x, const double *y)
{
   __shared__ double s_dot[MFEM_HIP_BLOCKS];
   const int n = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
   if (n>=N) { return; }
   const int bid = hipBlockIdx_x;
   const int tid = hipThreadIdx_x;
   const int bbd = bid*hipBlockDim_x;
   const int rid = bbd+tid;
   s_dot[tid] = x[n] * y[n];
   for (int workers=hipBlockDim_x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const int dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const int rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= hipBlockDim_x) { continue; }
      s_dot[tid] += s_dot[dualTid];
   }
   if (tid==0) { gdsr[bid] = s_dot[0]; }
}

static double hipVectorDot(const int N, const double *X, const double *Y)
{
   const int tpb = MFEM_HIP_BLOCKS;
   const int blockSize = MFEM_HIP_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int dot_sz = (N%tpb)==0 ? (N/tpb) : (1+N/tpb);
   cuda_reduce_buf.SetSize(dot_sz);
   Memory<double> &buf = cuda_reduce_buf.GetMemory();
   double *d_dot = buf.Write(MemoryClass::DEVICE, dot_sz);
   hipLaunchKernelGGL(hipKernelDot,gridSize,blockSize,0,0,N,d_dot,X,Y);
   MFEM_GPU_CHECK(hipGetLastError());
   const double *h_dot = buf.Read(MemoryClass::HOST, dot_sz);
   double dot = 0.0;
   for (int i = 0; i < dot_sz; i++) { dot += h_dot[i]; }
   return dot;
}
#endif // MFEM_USE_HIP

double Vector::operator*(const Vector &v) const
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
      return occa::linalg::dot<double,double,double>(
                OccaMemoryRead(data, size), OccaMemoryRead(v.data, size));
   }
#endif

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return cuVectorDot(size, m_data, v_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return hipVectorDot(size, m_data, v_data);
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
         double my_dot = 0.0;
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
      double prod = 0.0;
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

double Vector::Min() const
{
   if (size == 0) { return infinity(); }

   const bool use_dev = UseDevice();
   auto m_data = Read(use_dev);

   if (!use_dev) { goto vector_min_cpu; }

#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      return occa::linalg::min<double,double>(OccaMemoryRead(data, size));
   }
#endif

#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      return cuVectorMin(size, m_data);
   }
#endif

#ifdef MFEM_USE_HIP
   if (Device::Allows(Backend::HIP_MASK))
   {
      return hipVectorMin(size, m_data);
   }
#endif

#ifdef MFEM_USE_OPENMP
   if (Device::Allows(Backend::OMP_MASK))
   {
      double minimum = m_data[0];
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
         d_min[0] = (d_min[0]<m_data_[i])?d_min[0]:m_data_[i];
      });
      min.HostReadWrite();
      return min[0];
   }

vector_min_cpu:
   double minimum = data[0];
   for (int i = 1; i < size; i++)
   {
      if (m_data[i] < minimum)
      {
         minimum = m_data[i];
      }
   }
   return minimum;
}

}
