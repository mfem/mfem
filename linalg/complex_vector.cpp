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

#include "../general/forall.hpp"
#include "../general/reducers.hpp"
#include "complex_vector.hpp"

using namespace std;

namespace mfem
{

ComplexVector::ComplexVector(const ComplexVector &v)
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

ComplexVector::ComplexVector(const Vector &v)
{
   const int s = v.Size();
   size = s;
   if (s > 0)
   {
      MFEM_ASSERT(!v.data.Empty(), "invalid source vector");
      data.New(s, v.data.GetMemoryType());
      MFEM_FORALL(i, size, data[i] = v.data[i]; );
   }
   UseDevice(v.UseDevice());
}

ComplexVector::ComplexVector(ComplexVector &&v)
{
   *this = std::move(v);
}

complex_t &ComplexVector::Elem(int i)
{
   return operator()(i);
}

const complex_t &ComplexVector::Elem(int i) const
{
   return operator()(i);
}

complex_t ComplexVector::operator*(const complex_t *v) const
{
   HostRead();
   complex_t dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < size; i++)
   {
      dot += data[i] * v[i];
   }
   return dot;
}

complex_t ComplexVector::operator*(const real_t *v) const
{
   HostRead();
   complex_t dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < size; i++)
   {
      dot += data[i] * v[i];
   }
   return dot;
}

complex_t ComplexVector::operator*(const ComplexVector &v) const
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   if (size == 0) { return 0.0; }

   const bool use_dev = UseDevice() || v.UseDevice();
   const auto m_data = Read(use_dev), v_data = v.Read(use_dev);

   // The standard way of computing the dot product is non-deterministic
   complex_t prod = 0.0;
   for (int i = 0; i < size; i++)
   {
      prod += m_data[i] * v_data[i];
   }
   return prod;
}

complex_t ComplexVector::operator*(const Vector &v) const
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   if (size == 0) { return 0.0; }

   const bool use_dev = UseDevice() || v.UseDevice();
   const auto m_data = Read(use_dev);
   const auto v_data = v.Read(use_dev);

   // The standard way of computing the dot product is non-deterministic
   complex_t prod = 0.0;
   for (int i = 0; i < size; i++)
   {
      prod += m_data[i] * v_data[i];
   }
   return prod;
}

ComplexVector &ComplexVector::operator=(const complex_t *v)
{
   HostRead();
   MFEM_FORALL(i, size, data[i] = v[i]; );
   return *this;
}

ComplexVector &ComplexVector::operator=(const real_t *v)
{
   HostRead();
   MFEM_FORALL(i, size, data[i] = v[i]; );
   return *this;
}

ComplexVector &ComplexVector::operator=(const ComplexVector &v)
{
#if 0
   SetSize(v.Size(), v.data.GetMemoryType());
   data.CopyFrom(v.data, v.Size());
   UseDevice(v.UseDevice());
#else
   SetSize(v.Size());
   const bool vuse = v.UseDevice();
   const bool use_dev = UseDevice() || vuse;
   v.UseDevice(use_dev);
   // keep 'data' where it is, unless 'use_dev' is true
   if (use_dev) { Write(); }
   data.CopyFrom(v.data, v.Size());
   v.UseDevice(vuse);
#endif
   return *this;
}

ComplexVector &ComplexVector::operator=(const Vector &v)
{
   SetSize(v.Size());
   const bool vuse = v.UseDevice();
   const bool use_dev = UseDevice() || vuse;
   v.UseDevice(use_dev);
   // keep 'data' where it is, unless 'use_dev' is true
   if (use_dev) { Write(); }
   MFEM_FORALL(i, size, data[i] = v[i]; );
   v.UseDevice(vuse);
   return *this;
}

ComplexVector &ComplexVector::operator=(ComplexVector &&v)
{
   v.Swap(*this);
   if (this != &v) { v.Destroy(); }
   return *this;
}

ComplexVector &ComplexVector::operator=(complex_t value)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] = value; });
   return *this;
}

ComplexVector &ComplexVector::operator=(real_t value)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] = value; });
   return *this;
}

ComplexVector &ComplexVector::operator*=(complex_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= c; });
   return *this;
}

ComplexVector &ComplexVector::operator*=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= c; });
   return *this;
}

ComplexVector &ComplexVector::operator*=(const ComplexVector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator*=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator/=(complex_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   const complex_t m = conj(c) / norm(c);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= m; });
   return *this;
}

ComplexVector &ComplexVector::operator/=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   const real_t m = 1.0/c;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] *= m; });
   return *this;
}

ComplexVector &ComplexVector::operator/=(const ComplexVector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] /= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator/=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] /= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator-=(complex_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] -= c; });
   return *this;
}

ComplexVector &ComplexVector::operator-=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] -= c; });
   return *this;
}

ComplexVector &ComplexVector::operator-=(const ComplexVector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] -= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator-=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] -= x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator+=(complex_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] += c; });
   return *this;
}

ComplexVector &ComplexVector::operator+=(real_t c)
{
   const bool use_dev = UseDevice();
   const int N = size;
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] += c; });
   return *this;
}

ComplexVector &ComplexVector::operator+=(const ComplexVector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] += x[i]; });
   return *this;
}

ComplexVector &ComplexVector::operator+=(const Vector &v)
{
   MFEM_ASSERT(size == v.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || v.UseDevice();
   const int N = size;
   const auto x = v.Read(use_dev);
   auto y = ReadWrite(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] += x[i]; });
   return *this;
}

ComplexVector &ComplexVector::Set(const Vector &Vr, const Vector &Vi)
{
   MFEM_ASSERT(size == Vr.size && size == Vi.size, "incompatible Vectors!");

   const bool use_dev = UseDevice() || Vr.UseDevice() || Vi.UseDevice();
   const int N = size;
   const auto x = Vr.Read(use_dev);
   const auto y = Vi.Read(use_dev);
   auto z = Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { z[i] = complex_t(x[i], y[i]); });
   return *this;
}

const Vector &ComplexVector::real() const
{
   re_part.SetSize(size);
   const bool use_dev = UseDevice();
   const int N = size;
   const auto z = Read(use_dev);
   auto x = re_part.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { x[i] = z[i].real(); });
   return re_part;
}

const Vector &ComplexVector::imag() const
{
   im_part.SetSize(size);
   const bool use_dev = UseDevice();
   const int N = size;
   const auto z = Read(use_dev);
   auto y = im_part.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   { y[i] = z[i].imag(); });
   return im_part;
}

}
