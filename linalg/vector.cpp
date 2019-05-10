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

// Implementation of data type vector

#include "vector.hpp"
#include "dtensor.hpp"
#include "../general/forall.hpp"

#if defined(MFEM_USE_SUNDIALS) && defined(MFEM_USE_MPI)
#include <nvector/nvector_parallel.h>
#include <nvector/nvector_parhyp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace mfem
{

void Vector::Push() const
{
   mfem::Push(data, size*sizeof(double));
}

void Vector::Pull() const
{
   mfem::Pull(data, size*sizeof(double));
}

Vector::Vector(const Vector &v)
{
   int s = v.Size();

   if (s > 0)
   {
      MFEM_ASSERT(v.data, "invalid source vector");
      allocsize = size = s;
      data = mfem::New<double>(s);
      mfem::Memcpy(data, v.data, sizeof(double)*s);
   }
   else
   {
      allocsize = size = 0;
      data = NULL;
   }
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

   int p = 0;
   for (i = 0; i < np; i++)
      for (j = 0; j < dim[i]; j++)
      {
         *in[i] >> data[p++];
      }
}

void Vector::Load(std::istream &in, int Size)
{
   SetSize(Size);

   for (int i = 0; i < size; i++)
   {
      in >> data[i];
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
   return Dot(size, data, v);
}

double Vector::operator*(const Vector &v) const
{
#ifdef MFEM_DEBUG
   if (v.size != size)
   {
      mfem_error("Vector::operator*(const Vector &) const");
   }
#endif

   return operator*(v.data);
}

Vector &Vector::operator=(const double *v)
{
   if (data != v)
   {
      MFEM_ASSERT(data + size <= v || v + size <= data, "Vectors overlap!");
      mfem::Memcpy(data, v, sizeof(double)*size);
   }
   return *this;
}

Vector &Vector::operator=(const Vector &v)
{
   SetSize(v.Size());
   return operator=(v.data);
}

Vector &Vector::operator=(double value)
{
   DeviceVector y(data, size);
   MFEM_FORALL(i, size, y[i] = value;);
   return *this;
}

Vector &Vector::operator*=(double c)
{
   DeviceVector y(data, size);
   MFEM_FORALL(i, size, y[i] *= c;);
   return *this;
}

Vector &Vector::operator/=(double c)
{
   const double m = 1.0/c;
   DeviceVector y(data, size);
   MFEM_FORALL(i, size, y[i] *= m;);
   return *this;
}

Vector &Vector::operator-=(double c)
{
   DeviceVector y(data, size);
   MFEM_FORALL(i, size, y[i] -= c;);
   return *this;
}

Vector &Vector::operator-=(const Vector &v)
{
#ifdef MFEM_DEBUG
   if (size != v.size)
   {
      mfem_error("Vector::operator-=(const Vector &)");
   }
#endif
   const int N = size;
   DeviceVector y(data, N);
   const DeviceVector x(v, N);
   MFEM_FORALL(i, N, y[i] -= x[i];);
   return *this;
}

Vector &Vector::operator+=(const Vector &v)
{
#ifdef MFEM_DEBUG
   if (size != v.size)
   {
      mfem_error("Vector::operator+=(const Vector &)");
   }
#endif
   const int N = size;
   DeviceVector y(data, N);
   const DeviceVector x(v, N);
   MFEM_FORALL(i, N, y[i] += x[i];);
   return *this;
}

Vector &Vector::Add(const double a, const Vector &Va)
{
#ifdef MFEM_DEBUG
   if (size != Va.size)
   {
      mfem_error("Vector::Add(const double, const Vector &)");
   }
#endif
   if (a != 0.0)
   {
      const int N = size;
      DeviceVector y(data, N);
      const DeviceVector x(Va, N);
      MFEM_FORALL(i, N, y[i] += a * x[i];);
   }
   return *this;
}

Vector &Vector::Set(const double a, const Vector &Va)
{
#ifdef MFEM_DEBUG
   if (size != Va.size)
   {
      mfem_error("Vector::Set(const double, const Vector &)");
   }
#endif
   const int N = size;
   DeviceVector y(data, N);
   const DeviceVector x(Va, N);
   MFEM_FORALL(i, N, y[i] = a * x[i];);
   return *this;
}

void Vector::SetVector(const Vector &v, int offset)
{
   int vs = v.Size();
   double *vp = v.data, *p = data + offset;

#ifdef MFEM_DEBUG
   if (offset+vs > size)
   {
      mfem_error("Vector::SetVector(const Vector &, int)");
   }
#endif

   for (int i = 0; i < vs; i++)
   {
      p[i] = vp[i];
   }
}

void Vector::Neg()
{
   DeviceVector y(data, size);
   MFEM_FORALL(i, size, y[i] = -y[i];);
}

void add(const Vector &v1, const Vector &v2, Vector &v)
{
#ifdef MFEM_DEBUG
   if (v.size != v1.size || v.size != v2.size)
   {
      mfem_error("add(Vector &v1, Vector &v2, Vector &v)");
   }
#endif

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const int N = v.size;
   DeviceVector y(v, N);
   const DeviceVector x1(v1, N);
   const DeviceVector x2(v2, N);
   MFEM_FORALL(i, N, y[i] = x1[i] + x2[i];);
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
#ifdef MFEM_DEBUG
   if (v.size != v1.size || v.size != v2.size)
   {
      mfem_error ("add(Vector &v1, double alpha, Vector &v2, Vector &v)");
   }
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

      const int s = v.size;
#if !defined(MFEM_USE_LEGACY_OPENMP)
      const int N = s;
      DeviceVector d_z(vp, N);
      const DeviceVector d_x(v1p, N);
      const DeviceVector d_y(v2p, N);
      MFEM_FORALL(i, N, d_z[i] = d_x[i] + alpha * d_y[i];);
#else
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
      const int      s = x.size;
#if !defined(MFEM_USE_LEGACY_OPENMP)
      DeviceVector z(zp, s);
      const DeviceVector x(xp, s);
      const DeviceVector y(yp, s);
      MFEM_FORALL(i, s, z[i] = a * (x[i] + y[i]););
#else
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
      const int      s = x.size;

#if !defined(MFEM_USE_LEGACY_OPENMP)
      DeviceVector z(zp, s);
      const DeviceVector x(xp, s);
      const DeviceVector y(yp, s);
      MFEM_FORALL(i, s, z[i] = a * x[i] + b * y[i];);
#else
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
#ifdef MFEM_DEBUG
   if (x.size != y.size || x.size != z.size)
   {
      mfem_error ("subtract(const Vector &, const Vector &, Vector &)");
   }
#endif
   const double *xp = x.data;
   const double *yp = y.data;
   double       *zp = z.data;
   const int     s = x.size;

#if !defined(MFEM_USE_LEGACY_OPENMP)
   DeviceVector zd(zp, s);
   const DeviceVector xd(xp, s);
   const DeviceVector yd(yp, s);
   MFEM_FORALL(i, s, zd[i] = xd[i] - yd[i];);
#else
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      zp[i] = xp[i] - yp[i];
   }
#endif
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
      const int      s = x.size;

#if !defined(MFEM_USE_LEGACY_OPENMP)
      DeviceVector zd(zp, s);
      const DeviceVector xd(xp, s);
      const DeviceVector yd(yp, s);
      MFEM_FORALL(i, s, zd[i] = a * (xd[i] - yd[i]););
#else
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
   const int N = size;
   DeviceVector v(data, N);
   const DeviceVector l(lo, N);
   const DeviceVector h(hi, N);
   MFEM_FORALL(i, N,
   {
      if (v[i] < l[i])
      {
         v[i] = l[i];
      }
      else if (v[i] > h[i])
      {
         v[i] = h[i];
      }
   });
}

static void GetSubvector(const int N,
                         double *y, const double *x, const int* dofs)
{
   DeviceVector d_y(y, N);
   const DeviceVector d_x(x, N);
   const DeviceArray d_dofs(dofs, N);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[i] = dof_i >= 0 ? d_x[dof_i] : -d_x[-dof_i-1];
   });
}

void Vector::GetSubVector(const Array<int> &dofs, Vector &elemvect) const
{
   const int n = dofs.Size();
   elemvect.SetSize(n);
   mfem::GetSubvector(n, elemvect, data, dofs);
}

void Vector::GetSubVector(const Array<int> &dofs, double *elem_data) const
{
   mfem::GetSubvector(dofs.Size(), elem_data, data,dofs);
}

static void SetSubvector(const int N, double* y, const double d,
                         const int* dofs)
{
   DeviceVector d_y(y,N);
   const DeviceArray d_dofs(dofs,N);
   MFEM_FORALL(i, N,
   {
      const int j = d_dofs[i];
      if (j >= 0)
      {
         d_y[j] = d;
      }
      else
      {
         d_y[-1-j] = -d;
      }
   });
}

static void SetSubvector(const int N, double *y, const double *x,
                         const int* dofs)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   const DeviceArray d_dofs(dofs,N);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      if (dof_i >= 0)
      {
         d_y[dof_i] = d_x[i];
      }
      else
      {
         d_y[-1-dof_i] = -d_x[i];
      }
   });
}

void Vector::SetSubVector(const Array<int> &dofs, const double value)
{
   mfem::SetSubvector(dofs.Size(), data, value, dofs);
}

void Vector::SetSubVector(const Array<int> &dofs, const Vector &elemvect)
{
   mfem::SetSubvector(dofs.Size(), data, elemvect, dofs);
}

void Vector::SetSubVector(const Array<int> &dofs, double *elem_data)
{
   mfem::SetSubvector(dofs.Size(), data, elem_data, dofs);
}

static void AddElement(const int N, const int *dofs, const double *x, double *y)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   const DeviceArray d_dofs(dofs,N);
   MFEM_FORALL(i, N,
   {
      const int j = d_dofs[i];
      if (j >= 0)
         d_y[j] += d_x[i];
      else
      {
         d_y[-1-j] -= d_x[i];
      }
   });
}

void Vector::AddElementVector(const Array<int> &dofs, const Vector &elemvect)
{
   MFEM_ASSERT(dofs.Size() == elemvect.Size(), "Size mismatch: "
               "length of dofs is " << dofs.Size() <<
               ", length of elemvect is " << elemvect.Size());
   mfem::AddElement(dofs.Size(), dofs, elemvect.GetData(), data);
}

void Vector::AddElementVector(const Array<int> &dofs, double *elem_data)
{
   mfem::AddElement(dofs.Size(), dofs, elem_data, data);
}

void Vector::AddElementVector(const Array<int> &dofs, const double a,
                              const Vector &elemvect)
{
   const int N = dofs.Size();
   const double alpha = a;
   DeviceVector d_y(data, N);
   const DeviceVector d_x(elemvect, N);
   const DeviceArray d_dofs(dofs, N);
   MFEM_FORALL(i, N,
   {
      const int j = d_dofs[i];
      if (j >= 0)
         d_y[j] += alpha * d_x[i];
      else
      {
         d_y[-1-j] -= alpha * d_x[i];
      }
   });
}

void Vector::SetSubVectorComplement(const Array<int> &dofs, const double val)
{
   Vector dofs_vals;
   GetSubVector(dofs, dofs_vals);
   operator=(val);
   SetSubVector(dofs, dofs_vals);
}

void Vector::Print(std::ostream &out, int width) const
{
   if (!size) { return; }
   Pull();
   for (int i = 0; 1; )
   {
      out << data[i];
      i++;
      if (i == size)
      {
         break;
      }
      if ( i % width == 0 )
      {
         out << '\n';
      }
      else
      {
         out << ' ';
      }
   }
   out << '\n';
}

void Vector::Print_HYPRE(std::ostream &out) const
{
   int i;
   std::ios::fmtflags old_fmt = out.flags();
   out.setf(std::ios::scientific);
   std::streamsize old_prec = out.precision(14);

   out << size << '\n';  // number of rows

   for (i = 0; i < size; i++)
   {
      out << data[i] << '\n';
   }

   out.precision(old_prec);
   out.flags(old_fmt);
}

void Vector::Randomize(int seed)
{
   // static unsigned int seed = time(0);
   const double max = (double)(RAND_MAX) + 1.;

   if (seed == 0)
   {
      seed = (int)time(0);
   }

   // srand(seed++);
   srand((unsigned)seed);

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
            const double sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const double sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if data[i] != 0
   }
   return scale * std::sqrt(sum);
}

double Vector::Normlinf() const
{
   double max = 0.0;
   for (int i = 0; i < size; i++)
   {
      max = std::max(std::abs(data[i]), max);
   }
   return max;
}

double Vector::Norml1() const
{
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
   double max = data[0];

   for (int i = 1; i < size; i++)
      if (data[i] > max)
      {
         max = data[i];
      }

   return max;
}

double Vector::Min() const
{
   double min = data[0];

   for (int i = 1; i < size; i++)
      if (data[i] < min)
      {
         min = data[i];
      }

   return min;
}

double Vector::Sum() const
{
   double sum = 0.0;

   for (int i = 0; i < size; i++)
   {
      sum += data[i];
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

static double cuVectorMin(const int N, const double *X)
{
   const DeviceVector x(X, N);
   const int tpb = MFEM_CUDA_BLOCKS;
   const int blockSize = MFEM_CUDA_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int min_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const int bytes = min_sz*sizeof(double);
   static double *h_min = NULL;
   if (!h_min) { h_min = (double*)calloc(min_sz,sizeof(double)); }
   static void *gdsr = NULL;
   if (!gdsr) { MFEM_CUDA_CHECK(cudaMalloc(&gdsr, bytes)); }
   cuKernelMin<<<gridSize,blockSize>>>(N, (double*)gdsr, x);
   MFEM_CUDA_CHECK(cudaGetLastError());
   MFEM_CUDA_CHECK(cudaMemcpy(h_min, gdsr, bytes, cudaMemcpyDeviceToHost));
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
   const DeviceVector x(X, N);
   const DeviceVector y(Y, N);
   static int dot_block_sz = 0;
   const int tpb = MFEM_CUDA_BLOCKS;
   const int blockSize = MFEM_CUDA_BLOCKS;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int dot_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const int bytes = dot_sz*sizeof(double);
   static double *h_dot = NULL;
   if (!h_dot or dot_block_sz!=dot_sz)
   {
      if (h_dot) { free(h_dot); }
      h_dot = (double*)calloc(dot_sz,sizeof(double));
   }
   static void *gdsr = NULL;
   if (!gdsr or dot_block_sz!=dot_sz)
   {
      if (gdsr) { MFEM_CUDA_CHECK(cudaFree(gdsr)); }
      MFEM_CUDA_CHECK(cudaMalloc(&gdsr,bytes));
   }
   if (dot_block_sz!=dot_sz)
   {
      dot_block_sz = dot_sz;
   }
   cuKernelDot<<<gridSize,blockSize>>>(N, (double*)gdsr, x, y);
   MFEM_CUDA_CHECK(cudaGetLastError());
   MFEM_CUDA_CHECK(cudaMemcpy(h_dot, gdsr, bytes, cudaMemcpyDeviceToHost));
   double dot = 0.0;
   for (int i = 0; i < dot_sz; i++) { dot += h_dot[i]; }
   return dot;
}
#endif // MFEM_USE_CUDA

double Min(const int N, const double *x)
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
#ifdef MFEM_USE_CUDA
      return cuVectorMin(N, x);
#else
      mfem_error("Using Min on device w/o support");
#endif // MFEM_USE_CUDA
   }
   double min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < N; i++) { min = fmin(min, x[i]); }
   return min;
}

double Dot(const int N, const double *x, const double *y)
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
#ifdef MFEM_USE_CUDA
      return cuVectorDot(N, x, y);
#else
      mfem_error("Using Dot on device w/o support");
#endif // MFEM_USE_CUDA
   }
   double dot = 0.0;
#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for reduction(+:dot)
#endif
   for (int i = 0; i < N; i++) { dot += x[i] * y[i]; }
   return dot;
}

#ifdef MFEM_USE_SUNDIALS

#ifndef SUNTRUE
#define SUNTRUE TRUE
#endif
#ifndef SUNFALSE
#define SUNFALSE FALSE
#endif

Vector::Vector(N_Vector nv)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);
   switch (nvid)
   {
      case SUNDIALS_NVEC_SERIAL:
         SetDataAndSize(NV_DATA_S(nv), NV_LENGTH_S(nv));
         break;
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
         SetDataAndSize(NV_DATA_P(nv), NV_LOCLENGTH_P(nv));
         break;
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(nv)->local_vector;
         SetDataAndSize(hpv_local->data, hpv_local->size);
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << nvid << " is not supported");
   }
}

void Vector::ToNVector(N_Vector &nv)
{
   MFEM_ASSERT(nv, "N_Vector handle is NULL");
   N_Vector_ID nvid = N_VGetVectorID(nv);
   switch (nvid)
   {
      case SUNDIALS_NVEC_SERIAL:
         MFEM_ASSERT(NV_OWN_DATA_S(nv) == SUNFALSE, "invalid serial N_Vector");
         NV_DATA_S(nv) = data;
         NV_LENGTH_S(nv) = size;
         break;
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
         MFEM_ASSERT(NV_OWN_DATA_P(nv) == SUNFALSE, "invalid parallel N_Vector");
         NV_DATA_P(nv) = data;
         NV_LOCLENGTH_P(nv) = size;
         break;
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(nv)->local_vector;
         MFEM_ASSERT(hpv_local->owns_data == false, "invalid hypre N_Vector");
         hpv_local->data = data;
         hpv_local->size = size;
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << nvid << " is not supported");
   }
}

#endif // MFEM_USE_SUNDIALS

}
