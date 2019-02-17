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

#ifndef MFEM_OKINA_HPP
#define MFEM_OKINA_HPP

// *****************************************************************************
#include "../config/config.hpp"
#include "../general/error.hpp"

// *****************************************************************************
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>

// *****************************************************************************
uint32_t LOG2(uint32_t);
#define ISQRT(N) static_cast<unsigned>(sqrt(static_cast<float>(N)))
#define ICBRT(N) static_cast<unsigned>(cbrt(static_cast<float>(N)))
#define IROOT(D,N) ((D==1)?N:(D==2)?ISQRT(N):(D==3)?ICBRT(N):0)
#define MAX(a,b) ((a)>(b)?(a):(b))

// *****************************************************************************
#include "./cuda.hpp"
#include "./occa.hpp"

// *****************************************************************************
#include "mm.hpp"
#include "config.hpp"

// *****************************************************************************
// * GPU & HOST FOR_LOOP bodies wrapper
// *****************************************************************************
template <typename DBODY, typename HBODY>
void wrap(const size_t N, const size_t Nspt, DBODY &&d_body, HBODY &&h_body)
{
   const bool gpu = mfem::config::usingGpu();
   if (gpu)
   {
      const size_t smpb = mfem::config::SharedMemPerBlock();
      return cuWrap(N, Nspt, smpb, d_body);
   }
   else
   {
      const int Ns = Nspt;
      double cpu_mem_s[Ns];
      for (size_t k=0; k<N; k+=1) { h_body(k, cpu_mem_s); }
   }
}

// *****************************************************************************
// * MFEM_FORALL splitter
// *****************************************************************************
#define MFEM_FORALL(i,N,...) MFEM_FORALL_SHARED(i,N,0,__VA_ARGS__)
#define MFEM_FORALL_SEQ(...) MFEM_FORALL_SHARED(i,1,0,__VA_ARGS__)
#define MFEM_FORALL_SHARED(i,N,Nspt,...)                                \
   wrap(N, (const int) Nspt,                                                        \
        [=] __device__ (const size_t i,                                 \
                        double *__shared){__VA_ARGS__},                 \
        [&]            (const size_t i,                                 \
                        double *__shared){__VA_ARGS__})

// *****************************************************************************
#define GET_GPU const bool gpu = config::usingGpu();
#define GET_PTR(v) double *d_##v = (double*) mfem::mm::ptr(v)
#define GET_PTR_T(v,T) T *d_##v = (T*) mfem::mm::ptr(v)
#define GET_CONST_PTR(v) const double *d_##v = (const double*) mfem::mm::ptr(v)
#define GET_CONST_PTR_T(v,T) const T *d_##v = (const T*) mfem::mm::ptr(v)

// *****************************************************************************
#ifndef __NVCC__
#define MFEM_DEVICE
#define MFEM_HOST_DEVICE
#else
#define MFEM_DEVICE __device__
#define MFEM_HOST_DEVICE __host__ __device__
#endif

// *****************************************************************************
#define BUILTIN_TRAP __builtin_trap()
#define MFEM_SIGSEGV(ptr) for(int k=0;k<1024*1024;k+=1)((int*)ptr)[k]=0;
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_CPU_CANNOT_PASS {assert(FILE_LINE && false);}
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::usingGpu());}

// Offsets *********************************************************************
#define ijN(i,j,N) (i)+N*(j)
#define ijkN(i,j,k,N) (i)+N*((j)+N*(k))
#define ijklN(i,j,k,l,N) (i)+N*((j)+N*((k)+N*(l)))
#define ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define ijkNM(i,j,k,N,M) (i)+N*((j)+M*(k))
#define ijklNM(i,j,k,l,N,M) (i)+N*((j)+N*((k)+(M)*(l)))
// External offsets
#define jkliNM(i,j,k,l,N,M) (j)+N*((k)+N*((l)+M*(i)))
#define jklmiNM(i,j,k,l,m,N,M) (j)+N*((k)+N*((l)+N*((m)+M*(i))))
#define xyeijDQE(i,j,x,y,e,D,Q,E) (x)+Q*((y)+Q*((e)+E*((i)+D*j)))
#define xyzeijDQE(i,j,x,y,z,e,D,Q,E) (x)+Q*((y)+Q*((z)+Q*((e)+E*((i)+D*j))))

// *****************************************************************************
namespace mfem
{
namespace kernels
{
// *****************************************************************************
class Vector1
{
private:
   size_t N;
   double *data;
public:
   MFEM_HOST_DEVICE Vector1(const size_t n, double *d): N(n), data(d) {}
   MFEM_HOST_DEVICE double& operator()(const size_t i) { return data[i]; }
   MFEM_HOST_DEVICE double& operator()(const size_t i) const { return data[i]; }
};

// *****************************************************************************
class Vector2
{
private:
   size_t N,M;
   double *data;
public:
   MFEM_HOST_DEVICE Vector2(const size_t n, const size_t m, double *d)
      :N(n), M(m), data(d) {}
   MFEM_HOST_DEVICE double& operator()(const size_t i, const size_t j)
   {
      return data[i+N*j];
   }
   MFEM_HOST_DEVICE double& operator()(const size_t i, const size_t j) const
   {
      return data[i+N*j];
   }
};

// *****************************************************************************
class Vector3
{
private:
   size_t N,M,P;
   double *data;
public:
   __host__ __device__ Vector3(const size_t n,
                               const size_t m,
                               const size_t p,
                               double *d)
      :N(n), M(m), P(p), data(d) {}
   __host__ __device__ double& operator()(const size_t i,
                                          const size_t j,
                                          const size_t k)
   {
      const size_t ijkNM = (i)+N*((j)+M*(k));
      return data[ijkNM];
   }
   __host__ __device__ double& operator()(const size_t i,
                                          const size_t j,
                                          const size_t k) const
   {
      const size_t ijkNM = (i)+N*((j)+M*(k));
      return data[ijkNM];
   }
};
// *****************************************************************************
class Vector4
{
private:
   size_t N,M,P,Q;
   double *data;
public:
   __host__ __device__ Vector4(const size_t n,
                               const size_t m,
                               const size_t p,
                               const size_t q,
                               double *d)
      :N(n), M(m), P(p), Q(q), data(d) {}
   __host__ __device__ double& operator()(const size_t i,
                                          const size_t j,
                                          const size_t k,
                                          const size_t l)
   {
      const size_t ijklNM = (i)+N*((j)+M*((k)+(P)*(l)));
      return data[ijklNM];
   }
   __host__ __device__ double& operator()(const size_t i,
                                          const size_t j,
                                          const size_t k,
                                          const size_t l) const
   {
      const size_t ijklNM = (i)+N*((j)+M*((k)+(P)*(l)));
      return data[ijklNM];
   }
};

// *****************************************************************************
template <typename T> class XS
{
private:
   size_t N,M,P,Q;
   T *data;
public:
   // 1D
   XS(T *d): data(d) {}
   XS(const size_t n, const T *d) : N(n), data((T*)mfem::mm::ptr(d)) {}
   __host__ __device__ T operator[](const size_t i) { return data[i];}
   __host__ __device__ T operator[](const size_t i) const { return data[i];}
   __host__ __device__ T& operator()(const size_t i) { return data[i];}
   __host__ __device__ T& operator()(const size_t i) const { return data[i];}

   // 2D
   XS(const size_t n, const size_t m, const T *d)
      :N(n), M(m), data((T*)mfem::mm::ptr(d)) {}
   __host__ __device__ T& operator()(const size_t i, const size_t j)
   {
      return data[i+N*j];
   }
   __host__ __device__ T& operator()(const size_t i, const size_t j) const
   {
      return data[i+N*j];
   }

   // 3D
   XS(const size_t n, const size_t m, const size_t p, const T *d)
      :N(n), M(m), P(p), data((T*)mfem::mm::ptr(d)) {}
   __host__ __device__ T& operator()(const size_t i, const size_t j,
                                     const size_t k)
   {
      const size_t ijkNM = (i)+N*((j)+M*(k));
      return data[ijkNM];
   }
   __host__ __device__ T& operator()(const size_t i, const size_t j,
                                     const size_t k) const
   {
      const size_t ijkNM = (i)+N*((j)+M*(k));
      return data[ijkNM];
   }

   // 4D
   XS(const size_t n, const size_t m,
      const size_t p, const size_t q, const T *d)
      :N(n), M(m), P(p), Q(q), data((T*)mfem::mm::ptr(d)) {}
   __host__ __device__ T& operator()(const size_t i, const size_t j,
                                     const size_t k, const size_t l)
   {
      const size_t ijklNM = (i)+N*((j)+M*((k)+(P)*(l)));
      return data[ijklNM];
   }
   __host__ __device__ T& operator()(const size_t i, const size_t j,
                                     const size_t k, const size_t l) const
   {
      const size_t ijklNM = (i)+N*((j)+M*((k)+(P)*(l)));
      return data[ijklNM];
   }
};
typedef XS<double> Array;
typedef XS<double> Vector;
} // namespace kernels
} // namespace mfem

// *****************************************************************************
const char *strrnchr(const char*, const unsigned char, const int);
void dbg_F_L_F_N_A(const char*, const int, const char*, const int, ...);

// *****************************************************************************
#define _XA_(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,X,...) X
#define _NA_(...) _XA_(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define __FILENAME__ ({const char *f=strrnchr(__FILE__,'/',2);f?f+1:__FILE__;})
#define _F_L_F_ __FILENAME__,__LINE__,__FUNCTION__

// *****************************************************************************
#ifndef MFEM_DEBUG
#define dbg(...)
#else
#define dbg_stack(...) dbg_F_L_F_N_A(_F_L_F_,0)
#define dbg(...) dbg_F_L_F_N_A(_F_L_F_, _NA_(__VA_ARGS__),__VA_ARGS__)
#endif

#endif // MFEM_OKINA_HPP
