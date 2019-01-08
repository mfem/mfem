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

#include "../../general/okina.hpp"
#include "vector.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
#ifdef __NVCC__

// *****************************************************************************
#define CUDA_BLOCKSIZE 256

// *****************************************************************************
__global__ void cuKernelDot(const size_t N, double *gdsr,
                            const double *x, const double *y)
{
   __shared__ double s_dot[CUDA_BLOCKSIZE];
   const size_t n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const size_t bid = blockIdx.x;
   const size_t tid = threadIdx.x;
   const size_t bbd = bid*blockDim.x;
   const size_t rid = bbd+tid;
   s_dot[tid] = x[n] * y[n];
   for (size_t workers=blockDim.x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const size_t dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const size_t rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= blockDim.x) { continue; }
      s_dot[tid] += s_dot[dualTid];
   }
   if (tid==0) { gdsr[bid] = s_dot[0]; }
}

// *****************************************************************************
double cuVectorDot(const size_t N, const double *x, const double *y)
{
   const size_t tpb = CUDA_BLOCKSIZE;
   const size_t blockSize = CUDA_BLOCKSIZE;
   const size_t gridSize = (N+blockSize-1)/blockSize;
   const size_t dot_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const size_t bytes = dot_sz*sizeof(double);
   static double *h_dot = NULL;
   if (!h_dot) { h_dot = (double*)calloc(dot_sz,sizeof(double)); }
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr) { ::cuMemAlloc(&gdsr,bytes); }
   cuKernelDot<<<gridSize,blockSize>>>(N, (double*)gdsr, x, y);
   ::cuMemcpy((CUdeviceptr)h_dot,(CUdeviceptr)gdsr,bytes);
   double dot = 0.0;
   for (size_t i=0; i<dot_sz; i+=1) { dot += h_dot[i]; }
   return dot;
}
#endif // __NVCC__

// *****************************************************************************
__kernel double kVectorDot(const size_t N, const double *x, const double *y)
{
   if (config::usingGpu())
   {
#ifdef __NVCC__
      return cuVectorDot(N, x, y);
#endif // __NVCC__
   }

   double dot = 0.0;
   for (size_t i=0; i<N; i+=1) { dot += x[i] * y[i]; }
   return dot;
}

// *****************************************************************************
__kernel void kVectorMapDof(const int N,
                            double *v0, const double *v1, const int *dof)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dof[i];
      v0[dof_i] = v1[dof_i];
   });
}

// *****************************************************************************
__kernel void kVectorMapDof(double *v0, const double *v1,
                            const int dof, const int j)
{
   MFEM_FORALL(i, 1, v0[dof] = v1[j]; );
}

// *****************************************************************************
__kernel void kVectorSetDof(double *v0, const double alpha, const int dof)
{
   MFEM_FORALL(i, 1, v0[dof] = alpha; );
}

// *****************************************************************************
__kernel void kVectorSetDof(const int N, double *v0,
                            const double alpha, const int *dof)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dof[i];
      v0[dof_i] = alpha;
   });
}

// *****************************************************************************
__kernel void kVectorGetSubvector(const int N,
                                  double* y,
                                  const double* x,
                                  const int* dofs)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dofs[i];
      y[i] = dof_i >= 0 ? x[dof_i] : -x[-dof_i-1];
   });
}

// *****************************************************************************
__kernel void kVectorSetSubvector(const int N,
                                  double* x,
                                  const double* y,
                                  const int* dofs)
{
   MFEM_FORALL(i, N,
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         x[j] = y[i];
      }
      else {
         x[-1-j] = -y[i];
      }
   });
}

// *****************************************************************************
__kernel void kVectorSubtract(double *zp, const double *xp, const double *yp,
                              const size_t N)
{
   MFEM_FORALL(i, N, zp[i] = xp[i] - yp[i];);
}

// *****************************************************************************
__kernel void kVectorAlphaAdd(double *vp, const double* v1p,
                              const double alpha, const double *v2p,
                              const size_t N)
{
   MFEM_FORALL(i, N, vp[i] = v1p[i] + alpha * v2p[i];);
}

// *****************************************************************************
__kernel void kVectorPrint(const size_t N, const double *data)
{
   MFEM_FORALL(k, 1, // Sequential printf to get the same order as the host
   {
      for (size_t i=0; i<N; i+=1)
      {
         printf("\n\t%f",data[i]);
      }
   });
}

// *****************************************************************************
__kernel void kVectorAssign(const size_t N, const double* v, double *data)
{
   MFEM_FORALL(i, N, data[i] = v[i];);
}

// **************************************************************************
__kernel void kVectorSet(const size_t N,
                         const double value,
                         double *data)
{
   MFEM_FORALL(i, N, data[i] = value;);
}

// *****************************************************************************
__kernel void kVectorMultOp(const size_t N,
                            const double value,
                            double *data)
{
   MFEM_FORALL(i, N, data[i] *= value;);
}

// *****************************************************************************
__kernel void kVectorDotOpPlusEQ(const size_t size,
                                 const double *v, double *data)
{
   MFEM_FORALL(i, size, data[i] += v[i];);
}

// *****************************************************************************
__kernel void kVectorOpSubtract(const size_t size,
                                const double *v, double *data)
{
   MFEM_FORALL(i, size, data[i] -= v[i];);
}

// *****************************************************************************
__kernel void kAddElementVector(const size_t n, const int *dofs,
                                const double *elem_data, double *data)
{
   MFEM_FORALL(i, n,
   {
      const int j = dofs[i];
      if (j >= 0)
         data[j] += elem_data[i];
      else
      {
         data[-1-j] -= elem_data[i];
      }
   });
}

// *****************************************************************************
} // mfem
