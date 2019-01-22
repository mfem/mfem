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
#include <limits>

namespace mfem
{
namespace kernels
{
namespace vector
{

// *****************************************************************************
#ifdef __NVCC__

// *****************************************************************************
#define CUDA_BLOCKSIZE 256

// *****************************************************************************
__global__ void cuKernelMin(const size_t N, double *gdsr, const double *x)
{
   __shared__ double s_min[CUDA_BLOCKSIZE];
   const size_t n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const size_t bid = blockIdx.x;
   const size_t tid = threadIdx.x;
   const size_t bbd = bid*blockDim.x;
   const size_t rid = bbd+tid;
   s_min[tid] = x[n];
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
      s_min[tid] = fmin(s_min[tid], s_min[dualTid]);
   }
   if (tid==0) { gdsr[bid] = s_min[0]; }
}

// *****************************************************************************
static double cuVectorMin(const size_t N, const double *x)
{
   const size_t tpb = CUDA_BLOCKSIZE;
   const size_t blockSize = CUDA_BLOCKSIZE;
   const size_t gridSize = (N+blockSize-1)/blockSize;
   const size_t min_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const size_t bytes = min_sz*sizeof(double);
   static double *h_min = NULL;
   if (!h_min) { h_min = (double*)calloc(min_sz,sizeof(double)); }
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr) { ::cuMemAlloc(&gdsr,bytes); }
   cuKernelMin<<<gridSize,blockSize>>>(N, (double*)gdsr, x);
   ::cuMemcpy((CUdeviceptr)h_min,(CUdeviceptr)gdsr,bytes);
   double min = std::numeric_limits<double>::infinity();
   for (size_t i=0; i<min_sz; i+=1) { min = fmin(min, h_min[i]); }
   return min;
}

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
static double cuVectorDot(const size_t N, const double *x, const double *y)
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
double Min(const size_t N, const double *x)
{
   GET_CONST_PTR(x);
   if (config::usingGpu())
   {
#ifdef __NVCC__
      return cuVectorMin(N, d_x);
#endif // __NVCC__
   }
   double min = std::numeric_limits<double>::infinity();
   for (size_t i=0; i<N; i+=1) { min = fmin(min, d_x[i]); }
   return min;
}

// *****************************************************************************
double Dot(const size_t N, const double *x, const double *y)
{
   GET_CONST_PTR(x);
   GET_CONST_PTR(y);
   if (config::usingGpu())
   {
#ifdef __NVCC__
      return cuVectorDot(N, d_x, d_y);
#endif // __NVCC__
   }

   double dot = 0.0;
   for (size_t i=0; i<N; i+=1) { dot += d_x[i] * d_y[i]; }
   return dot;
}

// *****************************************************************************
__kernel void MapDof(const int N, double *y, const double *x, const int *dofs)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dofs[i];
      y[dof_i] = x[dof_i];
   });
}

// *****************************************************************************
__kernel void MapDof(double *y, const double *x, const int dof, const int j)
{
   MFEM_FORALL(i, 1, y[dof] = x[j];);
}

// *****************************************************************************
__kernel void SetDof(double *y, const double alpha, const int dof)
{
   MFEM_FORALL(i, 1, y[dof] = alpha;);
}

// *****************************************************************************
__kernel void SetDof(const int N, double *y, const double alpha, const int *dofs)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dofs[i];
      y[dof_i] = alpha;
   });
}

// *****************************************************************************
__kernel void GetSubvector(const int N, double *y, const double *x, const int* dofs)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dofs[i];
      y[i] = dof_i >= 0 ? x[dof_i] : -x[-dof_i-1];
   });
}

// *****************************************************************************
__kernel void SetSubvector(const int N, double *y, const double *x, const int* dofs)
{
   MFEM_FORALL(i, N,
   {
      const int dof_i = dofs[i];
      if (dof_i >= 0)
      {
         y[dof_i] = x[i];
      }
      else {
         y[-1-dof_i] = -x[i];
      }
   });
}

// *****************************************************************************
void SetSubvector(const int N, double* y, const double d, const int* dofs)
{
   MFEM_FORALL(i, N,
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         y[j] = d;
      }
      else {
         y[-1-j] = -d;
      }
   });
}

// *****************************************************************************
__kernel void AlphaAdd(double *z, const double *x,
              const double a, const double *y, const size_t N)
{
   MFEM_FORALL(i, N, z[i] = x[i] + a * y[i];);
}

// *****************************************************************************
__kernel void Subtract(double *z, const double *x, const double *y, const size_t N)
{
   MFEM_FORALL(i, N, z[i] = x[i] - y[i];);
}


// *****************************************************************************
__kernel void Print(const size_t N, const double *x)
{
   // Sequential printf to get the same order as on the host
   MFEM_FORALL(k, 1,
   {
      for (size_t i=0; i<N; i+=1)
      {
         printf("\n\t%f",x[i]);
      }
   });
}

// **************************************************************************
__kernel void Set(const size_t N, const double d, double *y)
{
   MFEM_FORALL(i, N, y[i] = d;);
}

// *****************************************************************************
__kernel void Assign(const size_t N, const double *x, double *y)
{
   MFEM_FORALL(i, N, y[i] = x[i];);
}

// *****************************************************************************
__kernel void OpMultEQ(const size_t N, const double d, double *y)
{
   MFEM_FORALL(i, N, y[i] *= d;);
}

// *****************************************************************************
__kernel void OpPlusEQ(const size_t N, const double *x, double *y)
{
   MFEM_FORALL(i, N, y[i] += x[i];);
}

// *****************************************************************************
__kernel void OpAddEQ(const size_t N, const double a, const double *x, double *y)
{
   MFEM_FORALL(i, N, y[i] += a * x[i];);
}

// *****************************************************************************
__kernel void OpSubtractEQ(const size_t N, const double *x, double *y)
{
   MFEM_FORALL(i, N, y[i] -= x[i];);
}

// *****************************************************************************
__kernel void AddElement(const size_t N, const int *dofs, const double *x, double *y)
{
   MFEM_FORALL(i, N,
   {
      const int j = dofs[i];
      if (j >= 0)
         y[j] += x[i];
      else
         y[-1-j] -= x[i];
   });
}

// *****************************************************************************
__kernel void AddElementAlpha(const size_t N, const int *dofs,
                              const double *x, double *y, const double alpha)
{
   MFEM_FORALL(i, N,
   {
      const int j = dofs[i];
      if (j >= 0)
         y[j] += alpha * x[i];
      else
      {
         y[-1-j] -= alpha * x[i];
      }
   });
}

} // namespace vector
} // namespace kernels
} // namespace mfem
