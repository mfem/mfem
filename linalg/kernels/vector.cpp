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
void MapDof(const int N, double *y, const double *x, const int *dofs)
{
   GET_PTR(v0);
   GET_CONST_PTR(v1);
   GET_CONST_PTR_T(dof,int);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[dof_i] = d_x[dof_i];
   });
}

// *****************************************************************************
void MapDof(double *y, const double *x, const int dof, const int j)
{
   GET_PTR(v0);
   GET_CONST_PTR(v1);
   MFEM_FORALL(i, 1, d_v0[dof] = d_v1[j]; );
}

// *****************************************************************************
void SetDof(double *y, const double alpha, const int dof)
{
   GET_PTR(v0);
   MFEM_FORALL(i, 1, d_v0[dof] = alpha; );
}

// *****************************************************************************
void SetDof(const int N, double *y, const double alpha, const int *dofs)
{
   GET_PTR(v0);
   GET_CONST_PTR_T(dof,int);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[dof_i] = alpha;
   });
}

// *****************************************************************************
void GetSubvector(const int N, double *y, const double *x, const int* dofs)
{
   GET_PTR(y);
   GET_CONST_PTR(x);
   GET_CONST_PTR_T(dofs,int);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[i] = dof_i >= 0 ? d_x[dof_i] : -d_x[-dof_i-1];
   });
}

// *****************************************************************************
void SetSubvector(const int N, double *y, const double *x, const int* dofs)
{
   GET_PTR(x);
   GET_CONST_PTR(y);
   GET_CONST_PTR_T(dofs,int);
   MFEM_FORALL(i, N,
   {
      const int j = d_dofs[i];
      if (j >= 0)
      {
         d_y[j] = d_x[i];
      }
      else {
         d_y[-1-j] = -d_x[i];
      }
   });
}

// *****************************************************************************
void AlphaAdd(double *z, const double *x,
              const double a, const double *y, const size_t N)
{
   GET_PTR(zp);
   GET_CONST_PTR(xp);
   GET_CONST_PTR(yp);
   MFEM_FORALL(i, N, d_zp[i] = d_xp[i] - d_yp[i];);
}

// *****************************************************************************
void Subtract(double *z, const double *x, const double *y, const size_t N)
{
   GET_PTR(vp);
   GET_CONST_PTR(v1p);
   GET_CONST_PTR(v2p);
   MFEM_FORALL(i, N, d_vp[i] = d_v1p[i] + alpha * d_v2p[i];);
}


// *****************************************************************************
void Print(const size_t N, const double *x)
{
   GET_CONST_PTR(data);
   MFEM_FORALL(k, 1, // Sequential printf to get the same order as the host
   {
      for (size_t i=0; i<N; i+=1)
      {
         printf("\n\t%f",d_x[i]);
      }
   });
}

// **************************************************************************
void Set(const size_t N, const double d, double *y)
{
   GET_PTR(data);
   GET_CONST_PTR(v);
   MFEM_FORALL(i, N, d_data[i] = d_v[i];);
}

// *****************************************************************************
void Assign(const size_t N, const double *x, double *y)
{
   GET_PTR(data);
   MFEM_FORALL(i, N, d_data[i] = value;);
}

// *****************************************************************************
void OpMultEQ(const size_t N, const double d, double *y)
{
   GET_PTR(data);
   MFEM_FORALL(i, N, d_data[i] *= value;);
}

// *****************************************************************************
void OpPlusEQ(const size_t size, const double *x, double *y)
{
   GET_CONST_PTR(v);
   GET_PTR(data);
   MFEM_FORALL(i, size, d_data[i] += d_v[i];);
}

// *****************************************************************************
void OpSubtractEQ(const size_t size, const double *x, double *y)
{
   GET_CONST_PTR(v);
   GET_PTR(data);
   MFEM_FORALL(i, size, d_data[i] -= d_v[i];);
}

// *****************************************************************************
void AddElement(const size_t n, const int *dofs, const double *x, double *y)
{
   GET_CONST_PTR_T(dofs,int);
   GET_CONST_PTR(elem_data);
   GET_PTR(data);
   MFEM_FORALL(i, n,
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

} // namespace vector
} // namespace kernels
} // namespace mfem
