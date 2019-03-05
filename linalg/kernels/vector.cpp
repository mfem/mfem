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
#include "../../linalg/device.hpp"
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
__global__ void cuKernelMin(const int N, double *gdsr, const double *x)
{
   __shared__ double s_min[CUDA_BLOCKSIZE];
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

// *****************************************************************************
static double cuVectorMin(const int N, const double *x)
{
   const int tpb = CUDA_BLOCKSIZE;
   const int blockSize = CUDA_BLOCKSIZE;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int min_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const int bytes = min_sz*sizeof(double);
   static double *h_min = NULL;
   if (!h_min) { h_min = (double*)calloc(min_sz,sizeof(double)); }
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr) { ::cuMemAlloc(&gdsr,bytes); }
   cuKernelMin<<<gridSize,blockSize>>>(N, (double*)gdsr, x);
   cuCheck(cudaGetLastError());
   ::cuMemcpy((CUdeviceptr)h_min,(CUdeviceptr)gdsr,bytes);
   double min = std::numeric_limits<double>::infinity();
   for (int i=0; i<min_sz; i+=1) { min = fmin(min, h_min[i]); }
   return min;
}

// *****************************************************************************
__global__ void cuKernelDot(const int N, double *gdsr,
                            const double *x, const double *y)
{
   __shared__ double s_dot[CUDA_BLOCKSIZE];
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

// *****************************************************************************
static double cuVectorDot(const int N, const double *x, const double *y)
{
   static int dot_block_sz = 0;
   const int tpb = CUDA_BLOCKSIZE;
   const int blockSize = CUDA_BLOCKSIZE;
   const int gridSize = (N+blockSize-1)/blockSize;
   const int dot_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const int bytes = dot_sz*sizeof(double);
   static double *h_dot = NULL;
   if (!h_dot or dot_block_sz!=dot_sz)
   {
      if (h_dot) { free(h_dot); }
      h_dot = (double*)calloc(dot_sz,sizeof(double));
   }
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr or dot_block_sz!=dot_sz)
   {
      if (gdsr) { cuCheck(::cuMemFree(gdsr)); }
      cuCheck(::cuMemAlloc(&gdsr,bytes));
   }
   if (dot_block_sz!=dot_sz)
   {
      dot_block_sz = dot_sz;
   }
   cuKernelDot<<<gridSize,blockSize>>>(N, (double*)gdsr, x, y);
   cuCheck(cudaGetLastError());
   cuCheck(::cuMemcpy((CUdeviceptr)h_dot,(CUdeviceptr)gdsr,bytes));
   double dot = 0.0;
   for (int i=0; i<dot_sz; i+=1) { dot += h_dot[i]; }
   return dot;
}
#endif // __NVCC__

// *****************************************************************************
double Min(const int N, const double *x)
{
   if (config::UsingDevice())
   {
#ifdef __NVCC__
      const DeviceVector d_x(x,N);
      return cuVectorMin(N, d_x);
#else
      mfem_error("Using Min on device w/o support");
#endif // __NVCC__
   }
   double min = std::numeric_limits<double>::infinity();
   for (int i=0; i<N; i+=1) { min = fmin(min, x[i]); }
   return min;
}

// *****************************************************************************
double Dot(const int N, const double *x, const double *y)
{
   if (config::UsingDevice())
   {
#ifdef __NVCC__
      const DeviceVector d_x(x,N);
      const DeviceVector d_y(y,N);
      return cuVectorDot(N, d_x, d_y);
#else
      mfem_error("Using Dot on device w/o support");
#endif // __NVCC__
   }
   double dot = 0.0;
   for (int i=0; i<N; i+=1) { dot += x[i] * y[i]; }
   return dot;
}

// *****************************************************************************
void MapDof(const int N, double *y, const double *x, const int *dofs)
{
   const DeviceArray d_dofs(dofs,N);
   const DeviceVector d_x(x,N);
   DeviceVector d_y(y,N);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[dof_i] = d_x[dof_i];
   });
}

// *****************************************************************************
void MapDof(double *y, const double *x, const int dof, const int j)
{
   const double *d_x = (double*) mm::ptr(x);
   double *d_y = (double*) mm::ptr(y);
   MFEM_FORALL(i, 1, d_y[dof] = d_x[j];);
}

// *****************************************************************************
void SetDof(double *y, const double alpha, const int dof)
{
   double *d_y = (double*) mm::ptr(y);
   MFEM_FORALL(i, 1, d_y[dof] = alpha;);
}

// *****************************************************************************
void SetDof(const int N, double *y, const double alpha, const int *dofs)
{
   DeviceVector d_y(y,N);
   const DeviceArray d_dofs(dofs,N);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[dof_i] = alpha;
   });
}

// *****************************************************************************
void GetSubvector(const int N, double *y, const double *x, const int* dofs)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   const DeviceArray d_dofs(dofs,N);
   MFEM_FORALL(i, N,
   {
      const int dof_i = d_dofs[i];
      d_y[i] = dof_i >= 0 ? d_x[dof_i] : -d_x[-dof_i-1];
   });
}

// *****************************************************************************
void SetSubvector(const int N, double *y, const double *x, const int* dofs)
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
      else {
         d_y[-1-dof_i] = -d_x[i];
      }
   });
}

// *****************************************************************************
void SetSubvector(const int N, double* y, const double d, const int* dofs)
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
      else {
         d_y[-1-j] = -d;
      }
   });
}

// *****************************************************************************
void AlphaAdd(double *z, const double *x,
              const double a, const double *y, const int N)
{
   DeviceVector d_z(z,N);
   const DeviceVector d_x(x,N);
   const DeviceVector d_y(y,N);
   MFEM_FORALL(i, N, d_z[i] = d_x[i] + a * d_y[i];);
}

// *****************************************************************************
void Subtract(double *z, const double *x, const double *y, const int N)
{
   DeviceVector d_z(z,N);
   const DeviceVector d_x(x,N);
   const DeviceVector d_y(y,N);
   MFEM_FORALL(i, N, d_z[i] = d_x[i] - d_y[i];);
}


// *****************************************************************************
void Print(const int N, const double *x)
{
   const DeviceVector d_x(x,N);
   // Sequential printf to get the same order as on the host
   MFEM_FORALL(k, 1,
   {
      for (int i=0; i<N; i+=1)
      {
         printf("\n\t%f",d_x[i]);
      }
   });
}

// **************************************************************************
void Set(const int N, const double d, double *y)
{
   DeviceVector d_y(y,N);
   MFEM_FORALL(i, N, d_y[i] = d;);
}

// *****************************************************************************
void Assign(const int N, const double *x, double *y)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   MFEM_FORALL(i, N, d_y[i] = d_x[i];);
}

// *****************************************************************************
void Assign(const int N, const int *x, int *y)
{
   DeviceArray d_y(y,N);
   const DeviceArray d_x(x,N);
   MFEM_FORALL(i, N, d_y[i] = d_x[i];);
}

// *****************************************************************************
void OpMultEQ(const int N, const double d, double *y)
{
   DeviceVector d_y(y,N);
   MFEM_FORALL(i, N, d_y[i] *= d;);
}

// *****************************************************************************
void OpPlusEQ(const int N, const double *x, double *y)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   MFEM_FORALL(i, N, d_y[i] += d_x[i];);
}

// *****************************************************************************
void OpAddEQ(const int N, const double a, const double *x, double *y)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   MFEM_FORALL(i, N, d_y[i] += a * d_x[i];);
}

// *****************************************************************************
void OpSubtractEQ(const int N, const double *x, double *y)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   MFEM_FORALL(i, N, d_y[i] -= d_x[i];);
}

// *****************************************************************************
void AddElement(const int N, const int *dofs, const double *x, double *y)
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

// *****************************************************************************
void AddElementAlpha(const int N, const int *dofs,
                     const double *x, double *y, const double alpha)
{
   DeviceVector d_y(y,N);
   const DeviceVector d_x(x,N);
   const DeviceArray d_dofs(dofs,N);
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

} // namespace vector
} // namespace kernels
} // namespace mfem
