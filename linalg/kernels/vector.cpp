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
#ifdef __NVCC__
#if __CUDA_ARCH__ > 300

// *****************************************************************************
#include <cub/cub.cuh>
__inline__ __device__ double4 operator*(double4 a, double4 b) {
   return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

// *****************************************************************************
static double cub_vector_dot(const int N,
                             const double* __restrict vec1,
                             const double* __restrict vec2) {
   push();
   static double *h_dot = NULL;
   if (!h_dot){
      dbg("!h_dot");
      void *ptr;
      cuMemHostAlloc(&ptr, sizeof(double), CU_MEMHOSTALLOC_PORTABLE);
      h_dot=(double*)ptr;
   }
   static double *d_dot = NULL;
   if (!d_dot) {
      dbg("!d_dot");
      cuMemAlloc((CUdeviceptr*)&d_dot, sizeof(double));
   }
   static void *d_storage = NULL;
   static size_t storage_bytes = 0;
   if (!d_storage){
      dbg("[cub_vector_dot] !d_storage");
      cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
      cuMemAlloc((CUdeviceptr*)&d_storage, storage_bytes*sizeof(double));
   }
   cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
   //mfem::mm::D2H(h_dot,d_dot,sizeof(double));
   checkCudaErrors(cuMemcpy((CUdeviceptr)h_dot,(CUdeviceptr)d_dot,sizeof(double)));
   //dbg("dot=%e",*h_dot);
   //assert(false);
   return *h_dot;
}
#else // __CUDA_ARCH__ > 300
// *****************************************************************************
void kdot(const size_t N, const double *d_x, const double *d_y, double *d_dot){
   forall(k, 1, {
         double l_dot = 0.0;
         for(int i=0;i<N;i+=1){
            l_dot += d_x[i] * d_y[i];
         }
         *d_dot = l_dot;
      });
}
#endif // __CUDA_ARCH__
#endif // __NVCC__

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
double kVectorDot(const size_t N, const double *x, const double *y){
   GET_CUDA;
   GET_CONST_ADRS(x);
   GET_CONST_ADRS(y);
#ifdef __NVCC__
#if __CUDA_ARCH__ > 300
   if (cuda) return cub_vector_dot(N, d_x, d_y);
#else // __CUDA_ARCH__
   if (cuda){
      static double *h_dot = NULL;
      if (!h_dot){
         void *ptr;
         cuMemHostAlloc(&ptr, sizeof(double), CU_MEMHOSTALLOC_PORTABLE);
         h_dot=(double*)ptr;
      }
      static double *d_dot = NULL;
      if (!d_dot) {
         dbg("!d_dot");
         cuMemAlloc((CUdeviceptr*)&d_dot, sizeof(double));
      }
      kdot(N,d_x,d_y,d_dot);
      mfem::mm::D2H(h_dot,d_dot,sizeof(double));
      return *h_dot;
   }
#endif // __CUDA_ARCH__
#endif // __NVCC__
   double dot = 0.0;
   for(size_t i=0;i<N;i+=1)
      dot += d_x[i] * d_y[i];
   return dot;
}

// *****************************************************************************
void kVectorMapDof(const int N, double *v0, const double *v1, const int *dof){
   GET_ADRS(v0);
   GET_CONST_ADRS(v1);
   GET_CONST_ADRS_T(dof,int);
   forall(i, N, {
         const int dof_i = d_dof[i];
         d_v0[dof_i] = d_v1[dof_i];
      });
}

// *****************************************************************************
void kVectorMapDof(double *v0, const double *v1, const int dof, const int j){
   GET_ADRS(v0);
   GET_CONST_ADRS(v1);
   forall(i, 1, d_v0[dof] = d_v1[j]; );
}

// *****************************************************************************
void kVectorSetDof(double *v0, const double alpha, const int dof){
   GET_ADRS(v0);
   forall(i, 1, d_v0[dof] = alpha; );
}

// *****************************************************************************
void kVectorSetDof(const int N, double *v0, const double alpha, const int *dof){
   GET_ADRS(v0);
   GET_CONST_ADRS_T(dof,int);
   forall(i, N, {
         const int dof_i = d_dof[i];
         d_v0[dof_i] = alpha;
      });
}

// *****************************************************************************
void kVectorGetSubvector(const int N,
                         double* v0,
                         const double* v1,
                         const int* v2){
   GET_ADRS(v0);
   GET_CONST_ADRS(v1);
   GET_CONST_ADRS_T(v2,int);
   forall(i, N, {
         const int dof_i = d_v2[i];
         //printf("\n[kVectorGetSubvector] N=%d, i=%ld, dof_i=%d",N,i,dof_i);
         assert(dof_i >= 0);
         d_v0[i] = dof_i >= 0 ? d_v1[dof_i] : -d_v1[-dof_i-1];
      });
}

// *****************************************************************************
void kVectorSetSubvector(const int N,
                         double* data,
                         const double* elemvect,
                         const int* dofs){
   GET_ADRS(data);
   GET_CONST_ADRS(elemvect);
   GET_CONST_ADRS_T(dofs,int);
   forall(i,N,{
         const int j = d_dofs[i];
         //printf("\n\tj=%d, set: %f",j,d_elemvect[i]);
         if (j >= 0) {
            d_data[j] = d_elemvect[i];
         } else {
            assert(false);
            d_data[-1-j] = -d_elemvect[i];
         }
      });
}

// *****************************************************************************
void kVectorSubtract(double *zp, const double *xp, const double *yp,
                     const size_t N){
   GET_ADRS(zp);
   GET_CONST_ADRS(xp);
   GET_CONST_ADRS(yp);
   forall(i, N, d_zp[i] = d_xp[i] - d_yp[i];);
}

// *****************************************************************************
void kVectorAlphaAdd(double *vp, const double* v1p,
                     const double alpha, const double *v2p, const size_t N){
   GET_ADRS(vp);
   GET_CONST_ADRS(v1p);
   GET_CONST_ADRS(v2p);
   forall(i, N, d_vp[i] = d_v1p[i] + alpha * d_v2p[i];);
}

// *****************************************************************************
void kVectorPrint(const size_t N, const double *data){
   GET_CONST_ADRS(data); // Sequential printf
   forall(k,1,for(int i=0;i<N;i+=1)printf("\n\t%f",d_data[i]););
}

// *****************************************************************************
void kVectorAssign(const size_t N, const double* v, double *data){
   GET_ADRS(data);
   GET_CONST_ADRS(v);
   forall(i, N, d_data[i] = d_v[i];);
}

// **************************************************************************
void kVectorSet(const size_t N,
                const double value,
                double *data){
   GET_ADRS(data);
   forall(i, N, d_data[i] = value;);
}

// *****************************************************************************
void kVectorMultOp(const size_t N,
                   const double value,
                   double *data){
   GET_ADRS(data);
   forall(i, N, d_data[i] *= value;);
}

// *****************************************************************************
void kVectorDotOpPlusEQ(const size_t size, const double *v, double *data){
   GET_CONST_ADRS(v);
   GET_ADRS(data);
   forall(i, size, d_data[i] += d_v[i];);
}
/*
// *****************************************************************************
void kSetSubVector(const size_t n, const int *dofs, const double *elemvect,
                   double *data){
   GET_CONST_ADRS_T(dofs,int);
   GET_CONST_ADRS(elemvect);
   GET_ADRS(data);
//#warning make sure we can work on this outer loop
   forall(i, n,{
         const int j = d_dofs[i];
         if (j >= 0) {
            d_data[j] = d_elemvect[i];
         }else{
            d_data[-1-j] = -d_elemvect[i];
         }
      });
      }*/

// *****************************************************************************
void kVectorOpSubtract(const size_t size, const double *v, double *data){
   GET_CONST_ADRS(v);
   GET_ADRS(data);
   forall(i, size, d_data[i] -= d_v[i];);
}

// *****************************************************************************
void kAddElementVector(const size_t n, const int *dofs,
                       const double *elem_data, double *data){
   GET_CONST_ADRS_T(dofs,int);
   GET_CONST_ADRS(elem_data);
   GET_ADRS(data);
   forall(k,1,{
         for (int i = 0; i < n; i++) {
            const int j = d_dofs[i];
            if (j >= 0){
               d_data[j] += d_elem_data[i];
            }else{
               d_data[-1-j] -= d_elem_data[i];
            }
         }
      });
}

// *****************************************************************************
MFEM_NAMESPACE_END
