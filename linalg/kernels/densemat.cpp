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
namespace densemat
{

// *****************************************************************************
template <class T> __device__ __host__
inline void Swap(T &a, T &b)
{
   T tmp(a);
   a = b;
   b = tmp;
}

// *****************************************************************************
__kernel void GetInverseMatrix(const int m, const int *ipiv,
                               const double *data, double *x){
   MFEM_GPU_CANNOT_PASS;
   MFEM_FORALL(_k_, 1,
      for (int k = 0; k < m; k++)
      {
         double *mx = &x[k*m];
         const double minus_x_k = -( mx[k] = 1.0/data[k+k*m] );
         for (int i = 0; i < k; i++)
         {
            mx[i] = data[i+k*m] * minus_x_k;
         }
         for (int j = k-1; j >= 0; j--)
         {
            const double x_j = ( mx[j] /= data[j+j*m] );
            for (int i = 0; i < j; i++)
            {
               mx[i] -= data[i+j*m] * x_j;
            }
         }
         //x += m;
      }
      // X <- X L^{-1} (use input only from the upper triangular part of X)
      {
         int k = m-1;
         for (int j = 0; j < k; j++)
         {
            const double minus_L_kj = -data[k+j*m];
            for (int i = 0; i <= j; i++)
            {
               x[i+j*m] += x[i+k*m] * minus_L_kj;
            }
            for (int i = j+1; i < m; i++)
            {
               x[i+j*m] = x[i+k*m] * minus_L_kj;
            }
         }
      }
      for (int k = m-2; k >= 0; k--)
      {
         for (int j = 0; j < k; j++)
         {
            const double L_kj = data[k+j*m];
            for (int i = 0; i < m; i++)
            {
               x[i+j*m] -= x[i+k*m] * L_kj;
            }
         }
      }
      // X <- X P
      for (int k = m-1; k >= 0; k--) {
         const int piv_k = ipiv[k];
         if (k != piv_k) {
            for (int i = 0; i < m; i++) {
               Swap<double>(x[i+k*m], x[i+piv_k*m]);
            }
         }
      }
   );
}

// *****************************************************************************
__kernel void LSolve(const int m, const int n,
                     const double *data, const int *ipiv, double *x)
{
   MFEM_GPU_CANNOT_PASS;
   MFEM_FORALL(k, n,
   {
      double *mx = &x[k*m];
      // X <- P X
      for (int i = 0; i < m; i++)
      {
	 //Swap<double>(mx[i], mx[ipiv[i]]);
         const double tmp = mx[i];
         mx[i] = mx[ipiv[i]];
         mx[ipiv[i]] = tmp;
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const double mx_j = mx[j];
         for (int i = j+1; i < m; i++)
         {
            mx[i] -= data[i+j*m] * mx_j;
         }
      }
   });
}

// *****************************************************************************
__kernel void USolve(const int m, const int n, const double *data, double *x)
{
   MFEM_GPU_CANNOT_PASS;
   MFEM_FORALL(k, n,
   {
      double *mx = &x[k*m];
      for (int j = m-1; j >= 0; j--)
      {
         const double x_j = ( mx[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            mx[i] -= data[i+j*m] * x_j;
         }
      }
   });
}

// *****************************************************************************
__kernel void FactorPrint(const int s, const double *x)
{
   MFEM_FORALL(i, s,
   {
      printf("\n\tx[%ld]=%f",i,x[i]);
   });
}

// *****************************************************************************
__kernel void FactorSet(const int s, const double *x, double *y)
{
   MFEM_FORALL(i, s, y[i] = x[i];);
}

// *****************************************************************************
__kernel void Factor(const int m, int *ipiv, double *data)
{
   MFEM_GPU_CANNOT_PASS;
   MFEM_FORALL(i, m,
   {
      // pivoting
      {
         int piv = i;
         double a = fabs(data[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const double b = fabs(data[j+i*m]);
            if (b > a)
            {
               a = b;
               piv = j;
            }
         }
         ipiv[i] = piv;
         if (piv != (int) i)
         {
            // swap rows i and piv in both L and U parts
            for (int j = 0; j < m; j++)
            {
               //Swap<double>(data[i+j*m], data[piv+j*m]);
               const double tmp = data[i+j*m];
               data[i+j*m] = data[piv+j*m];
               data[piv+j*m] = tmp;
            }
         }
      }
      const double diim = data[i+i*m];
      assert(diim != 0.0);
      const double a_ii_inv = 1.0/data[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         data[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const double a_ik = data[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            data[j+k*m] -= a_ik * data[j+i*m];
         }
      }
   });
}

// **************************************************************************
__kernel void Set(const double d, const size_t size, double *x)
{
   MFEM_FORALL(i, size, x[i] = d;);
}

// **************************************************************************
__kernel void Transpose(const size_t height, const size_t width,
                        double *y, const double *x)
{
   MFEM_FORALL(i, height,
   {
      for (size_t j=0; j<width; j+=1)
      {
         y[i+j*height] = x[j+i*height];
      }
   });
}

// *****************************************************************************
__kernel void MultAAt(const size_t height, const size_t width,
                      const double *a, double *aat)
{
   MFEM_FORALL(i, height,
   {
      for (size_t j=0; j<=i; j++)
      {
         double temp = 0.0;
         for (size_t k=0; k<width; k++)
         {
            temp += a[i+k*height] * a[j+k*height];
         }
         aat[j+i*height] = aat[i+j*height] = temp;
      }
   });
}

// *****************************************************************************
__kernel void GradToDiv(const size_t n, const double *x, double *y)
{
   MFEM_FORALL(i, n, y[i] = x[i];);
}

// *****************************************************************************
__kernel void AddMult_a_VVt(const size_t n, const double a, const double *v,
                            const size_t height, double *VVt)
{
   MFEM_FORALL(i, n,
   {
      double avi = a * v[i];
      for (size_t j = 0; j < i; j++)
      {
         double avivj = avi * v[j];
         VVt[i+j*height] += avivj;
         VVt[j+i*height] += avivj;
      }
      VVt[i+i*height] += avi * v[i];
   });
}

// *****************************************************************************
__kernel void MultWidth0(const size_t height, double *y)
{
   MFEM_FORALL(row, height, y[row] = 0.0;);
}

// *****************************************************************************
__kernel void Mult(const size_t height, const size_t width,
                   const double *data, const double *x, double *y)
{
   MFEM_FORALL(i, height,
   {
      double sum = 0.0;
      for (size_t j=0; j<width; j+=1)
      {
         sum += x[j]*data[i+j*height];
      }
      y[i] = sum;
   });
}

// *****************************************************************************
__kernel void Mult(const size_t ah, const size_t aw, const size_t bw,
                   const double *bd, const double *cd, double *ad)
{
   MFEM_FORALL(i, ah*aw, ad[i] = 0.0;);
   MFEM_FORALL(j, aw,
   {
      for (size_t k = 0; k < bw; k++)
      {
         for (size_t i = 0; i < ah; i++)
         {
            ad[i+j*ah] += bd[i+k*ah] * cd[k+j*bw];
         }
      }
   });
}

// *****************************************************************************
__kernel void OpEQ(const size_t hw, const double *x, double *y)
{
   MFEM_FORALL(i, hw, y[i] = x[i];);
}

// *****************************************************************************
__kernel void Diag(const size_t n, const size_t N, const double c, double *y)
{
   MFEM_FORALL(i, N, y[i] = 0.0;);
   MFEM_FORALL(i, n, y[i*(n+1)] = c;);
}


// *****************************************************************************
double Det2(const double *x)
{
   MFEM_GPU_CANNOT_PASS;
   GET_PTR(x);
   return d_x[0] * d_x[3] - d_x[1] * d_x[2];
}

// *****************************************************************************
double Det3(const double *data)
{
   MFEM_GPU_CANNOT_PASS;
   return
      data[0] * (data[4] * data[8] - data[5] * data[7]) +
      data[3] * (data[2] * data[7] - data[1] * data[8]) +
      data[6] * (data[1] * data[5] - data[2] * data[4]);
}

// *****************************************************************************
double FNormMax(const size_t hw, const double *x)
{
   MFEM_GPU_CANNOT_PASS;
   double max_norm = 0.0;
   for (size_t i = 0; i < hw; i++)
   {
      const double entry = fabs(x[i]);
      if (entry > max_norm)
      {
         max_norm = entry;
      }
   }
   return max_norm;
}

// *****************************************************************************
double FNorm2(const size_t hw, const double max_norm, const double *x)
{
   MFEM_GPU_CANNOT_PASS;
   double fnorm2 = 0.0;
   for (size_t i = 0; i < hw; i++)
   {
      const double entry = x[i] / max_norm;
      fnorm2 += entry * entry;
   }
   return fnorm2;
}

// *****************************************************************************
__kernel void CalcInverse2D(const double t, const double *a, double *inva)
{
   MFEM_GPU_CANNOT_PASS;
   inva[0+2*0] =  a[1+2*1] * t ;
   inva[0+2*1] = -a[0+2*1] * t ;
   inva[1+2*0] = -a[1+2*0] * t ;
   inva[1+2*1] =  a[0+2*0] * t ;
}

// *****************************************************************************
__kernel void CalcInverse3D(const double t, const double *a, double *inva)
{
   inva[1+3*0] = (a[1+3*2]*a[2+3*0]-a[1+3*0]*a[2+3*2])*t;
   inva[1+3*1] = (a[0+3*0]*a[2+3*2]-a[0+3*2]*a[2+3*0])*t;
   inva[1+3*2] = (a[0+3*2]*a[1+3*0]-a[0+3*0]*a[1+3*2])*t;

   inva[2+3*0] = (a[1+3*0]*a[2+3*1]-a[1+3*1]*a[2+3*0])*t;
   inva[2+3*1] = (a[0+3*1]*a[2+3*0]-a[0+3*0]*a[2+3*1])*t;
   inva[2+3*2] = (a[0+3*0]*a[1+3*1]-a[0+3*1]*a[1+3*0])*t;
}

} // namespace densemat
} // namespace kernels
} // namespace mfem
