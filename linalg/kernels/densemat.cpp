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

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
#ifdef MFEM_USE_LAPACK
#define ipiv_base 1
#else
#define ipiv_base 0
#endif

// *****************************************************************************
template <class T> __device__ __host__
inline void Swap(T &a, T &b)
{
   T tmp(a);
   a = b;
   b = tmp;
}

// *****************************************************************************
__kernel__ void LSolve(const int m,
                       const int n,
                       const double *data,
                       const int *ipiv,
                       double *x)
{
   for(int k=0; k<n; k+=1) {
      double *mx = &x[k*m];
      // X <- P X
      for (int i = 0; i < m; i++) {
         Swap<double>(mx[i], mx[ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++) {
         const double mx_j = mx[j];
         for (int i = j+1; i < m; i++) {
            mx[i] -= data[i+j*m] * mx_j;
         }
      }
   }
}

// *****************************************************************************
void kLSolve( const int m,
              const int n,
              const double *data, const int *ipiv, double *x)
{
   GET_CONST_ADRS(data);
   GET_CONST_ADRS_T(ipiv,int);
   GET_ADRS(x);
   forall(k, n,
   {
      double *d_mx = &d_x[k*m];
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         Swap<double>(d_mx[i], d_mx[d_ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const double d_mx_j = d_mx[j];
         for (int i = j+1; i < m; i++)
         {
            d_mx[i] -= d_data[i+j*m] * d_mx_j;
         }
      }
   });
}

// *****************************************************************************
void kUSolve(const int m, const int n, const double *data, double *x)
{
   GET_CONST_ADRS(data);
   GET_ADRS(x);
   forall(k, n,
   {
      double *d_mx = &d_x[k*m];
      for (int j = m-1; j >= 0; j--)
      {
         const double x_j = ( d_mx[j] /= d_data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            d_mx[i] -= d_data[i+j*m] * x_j;
         }
      }
   });
}

// *****************************************************************************
void kFactorPrint(const int s, const double *data)
{
   GET_CONST_ADRS(data);
   forall(i, s,
   {
      printf("\n\t\033[32md_data[%ld]=%f\033[m",i,d_data[i]);
   });
}

// *****************************************************************************
void kFactorSet(const int s, const double *adata, double *ludata)
{
   GET_CONST_ADRS(adata);
   GET_ADRS(ludata);
   forall(i, s,
   {
      d_ludata[i] = d_adata[i];
   });
}

// *****************************************************************************
void kFactor(const int m, int *ipiv, double *data)
{
   GET_ADRS_T(ipiv,int);
   GET_ADRS(data);
   forall(i,m,
   {
      // pivoting
      {
         size_t piv = i;
         double a = fabs(d_data[piv+i*m]);
         //printf("\n a=%f",a);
         for (int j = i+1; j < m; j++)
         {
            const double b = fabs(d_data[j+i*m]);
            //printf("\n\t b=%f",b);
            if (b > a)
            {
               a = b;
               piv = j;
            }
         }
         d_ipiv[i] = piv;
         if (piv != i)
         {
            // swap rows i and piv in both L and U parts
            for (int j = 0; j < m; j++)
            {
               //Swap<double>(d_data[i+j*m], d_data[piv+j*m]);
               const double a = d_data[i+j*m];
               const double b = d_data[piv+j*m];
               d_data[i+j*m] = b;
               d_data[piv+j*m] = a;
            }
         }
      }
      const double diim = d_data[i+i*m];
      //printf("\n\t\033[32;7mdiim=%f\033[m",diim);
      assert(diim != 0.0);
      const double a_ii_inv = 1.0/d_data[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         d_data[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const double a_ik = d_data[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            d_data[j+k*m] -= a_ik * d_data[j+i*m];
         }
      }
      //}
   });
}

// *****************************************************************************
void kMult(const int ah, const int aw, const int bw,
           const double *bd, const double *cd, double *ad)
{
   GET_CONST_ADRS(bd);
   GET_CONST_ADRS(cd);
   GET_ADRS(ad);
   forall(k,1,
   {
      for (int i = 0; i < ah*aw; i++)
      {
         d_ad[i] = 0.0;
      }
      for (int j = 0; j < aw; j++)
      {
         for (int k = 0; k < bw; k++)
         {
            for (int i = 0; i < ah; i++)
            {
               d_ad[i+j*ah] += d_bd[i+k*ah] * d_cd[k+j*bw];
            }
         }
      }
   });
}

// **************************************************************************
void DenseMatrixSet(const double d,
                    const size_t size,
                    double *data)
{
   GET_ADRS(data);
   forall(i, size, d_data[i] = d;);
}

// **************************************************************************
void DenseMatrixTranspose(const size_t height,
                          const size_t width,
                          double *data,
                          const double *mdata)
{
   GET_ADRS(data);
   GET_CONST_ADRS(mdata);
   forall(i,height,
   {
      for (size_t j=0; j<width; j+=1)
      {
         d_data[i+j*height] = d_mdata[j+i*height];
      }
   });
}

// *****************************************************************************
void kMultAAt(const size_t height, const size_t width,
              const double *a, double *aat)
{
   GET_CONST_ADRS(a);
   GET_ADRS(aat);
   forall(i, height,
   {
      for (size_t j=0; j<=i; j++)
      {
         double temp = 0.0;
         for (size_t k=0; k<width; k++)
         {
            temp += d_a[i+k*height] * d_a[j+k*height];
         }
         d_aat[j+i*height] = d_aat[i+j*height] = temp;
      }
   });
}

// *****************************************************************************
void kGradToDiv(const size_t n, const double *data, double *ddata)
{
   GET_CONST_ADRS(data);
   GET_ADRS(ddata);
   forall(i, n, d_ddata[i] = d_data[i];);
}

// *****************************************************************************
void kAddMult_a_VVt(const size_t n, const double a, const double *v,
                    const size_t height, double *VVt)
{
   GET_CONST_ADRS(v);
   GET_ADRS(VVt);
   forall(i, n,
   {
      double avi = a * d_v[i];
      for (size_t j = 0; j < i; j++)
      {
         double avivj = avi * d_v[j];
         d_VVt[i+j*height] += avivj;
         d_VVt[j+i*height] += avivj;
      }
      d_VVt[i+i*height] += avi * d_v[i];
   });

}

// *****************************************************************************
void kMult0(const size_t height, double *y)
{
   GET_ADRS(y);
   forall(row, height, d_y[row] = 0.0;);
}

// *****************************************************************************
void kMult(const size_t height, const size_t width,
           const double *data, const double *x, double *y)
{
   GET_CONST_ADRS(data);
   GET_CONST_ADRS(x);
   GET_ADRS(y);
   forall(i, height,
   {
      double sum = 0.0;
      for (size_t j=0; j<width; j+=1)
      {
         sum += d_x[j]*d_data[i+j*height];
      }
      d_y[i] = sum;
   });
}

// *****************************************************************************
void kDiag(const size_t n, const size_t N, const double c, double *data){
   GET_ADRS(data);
   forall(i, N, d_data[i] = 0.0;);
   forall(i, n, d_data[i*(n+1)] = c;);
}

// *****************************************************************************
void kOpEq(const size_t hw, const double *m, double *data){
   GET_CONST_ADRS(m);
   GET_ADRS(data);
   forall(i, hw, d_data[i] = d_m[i];);
}

// *****************************************************************************
double kDMDet2(const double *data){
   GET_ADRS(data);
   return d_data[0] * d_data[3] - d_data[1] * d_data[2];
}

// *****************************************************************************
double kDMDet3(const double *d){
   GET_ADRS(d);
   return
      d_d[0] * (d_d[4] * d_d[8] - d_d[5] * d_d[7]) +
      d_d[3] * (d_d[2] * d_d[7] - d_d[1] * d_d[8]) +
      d_d[6] * (d_d[1] * d_d[5] - d_d[2] * d_d[4]);
}

// *****************************************************************************
double kFNormMax(const size_t hw, const double *data){
   GET_ADRS(data);
   double max_norm = 0.0;
   for (size_t i = 0; i < hw; i++){
      const double entry = fabs(d_data[i]);
      if (entry > max_norm) {
         max_norm = entry;
      }
   }
   return max_norm;
}

// *****************************************************************************
double kFNorm2(const size_t hw, const double max_norm, const double *data){
   GET_ADRS(data);
   double fnorm2 = 0.0;
   for (size_t i = 0; i < hw; i++){
      const double entry = d_data[i] / max_norm;
      fnorm2 += entry * entry;
   }
   return fnorm2;
}

// *****************************************************************************
void kCalcInverse2D(const double t, const double *a, double *inva){
   GET_CONST_ADRS(a);
   GET_ADRS(inva);
   d_inva[0+2*0] = d_a[1+2*1] * t ;
   d_inva[0+2*1] = -d_a[0+2*1] * t ;
   d_inva[1+2*0] = -d_a[1+2*0] * t ;
   d_inva[1+2*1] = d_a[0+2*0] * t ;
}

// *****************************************************************************
void kCalcInverse3D(const double t, const double *a, double *inva){
   GET_CONST_ADRS(a);
   GET_ADRS(inva);
   
   d_inva[0+3*0] = (d_a[1+3*1]*d_a[2+3*2]-d_a[1+3*2]*d_a[2+3*1])*t;
   d_inva[0+3*1] = (d_a[0+3*2]*d_a[2+3*1]-d_a[0+3*1]*d_a[2+3*2])*t;
   d_inva[0+3*2] = (d_a[0+3*1]*d_a[1+3*2]-d_a[0+3*2]*d_a[1+3*1])*t;

   d_inva[1+3*0] = (d_a[1+3*2]*d_a[2+3*0]-d_a[1+3*0]*d_a[2+3*2])*t;
   d_inva[1+3*1] = (d_a[0+3*0]*d_a[2+3*2]-d_a[0+3*2]*d_a[2+3*0])*t;
   d_inva[1+3*2] = (d_a[0+3*2]*d_a[1+3*0]-d_a[0+3*0]*d_a[1+3*2])*t;

   d_inva[2+3*0] = (d_a[1+3*0]*d_a[2+3*1]-d_a[1+3*1]*d_a[2+3*0])*t;
   d_inva[2+3*1] = (d_a[0+3*1]*d_a[2+3*0]-d_a[0+3*0]*d_a[2+3*1])*t;
   d_inva[2+3*2] = (d_a[0+3*0]*d_a[1+3*1]-d_a[0+3*1]*d_a[1+3*0])*t;         
}

// *****************************************************************************
MFEM_NAMESPACE_END
