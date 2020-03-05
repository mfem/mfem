// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "catch.hpp"
#include <unordered_map>

#include "mfem.hpp"
#include "general/forall.hpp"

#ifdef MFEM_USE_MPI
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define PFesGetParMeshGetComm0(pfes) pfes.GetParMesh()->GetComm()
#else
typedef int MPI_Comm;
typedef int HYPRE_Int;
#define ParMesh Mesh
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Finalize()
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_INT int
#define MPI_LONG long
#define HYPRE_MPI_INT int
#define MPI_DOUBLE double
template<typename T>
void MPI_Reduce_(T *src, T *dst, const int n)
{
   for (int i=0; i<n; i++) { dst[i] = src[i]; }
}
#define MPI_Reduce(src, dst, n, T,...) MPI_Reduce_<T>(src,dst,n)
class MPI_Session
{
public:
   MPI_Session() {}
   MPI_Session(int argc, char **argv) {}
   bool Root() { return true; }
   int WorldRank() { return 0; }
   int WorldSize() { return 1; }
};
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

static void v0(const Vector &x, Vector &v) { v = 0.0; }
static double rho0(const Vector &x) { return 1.0; }
static double gamma(const Vector &x) { return 1.4; }

MFEM_HOST_DEVICE static inline
double norml2(const int size, const double *data)
{
   if (0 == size) { return 0.0; }
   if (1 == size) { return std::abs(data[0]); }
   double scale = 0.0;
   double sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (data[i] != 0.0)
      {
         const double absdata = fabs(data[i]);
         if (scale <= absdata)
         {
            const double sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         }
         //MFEM_VERIFY(scale>0.0,"");
         const double sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg);
      }
   }
   return scale * sqrt(sum);
}

MFEM_HOST_DEVICE static inline
void symmetrize(const int n, double *d)
{
   for (int i = 0; i<n; i++)
   {
      for (int j = 0; j<i; j++)
      {
         const double a = 0.5 * (d[i*n+j] + d[j*n+i]);
         d[j*n+i] = d[i*n+j] = a;
      }
   }
}

MFEM_HOST_DEVICE static inline
void multABt(const int ah, const int aw, const int bh,
             const double *A, const double *B, double *C)
{
   const int ah_x_bh = ah*bh;
   for (int i=0; i<ah_x_bh; i+=1)
   {
      C[i] = 0.0;
   }
   for (int k=0; k<aw; k+=1)
   {
      double *c = C;
      for (int j=0; j<bh; j+=1)
      {
         const double bjk = B[j];
         for (int i=0; i<ah; i+=1)
         {
            c[i] += A[i] * bjk;
         }
         c += ah;
      }
      A += ah;
      B += bh;
   }
}

namespace blas
{

MFEM_HOST_DEVICE static inline void Swap(double &a, double &b)
{
   double tmp = a;
   a = b;
   b = tmp;
}

const static double Epsilon = std::numeric_limits<double>::epsilon();

template<int dim> double Det(const double *d);

template<> MFEM_HOST_DEVICE inline double Det<2>(const double *d)
{
   return d[0] * d[3] - d[1] * d[2];
}

template<> MFEM_HOST_DEVICE inline double Det<3>(const double *d)
{
   return d[0] * (d[4] * d[8] - d[5] * d[7]) +
          d[3] * (d[2] * d[7] - d[1] * d[8]) +
          d[6] * (d[1] * d[5] - d[2] * d[4]);
}

template<int dim> void CalcInverse(const double *a, double *i);

template<> MFEM_HOST_DEVICE inline
void CalcInverse<2>(const double *a, double *inva)
{
   constexpr int n = 2;
   const double d = blas::Det<2>(a);
   const double t = 1.0 / d;
   inva[0*n+0] =  a[1*n+1] * t ;
   inva[0*n+1] = -a[0*n+1] * t ;
   inva[1*n+0] = -a[1*n+0] * t ;
   inva[1*n+1] =  a[0*n+0] * t ;
}

template<> MFEM_HOST_DEVICE inline
void CalcInverse<3>(const double *a, double *inva)
{
   constexpr int n = 3;
   const double d = blas::Det<3>(a);
   const double t = 1.0 / d;
   inva[0*n+0] = (a[1*n+1]*a[2*n+2]-a[1*n+2]*a[2*n+1])*t;
   inva[0*n+1] = (a[0*n+2]*a[2*n+1]-a[0*n+1]*a[2*n+2])*t;
   inva[0*n+2] = (a[0*n+1]*a[1*n+2]-a[0*n+2]*a[1*n+1])*t;

   inva[1*n+0] = (a[1*n+2]*a[2*n+0]-a[1*n+0]*a[2*n+2])*t;
   inva[1*n+1] = (a[0*n+0]*a[2*n+2]-a[0*n+2]*a[2*n+0])*t;
   inva[1*n+2] = (a[0*n+2]*a[1*n+0]-a[0*n+0]*a[1*n+2])*t;

   inva[2*n+0] = (a[1*n+0]*a[2*n+1]-a[1*n+1]*a[2*n+0])*t;
   inva[2*n+1] = (a[0*n+1]*a[2*n+0]-a[0*n+0]*a[2*n+1])*t;
   inva[2*n+2] = (a[0*n+0]*a[1*n+1]-a[0*n+1]*a[1*n+0])*t;
}

MFEM_HOST_DEVICE static inline
void Add(const int height, const int width, const double alpha,
         const double *A, const double *B, double *C)
{
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         const int n = i*width+j;
         C[n] = A[n] + alpha * B[n];
      }
   }
}

MFEM_HOST_DEVICE static inline
void Mult(const int ah, const int aw, const int bw,
          const double *B, const double *C, double *A)
{
   const int ah_x_aw = ah*aw;
   for (int i = 0; i < ah_x_aw; i++) { A[i] = 0.0; }
   for (int j = 0; j < aw; j++)
   {
      for (int k = 0; k < bw; k++)
      {
         for (int i = 0; i < ah; i++)
         {
            A[i+j*ah] += B[i+k*ah] * C[k+j*bw];
         }
      }
   }
}

MFEM_HOST_DEVICE static inline
void MultV(const int height, const int width,
           double *data, const double *x, double *y)
{
   if (width == 0)
   {
      for (int row = 0; row < height; row++)
      {
         y[row] = 0.0;
      }
      return;
   }
   double *d_col = data;
   double x_col = x[0];
   for (int row = 0; row < height; row++)
   {
      y[row] = x_col*d_col[row];
   }
   d_col += height;
   for (int col = 1; col < width; col++)
   {
      x_col = x[col];
      for (int row = 0; row < height; row++)
      {
         y[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

MFEM_HOST_DEVICE static inline
void Eigensystem2S(const double &d12, double &d1, double &d2,
                   double &c, double &s)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 == 0.0)
   {
      c = 1.;
      s = 0.;
   }
   else
   {
      double t;
      const double zeta = (d2 - d1)/(2*d12);
      const double azeta = fabs(zeta);
      if (azeta < sqrt_1_eps)
      {
         t = copysign(1./(azeta + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = copysign(0.5/azeta, zeta);
      }
      c = sqrt(1./(1. + t*t));
      s = c*t;
      t *= d12;
      d1 -= t;
      d2 += t;
   }
}

template<int dim>
void CalcEigenvalues(const double *d, double *lambda, double *vec);

template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<2>(const double *d, double *lambda, double *vec)
{
   double d0 = d[0];
   double d2 = d[2];
   double d3 = d[3];
   double c, s;
   Eigensystem2S(d2, d0, d3, c, s);
   if (d0 <= d3)
   {
      lambda[0] = d0;
      lambda[1] = d3;
      vec[0] =  c;
      vec[1] = -s;
      vec[2] =  s;
      vec[3] =  c;
   }
   else
   {
      lambda[0] = d3;
      lambda[1] = d0;
      vec[0] =  s;
      vec[1] =  c;
      vec[2] =  c;
      vec[3] = -s;
   }
}

MFEM_HOST_DEVICE static inline
void GetScalingFactor(const double &d_max, double &mult)
{
   int d_exp;
   if (d_max > 0.)
   {
      mult = frexp(d_max, &d_exp);
      if (d_exp == std::numeric_limits<double>::max_exponent)
      {
         mult *= std::numeric_limits<double>::radix;
      }
      mult = d_max/mult;
   }
   else
   {
      mult = 1.;
   }
}

MFEM_HOST_DEVICE static inline
bool KernelVector2G(const int &mode,
                    double &d1, double &d12, double &d21, double &d2)
{
   double n1 = fabs(d1) + fabs(d21);
   double n2 = fabs(d2) + fabs(d12);
   bool swap_columns = (n2 > n1);
   double mu;
   if (!swap_columns)
   {
      if (n1 == 0.)
      {
         return true;
      }

      if (mode == 0)
      {
         if (fabs(d1) > fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
      else
      {
         if (fabs(d1) < fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
   }
   else
   {
      if (mode == 0)
      {
         if (fabs(d12) > fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
      else
      {
         if (fabs(d12) < fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
   }
   n1 = hypot(d1, d21);
   if (d21 != 0.)
   {
      mu = copysign(n1, d1);
      n1 = -d21*(d21/(d1 + mu));
      d1 = mu;
      if (fabs(n1) <= fabs(d21))
      {
         n1 = n1/d21;
         mu = (2./(1. + n1*n1))*(n1*d12 + d2);
         d2  = d2  - mu;
         d12 = d12 - mu*n1;
      }
      else
      {
         n2 = d21/n1;
         mu = (2./(1. + n2*n2))*(d12 + n2*d2);
         d2  = d2  - mu*n2;
         d12 = d12 - mu;
      }
   }
   mu = -d12/d1;
   n2 = 1./(1. + fabs(mu));
   if (fabs(d1) <= n2*fabs(d2))
   {
      d2 = 0.;
      d1 = 1.;
   }
   else
   {
      d2 = n2;
      d1 = mu*n2;
   }
   if (swap_columns)
   {
      Swap(d1, d2);
   }
   return false;
}

MFEM_HOST_DEVICE static inline
void Vec_normalize3_aux(const double &x1, const double &x2,
                        const double &x3,
                        double &n1, double &n2, double &n3)
{
   double t, r;

   const double m = fabs(x1);
   r = x2/m;
   t = 1. + r*r;
   r = x3/m;
   t = sqrt(1./(t + r*r));
   n1 = copysign(t, x1);
   t /= m;
   n2 = x2*t;
   n3 = x3*t;
}

MFEM_HOST_DEVICE static inline
void Vec_normalize3(const double &x1, const double &x2, const double &x3,
                    double &n1, double &n2, double &n3)
{
   if (fabs(x1) >= fabs(x2))
   {
      if (fabs(x1) >= fabs(x3))
      {
         if (x1 != 0.)
         {
            Vec_normalize3_aux(x1, x2, x3, n1, n2, n3);
         }
         else
         {
            n1 = n2 = n3 = 0.;
         }
         return;
      }
   }
   else if (fabs(x2) >= fabs(x3))
   {
      Vec_normalize3_aux(x2, x1, x3, n2, n1, n3);
      return;
   }
   Vec_normalize3_aux(x3, x1, x2, n3, n1, n2);
}

MFEM_HOST_DEVICE static inline
int KernelVector3G_aux(const int &mode,
                       double &d1, double &d2, double &d3,
                       double &c12, double &c13, double &c23,
                       double &c21, double &c31, double &c32)
{
   int kdim;
   double mu, n1, n2, n3, s1, s2, s3;
   s1 = hypot(c21, c31);
   n1 = hypot(d1, s1);
   if (s1 != 0.)
   {
      mu = copysign(n1, d1);
      n1 = -s1*(s1/(d1 + mu));
      d1 = mu;
      if (fabs(n1) >= fabs(c21))
      {
         if (fabs(n1) >= fabs(c31))
         {
            s2 = c21/n1;
            s3 = c31/n1;
            mu = 2./(1. + s2*s2 + s3*s3);
            n2  = mu*(c12 + s2*d2  + s3*c32);
            n3  = mu*(c13 + s2*c23 + s3*d3);
            c12 = c12 -    n2;
            d2  = d2  - s2*n2;
            c32 = c32 - s3*n2;
            c13 = c13 -    n3;
            c23 = c23 - s2*n3;
            d3  = d3  - s3*n3;
            goto done_column_1;
         }
      }
      else if (fabs(c21) >= fabs(c31))
      {
         s1 = n1/c21;
         s3 = c31/c21;
         mu = 2./(1. + s1*s1 + s3*s3);
         n2  = mu*(s1*c12 + d2  + s3*c32);
         n3  = mu*(s1*c13 + c23 + s3*d3);
         c12 = c12 - s1*n2;
         d2  = d2  -    n2;
         c32 = c32 - s3*n2;
         c13 = c13 - s1*n3;
         c23 = c23 -    n3;
         d3  = d3  - s3*n3;
         goto done_column_1;
      }
      s1 = n1/c31;
      s2 = c21/c31;
      mu = 2./(1. + s1*s1 + s2*s2);
      n2  = mu*(s1*c12 + s2*d2  + c32);
      n3  = mu*(s1*c13 + s2*c23 + d3);
      c12 = c12 - s1*n2;
      d2  = d2  - s2*n2;
      c32 = c32 -    n2;
      c13 = c13 - s1*n3;
      c23 = c23 - s2*n3;
      d3  = d3  -    n3;
   }
done_column_1:
   if (KernelVector2G(mode, d2, c23, c32, d3))
   {
      d2 = c12/d1;
      d3 = c13/d1;
      d1 = 1.;
      kdim = 2;
   }
   else
   {
      d1 = -(c12*d2 + c13*d3)/d1;
      kdim = 1;
   }
   Vec_normalize3(d1, d2, d3, d1, d2, d3);
   return kdim;
}

MFEM_HOST_DEVICE static inline
int KernelVector3S(const int &mode, const double &d12,
                   const double &d13, const double &d23,
                   double &d1, double &d2, double &d3)
{
   double c12 = d12, c13 = d13, c23 = d23;
   double c21, c31, c32;
   int col, row;
   c32 = fabs(d1) + fabs(c12) + fabs(c13);
   c31 = fabs(d2) + fabs(c12) + fabs(c23);
   c21 = fabs(d3) + fabs(c13) + fabs(c23);
   if (c32 >= c21)
   {
      col = (c32 >= c31) ? 1 : 2;
   }
   else
   {
      col = (c31 >= c21) ? 2 : 3;
   }
   switch (col)
   {
      case 1:
         if (c32 == 0.)
         {
            return 3;
         }
         break;

      case 2:
         if (c31 == 0.)
         {
            return 3;
         }
         Swap(c13, c23);
         Swap(d1, d2);
         break;

      case 3:
         if (c21 == 0.)
         {
            return 3;
         }
         Swap(c12, c23);
         Swap(d1, d3);
   }
   if (mode == 0)
   {
      if (fabs(d1) <= fabs(c13))
      {
         row = (fabs(d1) <= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) <= fabs(c13)) ? 2 : 3;
      }
   }
   else
   {
      if (fabs(d1) >= fabs(c13))
      {
         row = (fabs(d1) >= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) >= fabs(c13)) ? 2 : 3;
      }
   }
   switch (row)
   {
      case 1:
         c21 = c12;
         c31 = c13;
         c32 = c23;
         break;

      case 2:
         c21 = d1;
         c31 = c13;
         c32 = c23;
         d1 = c12;
         c12 = d2;
         d2 = d1;
         c13 = c23;
         c23 = c31;
         break;

      case 3:
         c21 = c12;
         c31 = d1;
         c32 = c12;
         d1 = c13;
         c12 = c23;
         c13 = d3;
         d3 = d1;
   }
   row = KernelVector3G_aux(mode, d1, d2, d3, c12, c13, c23, c21, c31, c32);
   switch (col)
   {
      case 2:
         Swap(d1, d2);
         break;

      case 3:
         Swap(d1, d3);
   }
   return row;
}

MFEM_HOST_DEVICE static inline
int Reduce3S(const int &mode,
             double &d1, double &d2, double &d3,
             double &d12, double &d13, double &d23,
             double &z1, double &z2, double &z3,
             double &v1, double &v2, double &v3,
             double &g)
{
   int k;
   double s, w1, w2, w3;
   if (mode == 0)
   {
      if (fabs(z1) <= fabs(z3))
      {
         k = (fabs(z1) <= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) <= fabs(z3)) ? 2 : 3;
      }
   }
   else
   {
      if (fabs(z1) >= fabs(z3))
      {
         k = (fabs(z1) >= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) >= fabs(z3)) ? 2 : 3;
      }
   }
   switch (k)
   {
      case 2:
         Swap(d13, d23);
         Swap(d1, d2);
         Swap(z1, z2);
         break;

      case 3:
         Swap(d12, d23);
         Swap(d1, d3);
         Swap(z1, z3);
   }
   s = hypot(z2, z3);
   if (s == 0.)
   {
      v1 = v2 = v3 = 0.;
      g = 1.;
   }
   else
   {
      g = copysign(1., z1);
      v1 = -s*(s/(z1 + g));
      g = fabs(v1);
      if (fabs(z2) > g) { g = fabs(z2); }
      if (fabs(z3) > g) { g = fabs(z3); }
      v1 = v1/g;
      v2 = z2/g;
      v3 = z3/g;
      g = 2./(v1*v1 + v2*v2 + v3*v3);
      w1 = g*( d1*v1 + d12*v2 + d13*v3);
      w2 = g*(d12*v1 +  d2*v2 + d23*v3);
      w3 = g*(d13*v1 + d23*v2 +  d3*v3);
      s = (g/2)*(v1*w1 + v2*w2 + v3*w3);
      w1 -= s*v1;
      w2 -= s*v2;
      w3 -= s*v3;
      d1  -= 2*v1*w1;
      d2  -= 2*v2*w2;
      d23 -= v2*w3 + v3*w2;
      d3  -= 2*v3*w3;
   }
   switch (k)
   {
      case 2:
         Swap(z1, z2);
         break;
      case 3:
         Swap(z1, z3);
   }
   return k;
}

template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<3>(const double *d, double *lambda, double *vec)
{
   double d11 = d[0];
   double d12 = d[3];
   double d22 = d[4];
   double d13 = d[6];
   double d23 = d[7];
   double d33 = d[8];
   double mult;
   {
      double d_max = fabs(d11);
      if (d_max < fabs(d22)) { d_max = fabs(d22); }
      if (d_max < fabs(d33)) { d_max = fabs(d33); }
      if (d_max < fabs(d12)) { d_max = fabs(d12); }
      if (d_max < fabs(d13)) { d_max = fabs(d13); }
      if (d_max < fabs(d23)) { d_max = fabs(d23); }
      GetScalingFactor(d_max, mult);
   }
   d11 /= mult;  d22 /= mult;  d33 /= mult;
   d12 /= mult;  d13 /= mult;  d23 /= mult;
   double aa = (d11 + d22 + d33)/3;
   double c1 = d11 - aa;
   double c2 = d22 - aa;
   double c3 = d33 - aa;
   double Q, R;
   Q = (2*(d12*d12 + d13*d13 + d23*d23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(d23*d23 - c2*c3)+ d12*(d12*c3 - 2*d13*d23) + d13*d13*c2)/2;
   if (Q <= 0.)
   {
      lambda[0] = lambda[1] = lambda[2] = aa;
      vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
      vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
      vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
   }
   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      double r;
      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            r = 2*sqrtQ;
         }
         else
         {
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3);
         }
      }
      aa += r;
      c1 = d11 - aa;
      c2 = d22 - aa;
      c3 = d33 - aa;
      const int mode = 0;
      switch (KernelVector3S(mode, d12, d13, d23, c1, c2, c3))
      {
         case 3:
            lambda[0] = lambda[1] = lambda[2] = aa;
            vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
            vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
            vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
            goto done_3d;
         case 2:
         case 1:;
      }
      double v1, v2, v3, g;
      int k = Reduce3S(mode, d11, d22, d33, d12, d13, d23,
                       c1, c2, c3, v1, v2, v3, g);
      double c, s;
      Eigensystem2S(d23, d22, d33, c, s);
      double *vec_1, *vec_2, *vec_3;
      if (d11 <= d22)
      {
         if (d22 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d11 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
      }
      else
      {
         if (d11 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d22 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
      }
      vec_1[0] = c1;
      vec_1[1] = c2;
      vec_1[2] = c3;
      d22 = g*(v2*c - v3*s);
      d33 = g*(v2*s + v3*c);
      vec_2[0] =    - v1*d22;  vec_3[0] =   - v1*d33;
      vec_2[1] =  c - v2*d22;  vec_3[1] = s - v2*d33;
      vec_2[2] = -s - v3*d22;  vec_3[2] = c - v3*d33;
      switch (k)
      {
         case 2:
            Swap(vec_2[0], vec_2[1]);
            Swap(vec_3[0], vec_3[1]);
            break;

         case 3:
            Swap(vec_2[0], vec_2[2]);
            Swap(vec_3[0], vec_3[2]);
      }
   }
done_3d:
   lambda[0] *= mult;
   lambda[1] *= mult;
   lambda[2] *= mult;
}

template<int dim> double CalcSingularvalue(const double *d);

template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<2>(const double *d)
{
   constexpr int i = 2-1;
   double d0, d1, d2, d3;
   d0 = d[0];
   d1 = d[1];
   d2 = d[2];
   d3 = d[3];
   double mult;
   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      GetScalingFactor(d_max, mult);
   }
   d0 /= mult;
   d1 /= mult;
   d2 /= mult;
   d3 /= mult;
   double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
   double s = d0*d2 + d1*d3;
   s = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));
   if (s == 0.0)
   {
      return 0.0;
   }
   t = fabs(d0*d3 - d1*d2) / s;
   if (t > s)
   {
      if (i == 0)
      {
         return t*mult;
      }
      return s*mult;
   }
   if (i == 0)
   {
      return s*mult;
   }
   return t*mult;
}

MFEM_HOST_DEVICE static inline
void Eigenvalues2S(const double &d12, double &d1, double &d2)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 != 0.)
   {
      double t;
      const double zeta = (d2 - d1)/(2*d12);
      if (fabs(zeta) < sqrt_1_eps)
      {
         t = d12*copysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = d12*copysign(0.5/fabs(zeta), zeta);
      }
      d1 -= t;
      d2 += t;
   }
}

template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<3>(const double *d)
{
   constexpr int i = 3-1;
   double d0, d1, d2, d3, d4, d5, d6, d7, d8;
   d0 = d[0];  d3 = d[3];  d6 = d[6];
   d1 = d[1];  d4 = d[4];  d7 = d[7];
   d2 = d[2];  d5 = d[5];  d8 = d[8];
   double mult;
   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      if (d_max < fabs(d4)) { d_max = fabs(d4); }
      if (d_max < fabs(d5)) { d_max = fabs(d5); }
      if (d_max < fabs(d6)) { d_max = fabs(d6); }
      if (d_max < fabs(d7)) { d_max = fabs(d7); }
      if (d_max < fabs(d8)) { d_max = fabs(d8); }
      GetScalingFactor(d_max, mult);
   }
   d0 /= mult;  d1 /= mult;  d2 /= mult;
   d3 /= mult;  d4 /= mult;  d5 /= mult;
   d6 /= mult;  d7 /= mult;  d8 /= mult;
   double b11 = d0*d0 + d1*d1 + d2*d2;
   double b12 = d0*d3 + d1*d4 + d2*d5;
   double b13 = d0*d6 + d1*d7 + d2*d8;
   double b22 = d3*d3 + d4*d4 + d5*d5;
   double b23 = d3*d6 + d4*d7 + d5*d8;
   double b33 = d6*d6 + d7*d7 + d8*d8;
   double aa = (b11 + b22 + b33)/3;
   double c1, c2, c3;
   {
      double b11_b22 = ((d0-d3)*(d0+d3)+(d1-d4)*(d1+d4)+(d2-d5)*(d2+d5));
      double b22_b33 = ((d3-d6)*(d3+d6)+(d4-d7)*(d4+d7)+(d5-d8)*(d5+d8));
      double b33_b11 = ((d6-d0)*(d6+d0)+(d7-d1)*(d7+d1)+(d8-d2)*(d8+d2));
      c1 = (b11_b22 - b33_b11)/3;
      c2 = (b22_b33 - b11_b22)/3;
      c3 = (b33_b11 - b22_b33)/3;
   }
   double Q, R;
   Q = (2*(b12*b12 + b13*b13 + b23*b23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(b23*b23 - c2*c3)+ b12*(b12*c3 - 2*b13*b23) +b13*b13*c2)/2;
   if (Q <= 0.) { ; }
   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      double r;
      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            r = 2*sqrtQ;
         }
         else
         {
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;
         if (fabs(R) <= 0.9)
         {
            if (i == 2)
            {
               aa -= 2*sqrtQ*cos(acos(R)/3);
            }
            else if (i == 0)
            {
               aa -= 2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);
            }
            else
            {
               aa -= 2*sqrtQ*cos((acos(R) - 2.0*M_PI)/3);
            }
            goto have_aa;
         }
         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);
            if (i == 0)
            {
               aa += r;
               goto have_aa;
            }
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3);
            if (i == 2)
            {
               aa += r;
               goto have_aa;
            }
         }
      }
      c1 -= r;
      c2 -= r;
      c3 -= r;
      const int mode = 1;
      switch (KernelVector3S(mode, b12, b13, b23, c1, c2, c3))
      {
         case 3:
            aa += r;
            goto have_aa;
         case 2:
         case 1:;
      }
      double v1, v2, v3, g;
      Reduce3S(mode, b11, b22, b33, b12, b13, b23,
               c1, c2, c3, v1, v2, v3, g);
      Eigenvalues2S(b23, b22, b33);
      if (i == 2)
      {
         aa = fmin(fmin(b11, b22), b33);
      }
      else if (i == 1)
      {
         if (b11 <= b22)
         {
            aa = (b22 <= b33) ? b22 : fmax(b11, b33);
         }
         else
         {
            aa = (b11 <= b33) ? b11 : fmax(b33, b22);
         }
      }
      else
      {
         aa = fmax(fmax(b11, b22), b33);
      }
   }
have_aa:
   return sqrt(fabs(aa))*mult;
}
} // namespace blas

namespace hydrodynamics
{

struct QuadratureData
{
   DenseTensor Jac0inv, stressJinvT;
   Vector rho0DetJ0w;
   double h0, dt_est;
   QuadratureData(int dim, int nzones, int quads_per_zone)
      : Jac0inv(dim, dim, nzones * quads_per_zone),
        stressJinvT(nzones * quads_per_zone, dim, dim),
        rho0DetJ0w(nzones * quads_per_zone) { }
};

struct Tensors1D
{
   DenseMatrix HQshape1D, HQgrad1D, LQshape1D;
   Tensors1D(int H1order, int L2order, int nqp1D, bool bernstein_v)
      : HQshape1D(H1order + 1, nqp1D), HQgrad1D(H1order + 1, nqp1D),
        LQshape1D(L2order + 1, nqp1D)
   {
      const double *quad1D_pos =
         poly1d.GetPoints(nqp1D - 1, Quadrature1D::GaussLegendre);
      Poly_1D::Basis &basisH1 =
         poly1d.GetBasis(H1order, Quadrature1D::GaussLobatto);
      Vector col, grad_col;
      for (int q = 0; q < nqp1D; q++)
      {
         HQshape1D.GetColumnReference(q, col);
         HQgrad1D.GetColumnReference(q, grad_col);
         if (bernstein_v)
         {
            poly1d.CalcBernstein(H1order, quad1D_pos[q],
                                 col.GetData(), grad_col.GetData());
         }
         else { basisH1.Eval(quad1D_pos[q], col, grad_col); }
      }
      for (int q = 0; q < nqp1D; q++)
      {
         LQshape1D.GetColumnReference(q, col);
         poly1d.CalcBernstein(L2order, quad1D_pos[q], col);
      }
   }
};

template<int DIM, int D1D, int Q1D, int L1D, int H1D, int NBZ =1> static
void kSmemForceMult2D(const int NE,
                      const Array<double> &_B,
                      const Array<double> &_Bt,
                      const Array<double> &_Gt,
                      const DenseTensor &_sJit,
                      const Vector &_e,
                      Vector &_v)
{
   auto b = Reshape(_B.Read(), Q1D, L1D);
   auto bt = Reshape(_Bt.Read(), H1D, Q1D);
   auto gt = Reshape(_Gt.Read(), H1D, Q1D);
   auto sJit = Reshape(Read(_sJit.GetMemory(), Q1D*Q1D*NE*2*2),
                       Q1D,Q1D,NE,2,2);
   auto energy = Reshape(_e.Read(), L1D, L1D, NE);
   const double eps1 = std::numeric_limits<double>::epsilon();
   const double eps2 = eps1*eps1;
   auto velocity = Reshape(_v.Write(), D1D,D1D,2,NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][L1D];
      MFEM_SHARED double Bt[H1D][Q1D];
      MFEM_SHARED double Gt[H1D][Q1D];
      MFEM_SHARED double Ez[NBZ][L1D][L1D];
      double (*E)[L1D] = (double (*)[L1D])(Ez + z);
      MFEM_SHARED double LQz[2][NBZ][H1D][Q1D];
      double (*LQ0)[Q1D] = (double (*)[Q1D])(LQz[0] + z);
      double (*LQ1)[Q1D] = (double (*)[Q1D])(LQz[1] + z);
      MFEM_SHARED double QQz[3][NBZ][Q1D][Q1D];
      double (*QQ)[Q1D] = (double (*)[Q1D])(QQz[0] + z);
      double (*QQ0)[Q1D] = (double (*)[Q1D])(QQz[1] + z);
      double (*QQ1)[Q1D] = (double (*)[Q1D])(QQz[2] + z);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < H1D) { Bt[l][q] = bt(l,q); }
               if (l < H1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            E[lx][ly] = energy(lx,ly,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int lx = 0; lx < L1D; ++lx)
            {
               u += B[qx][lx] * E[lx][ly];
            }
            LQ0[ly][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int ly = 0; ly < L1D; ++ly)
            {
               u += B[qy][ly] * LQ0[ly][qx];
            }
            QQ[qy][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double esx = QQ[qy][qx] * sJit(qx,qy,e,0,c);
               const double esy = QQ[qy][qx] * sJit(qx,qy,e,1,c);
               QQ0[qy][qx] = esx;
               QQ1[qy][qx] = esy;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += Gt[dx][qx] * QQ0[qy][qx];
                  v += Bt[dx][qx] * QQ1[qy][qx];
               }
               LQ0[dx][qy] = u;
               LQ1[dx][qy] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += LQ0[dx][qy] * Bt[dy][qy];
                  v += LQ1[dx][qy] * Gt[dy][qy];
               }
               velocity(dx,dy,c,e) = u + v;
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
            {
               const double v = velocity(dx,dy,c,e);
               if (fabs(v) < eps2)
               {
                  velocity(dx,dy,c,e) = 0.0;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D> static
void kSmemForceMult3D(const int NE,
                      const Array<double> &_B,
                      const Array<double> &_Bt,
                      const Array<double> &_Gt,
                      const DenseTensor &_sJit,
                      const Vector &_e,
                      Vector &_v)
{
   auto b = Reshape(_B.Read(), Q1D, L1D);
   auto bt = Reshape(_Bt.Read(), H1D, Q1D);
   auto gt = Reshape(_Gt.Read(), H1D, Q1D);
   auto sJit = Reshape(Read(_sJit.GetMemory(), Q1D*Q1D*Q1D*NE*3*3),
                       Q1D,Q1D,Q1D,NE,3,3);
   auto energy = Reshape(_e.Read(), L1D, L1D, L1D, NE);
   const double eps1 = std::numeric_limits<double>::epsilon();
   const double eps2 = eps1*eps1;
   auto velocity = Reshape(_v.Write(), D1D, D1D, D1D, 3, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][L1D];
      MFEM_SHARED double Bt[H1D][Q1D];
      MFEM_SHARED double Gt[H1D][Q1D];
      MFEM_SHARED double E[L1D][L1D][L1D];
      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*MMQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+0);
      double (*MMQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*MMQ2)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+2);
      double (*MQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*MQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*MQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);
      MFEM_SHARED double QQQ[Q1D][Q1D][Q1D];
      double (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);
      double (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);
      double (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < H1D) { Bt[l][q] = bt(l,q); }
               if (l < H1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lz,z,L1D)
            {
               E[lx][ly][lz] = energy(lx,ly,lz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int lx = 0; lx < L1D; ++lx)
               {
                  u += B[qx][lx] * E[lx][ly][lz];
               }
               MMQ0[lz][ly][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int ly = 0; ly < L1D; ++ly)
               {
                  u += B[qy][ly] * MMQ0[lz][ly][qx];
               }
               MQQ0[lz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int lz = 0; lz < L1D; ++lz)
               {
                  u += B[qz][lz] * MQQ0[lz][qy][qx];
               }
               QQQ[qz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double esx = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const double esy = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const double esz = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ0[qz][qy][qx] = esx;
                  QQQ1[qz][qy][qx] = esy;
                  QQQ2[qz][qy][qx] = esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += Gt[hx][qx] * QQQ0[qz][qy][qx];
                     v += Bt[hx][qx] * QQQ1[qz][qy][qx];
                     w += Bt[hx][qx] * QQQ2[qz][qy][qx];
                  }
                  MQQ0[hx][qy][qz] = u;
                  MQQ1[hx][qy][qz] = v;
                  MQQ2[hx][qy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += MQQ0[hx][qy][qz] * Bt[hy][qy];
                     v += MQQ1[hx][qy][qz] * Gt[hy][qy];
                     w += MQQ2[hx][qy][qz] * Bt[hy][qy];
                  }
                  MMQ0[hx][hy][qz] = u;
                  MMQ1[hx][hy][qz] = v;
                  MMQ2[hx][hy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += MMQ0[hx][hy][qz] * Bt[hz][qz];
                     v += MMQ1[hx][hy][qz] * Bt[hz][qz];
                     w += MMQ2[hx][hy][qz] * Gt[hz][qz];
                  }
                  velocity(hx,hy,hz,c,e) = u + v + w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
               {
                  const double v = velocity(hx,hy,hz,c,e);
                  if (fabs(v) < eps2)
                  {
                     velocity(hx,hy,hz,c,e) = 0.0;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

typedef void (*fForceMult)(const int E,
                           const Array<double> &B,
                           const Array<double> &Bt,
                           const Array<double> &Gt,
                           const DenseTensor &stressJinvT,
                           const Vector &e,
                           Vector &v);

static void kForceMult(const int DIM,
                       const int D1D,
                       const int Q1D,
                       const int L1D,
                       const int H1D,
                       const int NE,
                       const Array<double> &B,
                       const Array<double> &Bt,
                       const Array<double> &Gt,
                       const DenseTensor &stressJinvT,
                       const Vector &e,
                       Vector &v)
{
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMult> call =
   {
      {0x234,&kSmemForceMult2D<2,3,4,2,3>},
      //{0x246,&kSmemForceMult2D<2,4,6,3,4>},
      //{0x258,&kSmemForceMult2D<2,5,8,4,5>},
      // 3D
      {0x334,&kSmemForceMult3D<3,3,4,2,3>},
      //{0x346,&kSmemForceMult3D<3,4,6,3,4>},
      //{0x358,&kSmemForceMult3D<3,5,8,4,5>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](NE, B, Bt, Gt, stressJinvT, e, v);
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D, int NBZ =1> static
void kSmemForceMultTranspose2D(const int NE,
                               const Array<double> &_Bt,
                               const Array<double> &_B,
                               const Array<double> &_G,
                               const DenseTensor &_sJit,
                               const Vector &_v,
                               Vector &_e)
{
   MFEM_VERIFY(D1D==H1D,"");
   auto b = Reshape(_B.Read(), Q1D,H1D);
   auto g = Reshape(_G.Read(), Q1D,H1D);
   auto bt = Reshape(_Bt.Read(), L1D,Q1D);
   auto sJit = Reshape(Read(_sJit.GetMemory(), Q1D*Q1D*NE*2*2),
                       Q1D, Q1D, NE, 2, 2);
   auto velocity = Reshape(_v.Read(), D1D,D1D,2,NE);
   auto energy = Reshape(_e.Write(), L1D, L1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED double Bt[L1D][Q1D];
      MFEM_SHARED double B[Q1D][H1D];
      MFEM_SHARED double G[Q1D][H1D];
      MFEM_SHARED double Vz[NBZ][D1D*D1D];
      double (*V)[D1D] = (double (*)[D1D])(Vz + z);
      MFEM_SHARED double DQz[2][NBZ][D1D*Q1D];
      double (*DQ0)[Q1D] = (double (*)[Q1D])(DQz[0] + z);
      double (*DQ1)[Q1D] = (double (*)[Q1D])(DQz[1] + z);
      MFEM_SHARED double QQz[3][NBZ][Q1D*Q1D];
      double (*QQ)[Q1D] = (double (*)[Q1D])(QQz[0] + z);
      double (*QQ0)[Q1D] = (double (*)[Q1D])(QQz[1] + z);
      double (*QQ1)[Q1D] = (double (*)[Q1D])(QQz[2] + z);
      MFEM_SHARED double QLz[NBZ][Q1D*L1D];
      double (*QL)[L1D] = (double (*)[L1D]) (QLz + z);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            QQ[qy][qx] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               V[dx][dy] = velocity(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < H1D; ++dx)
               {
                  const double input = V[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < H1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               QQ0[qy][qx] = u;
               QQ1[qy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double esx = QQ0[qy][qx] * sJit(qx,qy,e,0,c);
               const double esy = QQ1[qy][qx] * sJit(qx,qy,e,1,c);
               QQ[qy][qx] += esx + esy;
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
         {
            double u = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += QQ[qy][qx] * Bt[lx][qx];
            }
            QL[qy][lx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
         {
            double u = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += QL[qy][lx] * Bt[ly][qy];
            }
            energy(lx,ly,e) = u;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D> static
void kSmemForceMultTranspose3D(const int NE,
                               const Array<double> &_Bt,
                               const Array<double> &_B,
                               const Array<double> &_G,
                               const DenseTensor &_sJit,
                               const Vector &_v,
                               Vector &_e)
{
   MFEM_VERIFY(D1D==H1D,"");
   auto b = Reshape(_B.Read(), Q1D,H1D);
   auto g = Reshape(_G.Read(), Q1D,H1D);
   auto bt = Reshape(_Bt.Read(), L1D,Q1D);
   auto sJit = Reshape(Read(_sJit.GetMemory(), Q1D*Q1D*Q1D*NE*3*3),
                       Q1D, Q1D, Q1D, NE, 3, 3);
   auto velocity = Reshape(_v.Read(), D1D, D1D, D1D, 3, NE);
   auto energy = Reshape(_e.Write(), L1D, L1D, L1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED double Bt[L1D][Q1D];
      MFEM_SHARED double B[Q1D][H1D];
      MFEM_SHARED double G[Q1D][H1D];
      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*V)[D1D][D1D]    = (double (*)[D1D][D1D]) (sm0+0);
      double (*MMQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*MMQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+2);
      double (*MQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*MQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*MQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);
      double (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);
      double (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);
      double (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);
      MFEM_SHARED double QQQ[Q1D][Q1D][Q1D];
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               QQQ[qz][qy][qx] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  V[dx][dy][dz] = velocity(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < H1D; ++dx)
                  {
                     const double input = V[dx][dy][dz];
                     u += G[qx][dx] * input;
                     v += B[qx][dx] * input;
                  }
                  MMQ0[dz][dy][qx] = u;
                  MMQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < H1D; ++dy)
                  {
                     u += MMQ0[dz][dy][qx] * B[qy][dy];
                     v += MMQ1[dz][dy][qx] * G[qy][dy];
                     w += MMQ1[dz][dy][qx] * B[qy][dy];
                  }
                  MQQ0[dz][qy][qx] = u;
                  MQQ1[dz][qy][qx] = v;
                  MQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < H1D; ++dz)
                  {
                     u += MQQ0[dz][qy][qx] * B[qz][dz];
                     v += MQQ1[dz][qy][qx] * B[qz][dz];
                     w += MQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  QQQ0[qz][qy][qx] = u;
                  QQQ1[qz][qy][qx] = v;
                  QQQ2[qz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double esx = QQQ0[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const double esy = QQQ1[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const double esz = QQQ2[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ[qz][qy][qx] += esx + esy + esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ[qz][qy][qx] * Bt[lx][qx];
               }
               MQQ0[qz][qy][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += MQQ0[qz][qy][lx] * Bt[ly][qy];
               }
               MMQ0[qz][ly][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += MMQ0[qz][ly][lx] * Bt[lz][qz];
               }
               energy(lx,ly,lz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

typedef void (*fForceMultTranspose)(const int nzones,
                                    const Array<double> &Bt,
                                    const Array<double> &B,
                                    const Array<double> &G,
                                    const DenseTensor &sJit,
                                    const Vector &v,
                                    Vector &e);

static void kForceMultTranspose(const int DIM,
                                const int D1D,
                                const int Q1D,
                                const int L1D,
                                const int H1D,
                                const int nzones,
                                const Array<double> &L2QuadToDof,
                                const Array<double> &H1DofToQuad,
                                const Array<double> &H1DofToQuadD,
                                const DenseTensor &stressJinvT,
                                const Vector &v,
                                Vector &e)
{
   MFEM_VERIFY(D1D==H1D,"D1D!=H1D");
   MFEM_VERIFY(L1D==D1D-1, "L1D!=D1D-1");
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMultTranspose> call =
   {
      {0x234,&kSmemForceMultTranspose2D<2,3,4,2,3>},
      //{0x246,&kSmemForceMultTranspose2D<2,4,6,3,4>},
      //{0x258,&kSmemForceMultTranspose2D<2,5,8,4,5>},
      {0x334,&kSmemForceMultTranspose3D<3,3,4,2,3>},
      //{0x346,&kSmemForceMultTranspose3D<3,4,6,3,4>},
      //{0x358,&kSmemForceMultTranspose3D<3,5,8,4,5>}
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](nzones, L2QuadToDof, H1DofToQuad, H1DofToQuadD, stressJinvT, v, e);
}

class PAForceOperator : public Operator
{
private:
   const int dim, nzones;
   const QuadratureData &quad_data;
   const ParFiniteElementSpace &h1fes, &l2fes;
   const Operator *h1restrict, *l2restrict;
   const IntegrationRule &integ_rule, &ir1D;
   const int D1D, Q1D;
   const int L1D, H1D;
   const int h1sz, l2sz;
   const DofToQuad *l2D2Q, *h1D2Q;
   mutable Vector gVecL2, gVecH1;
public:
   PAForceOperator(const QuadratureData &qd,
                   const ParFiniteElementSpace &h1f,
                   const ParFiniteElementSpace &l2f,
                   const IntegrationRule &ir) :
      dim(h1f.GetMesh()->Dimension()),
      nzones(h1f.GetMesh()->GetNE()),
      quad_data(qd),
      h1fes(h1f),
      l2fes(l2f),
      h1restrict(h1f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      l2restrict(l2f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      integ_rule(ir),
      ir1D(IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder())),
      D1D(h1fes.GetFE(0)->GetOrder()+1),
      Q1D(ir1D.GetNPoints()),
      L1D(l2fes.GetFE(0)->GetOrder()+1),
      H1D(h1fes.GetFE(0)->GetOrder()+1),
      h1sz(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones),
      l2sz(l2fes.GetFE(0)->GetDof() * nzones),
      l2D2Q(&l2fes.GetFE(0)->GetDofToQuad(integ_rule, DofToQuad::TENSOR)),
      h1D2Q(&h1fes.GetFE(0)->GetDofToQuad(integ_rule, DofToQuad::TENSOR)),
      gVecL2(l2sz),
      gVecH1(h1sz)
   {
      MFEM_ASSERT(h1f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC),"");
      MFEM_ASSERT(l2f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC),"");
      gVecL2.SetSize(l2sz);
      gVecH1.SetSize(h1sz);
   }

   void Mult(const Vector &x, Vector &y) const
   {

      l2restrict->Mult(x, gVecL2);
      kForceMult(dim, D1D, Q1D, L1D, H1D, nzones,
                 l2D2Q->B, h1D2Q->Bt, h1D2Q->Gt, quad_data.stressJinvT,
                 gVecL2, gVecH1);
      h1restrict->MultTranspose(gVecH1, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const
   {
      h1restrict->Mult(x, gVecH1);
      kForceMultTranspose(dim, D1D, Q1D, L1D, H1D, nzones,
                          l2D2Q->Bt, h1D2Q->B, h1D2Q->G,
                          quad_data.stressJinvT,
                          gVecH1, gVecL2);
      l2restrict->MultTranspose(gVecL2, y);
   }
};

static void ComputeDiagonal2D(const int height, const int nzones,
                              const QuadratureData &quad_data,
                              const FiniteElementSpace &FESpace,
                              const Tensors1D *tensors1D,
                              Vector &diag)
{
   const TensorBasisElement *fe_H1 =
      dynamic_cast<const TensorBasisElement *>(FESpace.GetFE(0));
   const Array<int> &dof_map = fe_H1->GetDofMap();
   const DenseMatrix &HQs = tensors1D->HQshape1D;
   const int ndof1D = HQs.Height(), nqp1D = HQs.Width(), nqp = nqp1D * nqp1D;
   Vector dz(ndof1D * ndof1D);
   DenseMatrix HQ(ndof1D, nqp1D), D(dz.GetData(), ndof1D, ndof1D);
   Array<int> dofs;
   diag.SetSize(height);
   diag = 0.0;
   DenseMatrix HQs_sq(ndof1D, nqp1D);
   for (int i = 0; i < ndof1D; i++)
      for (int k = 0; k < nqp1D; k++)
      {
         HQs_sq(i, k) = HQs(i, k) * HQs(i, k);
      }
   for (int z = 0; z < nzones; z++)
   {
      DenseMatrix QQ(quad_data.rho0DetJ0w.GetData() + z*nqp, nqp1D, nqp1D);
      mfem::Mult(HQs_sq, QQ, HQ);
      MultABt(HQ, HQs_sq, D);
      FESpace.GetElementDofs(z, dofs);
      for (int j = 0; j < dz.Size(); j++)
      {
         diag[dofs[dof_map[j]]] += dz[j];
      }
   }
}

static void ComputeDiagonal3D(const int height, const int nzones,
                              const QuadratureData &quad_data,
                              const FiniteElementSpace &FESpace,
                              const Tensors1D *tensors1D,
                              Vector &diag)
{
   const TensorBasisElement *fe_H1 =
      dynamic_cast<const TensorBasisElement *>(FESpace.GetFE(0));
   const Array<int> &dof_map = fe_H1->GetDofMap();
   const DenseMatrix &HQs = tensors1D->HQshape1D;
   const int ndof1D = HQs.Height(), nqp1D = HQs.Width(),
             nqp = nqp1D * nqp1D * nqp1D;
   DenseMatrix HH_Q(ndof1D * ndof1D, nqp1D), Q_HQ(nqp1D, ndof1D*nqp1D);
   DenseMatrix H_HQ(HH_Q.GetData(), ndof1D, ndof1D*nqp1D);
   Vector dz(ndof1D * ndof1D * ndof1D);
   DenseMatrix D(dz.GetData(), ndof1D*ndof1D, ndof1D);
   Array<int> dofs;
   diag.SetSize(height);
   diag = 0.0;
   DenseMatrix HQs_sq(ndof1D, nqp1D);
   for (int i = 0; i < ndof1D; i++)
      for (int k = 0; k < nqp1D; k++)
      {
         HQs_sq(i, k) = HQs(i, k) * HQs(i, k);
      }
   for (int z = 0; z < nzones; z++)
   {
      DenseMatrix QQ_Q(quad_data.rho0DetJ0w.GetData() + z*nqp,
                       nqp1D * nqp1D, nqp1D);
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int i2 = 0; i2 < ndof1D; i2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               Q_HQ(k1, i2 + ndof1D*k3) = 0.0;
               for (int k2 = 0; k2 < nqp1D; k2++)
               {
                  Q_HQ(k1, i2 + ndof1D*k3) +=
                     QQ_Q(k1 + nqp1D*k2, k3) * HQs_sq(i2, k2);
               }
            }
         }
      }
      mfem::Mult(HQs_sq, Q_HQ, H_HQ);
      MultABt(HH_Q, HQs_sq, D);
      FESpace.GetElementDofs(z, dofs);
      for (int j = 0; j < dz.Size(); j++)
      {
         diag[dofs[dof_map[j]]] += dz[j];
      }
   }
}

class PAMassOperator : public Operator
{
private:
#ifdef MFEM_USE_MPI
   const MPI_Comm comm;
#endif
   const int dim, nzones;
   const QuadratureData &quad_data;
   FiniteElementSpace &FESpace;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr massOperator;
   Tensors1D *tensors1D;
public:
   PAMassOperator(Coefficient &Q,
                  const QuadratureData &qd,
                  ParFiniteElementSpace &pfes,
                  const IntegrationRule &ir,
                  Tensors1D *t1D) :
      Operator(pfes.GetTrueVSize()),
#ifdef MFEM_USE_MPI
      comm(PFesGetParMeshGetComm0(pfes)),
#endif
      dim(pfes.GetMesh()->Dimension()),
      nzones(pfes.GetMesh()->GetNE()),
      quad_data(qd),
      FESpace(pfes),
      pabf(&pfes),
      ess_tdofs_count(0),
      ess_tdofs(0),
      tensors1D(t1D)
   {
      pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q,&ir));
      pabf.Assemble();
      pabf.FormSystemMatrix(mfem::Array<int>(), massOperator);
   }

   void Mult(const Vector &x, Vector &y) const
   {
      ParGridFunction X;
      X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
      if (ess_tdofs_count) { X.SetSubVector(ess_tdofs, 0.0); }
      massOperator->Mult(X, y);
      if (ess_tdofs_count) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   void ComputeDiagonal2D(Vector &diag) const
   {
      return hydrodynamics::ComputeDiagonal2D(FESpace.GetVSize(), nzones,
                                              quad_data, FESpace, tensors1D,
                                              diag);
   }
   void ComputeDiagonal3D(Vector &diag) const
   {
      return hydrodynamics::ComputeDiagonal3D(FESpace.GetVSize(), nzones,
                                              quad_data, FESpace, tensors1D,
                                              diag);
   }

   const Operator *GetProlongation() const
   { return FESpace.GetProlongationMatrix(); }

   const Operator *GetRestriction() const
   { return FESpace.GetRestrictionMatrix(); }

   void SetEssentialTrueDofs(Array<int> &dofs)
   {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs.Size()==0)
      {
         int global_ess_tdofs_count;
         MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                       1, MPI_INT, MPI_SUM, comm);
         MFEM_VERIFY(global_ess_tdofs_count>0, "!(global_ess_tdofs_count>0)");
         ess_tdofs.SetSize(global_ess_tdofs_count);
      }
      if (ess_tdofs_count == 0)
      {
         return;
      }
      ess_tdofs = dofs;
   }

   void EliminateRHS(Vector &b) const
   {
      if (ess_tdofs_count > 0)
      {
         b.SetSubVector(ess_tdofs, 0.0);
      }
   }
};

class DiagonalSolver : public Solver
{
private:
   Vector diag;
   FiniteElementSpace &FESpace;
public:
   DiagonalSolver(FiniteElementSpace &fes)
      : Solver(fes.GetVSize()), diag(), FESpace(fes) { }
   void SetDiagonal(Vector &d)
   {
      const Operator *P = FESpace.GetProlongationMatrix();
      if (P == NULL) { diag = d; return; }
      diag.SetSize(P->Width());
      P->MultTranspose(d, diag);
   }
   void Mult(const Vector &x, Vector &y) const
   {
      const int N = x.Size();
      auto d_diag = diag.Read();
      auto d_x = x.Read();
      auto d_y = y.Write();
      MFEM_FORALL(i, N, d_y[i] = d_x[i] / d_diag[i];);
   }
   void SetOperator(const Operator &op) { }
};

struct TimingData
{
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;
   const HYPRE_Int L2dof;
   HYPRE_Int H1iter, L2iter, quad_tstep;
   TimingData(const HYPRE_Int l2d) :
      L2dof(l2d), H1iter(0), L2iter(0), quad_tstep(0) { }
};

class QUpdate
{
private:
   const int dim, NQ, NE;
   const bool use_viscosity;
   const double cfl, gamma;
   TimingData *timer;
   const IntegrationRule &ir;
   ParFiniteElementSpace &H1, &L2;
   const Operator *H1ER;
   const int vdim;
   Vector d_dt_est;
   Vector d_l2_e_quads_data;
   Vector d_h1_v_local_in, d_h1_grad_x_data, d_h1_grad_v_data;
   const QuadratureInterpolator *q1,*q2;
public:
   QUpdate(const int d, const int ne, const bool uv,
           const double c, const double g, TimingData *t,
           const IntegrationRule &i,
           ParFiniteElementSpace &h1, ParFiniteElementSpace &l2):
      dim(d), NQ(i.GetNPoints()), NE(ne), use_viscosity(uv), cfl(c), gamma(g),
      timer(t), ir(i), H1(h1), L2(l2),
      H1ER(H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      vdim(H1.GetVDim()),
      d_dt_est(NE*NQ),
      d_l2_e_quads_data(NE*NQ),
      d_h1_v_local_in(NQ*NE*vdim),
      d_h1_grad_x_data(NQ*NE*vdim*vdim),
      d_h1_grad_v_data(NQ*NE*vdim*vdim),
      q1(H1.GetQuadratureInterpolator(ir)),
      q2(L2.GetQuadratureInterpolator(ir)) { }

   void UpdateQuadratureData(const Vector &S,
                             bool &quad_data_is_current,
                             QuadratureData &quad_data,
                             const Tensors1D *tensors1D);
};

void ComputeRho0DetJ0AndVolume(const int dim,
                               const int NE,
                               const IntegrationRule &ir,
                               ParMesh *mesh,
                               ParFiniteElementSpace &l2_fes,
                               ParGridFunction &rho0,
                               QuadratureData &quad_data,
                               double &loc_area)
{
   const int NQ = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(ir, flags);
   Vector rho0Q(NQ*NE);
   rho0Q.UseDevice(true);
   Vector j, detj;
   const QuadratureInterpolator *qi = l2_fes.GetQuadratureInterpolator(ir);
   qi->Mult(rho0, QuadratureInterpolator::VALUES, rho0Q, j, detj);
   auto W = ir.GetWeights().Read();
   auto R = Reshape(rho0Q.Read(), NQ, NE);
   auto J = Reshape(geom->J.Read(), NQ, dim, dim, NE);
   auto detJ = Reshape(geom->detJ.Read(), NQ, NE);
   auto V = Reshape(quad_data.rho0DetJ0w.Write(), NQ, NE);
   Memory<double> &Jinv_m = quad_data.Jac0inv.GetMemory();
   auto invJ = Reshape(Jinv_m.Write(Device::GetDeviceMemoryClass(),
                                    quad_data.Jac0inv.TotalSize()),
                       dim, dim, NQ, NE);
   Vector area(NE*NQ), one(NE*NQ);
   auto A = Reshape(area.Write(), NQ, NE);
   auto O = Reshape(one.Write(), NQ, NE);
   if (dim==2)
   {
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + qy * Q1D;
               const double J11 = J(q,0,0,e);
               const double J12 = J(q,1,0,e);
               const double J21 = J(q,0,1,e);
               const double J22 = J(q,1,1,e);
               const double det = detJ(q,e);
               V(q,e) =  W[q] * R(q,e) * det;
               const double r_idetJ = 1.0 / det;
               invJ(0,0,q,e) =  J22 * r_idetJ;
               invJ(1,0,q,e) = -J12 * r_idetJ;
               invJ(0,1,q,e) = -J21 * r_idetJ;
               invJ(1,1,q,e) =  J11 * r_idetJ;
               A(q,e) = W[q] * det;
               O(q,e) = 1.0;
            }
         }
      });
   }
   else
   {
      MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const double J11 = J(q,0,0,e), J12 = J(q,0,1,e), J13 = J(q,0,2,e);
                  const double J21 = J(q,1,0,e), J22 = J(q,1,1,e), J23 = J(q,1,2,e);
                  const double J31 = J(q,2,0,e), J32 = J(q,2,1,e), J33 = J(q,2,2,e);
                  const double det = detJ(q,e);
                  V(q,e) = W[q] * R(q,e) * det;
                  const double r_idetJ = 1.0 / det;
                  invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
                  invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
                  invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
                  invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
                  invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
                  invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
                  invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
                  invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
                  invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
                  A(q,e) = W[q] * det;
                  O(q,e) = 1.0;
               }
            }
         }
      });
   }
   quad_data.rho0DetJ0w.HostRead();
   loc_area = area * one;
}

class TaylorCoefficient : public Coefficient
{
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};

template<int D1D, int Q1D, int NBZ> static inline
void D2QValues2D(const int NE, const Array<double> &b_,
                 const Vector &x_, Vector &y_)
{
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int zid = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double DDz[NBZ][D1D*D1D];
      double (*DD)[D1D] = (double (*)[D1D])(DDz + zid);
      MFEM_SHARED double DQz[NBZ][D1D*Q1D];
      double (*DQ)[Q1D] = (double (*)[Q1D])(DQz + zid);
      if (zid == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            DD[dy][dx] = x(dx,dy,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double dq = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               dq += B[qx][dx] * DD[dy][dx];
            }
            DQ[dy][qx] = dq;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double qq = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               qq += DQ[dy][qx] * B[qy][dy];
            }
            y(qx,qy,e) = qq;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int D1D, int Q1D> static inline
void D2QValues3D(const int NE, const Array<double> &b_,
                 const Vector &x_, Vector &y_)
{
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, Q1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double sm0[Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[Q1D*Q1D*Q1D];
      double (*X)[D1D][D1D]   = (double (*)[D1D][D1D]) sm0;
      double (*DDQ)[D1D][Q1D] = (double (*)[D1D][Q1D]) sm1;
      double (*DQQ)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) sm0;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  u += B[qx][dx] * X[dz][dy][dx];
               }
               DDQ[dz][dy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ[dz][dy][qx] * B[qy][dy];
               }
               DQQ[dz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ[dz][qy][qx] * B[qz][dz];
               }
               y(qx,qy,qz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

typedef void (*fD2QValues)(const int NE,
                           const Array<double> &B,
                           const Vector &e_vec,
                           Vector &q_val);

static void D2QValues(const FiniteElementSpace &fes,
                      const DofToQuad *maps,
                      const IntegrationRule& ir,
                      const Vector &e_vec,
                      Vector &q_val)
{
   const int dim = fes.GetMesh()->Dimension();
   const int nzones = fes.GetNE();
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int id = (dim<<8)|(dofs1D<<4)|(quad1D);
   static std::unordered_map<int, fD2QValues> call =
   {
      // 2D
      {0x224,&D2QValues2D<2,4,8>},
      //{0x236,&D2QValues2D<3,6,4>}, {0x248,&D2QValues2D<4,8,2>},
      // 3D
      {0x324,&D2QValues3D<2,4>},
      //{0x336,&D2QValues3D<3,6>}, {0x348,&D2QValues3D<4,8>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](nzones, maps->B, e_vec, q_val);
}

void Values(FiniteElementSpace *fespace, const IntegrationRule &ir,
            const Vector &e_vec, Vector &q_val)
{
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QValues(*fespace, &d2q,ir, e_vec, q_val);
}

template<int D1D, int Q1D, int NBZ> static inline
void D2QGrad2D(const int NE,
               const Array<double> &b_,
               const Array<double> &g_,
               const Vector &x_,
               Vector &y_)
{
   constexpr int VDIM = 2;
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, VDIM, Q1D, Q1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];
      MFEM_SHARED double Xz[NBZ][D1D][D1D];
      double (*X)[D1D] = (double (*)[D1D])(Xz + tidz);
      MFEM_SHARED double GD[2][NBZ][D1D][Q1D];
      double (*DQ0)[Q1D] = (double (*)[Q1D])(GD[0] + tidz);
      double (*DQ1)[Q1D] = (double (*)[Q1D])(GD[1] + tidz);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               X[dx][dy] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int D1D, int Q1D> static inline
void D2QGrad3D(const int NE,
               const Array<double> &b_, const Array<double> &g_,
               const Vector &x_, Vector &y_)
{
   constexpr int VDIM = 3;
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, VDIM, Q1D, Q1D, Q1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];
      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*X)[D1D][D1D]    = (double (*)[D1D][D1D]) (sm0+2);
      double (*DDQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+0);
      double (*DDQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*DQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*DQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*DQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {

                  X[dx][dy][dz] = x(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double coords = X[dx][dy][dz];
                     u += coords * B[qx][dx];
                     v += coords * G[qx][dx];
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ1[dz][dy][qx] * B[qy][dy];
                     v += DDQ0[dz][dy][qx] * G[qy][dy];
                     w += DDQ0[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ0[dz][qy][qx] * B[qz][dz];
                     v += DQQ1[dz][qy][qx] * B[qz][dz];
                     w += DQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  y(c,0,qx,qy,qz,e) = u;
                  y(c,1,qx,qy,qz,e) = v;
                  y(c,2,qx,qy,qz,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

typedef void (*fD2QGrad)(const int NE,
                         const Array<double> &B,
                         const Array<double> &G,
                         const Vector &e_vec,
                         Vector &q_der);

static void D2QGrad(const FiniteElementSpace &fes,
                    const DofToQuad *maps,
                    const IntegrationRule& ir,
                    const Vector &e_vec,
                    Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int D1D = fes.GetFE(0)->GetOrder() + 1;
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int id = (dim<<8)|(D1D<<4)|(Q1D);
   static std::unordered_map<int, fD2QGrad> call =
   {
      // 2D
      {0x234,&D2QGrad2D<3,4,8>},
      //{0x246,&D2QGrad2D<4,6,4>},{0x258,&D2QGrad2D<5,8,2>},
      // 3D
      {0x334,&D2QGrad3D<3,4>},
      //{0x346,&D2QGrad3D<4,6>},{0x358,&D2QGrad3D<5,8>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](NE, maps->B, maps->G, e_vec, q_der);
}

void Derivatives(FiniteElementSpace *fespace,  const IntegrationRule &ir,
                 const Vector &e_vec, Vector &q_der)
{
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QGrad(*fespace, &d2q, ir, e_vec, q_der);
}

MFEM_HOST_DEVICE inline double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

template<int dim> MFEM_HOST_DEVICE static inline
void QBody(const int nzones, const int z,
           const int nqp, const int q,
           const double gamma,
           const bool use_viscosity,
           const double h0,
           const double h1order,
           const double cfl,
           const double infinity,
           double *Jinv,
           double *stress,
           double *sgrad_v,
           double *eig_val_data,
           double *eig_vec_data,
           double *compr_dir,
           double *Jpi,
           double *ph_dir,
           double *stressJiT,
           const double *d_weights,
           const double *d_Jacobians,
           const double *d_rho0DetJ0w,
           const double *d_e_quads,
           const double *d_grad_v_ext,
           const double *d_Jac0inv,
           double *d_dt_est,
           double *d_stressJinvT)
{
   constexpr int dim2 = dim*dim;
   double min_detJ = infinity;
   const int zq = z * nqp + q;
   const double weight =  d_weights[q];
   const double inv_weight = 1. / weight;
   const double *J = d_Jacobians + dim2*(nqp*z + q);
   const double detJ = mfem::blas::Det<dim>(J);
   min_detJ = std::fmin(min_detJ,detJ);
   blas::CalcInverse<dim>(J,Jinv);
   const double rho = inv_weight * d_rho0DetJ0w[zq] / detJ;
   const double e   = std::fmax(0.0, d_e_quads[zq]);
   const double p   = (gamma - 1.0) * rho * e;
   const double sound_speed = std::sqrt(gamma * (gamma-1.0) * e);
   for (int k = 0; k < dim2; k+=1) { stress[k] = 0.0; }
   for (int d = 0; d < dim; d++) { stress[d*dim+d] = -p; }
   double visc_coeff = 0.0;
   if (use_viscosity)
   {
      const double *dV = d_grad_v_ext + dim2*(nqp*z + q);
      blas::Mult(dim, dim, dim, dV, Jinv, sgrad_v);
      symmetrize(dim,sgrad_v);
      if (dim==1)
      {
         eig_val_data[0] = sgrad_v[0];
         eig_vec_data[0] = 1.;
      }
      else
      {
         blas::CalcEigenvalues<dim>(sgrad_v, eig_val_data, eig_vec_data);
      }
      for (int k=0; k<dim; k+=1) { compr_dir[k]=eig_vec_data[k]; }
      blas::Mult(dim, dim, dim, J, d_Jac0inv+zq*dim*dim, Jpi);
      blas::MultV(dim, dim, Jpi, compr_dir, ph_dir);
      const double ph_dir_nl2 = norml2(dim,ph_dir);
      const double compr_dir_nl2 = norml2(dim, compr_dir);
      const double h = h0 * ph_dir_nl2 / compr_dir_nl2;
      const double mu = eig_val_data[0];
      visc_coeff = 2.0 * rho * h * h * std::fabs(mu);
      const double eps = 1e-12;
      visc_coeff += 0.5 * rho * h * sound_speed *
                    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
      blas::Add(dim, dim, visc_coeff, stress, sgrad_v, stress);
   }
   const double sv = blas::CalcSingularvalue<dim>(J);
   const double h_min = sv / h1order;
   const double inv_h_min = 1. / h_min;
   const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
   const double inv_dt = sound_speed * inv_h_min
                         + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
   if (min_detJ < 0.0)
   {
      d_dt_est[zq] = 0.0;
   }
   else
   {
      if (inv_dt>0.0)
      {
         const double cfl_inv_dt = cfl / inv_dt;
         d_dt_est[zq] = std::fmin(d_dt_est[zq], cfl_inv_dt);
      }
   }
   mfem::multABt(dim, dim, dim, stress, Jinv, stressJiT);
   for (int k=0; k<dim2; k+=1) { stressJiT[k] *= weight * detJ; }
   for (int vd = 0 ; vd < dim; vd++)
   {
      for (int gd = 0; gd < dim; gd++)
      {
         const int offset = zq + nqp*nzones*(gd+vd*dim);
         d_stressJinvT[offset] = stressJiT[vd+gd*dim];
      }
   }
}

template<int dim, int Q1D> static inline
void QKernel(const int nzones,
             const int nqp,
             const int nqp1D,
             const double gamma,
             const bool use_viscosity,
             const double h0,
             const double h1order,
             const double cfl,
             const double infinity,
             const Array<double> &weights,
             const Vector &Jacobians,
             const Vector &rho0DetJ0w,
             const Vector &e_quads,
             const Vector &grad_v_ext,
             const DenseTensor &Jac0inv,
             Vector &dt_est,
             DenseTensor &stressJinvT)
{
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   if (dim==2)
   {
      MFEM_FORALL_2D(z, nzones, Q1D, Q1D, 1,
      {
         constexpr int DIM = dim;
         constexpr int DIM2 = dim*dim;
         double Jinv[DIM2];
         double stress[DIM2];
         double sgrad_v[DIM2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[DIM];
         double Jpi[DIM2];
         double ph_dir[DIM];
         double stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               QBody<dim>(nzones, z, nqp, qx + qy * Q1D,
               gamma, use_viscosity, h0, h1order, cfl, infinity,
               Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
               compr_dir,Jpi,ph_dir,stressJiT,
               d_weights, d_Jacobians, d_rho0DetJ0w,
               d_e_quads, d_grad_v_ext, d_Jac0inv,
               d_dt_est, d_stressJinvT);
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
   if (dim==3)
   {
      MFEM_FORALL_3D(z, nzones, Q1D, Q1D, Q1D,
      {
         constexpr int DIM = dim;
         constexpr int DIM2 = dim*dim;
         double Jinv[DIM2];
         double stress[DIM2];
         double sgrad_v[DIM2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[DIM];
         double Jpi[DIM2];
         double ph_dir[DIM];
         double stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  QBody<dim>(nzones, z, nqp, qx + Q1D * (qy + qz * Q1D),
                  gamma, use_viscosity, h0, h1order, cfl, infinity,
                  Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
                  compr_dir,Jpi,ph_dir,stressJiT,
                  d_weights, d_Jacobians, d_rho0DetJ0w,
                  d_e_quads, d_grad_v_ext, d_Jac0inv,
                  d_dt_est, d_stressJinvT);
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
}

void QUpdate::UpdateQuadratureData(const Vector &S,
                                   bool &quad_data_is_current,
                                   QuadratureData &quad_data,
                                   const Tensors1D *tensors1D)
{
   if (quad_data_is_current) { return; }
   timer->sw_qdata.Start();
   Vector* S_p = const_cast<Vector*>(&S);
   const int H1_size = H1.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();
   const double h1order = (double) H1.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();
   GridFunction d_x, d_v, d_e;
   d_x.MakeRef(&H1,*S_p, 0);
   H1ER->Mult(d_x, d_h1_v_local_in);
   Derivatives(&H1, ir, d_h1_v_local_in, d_h1_grad_x_data);
   d_v.MakeRef(&H1,*S_p, H1_size);
   H1ER->Mult(d_v, d_h1_v_local_in);
   Derivatives(&H1, ir, d_h1_v_local_in, d_h1_grad_v_data);
   d_e.MakeRef(&L2, *S_p, 2*H1_size);
   Values(&L2, ir, d_e, d_l2_e_quads_data);
   d_dt_est = quad_data.dt_est;
   const int id = (dim<<4) | nqp1D;
   typedef void (*fQKernel)(const int NE, const int NQ, const int Q1D,
                            const double gamma, const bool use_viscosity,
                            const double h0, const double h1order,
                            const double cfl, const double infinity,
                            const Array<double> &weights,
                            const Vector &Jacobians, const Vector &rho0DetJ0w,
                            const Vector &e_quads, const Vector &grad_v_ext,
                            const DenseTensor &Jac0inv,
                            Vector &dt_est, DenseTensor &stressJinvT);
   static std::unordered_map<int, fQKernel> qupdate =
   {
      {0x24,&QKernel<2,4>}, //{0x26,&QKernel<2,6>}, {0x28,&QKernel<2,8>},
      {0x34,&QKernel<3,4>}, //{0x36,&QKernel<3,6>}, {0x38,&QKernel<3,8>}
   };
   if (!qupdate[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   qupdate[id](NE, NQ, nqp1D, gamma, use_viscosity, quad_data.h0,
               h1order, cfl, infinity, ir.GetWeights(), d_h1_grad_x_data,
               quad_data.rho0DetJ0w, d_l2_e_quads_data, d_h1_grad_v_data,
               quad_data.Jac0inv, d_dt_est, quad_data.stressJinvT);
   quad_data.dt_est = d_dt_est.Min();
   quad_data_is_current = true;
   timer->sw_qdata.Stop();
   timer->quad_tstep += NE;
}

class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace, &L2FESpace;
   mutable ParFiniteElementSpace H1compFESpace;
   const int H1Vsize;
   const int H1TVSize;
   const HYPRE_Int H1GTVSize;
   const int H1compTVSize;
   const int L2Vsize;
   const int L2TVSize;
   const HYPRE_Int L2GTVSize;
   Array<int> block_offsets;
   mutable ParGridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int dim, nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const double cfl;
   const bool use_viscosity;
   const double cg_rel_tol;
   const int cg_max_iter;
   const double ftz_tol;
   Coefficient *material_pcf;
   mutable ParBilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   const IntegrationRule &integ_rule;
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current, forcemat_is_assembled;
   Tensors1D T1D;
   mutable MixedBilinearForm Force;
   PAForceOperator *ForcePA;
   PAMassOperator *VMassPA, *EMassPA;
   mutable DiagonalSolver VMassPA_prec;
   CGSolver CG_VMass, CG_EMass, locCG;
   mutable TimingData timer;
   const double gamma;
   mutable QUpdate Q;
   mutable Vector X, B, one, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   void UpdateQuadratureData(const Vector &S) const
   {
      return Q.UpdateQuadratureData(S, quad_data_is_current, quad_data, &T1D);
   }

public:
   LagrangianHydroOperator(Coefficient &rho_coeff,
                           const int size,
                           ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &essential_tdofs,
                           ParGridFunction &rho0,
                           const int source_type_,
                           const double cfl_,
                           Coefficient *material_,
                           const bool visc,
                           const double cgt,
                           const int cgiter,
                           double ftz,
                           const int order_q,
                           const double gm,
                           int h1_basis_type):
      TimeDependentOperator(size),
      H1FESpace(h1_fes), L2FESpace(l2_fes),
      H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(), 1),
      H1Vsize(H1FESpace.GetVSize()),
      H1TVSize(H1FESpace.GetTrueVSize()),
      H1GTVSize(H1FESpace.GlobalTrueVSize()),
      H1compTVSize(H1compFESpace.GetTrueVSize()),
      L2Vsize(L2FESpace.GetVSize()),
      L2TVSize(L2FESpace.GetTrueVSize()),
      L2GTVSize(L2FESpace.GlobalTrueVSize()),
      block_offsets(4),
      x_gf(&H1FESpace),
      ess_tdofs(essential_tdofs),
      dim(h1_fes.GetMesh()->Dimension()),
      nzones(h1_fes.GetMesh()->GetNE()),
      l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
      h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
      source_type(source_type_), cfl(cfl_),
      use_viscosity(visc),
      cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
      material_pcf(material_),
      Mv(&h1_fes), Mv_spmat_copy(),
      Me(l2dofs_cnt, l2dofs_cnt, nzones),
      Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
      integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(0),
                              (order_q>0)? order_q :
                              3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
      quad_data(dim, nzones, integ_rule.GetNPoints()),
      quad_data_is_current(false), forcemat_is_assembled(false),
      T1D(H1FESpace.GetFE(0)->GetOrder(), L2FESpace.GetFE(0)->GetOrder(),
          int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))),
          h1_basis_type == BasisType::Positive),
      Force(&l2_fes, &h1_fes),
      VMassPA_prec(H1compFESpace),
      CG_VMass(PFesGetParMeshGetComm(H1FESpace)),
      CG_EMass(PFesGetParMeshGetComm(L2FESpace)),
      locCG(),
      timer(L2TVSize),
      gamma(gm),
      Q(dim, nzones, use_viscosity, cfl, gamma,
        &timer, integ_rule, H1FESpace, L2FESpace),
      X(H1compFESpace.GetTrueVSize()),
      B(H1compFESpace.GetTrueVSize()),
      one(L2Vsize),
      rhs(H1Vsize),
      e_rhs(L2Vsize),
      rhs_c_gf(&H1compFESpace),
      dvc_gf(&H1compFESpace)
   {
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + H1Vsize;
      block_offsets[2] = block_offsets[1] + H1Vsize;
      block_offsets[3] = block_offsets[2] + L2Vsize;
      one.UseDevice(true);
      one = 1.0;
      ForcePA = new PAForceOperator(quad_data, h1_fes,l2_fes, integ_rule);
      VMassPA = new PAMassOperator(rho_coeff, quad_data, H1compFESpace,
                                   integ_rule, &T1D);
      EMassPA = new PAMassOperator(rho_coeff, quad_data, L2FESpace,
                                   integ_rule, &T1D);
      H1FESpace.GetParMesh()->GetNodes()->ReadWrite();
      const int bdr_attr_max = H1FESpace.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < dim; c++)
      {
         ess_bdr = 0; ess_bdr[c] = 1;
         H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
      GridFunctionCoefficient rho_coeff_gf(&rho0);
      double loc_area = 0.0, glob_area;
      int loc_z_cnt = nzones, glob_z_cnt;
      ParMesh *pm = H1FESpace.GetParMesh();
      ComputeRho0DetJ0AndVolume(dim, nzones, integ_rule,
                                H1FESpace.GetParMesh(),
                                l2_fes, rho0, quad_data, loc_area);
      MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
      MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
      switch (pm->GetElementBaseGeometry(0))
      {
         case Geometry::SQUARE:
            quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
         case Geometry::CUBE:
            quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
         default: MFEM_ABORT("Unknown zone type!");
      }
      quad_data.h0 /= (double) H1FESpace.GetOrder(0);
      {
         Vector d;
         (dim == 2) ? VMassPA->ComputeDiagonal2D(d) : VMassPA->ComputeDiagonal3D(d);
         VMassPA_prec.SetDiagonal(d);
      }
      CG_VMass.SetPreconditioner(VMassPA_prec);
      CG_VMass.SetOperator(*VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(0);

      CG_EMass.SetOperator(*EMassPA);
      CG_EMass.iterative_mode = false;
      CG_EMass.SetRelTol(1e-8);
      CG_EMass.SetAbsTol(1e-8 * std::numeric_limits<double>::epsilon());
      CG_EMass.SetMaxIter(200);
      CG_EMass.SetPrintLevel(-1);
   }

   ~LagrangianHydroOperator()
   {
      delete EMassPA;
      delete VMassPA;
      delete ForcePA;
   }

   virtual void Mult(const Vector &S, Vector &dS_dt) const
   {
      UpdateMesh(S);
      Vector* sptr = const_cast<Vector*>(&S);
      ParGridFunction v;
      const int VsizeH1 = H1FESpace.GetVSize();
      v.MakeRef(&H1FESpace, *sptr, VsizeH1);
      ParGridFunction dx;
      dx.MakeRef(&H1FESpace, dS_dt, 0);
      dx = v;
      SolveVelocity(S, dS_dt);
      SolveEnergy(S, v, dS_dt);
      quad_data_is_current = false;
   }

   MemoryClass GetMemoryClass() const  { return Device::GetDeviceMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      ParGridFunction dv;
      dv.MakeRef(&H1FESpace, dS_dt, H1Vsize);
      dv = 0.0;
      timer.sw_force.Start();
      ForcePA->Mult(one, rhs);
      if (ftz_tol>0.0)
      {
         for (int i = 0; i < H1Vsize; i++)
         {
            if (fabs(rhs[i]) < ftz_tol)
            {
               rhs[i] = 0.0;
            }
         }
      }
      timer.sw_force.Stop();
      rhs.Neg();
      const int size = H1compFESpace.GetVSize();
      const Operator *Pconf = H1compFESpace.GetProlongationMatrix();
      const Operator *Rconf = H1compFESpace.GetRestrictionMatrix();
      PAMassOperator *kVMassPA = VMassPA;
      for (int c = 0; c < dim; c++)
      {
         dvc_gf.MakeRef(&H1compFESpace, dS_dt, H1Vsize + c*size);
         rhs_c_gf.MakeRef(&H1compFESpace, rhs, c*size);
         if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
         else { B = rhs_c_gf; }
         if (Rconf) { Rconf->Mult(dvc_gf, X); }
         else { X = dvc_gf; }
         kVMassPA->SetEssentialTrueDofs(c_tdofs[c]);
         kVMassPA->EliminateRHS(B);
         timer.sw_cgH1.Start();
         CG_VMass.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1iter += CG_VMass.GetNumIterations();
         if (Pconf) { Pconf->Mult(X, dvc_gf); }
         else { dvc_gf = X; }
         dvc_gf.GetMemory().SyncAlias(dS_dt.GetMemory(), dvc_gf.Size());
      }
   }

   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      ParGridFunction de;
      de.MakeRef(&L2FESpace, dS_dt, H1Vsize*2);
      de = 0.0;
      LinearForm *e_source = NULL;
      MFEM_VERIFY(source_type!=1,"");
      Array<int> l2dofs;
      timer.sw_force.Start();
      ForcePA->MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      timer.sw_cgL2.Start();
      CG_EMass.Mult(e_rhs, de);
      timer.sw_cgL2.Stop();
      const int cg_num_iter = CG_EMass.GetNumIterations();
      timer.L2iter += (cg_num_iter==0) ? 1 : cg_num_iter;
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
      delete e_source;
   }

   void UpdateMesh(const Vector &S) const
   {
      Vector* sptr = const_cast<Vector*>(&S);
      x_gf.MakeRef(&H1FESpace, *sptr, 0);
      H1FESpace.GetParMesh()->NewNodes(x_gf, false);
   }

   double GetTimeStepEstimate(const Vector &S) const
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);
      double glob_dt_est;
      MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                    H1FESpace.GetParMesh()->GetComm());
      return glob_dt_est;
   }

   void ResetTimeStepEstimate() const
   {
      quad_data.dt_est = std::numeric_limits<double>::infinity();
   }

   void ResetQuadratureData() const { quad_data_is_current = false; }

   void ComputeDensity(ParGridFunction &rho) const
   {
      rho.SetSpace(&L2FESpace);
      DenseMatrix Mrho(l2dofs_cnt);
      Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
      Array<int> dofs(l2dofs_cnt);
      for (int i = 0; i < nzones; i++)
      {
         L2FESpace.GetElementDofs(i, dofs);
         rho.SetSubVector(dofs, rho_z);
      }
   }
};
} // namespace hydrodynamics

int sedov(MPI_Session &mpi, int argc, char *argv[])
{
   const int myid = mpi.WorldRank();

   const int problem = 1;
   const char *mesh_file = "data/cube.mesh";
   int rs_levels = 0;
   const int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-14;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return -1;
   }
   //if (mpi.Root()) { args.PrintOptions(cout); }

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   ParMesh *pmesh = NULL;
#ifdef MFEM_USE_MPI
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
#else
   pmesh = new Mesh(*mesh);
#endif
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }
   int nzones = pmesh->GetNE(), nzones_min, nzones_max;
   MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl; }

   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }
   ODESolver *ode_solver = new RK4Solver;
   const HYPRE_Int H1GTVSize = H1FESpace.GlobalTrueVSize();
   const HYPRE_Int L2GTVSize = L2FESpace.GlobalTrueVSize();
   const int H1Vsize = H1FESpace.GetVSize();
   const int L2Vsize = L2FESpace.GetVSize();
   if (mpi.Root())
   {
      cout << "Number of local/global kinematic (position, velocity) dofs: "
           << H1Vsize << "/" << H1GTVSize << endl;
      cout << "Number of local/global specific internal energy dofs: "
           << L2Vsize << "/" << L2GTVSize << endl;
   }
   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + H1Vsize;
   true_offset[2] = true_offset[1] + H1Vsize;
   true_offset[3] = true_offset[2] + L2Vsize;
   BlockVector S(true_offset, Device::GetDeviceMemoryType());
   S.UseDevice(true);
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);
   pmesh->SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   v_gf.SyncAliasMemory(S);
   ParGridFunction rho(&L2FESpace);
   FunctionCoefficient rho_fct_coeff(rho0);
   ConstantCoefficient rho_coeff(1.0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_fct_coeff);
   rho.ProjectGridFunction(l2_rho);
   DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                            blast_position[2], blast_energy);
   l2_e.ProjectCoefficient(e_coeff);
   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);
   const int source = 0; bool visc = true;

   mfem::hydrodynamics::LagrangianHydroOperator oper(rho_coeff, S.Size(),
                                                     H1FESpace, L2FESpace,
                                                     ess_tdofs, rho, source,
                                                     cfl, mat_gf_coeff,
                                                     visc, cg_tol, cg_max_iter,
                                                     ftz_tol, order_q,
                                                     gamma(S),
                                                     H1FEC.GetBasisType());

   ode_solver->Init(oper);
   oper.ResetTimeStepEstimate();
   double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   int checks = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();
      ode_solver->Step(S, t, dt);
      steps++;
      const double dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         dt *= 0.85;
         if (dt < numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      pmesh->NewNodes(x_gf, false);
      if (last_step || (ti % vis_steps) == 0)
      {
         double loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                       pmesh->GetComm());
         if (mpi.Root())
         {
            const double sqrt_tot_norm = sqrt(tot_norm);
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt
                 << ",\t|e| = " << setprecision(10)
                 << sqrt_tot_norm;
            cout << endl;
         }
      }
      REQUIRE(problem==1);
      double loc_norm = e_gf * e_gf, tot_norm;
      MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                    pmesh->GetComm());
      const double stm = sqrt(tot_norm);
      //printf("\n\033[33m%.15e\033[m", stm); fflush(0);
      REQUIRE((rs_levels==0 || rs_levels==1));
      REQUIRE(rp_levels==0);
      REQUIRE(order_v==2);
      REQUIRE(order_e==1);
      REQUIRE(ode_solver_type==4);
      REQUIRE(t_final==Approx(0.6));
      REQUIRE(cfl==Approx(0.5));
      REQUIRE(cg_tol==Approx(1.e-14));
      const int dim = strcmp(mesh_file,"data/square.mesh")==0?2:
                      strcmp(mesh_file,"data/cube.mesh")==0?3:1;
      REQUIRE((dim==2 || dim==3));
      if (dim==2)
      {
         const double p1_05[2] = {3.508254945225794e+00,
                                  1.403249766367977e+01
                                 };
         const double p1_15[2] = {2.756444596823211e+00,
                                  1.104093401469385e+01
                                 };
         if (ti==05) {checks++; REQUIRE(stm==Approx(p1_05[rs_levels]));}
         if (ti==15) {checks++; REQUIRE(stm==Approx(p1_15[rs_levels]));}
      }
      if (dim==3)
      {
         const double p1_05[2] = {1.339163718592567e+01,
                                  1.071277540097426e+02
                                 };
         const double p1_28[2] = {7.521073677398005e+00,
                                  5.985720905709158e+01
                                 };
         if (ti==05) {checks++; REQUIRE(stm==Approx(p1_05[rs_levels]));}
         if (ti==28) {checks++; REQUIRE(stm==Approx(p1_28[rs_levels]));}
      }
   }
   REQUIRE(checks==2);
   REQUIRE(ode_solver_type==4);
   steps *= 4;
   //oper.PrintTimingData(mpi.Root(), steps, fom);
   delete ode_solver;
   delete pmesh;
   delete mat_gf_coeff;
   return 0;
}
} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void sedov_tests(MPI_Session &mpi)
{
   const char *argv2D[]= {"sedov_tests",
                          "-m", "data/square.mesh",
                          nullptr
                         };
   REQUIRE(sedov(mpi, argn(argv2D), const_cast<char**>(argv2D))==0);

   const char *argv2Drs1[]= {"sedov_tests",
                             "-rs", "1", "-ms", "20",
                             "-m", "data/square.mesh",
                             nullptr
                            };
   REQUIRE(sedov(mpi, argn(argv2Drs1), const_cast<char**>(argv2Drs1))==0);

   const char *argv3D[]= {"sedov_tests",
                          "-m", "data/cube.mesh",
                          nullptr
                         };
   REQUIRE(sedov(mpi, argn(argv3D), const_cast<char**>(argv3D))==0);

   const char *argv3Drs1[]= {"sedov_tests",
                             "-rs", "1", "-ms", "28",
                             "-m", "data/cube.mesh",
                             nullptr
                            };
   REQUIRE(sedov(mpi, argn(argv3Drs1), const_cast<char**>(argv3Drs1))==0);

}

#ifndef MFEM_SEDOV_TESTS

TEST_CASE("Sedov", "[Sedov], [Parallel]")
{
#ifdef MFEM_USE_MPI
   MPI_Session &mpi = *GlobalMPISession;
#else
   MPI_Session mpi;
#endif
   sedov_tests(mpi);
}

#else

TEST_CASE("Sedov", "[Sedov], [Parallel]")
{
#ifdef MFEM_USE_MPI
   MPI_Session &mpi = *GlobalMPISession;
#else
   MPI_Session mpi;
#endif
   Device device;
   device.Configure(MFEM_SEDOV_DEVICE);
   if (mpi.Root()) {device.Print();}
   sedov_tests(mpi);
}

#endif // MFEM_SEDOV_TESTS
