// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_KERNELS_HPP
#define MFEM_KERNELS_HPP

#include "../config/config.hpp"
#include "../general/backends.hpp"
#include "../general/globals.hpp"

#include "matrix.hpp"
#include "tmatrix.hpp"
#include "tlayout.hpp"
#include "ttensor.hpp"

// This header contains stand-alone functions for "small" dense linear algebra
// (at quadrature point or element-level) designed to be inlined directly into
// device kernels.

// Many methods of the DenseMatrix class and some of the Vector class call these
// kernels directly on the host, see the implementations in linalg/densemat.cpp
// and linalag.vector.cpp.

namespace mfem
{

namespace kernels
{

/// Returns the l2 norm of the Vector with given @a size and @a data.
template<typename T>
MFEM_HOST_DEVICE inline
double Norml2(const int size, const T *data)
{
   if (0 == size) { return 0.0; }
   if (1 == size) { return std::abs(data[0]); }
   T scale = 0.0;
   T sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (data[i] != 0.0)
      {
         const T absdata = fabs(data[i]);
         if (scale <= absdata)
         {
            const T sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const T sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if data[i] != 0
   }
   return scale * sqrt(sum);
}

/** @brief Matrix vector multiplication: y = A x, where the matrix A is of size
    @a height x @a width with given @a data, while @a x and @a y specify the
    data of the input and output vectors. */
template<typename TA, typename TX, typename TY>
MFEM_HOST_DEVICE inline
void Mult(const int height, const int width, TA *data, const TX *x, TY *y)
{
   if (width == 0)
   {
      for (int row = 0; row < height; row++)
      {
         y[row] = 0.0;
      }
      return;
   }
   TA *d_col = data;
   TX x_col = x[0];
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

/// Symmetrize a square matrix with given @a size and @a data: A -> (A+A^T)/2.
template<typename T>
MFEM_HOST_DEVICE inline
void Symmetrize(const int size, T *data)
{
   for (int i = 0; i < size; i++)
   {
      for (int j = 0; j < i; j++)
      {
         const T a = 0.5 * (data[i*size+j] + data[j*size+i]);
         data[j*size+i] = data[i*size+j] = a;
      }
   }
}

/// Compute the determinant of a square matrix of size dim with given @a data.
template<int dim, typename T>
MFEM_HOST_DEVICE inline T Det(const T *data)
{
   return TDetHD<T>(ColumnMajorLayout2D<dim,dim>(), data);
}

/** @brief Return the inverse a matrix with given @a size and @a data into the
    matrix with data @a inv_data. */
template<int dim, typename T>
MFEM_HOST_DEVICE inline
void CalcInverse(const T *data, T *inv_data)
{
   typedef ColumnMajorLayout2D<dim,dim> layout_t;
   const T det = TAdjDetHD<T>(layout_t(), data, layout_t(), inv_data);
   TAssignHD<AssignOp::Mult>(layout_t(), inv_data, static_cast<T>(1.0)/det);
}

/** @brief Compute C = A + alpha*B, where the matrices A, B and C are of size @a
    height x @a width with data @a Adata, @a Bdata and @a Cdata. */
template<typename TALPHA, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void Add(const int height, const int width, const TALPHA alpha,
         const TA *Adata, const TB *Bdata, TC *Cdata)
{
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         const int n = i*width+j;
         Cdata[n] = Adata[n] + alpha * Bdata[n];
      }
   }
}

/** @brief Matrix-matrix multiplication: A = B * C, where the matrices A, B and
    C are of sizes @a Aheight x @a Awidth, @a Aheight x @a Bwidth and @a Bwidth
    x @a Awidth, respectively. */
template<typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void Mult(const int Aheight, const int Awidth, const int Bwidth,
          const TB *Bdata, const TC *Cdata, TA *Adata)
{
   const int ah_x_aw = Aheight * Awidth;
   for (int i = 0; i < ah_x_aw; i++) { Adata[i] = 0.0; }
   for (int j = 0; j < Awidth; j++)
   {
      for (int k = 0; k < Bwidth; k++)
      {
         for (int i = 0; i < Aheight; i++)
         {
            Adata[i+j*Aheight] += Bdata[i+k*Aheight] * Cdata[k+j*Bwidth];
         }
      }
   }
}

/** @brief Multiply a matrix of size @a Aheight x @a Awidth and data @a Adata
    with the transpose of a matrix of size @a Bheight x @a Awidth and data @a
    Bdata: A * Bt. Return the result in a matrix with data @a ABtdata. */
template<typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void MultABt(const int Aheight, const int Awidth, const int Bheight,
             const TA *Adata, const TB *Bdata, TC *ABtdata)
{
   const int ah_x_bh = Aheight * Bheight;
   for (int i = 0; i < ah_x_bh; i++) { ABtdata[i] = 0.0; }
   for (int k = 0; k < Awidth; k++)
   {
      TC *c = ABtdata;
      for (int j = 0; j < Bheight; j++)
      {
         const double bjk = Bdata[j];
         for (int i = 0; i < Aheight; i++)
         {
            c[i] += Adata[i] * bjk;
         }
         c += Aheight;
      }
      Adata += Aheight;
      Bdata += Bheight;
   }
}

/// Compute the spectrum of the matrix of size dim with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<int dim> MFEM_HOST_DEVICE
void CalcEigenvalues(const double *data, double *lambda, double *vec);

/// Return the i'th singular value of the matrix of size dim with given @a data.
template<int dim> MFEM_HOST_DEVICE
double CalcSingularvalue(const double *data, const int i);


// Utility functions for CalcEigenvalues and CalcSingularvalue
namespace internal
{

/// Utility function to swap the values of @a a and @a b.
template<typename T>
MFEM_HOST_DEVICE static inline
void Swap(T &a, T &b)
{
   T tmp = a;
   a = b;
   b = tmp;
}

const double Epsilon = std::numeric_limits<double>::epsilon();

/// Utility function used in CalcSingularvalue<3>.
MFEM_HOST_DEVICE static inline
void Eigenvalues2S(const double &d12, double &d1, double &d2)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 != 0.)
   {
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
      double t;
      const double zeta = (d2 - d1)/(2*d12); // inf/inf from overflows?
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

/// Utility function used in CalcEigenvalues().
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
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
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


/// Utility function used in CalcEigenvalues<3>.
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
   // mult = 2^d_exp is such that d_max/mult is in [0.5,1) or in other words
   // d_max is in the interval [0.5,1)*mult
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
bool KernelVector2G(const int &mode,
                    double &d1, double &d12, double &d21, double &d2)
{
   // Find a vector (z1,z2) in the "near"-kernel of the matrix
   // |  d1  d12 |
   // | d21   d2 |
   // using QR factorization.
   // The vector (z1,z2) is returned in (d1,d2). Return 'true' if the matrix
   // is zero without setting (d1,d2).
   // Note: in the current implementation |z1| + |z2| = 1.

   // l1-norms of the columns
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

      if (mode == 0) // eliminate the larger entry in the column
      {
         if (fabs(d1) > fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
      else // eliminate the smaller entry in the column
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
      // n2 > n1, swap columns 1 and 2
      if (mode == 0) // eliminate the larger entry in the column
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
      else // eliminate the smaller entry in the column
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
      // v = (n1, n2)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, d21)^t = (mu, 0)^t
      mu = copysign(n1, d1);
      n1 = -d21*(d21/(d1 + mu)); // = d1 - mu
      d1 = mu;
      // normalize (n1,d21) to avoid overflow/underflow
      // normalize (n1,d21) by the max-norm to avoid the sqrt call
      if (fabs(n1) <= fabs(d21))
      {
         // (n1,n2) <-- (n1/d21,1)
         n1 = n1/d21;
         mu = (2./(1. + n1*n1))*(n1*d12 + d2);
         d2  = d2  - mu;
         d12 = d12 - mu*n1;
      }
      else
      {
         // (n1,n2) <-- (1,d21/n1)
         n2 = d21/n1;
         mu = (2./(1. + n2*n2))*(d12 + n2*d2);
         d2  = d2  - mu*n2;
         d12 = d12 - mu;
      }
   }

   // Solve:
   // | d1 d12 | | z1 | = | 0 |
   // |  0  d2 | | z2 |   | 0 |

   // choose (z1,z2) to minimize |d1*z1 + d12*z2| + |d2*z2|
   // under the condition |z1| + |z2| = 1, z2 >= 0 (for uniqueness)
   // set t = z1, z2 = 1 - |t|, -1 <= t <= 1
   // objective function is:
   // |d1*t + d12*(1 - |t|)| + |d2|*(1 - |t|) -- piecewise linear with
   // possible minima are -1,0,1,t1 where t1: d1*t1 + d12*(1 - |t1|) = 0
   // values: @t=+/-1 -> |d1|, @t=0 -> |n1| + |d2|, @t=t1 -> |d2|*(1 - |t1|)

   // evaluate z2 @t=t1
   mu = -d12/d1;
   // note: |mu| <= 1,       if using l2-norm for column pivoting
   //       |mu| <= sqrt(2), if using l1-norm
   n2 = 1./(1. + fabs(mu));
   // check if |d1|<=|d2|*z2
   if (fabs(d1) <= n2*fabs(d2))
   {
      d2 = 0.;
      d1 = 1.;
   }
   else
   {
      d2 = n2;
      // d1 = (n2 < 0.5) ? copysign(1. - n2, mu) : mu*n2;
      d1 = mu*n2;
   }

   if (swap_columns)
   {
      Swap(d1, d2);
   }

   return false;
}

/// Utility function used in CalcEigenvalues<3>.
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

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
void Vec_normalize3(const double &x1, const double &x2, const double &x3,
                    double &n1, double &n2, double &n3)
{
   // should work ok when xk is the same as nk for some or all k
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

/// Utility function used in CalcEigenvalues<3>.
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
      // v = (s1, s2, s3)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, c12, c13)^t = (mu, 0, 0)^t
      mu = copysign(n1, d1);
      n1 = -s1*(s1/(d1 + mu)); // = d1 - mu
      d1 = mu;

      // normalize (n1,c21,c31) to avoid overflow/underflow
      // normalize (n1,c21,c31) by the max-norm to avoid the sqrt call
      if (fabs(n1) >= fabs(c21))
      {
         if (fabs(n1) >= fabs(c31))
         {
            // n1 is max, (s1,s2,s3) <-- (1,c21/n1,c31/n1)
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
         // c21 is max, (s1,s2,s3) <-- (n1/c21,1,c31/c21)
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
      // c31 is max, (s1,s2,s3) <-- (n1/c31,c21/c31,1)
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

   // Solve:
   // |  d2 c23 | | z2 | = | 0 |
   // | c32  d3 | | z3 |   | 0 |
   if (KernelVector2G(mode, d2, c23, c32, d3))
   {
      // Have two solutions:
      // two vectors in the kernel are P (-c12/d1, 1, 0)^t and
      // P (-c13/d1, 0, 1)^t where P is the permutation matrix swapping
      // entries 1 and col.

      // A vector orthogonal to both these vectors is P (1, c12/d1, c13/d1)^t
      d2 = c12/d1;
      d3 = c13/d1;
      d1 = 1.;
      kdim = 2;
   }
   else
   {
      // solve for z1:
      // note: |z1| <= a since |z2| + |z3| = 1, and
      // max{|c12|,|c13|} <= max{norm(col. 2),norm(col. 3)}
      //                  <= norm(col. 1) <= a |d1|
      // a = 1,       if using l2-norm for column pivoting
      // a = sqrt(3), if using l1-norm
      d1 = -(c12*d2 + c13*d3)/d1;
      kdim = 1;
   }

   Vec_normalize3(d1, d2, d3, d1, d2, d3);

   return kdim;
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
int KernelVector3S(const int &mode, const double &d12,
                   const double &d13, const double &d23,
                   double &d1, double &d2, double &d3)
{
   // Find a unit vector (z1,z2,z3) in the "near"-kernel of the matrix
   // |  d1  d12  d13 |
   // | d12   d2  d23 |
   // | d13  d23   d3 |
   // using QR factorization.
   // The vector (z1,z2,z3) is returned in (d1,d2,d3).
   // Returns the dimension of the kernel, kdim, but never zero.
   // - if kdim == 3, then (d1,d2,d3) is not defined,
   // - if kdim == 2, then (d1,d2,d3) is a vector orthogonal to the kernel,
   // - otherwise kdim == 1 and (d1,d2,d3) is a vector in the "near"-kernel.

   double c12 = d12, c13 = d13, c23 = d23;
   double c21, c31, c32;
   int col, row;

   // l1-norms of the columns:
   c32 = fabs(d1) + fabs(c12) + fabs(c13);
   c31 = fabs(d2) + fabs(c12) + fabs(c23);
   c21 = fabs(d3) + fabs(c13) + fabs(c23);

   // column pivoting: choose the column with the largest norm
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
         if (c32 == 0.) // zero matrix
         {
            return 3;
         }
         break;

      case 2:
         if (c31 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c13, c23);
         Swap(d1, d2);
         break;

      case 3:
         if (c21 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c12, c23);
         Swap(d1, d3);
   }

   // row pivoting depending on 'mode'
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
   // row is kdim

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

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
int Reduce3S(const int &mode,
             double &d1, double &d2, double &d3,
             double &d12, double &d13, double &d23,
             double &z1, double &z2, double &z3,
             double &v1, double &v2, double &v3,
             double &g)
{
   // Given the matrix
   //     |  d1  d12  d13 |
   // A = | d12   d2  d23 |
   //     | d13  d23   d3 |
   // and a unit eigenvector z=(z1,z2,z3), transform the matrix A into the
   // matrix B = Q P A P Q that has the form
   //                 | b1   0   0 |
   // B = Q P A P Q = | 0   b2 b23 |
   //                 | 0  b23  b3 |
   // where P is the permutation matrix switching entries 1 and k, and
   // Q is the reflection matrix Q = I - g v v^t, defined by: set y = P z and
   // v = c(y - e_1); if y = e_1, then v = 0 and Q = I.
   // Note: Q y = e_1, Q e_1 = y ==> Q P A P Q e_1 = ... = lambda e_1.
   // The entries (b1,b2,b3,b23) are returned in (d1,d2,d3,d23), and the
   // return value of the function is k. The variable g = 2/(v1^2+v2^2+v3^3).

   int k;
   double s, w1, w2, w3;

   if (mode == 0)
   {
      // choose k such that z^t e_k = zk has the smallest absolute value, i.e.
      // the angle between z and e_k is closest to pi/2
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
      // choose k such that zk is the largest by absolute value
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
      // s can not be zero, if zk is the smallest (mode == 0)
      v1 = v2 = v3 = 0.;
      g = 1.;
   }
   else
   {
      g = copysign(1., z1);
      v1 = -s*(s/(z1 + g)); // = z1 - g
      // normalize (v1,z2,z3) by its max-norm, avoiding the sqrt call
      g = fabs(v1);
      if (fabs(z2) > g) { g = fabs(z2); }
      if (fabs(z3) > g) { g = fabs(z3); }
      v1 = v1/g;
      v2 = z2/g;
      v3 = z3/g;
      g = 2./(v1*v1 + v2*v2 + v3*v3);

      // Compute Q A Q = A - v w^t - w v^t, where
      // w = u - (g/2)(v^t u) v, and u = g A v
      // set w = g A v
      w1 = g*( d1*v1 + d12*v2 + d13*v3);
      w2 = g*(d12*v1 +  d2*v2 + d23*v3);
      w3 = g*(d13*v1 + d23*v2 +  d3*v3);
      // w := w - (g/2)(v^t w) v
      s = (g/2)*(v1*w1 + v2*w2 + v3*w3);
      w1 -= s*v1;
      w2 -= s*v2;
      w3 -= s*v3;
      // dij -= vi*wj + wi*vj
      d1  -= 2*v1*w1;
      d2  -= 2*v2*w2;
      d23 -= v2*w3 + v3*w2;
      d3  -= 2*v3*w3;
      // compute the off-diagonal entries on the first row/column of B which
      // should be zero (for debugging):
#if 0
      s = d12 - v1*w2 - v2*w1;  // b12 = 0
      s = d13 - v1*w3 - v3*w1;  // b13 = 0
#endif
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

} // namespace kernels::internal


// Implementations of CalcEigenvalues and CalcSingularvalue for dim = 2, 3.

/// Compute the spectrum of the matrix of size 2 with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<2>(const double *data, double *lambda, double *vec)
{
   double d0 = data[0];
   double d2 = data[2]; // use the upper triangular entry
   double d3 = data[3];
   double c, s;
   internal::Eigensystem2S(d2, d0, d3, c, s);
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

/// Compute the spectrum of the matrix of size 3 with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<3>(const double *data, double *lambda, double *vec)
{
   double d11 = data[0];
   double d12 = data[3]; // use the upper triangular entries
   double d22 = data[4];
   double d13 = data[6];
   double d23 = data[7];
   double d33 = data[8];

   double mult;
   {
      double d_max = fabs(d11);
      if (d_max < fabs(d22)) { d_max = fabs(d22); }
      if (d_max < fabs(d33)) { d_max = fabs(d33); }
      if (d_max < fabs(d12)) { d_max = fabs(d12); }
      if (d_max < fabs(d13)) { d_max = fabs(d13); }
      if (d_max < fabs(d23)) { d_max = fabs(d23); }

      internal::GetScalingFactor(d_max, mult);
   }

   d11 /= mult;  d22 /= mult;  d33 /= mult;
   d12 /= mult;  d13 /= mult;  d23 /= mult;

   double aa = (d11 + d22 + d33)/3;  // aa = tr(A)/3
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
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;
      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
         }
      }

      aa += r;
      c1 = d11 - aa;
      c2 = d22 - aa;
      c3 = d33 - aa;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      // mode == 1: largest absolute value --> angle farthest from pi/2
      // Observations:
      // mode == 0 produces better eigenvectors, less accurate eigenvalues?
      // mode == 1 produces better eigenvalues, less accurate eigenvectors?
      const int mode = 0;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  d12  d13 |
      //  | d12   c2  d23 | = A - aa*I
      //  | d13  d23   c3 |
      // This vector is also an eigenvector for A corresponding to aa.
      // The vector z overwrites (c1,c2,c3).
      switch (internal::KernelVector3S(mode, d12, d13, d23, c1, c2, c3))
      {
         case 3:
            // 'aa' is a triple eigenvalue
            lambda[0] = lambda[1] = lambda[2] = aa;
            vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
            vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
            vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
            goto done_3d;

         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c=(c1,c2,c3) transform A into
      //                   | d11   0   0 |
      // A <-- Q P A P Q = |  0  d22 d23 |
      //                   |  0  d23 d33 |
      double v1, v2, v3, g;
      int k = internal::Reduce3S(mode, d11, d22, d33, d12, d13, d23,
                                 c1, c2, c3, v1, v2, v3, g);
      // Q = I - 2 v v^t
      // P - permutation matrix switching entries 1 and k

      // find the eigenvalues and eigenvectors for
      // | d22 d23 |
      // | d23 d33 |
      double c, s;
      internal::Eigensystem2S(d23, d22, d33, c, s);
      // d22 <-> P Q (0, c, -s), d33 <-> P Q (0, s, c)

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
            internal::Swap(vec_2[0], vec_2[1]);
            internal::Swap(vec_3[0], vec_3[1]);
            break;

         case 3:
            internal::Swap(vec_2[0], vec_2[2]);
            internal::Swap(vec_3[0], vec_3[2]);
      }
   }

done_3d:
   lambda[0] *= mult;
   lambda[1] *= mult;
   lambda[2] *= mult;
}

/// Return the i'th singular value of the matrix of size 2 with given @a data.
template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<2>(const double *data, const int i)
{
   double d0, d1, d2, d3;
   d0 = data[0];
   d1 = data[1];
   d2 = data[2];
   d3 = data[3];
   double mult;

   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      internal::GetScalingFactor(d_max, mult);
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

/// Return the i'th singular value of the matrix of size 3 with given @a data.
template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<3>(const double *data, const int i)
{
   double d0, d1, d2, d3, d4, d5, d6, d7, d8;
   d0 = data[0];  d3 = data[3];  d6 = data[6];
   d1 = data[1];  d4 = data[4];  d7 = data[7];
   d2 = data[2];  d5 = data[5];  d8 = data[8];
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
      internal::GetScalingFactor(d_max, mult);
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

   // double a, b, c;
   // a = -(b11 + b22 + b33);
   // b = b11*(b22 + b33) + b22*b33 - b12*b12 - b13*b13 - b23*b23;
   // c = b11*(b23*b23 - b22*b33) + b12*(b12*b33 - 2*b13*b23) + b13*b13*b22;

   // double Q = (a * a - 3 * b) / 9;
   // double Q = (b12*b12 + b13*b13 + b23*b23 +
   //             ((b11 - b22)*(b11 - b22) +
   //              (b11 - b33)*(b11 - b33) +
   //              (b22 - b33)*(b22 - b33))/6)/3;
   // Q = (3*(b12^2 + b13^2 + b23^2) +
   //      ((b11 - b22)^2 + (b11 - b33)^2 + (b22 - b33)^2)/2)/9
   //   or
   // Q = (1/6)*|B-tr(B)/3|_F^2
   // Q >= 0 and
   // Q = 0  <==> B = scalar * I
   // double R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
   double aa = (b11 + b22 + b33)/3;  // aa = tr(B)/3
   double c1, c2, c3;
   // c1 = b11 - aa; // ((b11 - b22) + (b11 - b33))/3
   // c2 = b22 - aa; // ((b22 - b11) + (b22 - b33))/3
   // c3 = b33 - aa; // ((b33 - b11) + (b33 - b22))/3
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
   // R = (-1/2)*det(B-(tr(B)/3)*I)
   // Note: 54*(det(S))^2 <= |S|_F^6, when S^t=S and tr(S)=0, S is 3x3
   // Therefore: R^2 <= Q^3

   if (Q <= 0.) { ; }

   // else if (fabs(R) >= sqrtQ3)
   // {
   //    double det = (d[0] * (d[4] * d[8] - d[5] * d[7]) +
   //                  d[3] * (d[2] * d[7] - d[1] * d[8]) +
   //                  d[6] * (d[1] * d[5] - d[2] * d[4]));
   //
   //    if (R > 0.)
   //    {
   //       if (i == 2)
   //          // aa -= 2*sqrtQ;
   //          return fabs(det)/(aa + sqrtQ);
   //       else
   //          aa += sqrtQ;
   //    }
   //    else
   //    {
   //       if (i != 0)
   //          aa -= sqrtQ;
   //          // aa = fabs(det)/sqrt(aa + 2*sqrtQ);
   //       else
   //          aa += 2*sqrtQ;
   //    }
   // }

   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;

      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         // if (fabs(R) <= 0.95)
         if (fabs(R) <= 0.9)
         {
            if (i == 2)
            {
               aa -= 2*sqrtQ*cos(acos(R)/3);   // min
            }
            else if (i == 0)
            {
               aa -= 2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);   // max
            }
            else
            {
               aa -= 2*sqrtQ*cos((acos(R) - 2.0*M_PI)/3);   // mid
            }
            goto have_aa;
         }

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
            if (i == 0)
            {
               aa += r;
               goto have_aa;
            }
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
            if (i == 2)
            {
               aa += r;
               goto have_aa;
            }
         }
      }

      // (tr(B)/3 + r) is the root which is separated from the other
      // two roots which are close to each other when |R| is close to 1

      c1 -= r;
      c2 -= r;
      c3 -= r;
      // aa += r;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      //            (eliminate large entries)
      // mode == 1: largest absolute value --> angle farthest from pi/2
      //            (eliminate small entries)
      const int mode = 1;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  b12  b13 |
      //  | b12   c2  b23 | = B - aa*I
      //  | b13  b23   c3 |
      // This vector is also an eigenvector for B corresponding to aa
      // The vector z overwrites (c1,c2,c3).
      switch (internal::KernelVector3S(mode, b12, b13, b23, c1, c2, c3))
      {
         case 3:
            aa += r;
            goto have_aa;
         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c = (c1,c2,c3) to transform B into
      //                   | b11   0   0 |
      // B <-- Q P B P Q = |  0  b22 b23 |
      //                   |  0  b23 b33 |
      double v1, v2, v3, g;
      internal::Reduce3S(mode, b11, b22, b33, b12, b13, b23,
                         c1, c2, c3, v1, v2, v3, g);
      // Q = I - g v v^t
      // P - permutation matrix switching rows and columns 1 and k

      // find the eigenvalues of
      //  | b22 b23 |
      //  | b23 b33 |
      internal::Eigenvalues2S(b23, b22, b33);

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

   return sqrt(fabs(aa))*mult; // take abs before we sort?
}


/// Assuming L.U = P.A for a factored matrix (m x m),
//  compute x <- A x
//
// @param [in] data LU factorization of A
// @param [in] m square matrix height
// @param [in] ipiv array storing pivot information
// @param [in, out] x vector storing right-hand side and then solution
MFEM_HOST_DEVICE
inline void LUSolve(const double *data, const int m, const int *ipiv,
                    double *x)
{
   // X <- P X
   for (int i = 0; i < m; i++)
   {
      internal::Swap<double>(x[i], x[ipiv[i]]);
   }

   // X <- L^{-1} X
   for (int j = 0; j < m; j++)
   {
      const double x_j = x[j];
      for (int i = j + 1; i < m; i++)
      {
         x[i] -= data[i + j * m] * x_j;
      }
   }

   // X <- U^{-1} X
   for (int j = m - 1; j >= 0; j--)
   {
      const double x_j = (x[j] /= data[j + j * m]);
      for (int i = 0; i < j; i++)
      {
         x[i] -= data[i + j * m] * x_j;
      }
   }
}

} // namespace kernels

} // namespace mfem

#endif // MFEM_KERNELS_HPP
