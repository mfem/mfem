// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


// Implementation of data types dense matrix, inverse dense matrix


#include "kernels.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "densemat.hpp"
#include "lapack.hpp"
#include "batched/batched.hpp"
#include "../general/forall.hpp"
#include "../general/table.hpp"
#include "../general/globals.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cstdlib>
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
#define copysign _copysign
#endif


namespace mfem
{

using namespace std;

DenseMatrix::DenseMatrix() : Matrix(0) { }

DenseMatrix::DenseMatrix(int s) : Matrix(s)
{
   MFEM_ASSERT(s >= 0, "invalid DenseMatrix size: " << s);
   if (s > 0)
   {
      data.SetSize(s*s);
      *this = 0.0; // init with zeroes
   }
}

DenseMatrix::DenseMatrix(int m, int n) : Matrix(m, n)
{
   MFEM_ASSERT(m >= 0 && n >= 0,
               "invalid DenseMatrix size: " << m << " x " << n);
   const int capacity = m*n;
   if (capacity > 0)
   {
      data.SetSize(capacity);
      *this = 0.0; // init with zeroes
   }
}

DenseMatrix::DenseMatrix(const DenseMatrix &mat, char ch)
   : Matrix(mat.width, mat.height)
{
   MFEM_CONTRACT_VAR(ch);
   const int capacity = height*width;
   if (capacity > 0)
   {
      data.SetSize(capacity);

      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < width; j++)
         {
            (*this)(i,j) = mat(j,i);
         }
      }
   }
}

void DenseMatrix::SetSize(int h, int w)
{
   MFEM_ASSERT(h >= 0 && w >= 0,
               "invalid DenseMatrix size: " << h << " x " << w);
   if (Height() == h && Width() == w)
   {
      return;
   }
   height = h;
   width = w;
   data.SetSize(h*w, 0.0);
}

real_t &DenseMatrix::Elem(int i, int j)
{
   return (*this)(i,j);
}

const real_t &DenseMatrix::Elem(int i, int j) const
{
   return (*this)(i,j);
}

void DenseMatrix::Mult(const real_t *x, real_t *y) const
{
   kernels::Mult(height, width, HostRead(), x, y);
}

void DenseMatrix::Mult(const real_t *x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size(), "incompatible dimensions");

   Mult(x, y.HostWrite());
}

void DenseMatrix::Mult(const Vector &x, real_t *y) const
{
   MFEM_ASSERT(width == x.Size(), "incompatible dimensions");

   Mult(x.HostRead(), y);
}

void DenseMatrix::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   Mult(x.HostRead(), y.HostWrite());
}

void DenseMatrix::AbsMult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   kernels::AbsMult(height, width, HostRead(), x.HostRead(), y.HostWrite());
}

real_t DenseMatrix::operator *(const DenseMatrix &m) const
{
   MFEM_ASSERT(Height() == m.Height() && Width() == m.Width(),
               "incompatible dimensions");

   const int hw = height * width;
   real_t a = 0.0;
   for (int i = 0; i < hw; i++)
   {
      a += data[i] * m.data[i];
   }

   return a;
}

void DenseMatrix::MultTranspose(const real_t *x, real_t *y) const
{
   kernels::MultTranspose(height, width, HostRead(), x, y);
}

void DenseMatrix::MultTranspose(const real_t *x, Vector &y) const
{
   MFEM_ASSERT(width == y.Size(), "incompatible dimensions");

   MultTranspose(x, y.HostWrite());
}

void DenseMatrix::MultTranspose(const Vector &x, real_t *y) const
{
   MFEM_ASSERT(height == x.Size(), "incompatible dimensions");

   MultTranspose(x.HostRead(), y);
}

void DenseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   MultTranspose(x.HostRead(), y.HostWrite());
}

void DenseMatrix::AbsMultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   kernels::AbsMultTranspose(height, width, HostRead(),
                             x.HostRead(), y.HostWrite());
}

void DenseMatrix::AddMult(const Vector &x, Vector &y, const real_t a) const
{
   if (a != 1.0)
   {
      AddMult_a(a, x, y);
      return;
   }
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   const real_t *xp = x.GetData(), *d_col = data;
   real_t *yp = y.GetData();
   for (int col = 0; col < width; col++)
   {
      real_t x_col = xp[col];
      for (int row = 0; row < height; row++)
      {
         yp[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

void DenseMatrix::AddMultTranspose(const Vector &x, Vector &y,
                                   const real_t a) const
{
   if (a != 1.0)
   {
      AddMultTranspose_a(a, x, y);
      return;
   }
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   const real_t *d_col = data;
   for (int col = 0; col < width; col++)
   {
      real_t y_col = 0.0;
      for (int row = 0; row < height; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] += y_col;
      d_col += height;
   }
}

void DenseMatrix::AddMult_a(real_t a, const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   HostRead();
   x.HostRead();
   y.HostReadWrite();
   const real_t *xp = x.GetData(), *d_col = data;
   real_t *yp = y.GetData();
   for (int col = 0; col < width; col++)
   {
      const real_t x_col = a*xp[col];
      for (int row = 0; row < height; row++)
      {
         yp[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

void DenseMatrix::AddMultTranspose_a(real_t a, const Vector &x,
                                     Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   const real_t *d_col = data;
   for (int col = 0; col < width; col++)
   {
      real_t y_col = 0.0;
      for (int row = 0; row < height; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] += a * y_col;
      d_col += height;
   }
}

real_t DenseMatrix::InnerProduct(const real_t *x, const real_t *y) const
{
   real_t prod = 0.0;

   for (int i = 0; i < height; i++)
   {
      real_t Axi = 0.0;
      for (int j = 0; j < width; j++)
      {
         Axi += (*this)(i,j) * x[j];
      }
      prod += y[i] * Axi;
   }

   return prod;
}

// LeftScaling this = diag(s) * this
void DenseMatrix::LeftScaling(const Vector & s)
{
   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= s(i);
      }
   }
}

// InvLeftScaling this = diag(1./s) * this
void DenseMatrix::InvLeftScaling(const Vector & s)
{
   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) /= s(i);
      }
   }
}

// RightScaling: this = this * diag(s);
void DenseMatrix::RightScaling(const Vector & s)
{
   real_t sj;
   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      sj = s(j);
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= sj;
      }
   }
}

// InvRightScaling: this = this * diag(1./s);
void DenseMatrix::InvRightScaling(const Vector & s)
{
   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      const real_t sj = 1./s(j);
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= sj;
      }
   }
}

// SymmetricScaling this = diag(sqrt(s)) * this * diag(sqrt(s))
void DenseMatrix::SymmetricScaling(const Vector & s)
{
   if (height != width || s.Size() != height)
   {
      mfem_error("DenseMatrix::SymmetricScaling: dimension mismatch");
   }

   real_t * ss = new real_t[width];
   real_t * it_s = s.GetData();
   real_t * it_ss = ss;
   for ( real_t * end_s = it_s + width; it_s != end_s; ++it_s)
   {
      *(it_ss++) = sqrt(*it_s);
   }

   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= ss[i]*ss[j];
      }
   }

   delete[] ss;
}

// InvSymmetricScaling this = diag(sqrt(1./s)) * this * diag(sqrt(1./s))
void DenseMatrix::InvSymmetricScaling(const Vector & s)
{
   if (height != width || s.Size() != width)
   {
      mfem_error("DenseMatrix::InvSymmetricScaling: dimension mismatch");
   }

   real_t * ss = new real_t[width];
   real_t * it_s = s.GetData();
   real_t * it_ss = ss;
   for (real_t * end_s = it_s + width; it_s != end_s; ++it_s)
   {
      *(it_ss++) = 1./sqrt(*it_s);
   }

   real_t * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= ss[i]*ss[j];
      }
   }

   delete[] ss;
}

real_t DenseMatrix::Trace() const
{
#ifdef MFEM_DEBUG
   if (Width() != Height())
   {
      mfem_error("DenseMatrix::Trace() : not a square matrix!");
   }
#endif

   real_t t = 0.0;

   for (int i = 0; i < width; i++)
   {
      t += (*this)(i, i);
   }

   return t;
}

MatrixInverse *DenseMatrix::Inverse() const
{
   return new DenseMatrixInverse(*this);
}

void DenseMatrix::Exponential()
{
   MFEM_ASSERT(Height() == Width() && Height() <= 2,
               "The matrix must be square and "
               << "of size less than or equal to 2."
               << "  Height() = " << Height()
               << ", Width() = " << Width());

   switch (Height())
   {
      case 1:
      {
         data[0] = std::exp(data[0]);
         break;
      }
      case 2:
      {
         /// Formulas from Corollary 2.4 of doi:10.1109/9.233156
         /// Note typo in the paper, in the prefactor in the equation under (i).
         const real_t a = data[0];
         const real_t b = data[1];
         const real_t c = data[2];
         const real_t d = data[3];
         const real_t e = (a - d)*(a - d) + 4*b*c;
         const real_t f = std::exp((a + d)/2.0);
         const real_t g = std::sqrt(std::abs(e)) / 2.0;

         if (e == 0)
         {
            data[0] = 1.0 + (a - d)/2.0;
            data[3] = 1.0 - (a - d)/2.0;
         }
         else if (e > 0)
         {
            data[0] = std::cosh(g) + (a - d)/2 * std::sinh(g) / g;
            data[1] = b * std::sinh(g) / g;
            data[2] = c * std::sinh(g) / g;
            data[3] = std::cosh(g) - (a - d)/2 * std::sinh(g) / g;
         }
         else
         {
            data[0] = std::cos(g) + (a - d)/2 * std::sin(g) / g;
            data[1] = b * std::sin(g) / g;
            data[2] = c * std::sin(g) / g;
            data[3] = std::cos(g) - (a - d)/2 * std::sin(g) / g;
         }
         for (int i = 0; i < 4; i++)
         {
            data[i] *= f;
         }
         break;
      }
      case 3:
      {
         MFEM_ABORT("3x3 matrices are not currently supported");
      }
      default:
      {
         MFEM_ABORT("Only 1x1 and 2x2 matrices are currently supported");
      }
   }
}

real_t DenseMatrix::Det() const
{
   MFEM_ASSERT(Height() == Width() && Height() > 0,
               "The matrix must be square and "
               << "sized larger than zero to compute the determinant."
               << "  Height() = " << Height()
               << ", Width() = " << Width());

   switch (Height())
   {
      case 1:
         return data[0];

      case 2:
         return data[0] * data[3] - data[1] * data[2];

      case 3:
      {
         const real_t *d = data;
         return
            d[0] * (d[4] * d[8] - d[5] * d[7]) +
            d[3] * (d[2] * d[7] - d[1] * d[8]) +
            d[6] * (d[1] * d[5] - d[2] * d[4]);
      }
      case 4:
      {
         const real_t *d = data;
         return
            d[ 0] * (d[ 5] * (d[10] * d[15] - d[11] * d[14]) -
                     d[ 9] * (d[ 6] * d[15] - d[ 7] * d[14]) +
                     d[13] * (d[ 6] * d[11] - d[ 7] * d[10])
                    ) -
            d[ 4] * (d[ 1] * (d[10] * d[15] - d[11] * d[14]) -
                     d[ 9] * (d[ 2] * d[15] - d[ 3] * d[14]) +
                     d[13] * (d[ 2] * d[11] - d[ 3] * d[10])
                    ) +
            d[ 8] * (d[ 1] * (d[ 6] * d[15] - d[ 7] * d[14]) -
                     d[ 5] * (d[ 2] * d[15] - d[ 3] * d[14]) +
                     d[13] * (d[ 2] * d[ 7] - d[ 3] * d[ 6])
                    ) -
            d[12] * (d[ 1] * (d[ 6] * d[11] - d[ 7] * d[10]) -
                     d[ 5] * (d[ 2] * d[11] - d[ 3] * d[10]) +
                     d[ 9] * (d[ 2] * d[ 7] - d[ 3] * d[ 6])
                    );
      }
      default:
      {
         // In the general case we compute the determinant from the LU
         // decomposition.
         DenseMatrixInverse lu_factors(*this);

         return lu_factors.Det();
      }
   }
   // not reachable
}

real_t DenseMatrix::Weight() const
{
   if (Height() == Width())
   {
      // return fabs(Det());
      return Det();
   }
   else if ((Height() == 2) && (Width() == 1))
   {
      return sqrt(data[0] * data[0] + data[1] * data[1]);
   }
   else if ((Height() == 3) && (Width() == 1))
   {
      return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
   }
   else if ((Height() == 3) && (Width() == 2))
   {
      const real_t *d = data;
      real_t E = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      real_t G = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
      real_t F = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
      return sqrt(E * G - F * F);
   }
   mfem_error("DenseMatrix::Weight(): mismatched or unsupported dimensions");
   return 0.0;
}

void DenseMatrix::Set(real_t alpha, const real_t *A)
{
   const int s = Width()*Height();
   for (int i = 0; i < s; i++)
   {
      data[i] = alpha*A[i];
   }
}

void DenseMatrix::Add(const real_t c, const DenseMatrix &A)
{
   for (int j = 0; j < Width(); j++)
   {
      for (int i = 0; i < Height(); i++)
      {
         (*this)(i,j) += c * A(i,j);
      }
   }
}

void DenseMatrix::Add(const real_t c, const real_t *A)
{
   const int s = Width()*Height();
   for (int i = 0; i < s; i++)
   {
      data[i] += c*A[i];
   }
}

DenseMatrix &DenseMatrix::operator=(real_t c)
{
   const int s = Height()*Width();
   for (int i = 0; i < s; i++)
   {
      data[i] = c;
   }
   return *this;
}

DenseMatrix &DenseMatrix::operator=(const real_t *d)
{
   const int s = Height()*Width();
   for (int i = 0; i < s; i++)
   {
      data[i] = d[i];
   }
   return *this;
}

DenseMatrix &DenseMatrix::operator+=(const real_t *m)
{
   kernels::Add(Height(), Width(), m, (real_t*)data);
   return *this;
}

DenseMatrix &DenseMatrix::operator+=(const DenseMatrix &m)
{
   MFEM_ASSERT(Height() == m.Height() && Width() == m.Width(),
               "incompatible matrix sizes.");
   return *this += m.GetData();
}

DenseMatrix &DenseMatrix::operator-=(const DenseMatrix &m)
{
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         (*this)(i, j) -= m(i, j);
      }
   }

   return *this;
}

DenseMatrix &DenseMatrix::operator*=(real_t c)
{
   int s = Height()*Width();
   for (int i = 0; i < s; i++)
   {
      data[i] *= c;
   }
   return *this;
}

void DenseMatrix::Neg()
{
   const int hw = Height() * Width();
   for (int i = 0; i < hw; i++)
   {
      data[i] = -data[i];
   }
}

void DenseMatrix::Invert()
{
#ifdef MFEM_DEBUG
   if (Height() <= 0 || Height() != Width())
   {
      mfem_error("DenseMatrix::Invert(): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[width];
   int    lwork = -1;
   real_t qwork, *work;
   int    info;

   MFEM_LAPACK_PREFIX(getrf_)(&width, &width, data, &width, ipiv, &info);

   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in DGETRF");
   }

   MFEM_LAPACK_PREFIX(getri_)(&width, data, &width, ipiv, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new real_t[lwork];

   MFEM_LAPACK_PREFIX(getri_)(&width, data, &width, ipiv, work, &lwork, &info);

   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in DGETRI");
   }

   delete [] work;
   delete [] ipiv;
#else
   int c, i, j, n = Width();
   real_t a, b;
   Array<int> piv(n);

   for (c = 0; c < n; c++)
   {
      a = fabs((*this)(c, c));
      i = c;
      for (j = c + 1; j < n; j++)
      {
         b = fabs((*this)(j, c));
         if (a < b)
         {
            a = b;
            i = j;
         }
      }
      if (a == 0.0)
      {
         mfem_error("DenseMatrix::Invert() : singular matrix");
      }
      piv[c] = i;
      for (j = 0; j < n; j++)
      {
         mfem::Swap<real_t>((*this)(c, j), (*this)(i, j));
      }

      a = (*this)(c, c) = 1.0 / (*this)(c, c);
      for (j = 0; j < c; j++)
      {
         (*this)(c, j) *= a;
      }
      for (j++; j < n; j++)
      {
         (*this)(c, j) *= a;
      }
      for (i = 0; i < c; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
         {
            (*this)(i, j) += b * (*this)(c, j);
         }
         for (j++; j < n; j++)
         {
            (*this)(i, j) += b * (*this)(c, j);
         }
      }
      for (i++; i < n; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
         {
            (*this)(i, j) += b * (*this)(c, j);
         }
         for (j++; j < n; j++)
         {
            (*this)(i, j) += b * (*this)(c, j);
         }
      }
   }

   for (c = n - 1; c >= 0; c--)
   {
      j = piv[c];
      for (i = 0; i < n; i++)
      {
         mfem::Swap<real_t>((*this)(i, c), (*this)(i, j));
      }
   }
#endif
}

void DenseMatrix::SquareRootInverse()
{
   // Square root inverse using Denman--Beavers
#ifdef MFEM_DEBUG
   if (Height() <= 0 || Height() != Width())
   {
      mfem_error("DenseMatrix::SquareRootInverse() matrix not square.");
   }
#endif

   DenseMatrix tmp1(Height());
   DenseMatrix tmp2(Height());
   DenseMatrix tmp3(Height());

   tmp1 = (*this);
   (*this) = 0.0;
   for (int v = 0; v < Height() ; v++) { (*this)(v,v) = 1.0; }

   for (int j = 0; j < 10; j++)
   {
      for (int i = 0; i < 10; i++)
      {
         tmp2 = tmp1;
         tmp3 = (*this);

         tmp2.Invert();
         tmp3.Invert();

         tmp1 += tmp3;
         (*this) += tmp2;

         tmp1 *= 0.5;
         (*this) *= 0.5;
      }
      mfem::Mult((*this), tmp1, tmp2);
      for (int v = 0; v < Height() ; v++) { tmp2(v,v) -= 1.0; }
      if (tmp2.FNorm() < 1e-10) { break; }
   }

   if (tmp2.FNorm() > 1e-10)
   {
      mfem_error("DenseMatrix::SquareRootInverse not converged");
   }
}

void DenseMatrix::Norm2(real_t *v) const
{
   for (int j = 0; j < Width(); j++)
   {
      v[j] = 0.0;
      for (int i = 0; i < Height(); i++)
      {
         v[j] += (*this)(i,j)*(*this)(i,j);
      }
      v[j] = sqrt(v[j]);
   }
}

real_t DenseMatrix::MaxMaxNorm() const
{
   int hw = Height()*Width();
   const real_t *d = data;
   real_t norm = 0.0, abs_entry;

   for (int i = 0; i < hw; i++)
   {
      abs_entry = fabs(d[i]);
      if (norm < abs_entry)
      {
         norm = abs_entry;
      }
   }

   return norm;
}

void DenseMatrix::FNorm(real_t &scale_factor, real_t &scaled_fnorm2) const
{
   int i, hw = Height() * Width();
   real_t max_norm = 0.0, entry, fnorm2;

   for (i = 0; i < hw; i++)
   {
      entry = fabs(data[i]);
      if (entry > max_norm)
      {
         max_norm = entry;
      }
   }

   if (max_norm == 0.0)
   {
      scale_factor = scaled_fnorm2 = 0.0;
      return;
   }

   fnorm2 = 0.0;
   for (i = 0; i < hw; i++)
   {
      entry = data[i] / max_norm;
      fnorm2 += entry * entry;
   }

   scale_factor = max_norm;
   scaled_fnorm2 = fnorm2;
}

void dsyevr_Eigensystem(DenseMatrix &a, Vector &ev, DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK
   ev.SetSize(a.Width());

   char      JOBZ     = 'N';
   char      RANGE    = 'A';
   char      UPLO     = 'U';
   int       N        = a.Width();
   real_t   *A        = new real_t[N*N];
   int       LDA      = N;
   real_t    VL       = 0.0;
   real_t    VU       = 1.0;
   int       IL       = 0;
   int       IU       = 1;
   real_t    ABSTOL   = 0.0;
   int       M;
   real_t   *W        = ev.GetData();
   real_t   *Z        = NULL;
   int       LDZ      = 1;
   int      *ISUPPZ   = new int[2*N];
   int       LWORK    = -1; // query optimal (double) workspace size
   real_t    QWORK;
   real_t   *WORK     = NULL;
   int       LIWORK   = -1; // query optimal (int) workspace size
   int       QIWORK;
   int      *IWORK    = NULL;
   int       INFO;

   if (evect) // Compute eigenvectors too
   {
      evect->SetSize(N);

      JOBZ     = 'V';
      Z        = evect->Data();
      LDZ      = N;
   }

   int hw = a.Height() * a.Width();
   real_t *data = a.Data();

   for (int i = 0; i < hw; i++)
   {
      A[i] = data[i];
   }

   MFEM_LAPACK_PREFIX(syevr_)(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL,
                              &IU, &ABSTOL, &M, W, Z, &LDZ, ISUPPZ, &QWORK,
                              &LWORK, &QIWORK, &LIWORK, &INFO);

   LWORK  = (int) QWORK;
   LIWORK = QIWORK;

   WORK  = new real_t[LWORK];
   IWORK = new int[LIWORK];

   MFEM_LAPACK_PREFIX(syevr_)(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL,
                              &IU, &ABSTOL, &M, W, Z, &LDZ, ISUPPZ, WORK,
                              &LWORK, IWORK, &LIWORK, &INFO);

   if (INFO != 0)
   {
      mfem::err << "dsyevr_Eigensystem(...): DSYEVR error code: "
                << INFO << endl;
      mfem_error();
   }

#ifdef MFEM_DEBUG
   if (M < N)
   {
      mfem::err << "dsyevr_Eigensystem(...):\n"
                << " DSYEVR did not find all eigenvalues "
                << M << "/" << N << endl;
      mfem_error();
   }
   if (CheckFinite(W, N) > 0)
   {
      mfem_error("dsyevr_Eigensystem(...): inf/nan values in W");
   }
   if (CheckFinite(Z, N*N) > 0)
   {
      mfem_error("dsyevr_Eigensystem(...): inf/nan values in Z");
   }
   VU = 0.0;
   for (IL = 0; IL < N; IL++)
      for (IU = 0; IU <= IL; IU++)
      {
         VL = 0.0;
         for (M = 0; M < N; M++)
         {
            VL += Z[M+IL*N] * Z[M+IU*N];
         }
         if (IU < IL)
         {
            VL = fabs(VL);
         }
         else
         {
            VL = fabs(VL-1.0);
         }
         if (VL > VU)
         {
            VU = VL;
         }
         if (VU > 0.5)
         {
            mfem::err << "dsyevr_Eigensystem(...):"
                      << " Z^t Z - I deviation = " << VU
                      << "\n W[max] = " << W[N-1] << ", W[min] = "
                      << W[0] << ", N = " << N << endl;
            mfem_error();
         }
      }
   if (VU > 1e-9)
   {
      mfem::err << "dsyevr_Eigensystem(...):"
                << " Z^t Z - I deviation = " << VU
                << "\n W[max] = " << W[N-1] << ", W[min] = "
                << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
   {
      mfem_error("dsyevr_Eigensystem(...): ERROR: ...");
   }
   VU = 0.0;
   for (IL = 0; IL < N; IL++)
      for (IU = 0; IU < N; IU++)
      {
         VL = 0.0;
         for (M = 0; M < N; M++)
         {
            VL += Z[IL+M*N] * W[M] * Z[IU+M*N];
         }
         VL = fabs(VL-data[IL+N*IU]);
         if (VL > VU)
         {
            VU = VL;
         }
      }
   if (VU > 1e-9)
   {
      mfem::err << "dsyevr_Eigensystem(...):"
                << " max matrix deviation = " << VU
                << "\n W[max] = " << W[N-1] << ", W[min] = "
                << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
   {
      mfem_error("dsyevr_Eigensystem(...): ERROR: ...");
   }
#endif

   delete [] IWORK;
   delete [] WORK;
   delete [] ISUPPZ;
   delete [] A;
#else
   MFEM_CONTRACT_VAR(a);
   MFEM_CONTRACT_VAR(ev);
   MFEM_CONTRACT_VAR(evect);
#endif
}

void dsyev_Eigensystem(DenseMatrix &a, Vector &ev, DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK
   int   N      = a.Width();
   char  JOBZ   = 'N';
   char  UPLO   = 'U';
   int   LDA    = N;
   int   LWORK  = -1; /* query optimal workspace size */
   int   INFO;

   ev.SetSize(N);

   real_t *A    = NULL;
   real_t *W    = ev.GetData();
   real_t *WORK = NULL;
   real_t  QWORK;

   if (evect)
   {
      JOBZ = 'V';
      evect->SetSize(N);
      A = evect->Data();
   }
   else
   {
      A = new real_t[N*N];
   }

   int hw = a.Height() * a.Width();
   real_t *data = a.Data();
   for (int i = 0; i < hw; i++)
   {
      A[i] = data[i];
   }

   MFEM_LAPACK_PREFIX(syev_)(&JOBZ, &UPLO, &N, A, &LDA, W, &QWORK, &LWORK, &INFO);

   LWORK = (int) QWORK;
   WORK = new real_t[LWORK];

   MFEM_LAPACK_PREFIX(syev_)(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);

   if (INFO != 0)
   {
      mfem::err << "dsyev_Eigensystem: DSYEV error code: " << INFO << endl;
      mfem_error();
   }

   delete [] WORK;
   if (evect == NULL) { delete [] A; }
#else
   MFEM_CONTRACT_VAR(a);
   MFEM_CONTRACT_VAR(ev);
   MFEM_CONTRACT_VAR(evect);
#endif
}

void DenseMatrix::Eigensystem(Vector &ev, DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK

   // dsyevr_Eigensystem(*this, ev, evect);

   dsyev_Eigensystem(*this, ev, evect);

#else

   MFEM_CONTRACT_VAR(ev);
   MFEM_CONTRACT_VAR(evect);
   mfem_error("DenseMatrix::Eigensystem: Compiled without LAPACK");

#endif
}

void dsygv_Eigensystem(DenseMatrix &a, DenseMatrix &b, Vector &ev,
                       DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK
   int   N      = a.Width();
   int   ITYPE  = 1;
   char  JOBZ   = 'N';
   char  UPLO   = 'U';
   int   LDA    = N;
   int   LDB    = N;
   int   LWORK  = -1; /* query optimal workspace size */
   int   INFO;

   ev.SetSize(N);

   real_t *A    = NULL;
   real_t *B    = new real_t[N*N];
   real_t *W    = ev.GetData();
   real_t *WORK = NULL;
   real_t  QWORK;

   if (evect)
   {
      JOBZ = 'V';
      evect->SetSize(N);
      A = evect->Data();
   }
   else
   {
      A = new real_t[N*N];
   }

   int hw = a.Height() * a.Width();
   real_t *a_data = a.Data();
   real_t *b_data = b.Data();
   for (int i = 0; i < hw; i++)
   {
      A[i] = a_data[i];
      B[i] = b_data[i];
   }

   MFEM_LAPACK_PREFIX(sygv_)(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W,
                             &QWORK, &LWORK, &INFO);

   LWORK = (int) QWORK;
   WORK = new real_t[LWORK];

   MFEM_LAPACK_PREFIX(sygv_)(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, WORK,
                             &LWORK, &INFO);

   if (INFO != 0)
   {
      mfem::err << "dsygv_Eigensystem: DSYGV error code: " << INFO << endl;
      mfem_error();
   }

   delete [] WORK;
   delete [] B;
   if (evect == NULL) { delete [] A; }
#else
   MFEM_CONTRACT_VAR(a);
   MFEM_CONTRACT_VAR(b);
   MFEM_CONTRACT_VAR(ev);
   MFEM_CONTRACT_VAR(evect);
#endif
}

void DenseMatrix::Eigensystem(DenseMatrix &b, Vector &ev,
                              DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK

   dsygv_Eigensystem(*this, b, ev, evect);

#else
   MFEM_CONTRACT_VAR(b);
   MFEM_CONTRACT_VAR(ev);
   MFEM_CONTRACT_VAR(evect);
   mfem_error("DenseMatrix::Eigensystem(generalized): Compiled without LAPACK");
#endif
}

void DenseMatrix::SingularValues(Vector &sv) const
{
#ifdef MFEM_USE_LAPACK
   DenseMatrix copy_of_this = *this;
   char        jobu         = 'N';
   char        jobvt        = 'N';
   int         m            = Height();
   int         n            = Width();
   real_t      *a           = copy_of_this.data;
   sv.SetSize(min(m, n));
   real_t      *s           = sv.GetData();
   real_t      *u           = NULL;
   real_t      *vt          = NULL;
   real_t      *work        = NULL;
   int         lwork        = -1;
   int         info;
   real_t      qwork;

   MFEM_LAPACK_PREFIX(gesvd_)(&jobu, &jobvt, &m, &n, a, &m, s, u, &m, vt, &n,
                              &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new real_t[lwork];

   MFEM_LAPACK_PREFIX(gesvd_)(&jobu, &jobvt, &m, &n, a, &m, s, u, &m, vt, &n,
                              work, &lwork, &info);

   delete [] work;
   if (info)
   {
      mfem::err << "DenseMatrix::SingularValues : info = " << info << endl;
      mfem_error();
   }
#else
   MFEM_CONTRACT_VAR(sv);
   // compiling without lapack
   mfem_error("DenseMatrix::SingularValues: Compiled without LAPACK");
#endif
}

int DenseMatrix::Rank(real_t tol) const
{
   int rank=0;
   Vector sv(min(Height(), Width()));
   SingularValues(sv);

   for (int i=0; i < sv.Size(); ++i)
      if (sv(i) >= tol)
      {
         ++rank;
      }

   return rank;
}

real_t DenseMatrix::CalcSingularvalue(const int i) const
{
   MFEM_ASSERT(Height() == Width() && Height() > 0 && Height() < 4,
               "The matrix must be square and sized 1, 2, or 3 to compute the"
               " singular values."
               << "  Height() = " << Height()
               << ", Width() = " << Width());

   const int n = Height();
   const real_t *d = data;

   if (n == 1)
   {
      return d[0];
   }
   else if (n == 2)
   {
      return kernels::CalcSingularvalue<2>(d,i);
   }
   else
   {
      return kernels::CalcSingularvalue<3>(d,i);
   }
}

void DenseMatrix::CalcEigenvalues(real_t *lambda, real_t *vec) const
{
#ifdef MFEM_DEBUG
   if (Height() != Width() || Height() < 2 || Height() > 3)
   {
      mfem_error("DenseMatrix::CalcEigenvalues");
   }
#endif

   const int n = Height();
   const real_t *d = data;

   if (n == 2)
   {
      kernels::CalcEigenvalues<2>(d, lambda, vec);
   }
   else
   {
      kernels::CalcEigenvalues<3>(d, lambda, vec);
   }
}

void DenseMatrix::GetRow(int r, Vector &row) const
{
   int m = Height();
   int n = Width();
   row.SetSize(n);

   const real_t* rp = data + r;
   real_t* vp = row.GetData();

   for (int i = 0; i < n; i++)
   {
      vp[i] = *rp;
      rp += m;
   }
}

void DenseMatrix::GetColumn(int c, Vector &col) const
{
   int m = Height();
   col.SetSize(m);

   real_t *cp = Data() + c * m;
   real_t *vp = col.GetData();

   for (int i = 0; i < m; i++)
   {
      vp[i] = cp[i];
   }
}

void DenseMatrix::GetDiag(Vector &d) const
{
   if (height != width)
   {
      mfem_error("DenseMatrix::GetDiag\n");
   }
   d.SetSize(height);

   for (int i = 0; i < height; ++i)
   {
      d(i) = (*this)(i,i);
   }
}

void DenseMatrix::Getl1Diag(Vector &l) const
{
   if (height != width)
   {
      mfem_error("DenseMatrix::Getl1Diag\n");
   }
   l.SetSize(height);

   l = 0.0;

   for (int j = 0; j < width; ++j)
      for (int i = 0; i < height; ++i)
      {
         l(i) += fabs((*this)(i,j));
      }
}

void DenseMatrix::GetRowl1(Vector &l) const
{
   l.SetSize(height);
   l = 0.0;

   for (int j = 0; j < width; ++j)
      for (int i = 0; i < height; ++i)
      {
         l(i) += fabs((*this)(i,j));
      }
}

void DenseMatrix::GetRowl2(Vector &l) const
{
   l.SetSize(height);
   l = 0.0;

   for (int j = 0; j < width; ++j)
      for (int i = 0; i < height; ++i)
      {
         l[i] += operator()(i,j)*operator()(i,j);
      }

   for (int i = 0; i < height; ++i)
   {
      l[i] = sqrt(l[i]);
   }
}

void DenseMatrix::GetRowSums(Vector &l) const
{
   l.SetSize(height);
   for (int i = 0; i < height; i++)
   {
      real_t d = 0.0;
      for (int j = 0; j < width; j++)
      {
         d += operator()(i, j);
      }
      l(i) = d;
   }
}

void DenseMatrix::Diag(real_t c, int n)
{
   SetSize(n);

   const int N = n*n;
   for (int i = 0; i < N; i++)
   {
      data[i] = 0.0;
   }
   for (int i = 0; i < n; i++)
   {
      data[i*(n+1)] = c;
   }
}

void DenseMatrix::Diag(real_t *diag, int n)
{
   SetSize(n);

   int i, N = n*n;
   for (i = 0; i < N; i++)
   {
      data[i] = 0.0;
   }
   for (i = 0; i < n; i++)
   {
      data[i*(n+1)] = diag[i];
   }
}

void DenseMatrix::Transpose()
{
   int i, j;
   real_t t;

   if (Width() == Height())
   {
      for (i = 0; i < Height(); i++)
         for (j = i+1; j < Width(); j++)
         {
            t = (*this)(i,j);
            (*this)(i,j) = (*this)(j,i);
            (*this)(j,i) = t;
         }
   }
   else
   {
      DenseMatrix T(*this,'t');
      (*this) = T;
   }
}

void DenseMatrix::Transpose(const DenseMatrix &A)
{
   SetSize(A.Width(),A.Height());

   for (int i = 0; i < Height(); i++)
      for (int j = 0; j < Width(); j++)
      {
         (*this)(i,j) = A(j,i);
      }
}

void DenseMatrix::Symmetrize()
{
#ifdef MFEM_DEBUG
   if (Width() != Height())
   {
      mfem_error("DenseMatrix::Symmetrize() : not a square matrix!");
   }
#endif
   kernels::Symmetrize(Height(), Data());
}

void DenseMatrix::Lump()
{
   for (int i = 0; i < Height(); i++)
   {
      real_t L = 0.0;
      for (int j = 0; j < Width(); j++)
      {
         L += (*this)(i, j);
         (*this)(i, j) = 0.0;
      }
      (*this)(i, i) = L;
   }
}

void DenseMatrix::GradToCurl(DenseMatrix &curl)
{
   int n = Height();

#ifdef MFEM_DEBUG
   if ((Width() != 2 || curl.Width() != 1 || 2*n != curl.Height()) &&
       (Width() != 3 || curl.Width() != 3 || 3*n != curl.Height()))
   {
      mfem_error("DenseMatrix::GradToCurl(...): dimension mismatch");
   }
#endif

   if (Width() == 2)
   {
      for (int i = 0; i < n; i++)
      {
         // (x,y) is grad of Ui
         real_t x = (*this)(i,0);
         real_t y = (*this)(i,1);

         int j = i+n;

         // curl of (Ui,0)
         curl(i,0) = -y;

         // curl of (0,Ui)
         curl(j,0) = x;
      }
   }
   else
   {
      for (int i = 0; i < n; i++)
      {
         // (x,y,z) is grad of Ui
         real_t x = (*this)(i,0);
         real_t y = (*this)(i,1);
         real_t z = (*this)(i,2);

         int j = i+n;
         int k = j+n;

         // curl of (Ui,0,0)
         curl(i,0) =  0.;
         curl(i,1) =  z;
         curl(i,2) = -y;

         // curl of (0,Ui,0)
         curl(j,0) = -z;
         curl(j,1) =  0.;
         curl(j,2) =  x;

         // curl of (0,0,Ui)
         curl(k,0) =  y;
         curl(k,1) = -x;
         curl(k,2) =  0.;
      }
   }
}

void DenseMatrix::GradToVectorCurl2D(DenseMatrix &curl)
{
   MFEM_VERIFY(Width() == 2,
               "DenseMatrix::GradToVectorCurl2D(...): dimension must be 2")

   int n = Height();
   // rotate gradient
   for (int i = 0; i < n; i++)
   {
      curl(i,0) = (*this)(i,1);
      curl(i,1) = -(*this)(i,0);
   }
}

void DenseMatrix::GradToDiv(Vector &div)
{
   MFEM_ASSERT(Width()*Height() == div.Size(), "incompatible Vector 'div'!");

   // div(dof*j+i) <-- (*this)(i,j)

   const int n = height * width;
   real_t *ddata = div.GetData();

   for (int i = 0; i < n; i++)
   {
      ddata[i] = data[i];
   }
}

void DenseMatrix::CopyRows(const DenseMatrix &A, int row1, int row2)
{
   SetSize(row2 - row1 + 1, A.Width());

   for (int j = 0; j < Width(); j++)
   {
      for (int i = row1; i <= row2; i++)
      {
         (*this)(i-row1,j) = A(i,j);
      }
   }
}

void DenseMatrix::CopyCols(const DenseMatrix &A, int col1, int col2)
{
   SetSize(A.Height(), col2 - col1 + 1);

   for (int j = col1; j <= col2; j++)
   {
      for (int i = 0; i < Height(); i++)
      {
         (*this)(i,j-col1) = A(i,j);
      }
   }
}

void DenseMatrix::CopyMN(const DenseMatrix &A, int m, int n, int Aro, int Aco)
{
   SetSize(m,n);

   for (int j = 0; j < n; j++)
   {
      for (int i = 0; i < m; i++)
      {
         (*this)(i,j) = A(Aro+i,Aco+j);
      }
   }
}

void DenseMatrix::CopyMN(const DenseMatrix &A, int row_offset, int col_offset)
{
   real_t *v = A.Data();

   for (int j = 0; j < A.Width(); j++)
   {
      for (int i = 0; i < A.Height(); i++)
      {
         (*this)(row_offset+i,col_offset+j) = *(v++);
      }
   }
}

void DenseMatrix::CopyMNt(const DenseMatrix &A, int row_offset, int col_offset)
{
   real_t *v = A.Data();

   for (int i = 0; i < A.Width(); i++)
   {
      for (int j = 0; j < A.Height(); j++)
      {
         (*this)(row_offset+i,col_offset+j) = *(v++);
      }
   }
}

void DenseMatrix::CopyMN(const DenseMatrix &A, int m, int n, int Aro, int Aco,
                         int row_offset, int col_offset)
{
   MFEM_VERIFY(row_offset+m <= this->Height() && col_offset+n <= this->Width(),
               "this DenseMatrix is too small to accommodate the submatrix.  "
               << "row_offset = " << row_offset
               << ", m = " << m
               << ", this->Height() = " << this->Height()
               << ", col_offset = " << col_offset
               << ", n = " << n
               << ", this->Width() = " << this->Width()
              );
   MFEM_VERIFY(Aro+m <= A.Height() && Aco+n <= A.Width(),
               "The A DenseMatrix is too small to accommodate the submatrix.  "
               << "Aro = " << Aro
               << ", m = " << m
               << ", A.Height() = " << A.Height()
               << ", Aco = " << Aco
               << ", n = " << n
               << ", A.Width() = " << A.Width()
              );

   for (int j = 0; j < n; j++)
   {
      for (int i = 0; i < m; i++)
      {
         (*this)(row_offset+i,col_offset+j) = A(Aro+i,Aco+j);
      }
   }
}

void DenseMatrix::CopyMNDiag(real_t c, int n, int row_offset, int col_offset)
{
   for (int i = 0; i < n; i++)
   {
      for (int j = i+1; j < n; j++)
      {
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;
      }
   }

   for (int i = 0; i < n; i++)
   {
      (*this)(row_offset+i,col_offset+i) = c;
   }
}

void DenseMatrix::CopyMNDiag(real_t *diag, int n, int row_offset,
                             int col_offset)
{
   for (int i = 0; i < n; i++)
   {
      for (int j = i+1; j < n; j++)
      {
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;
      }
   }

   for (int i = 0; i < n; i++)
   {
      (*this)(row_offset+i,col_offset+i) = diag[i];
   }
}

void DenseMatrix::CopyExceptMN(const DenseMatrix &A, int m, int n)
{
   SetSize(A.Width()-1,A.Height()-1);

   int i, j, i_off = 0, j_off = 0;

   for (j = 0; j < A.Width(); j++)
   {
      if ( j == n )
      {
         j_off = 1;
         continue;
      }
      for (i = 0; i < A.Height(); i++)
      {
         if ( i == m )
         {
            i_off = 1;
            continue;
         }
         (*this)(i-i_off,j-j_off) = A(i,j);
      }
      i_off = 0;
   }
}

void DenseMatrix::AddMatrix(DenseMatrix &A, int ro, int co)
{
   int h, ah, aw;
   real_t *p, *ap;

   h  = Height();
   ah = A.Height();
   aw = A.Width();

#ifdef MFEM_DEBUG
   if (co+aw > Width() || ro+ah > h)
   {
      mfem_error("DenseMatrix::AddMatrix(...) 1 : dimension mismatch");
   }
#endif

   p  = data + ro + co * h;
   ap = A.data;

   for (int c = 0; c < aw; c++)
   {
      for (int r = 0; r < ah; r++)
      {
         p[r] += ap[r];
      }
      p  += h;
      ap += ah;
   }
}

void DenseMatrix::AddMatrix(real_t a, const DenseMatrix &A, int ro, int co)
{
   int h, ah, aw;
   real_t *p, *ap;

   h  = Height();
   ah = A.Height();
   aw = A.Width();

#ifdef MFEM_DEBUG
   if (co+aw > Width() || ro+ah > h)
   {
      mfem_error("DenseMatrix::AddMatrix(...) 2 : dimension mismatch");
   }
#endif

   p  = data + ro + co * h;
   ap = A.Data();

   for (int c = 0; c < aw; c++)
   {
      for (int r = 0; r < ah; r++)
      {
         p[r] += a * ap[r];
      }
      p  += h;
      ap += ah;
   }
}

void DenseMatrix::GetSubMatrix(const Array<int> & idx, DenseMatrix & A) const
{
   int k = idx.Size();
   int idx_max = idx.Max();
   MFEM_VERIFY(idx.Min() >=0 && idx_max < this->height && idx_max < this->width,
               "DenseMatrix::GetSubMatrix: Index out of bounds");
   A.SetSize(k);
   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx[i];
      for (int j = 0; j<k; j++)
      {
         jj = idx[j];
         adata[i+j*k] = this->data[ii+jj*height];
      }
   }
}

void DenseMatrix::GetSubMatrix(const Array<int> & idx_i,
                               const Array<int> & idx_j, DenseMatrix & A) const
{
   int k = idx_i.Size();
   int l = idx_j.Size();

   MFEM_VERIFY(idx_i.Min() >=0 && idx_i.Max() < this->height,
               "DenseMatrix::GetSubMatrix: Row index out of bounds");
   MFEM_VERIFY(idx_j.Min() >=0 && idx_j.Max() < this->width,
               "DenseMatrix::GetSubMatrix: Col index out of bounds");

   A.SetSize(k,l);
   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx_i[i];
      for (int j = 0; j<l; j++)
      {
         jj = idx_j[j];
         adata[i+j*k] = this->data[ii+jj*height];
      }
   }
}

void DenseMatrix::GetSubMatrix(int ibeg, int iend, DenseMatrix & A)
{
   MFEM_VERIFY(iend >= ibeg, "DenseMatrix::GetSubMatrix: Inconsistent range");
   MFEM_VERIFY(ibeg >=0,
               "DenseMatrix::GetSubMatrix: Negative index");
   MFEM_VERIFY(iend <= this->height && iend <= this->width,
               "DenseMatrix::GetSubMatrix: Index bigger than upper bound");

   int k = iend - ibeg;
   A.SetSize(k);
   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<k; j++)
      {
         jj = ibeg + j;
         adata[i+j*k] = this->data[ii+jj*height];
      }
   }
}

void DenseMatrix::GetSubMatrix(int ibeg, int iend, int jbeg, int jend,
                               DenseMatrix & A)
{
   MFEM_VERIFY(iend >= ibeg,
               "DenseMatrix::GetSubMatrix: Inconsistent row range");
   MFEM_VERIFY(jend >= jbeg,
               "DenseMatrix::GetSubMatrix: Inconsistent col range");
   MFEM_VERIFY(ibeg >=0,
               "DenseMatrix::GetSubMatrix: Negative row index");
   MFEM_VERIFY(jbeg >=0,
               "DenseMatrix::GetSubMatrix: Negative row index");
   MFEM_VERIFY(iend <= this->height,
               "DenseMatrix::GetSubMatrix: Index bigger than row upper bound");
   MFEM_VERIFY(jend <= this->width,
               "DenseMatrix::GetSubMatrix: Index bigger than col upper bound");

   int k = iend - ibeg;
   int l = jend - jbeg;
   A.SetSize(k,l);
   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<l; j++)
      {
         jj = jbeg + j;
         adata[i+j*k] = this->data[ii+jj*height];
      }
   }
}

void DenseMatrix::SetSubMatrix(const Array<int> & idx, const DenseMatrix & A)
{
   int k = idx.Size();
   MFEM_VERIFY(A.Height() == k && A.Width() == k,
               "DenseMatrix::SetSubMatrix:Inconsistent matrix dimensions");

   int idx_max = idx.Max();

   MFEM_VERIFY(idx.Min() >=0,
               "DenseMatrix::SetSubMatrix: Negative index");
   MFEM_VERIFY(idx_max < this->height,
               "DenseMatrix::SetSubMatrix: Index bigger than row upper bound");
   MFEM_VERIFY(idx_max < this->width,
               "DenseMatrix::SetSubMatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx[i];
      for (int j = 0; j<k; j++)
      {
         jj = idx[j];
         this->data[ii+jj*height] = adata[i+j*k];
      }
   }
}

void DenseMatrix::SetSubMatrix(const Array<int> & idx_i,
                               const Array<int> & idx_j, const DenseMatrix & A)
{
   int k = idx_i.Size();
   int l = idx_j.Size();
   MFEM_VERIFY(k == A.Height() && l == A.Width(),
               "DenseMatrix::SetSubMatrix:Inconsistent matrix dimensions");
   MFEM_VERIFY(idx_i.Min() >=0,
               "DenseMatrix::SetSubMatrix: Negative row index");
   MFEM_VERIFY(idx_j.Min() >=0,
               "DenseMatrix::SetSubMatrix: Negative col index");
   MFEM_VERIFY(idx_i.Max() < this->height,
               "DenseMatrix::SetSubMatrix: Index bigger than row upper bound");
   MFEM_VERIFY(idx_j.Max() < this->width,
               "DenseMatrix::SetSubMatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx_i[i];
      for (int j = 0; j<l; j++)
      {
         jj = idx_j[j];
         this->data[ii+jj*height] = adata[i+j*k];
      }
   }
}

void DenseMatrix::SetSubMatrix(int ibeg, const DenseMatrix & A)
{
   int k = A.Height();

   MFEM_VERIFY(A.Width() == k, "DenseMatrix::SetSubmatrix: A is not square");
   MFEM_VERIFY(ibeg >=0,
               "DenseMatrix::SetSubmatrix: Negative index");
   MFEM_VERIFY(ibeg + k <= this->height,
               "DenseMatrix::SetSubmatrix: index bigger than row upper bound");
   MFEM_VERIFY(ibeg + k <= this->width,
               "DenseMatrix::SetSubmatrix: index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<k; j++)
      {
         jj = ibeg + j;
         this->data[ii+jj*height] = adata[i+j*k];
      }
   }
}

void DenseMatrix::SetSubMatrix(int ibeg, int jbeg, const DenseMatrix & A)
{
   int k = A.Height();
   int l = A.Width();

   MFEM_VERIFY(ibeg>=0,
               "DenseMatrix::SetSubmatrix: Negative row index");
   MFEM_VERIFY(jbeg>=0,
               "DenseMatrix::SetSubmatrix: Negative col index");
   MFEM_VERIFY(ibeg + k <= this->height,
               "DenseMatrix::SetSubmatrix: Index bigger than row upper bound");
   MFEM_VERIFY(jbeg + l <= this->width,
               "DenseMatrix::SetSubmatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<l; j++)
      {
         jj = jbeg + j;
         this->data[ii+jj*height] = adata[i+j*k];
      }
   }
}

void DenseMatrix::AddSubMatrix(const Array<int> & idx, const DenseMatrix & A)
{
   int k = idx.Size();
   MFEM_VERIFY(A.Height() == k && A.Width() == k,
               "DenseMatrix::AddSubMatrix:Inconsistent matrix dimensions");

   int idx_max = idx.Max();

   MFEM_VERIFY(idx.Min() >=0, "DenseMatrix::AddSubMatrix: Negative index");
   MFEM_VERIFY(idx_max < this->height,
               "DenseMatrix::AddSubMatrix: Index bigger than row upper bound");
   MFEM_VERIFY(idx_max < this->width,
               "DenseMatrix::AddSubMatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx[i];
      for (int j = 0; j<k; j++)
      {
         jj = idx[j];
         this->data[ii+jj*height] += adata[i+j*k];
      }
   }
}

void DenseMatrix::AddSubMatrix(const Array<int> & idx_i,
                               const Array<int> & idx_j, const DenseMatrix & A)
{
   int k = idx_i.Size();
   int l = idx_j.Size();
   MFEM_VERIFY(k == A.Height() && l == A.Width(),
               "DenseMatrix::AddSubMatrix:Inconsistent matrix dimensions");

   MFEM_VERIFY(idx_i.Min() >=0,
               "DenseMatrix::AddSubMatrix: Negative row index");
   MFEM_VERIFY(idx_j.Min() >=0,
               "DenseMatrix::AddSubMatrix: Negative col index");
   MFEM_VERIFY(idx_i.Max() < this->height,
               "DenseMatrix::AddSubMatrix: Index bigger than row upper bound");
   MFEM_VERIFY(idx_j.Max() < this->width,
               "DenseMatrix::AddSubMatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = idx_i[i];
      for (int j = 0; j<l; j++)
      {
         jj = idx_j[j];
         this->data[ii+jj*height] += adata[i+j*k];
      }
   }
}

void DenseMatrix::AddSubMatrix(int ibeg, const DenseMatrix & A)
{
   int k = A.Height();
   MFEM_VERIFY(A.Width() == k, "DenseMatrix::AddSubmatrix: A is not square");

   MFEM_VERIFY(ibeg>=0,
               "DenseMatrix::AddSubmatrix: Negative index");
   MFEM_VERIFY(ibeg + k <= this->Height(),
               "DenseMatrix::AddSubmatrix: Index bigger than row upper bound");
   MFEM_VERIFY(ibeg + k <= this->Width(),
               "DenseMatrix::AddSubmatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<k; j++)
      {
         jj = ibeg + j;
         this->data[ii+jj*height] += adata[i+j*k];
      }
   }
}

void DenseMatrix::AddSubMatrix(int ibeg, int jbeg, const DenseMatrix & A)
{
   int k = A.Height();
   int l = A.Width();

   MFEM_VERIFY(ibeg>=0,
               "DenseMatrix::AddSubmatrix: Negative row index");
   MFEM_VERIFY(jbeg>=0,
               "DenseMatrix::AddSubmatrix: Negative col index");
   MFEM_VERIFY(ibeg + k <= this->height,
               "DenseMatrix::AddSubmatrix: Index bigger than row upper bound");
   MFEM_VERIFY(jbeg + l <= this->width,
               "DenseMatrix::AddSubmatrix: Index bigger than col upper bound");

   real_t * adata = A.Data();

   int ii, jj;
   for (int i = 0; i<k; i++)
   {
      ii = ibeg + i;
      for (int j = 0; j<l; j++)
      {
         jj = jbeg + j;
         this->data[ii+jj*height] += adata[i+j*k];
      }
   }
}

void DenseMatrix::AddToVector(int offset, Vector &v) const
{
   const int n = height * width;
   real_t *vdata = v.GetData() + offset;

   for (int i = 0; i < n; i++)
   {
      vdata[i] += data[i];
   }
}

void DenseMatrix::GetFromVector(int offset, const Vector &v)
{
   const int n = height * width;
   const real_t *vdata = v.GetData() + offset;

   for (int i = 0; i < n; i++)
   {
      data[i] = vdata[i];
   }
}

void DenseMatrix::AdjustDofDirection(Array<int> &dofs)
{
   const int n = Height();

#ifdef MFEM_DEBUG
   if (dofs.Size() != n || Width() != n)
   {
      mfem_error("DenseMatrix::AdjustDofDirection(...): dimension mismatch");
   }
#endif

   int *dof = dofs;
   for (int i = 0; i < n-1; i++)
   {
      const int s = (dof[i] < 0) ? (-1) : (1);
      for (int j = i+1; j < n; j++)
      {
         const int t = (dof[j] < 0) ? (-s) : (s);
         if (t < 0)
         {
            (*this)(i,j) = -(*this)(i,j);
            (*this)(j,i) = -(*this)(j,i);
         }
      }
   }
}

void DenseMatrix::SetRow(int row, real_t value)
{
   for (int j = 0; j < Width(); j++)
   {
      (*this)(row, j) = value;
   }
}

void DenseMatrix::SetCol(int col, real_t value)
{
   for (int i = 0; i < Height(); i++)
   {
      (*this)(i, col) = value;
   }
}

void DenseMatrix::SetRow(int r, const real_t* row)
{
   MFEM_ASSERT(row != nullptr, "supplied row pointer is null");
   for (int j = 0; j < Width(); j++)
   {
      (*this)(r, j) = row[j];
   }
}

void DenseMatrix::SetRow(int r, const Vector &row)
{
   MFEM_ASSERT(Width() == row.Size(), "");
   SetRow(r, row.GetData());
}

void DenseMatrix::SetCol(int c, const real_t* col)
{
   MFEM_ASSERT(col != nullptr, "supplied column pointer is null");
   for (int i = 0; i < Height(); i++)
   {
      (*this)(i, c) = col[i];
   }
}

void DenseMatrix::SetCol(int c, const Vector &col)
{
   MFEM_ASSERT(Height() == col.Size(), "");
   SetCol(c, col.GetData());
}

void DenseMatrix::Threshold(real_t eps)
{
   for (int col = 0; col < Width(); col++)
   {
      for (int row = 0; row < Height(); row++)
      {
         if (std::abs(operator()(row,col)) <= eps)
         {
            operator()(row,col) = 0.0;
         }
      }
   }
}

void DenseMatrix::Print(std::ostream &os, int width_) const
{
   // save current output flags
   ios::fmtflags old_flags = os.flags();
   // output flags = scientific + show sign
   os << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      os << "[row " << i << "]\n";
      for (int j = 0; j < width; j++)
      {
         os << (*this)(i,j);
         if (j+1 == width || (j+1) % width_ == 0)
         {
            os << '\n';
         }
         else
         {
            os << ' ';
         }
      }
   }
   // reset output flags to original values
   os.flags(old_flags);
}

void DenseMatrix::PrintMatlab(std::ostream &os) const
{
   // save current output flags
   ios::fmtflags old_flags = os.flags();
   // output flags = scientific + show sign
   os << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         os << (*this)(i,j);
         os << ' ';
      }
      os << "\n";
   }
   // reset output flags to original values
   os.flags(old_flags);
}

void DenseMatrix::PrintMathematica(std::ostream &os) const
{
   ios::fmtflags old_fmt = os.flags();
   os.setf(ios::scientific);
   std::streamsize old_prec = os.precision(14);

   os << "(* Read file into Mathematica using: "
      << "myMat = Get[\"this_file_name\"] *)\n";
   os << "{\n";

   for (int i = 0; i < height; i++)
   {
      os << "{\n";
      for (int j = 0; j < width; j++)
      {
         os << "Internal`StringToMReal[\"" << (*this)(i,j) << "\"]";
         if (j < width - 1) { os << ','; }
         os << '\n';
      }
      os << '}';
      if (i < height - 1) { os << ','; }
      os << '\n';
   }
   os << "}\n";

   os.precision(old_prec);
   os.flags(old_fmt);
}

void DenseMatrix::PrintT(std::ostream &os, int width_) const
{
   // save current output flags
   ios::fmtflags old_flags = os.flags();
   // output flags = scientific + show sign
   os << setiosflags(ios::scientific | ios::showpos);
   for (int j = 0; j < width; j++)
   {
      os << "[col " << j << "]\n";
      for (int i = 0; i < height; i++)
      {
         os << (*this)(i,j);
         if (i+1 == height || (i+1) % width_ == 0)
         {
            os << '\n';
         }
         else
         {
            os << ' ';
         }
      }
   }
   // reset output flags to original values
   os.flags(old_flags);
}

void DenseMatrix::TestInversion()
{
   DenseMatrix copy(*this), C(width);
   Invert();
   mfem::Mult(*this, copy, C);

   for (int i = 0; i < width; i++)
   {
      C(i,i) -= 1.0;
   }
   mfem::out << "size = " << width << ", i_max = " << C.MaxMaxNorm()
             << ", cond_F = " << FNorm()*copy.FNorm() << endl;
}

void DenseMatrix::Swap(DenseMatrix &other)
{
   mfem::Swap(*this, other);
}


void Add(const DenseMatrix &A, const DenseMatrix &B,
         real_t alpha, DenseMatrix &C)
{
   kernels::Add(C.Height(), C.Width(), alpha, A.Data(), B.Data(), C.Data());
}

void Add(real_t alpha, const real_t *A,
         real_t beta,  const real_t *B, DenseMatrix &C)
{
   kernels::Add(C.Height(), C.Width(), alpha, A, beta, B, C.Data());
}

void Add(real_t alpha, const DenseMatrix &A,
         real_t beta,  const DenseMatrix &B, DenseMatrix &C)
{
   MFEM_ASSERT(A.Height() == C.Height(), "");
   MFEM_ASSERT(B.Height() == C.Height(), "");
   MFEM_ASSERT(A.Width() == C.Width(), "");
   MFEM_ASSERT(B.Width() == C.Width(), "");
   Add(alpha, A.GetData(), beta, B.GetData(), C);
}

bool LinearSolve(DenseMatrix& A, real_t* X, real_t TOL)
{
   MFEM_VERIFY(A.IsSquare(), "A must be a square matrix!");
   MFEM_ASSERT(A.NumCols() > 0, "supplied matrix, A, is empty!");
   MFEM_ASSERT(X != nullptr, "supplied vector, X, is null!");

   int N = A.NumCols();

   switch (N)
   {
      case 1:
      {
         real_t det = A(0,0);
         if (std::abs(det) <= TOL) { return false; } // singular

         X[0] /= det;
         break;
      }
      case 2:
      {
         real_t det = A.Det();
         if (std::abs(det) <= TOL) { return false; } // singular

         real_t invdet = 1. / det;

         real_t b0 = X[0];
         real_t b1 = X[1];

         X[0] = ( A(1,1)*b0 - A(0,1)*b1) * invdet;
         X[1] = (-A(1,0)*b0 + A(0,0)*b1) * invdet;
         break;
      }
      default:
      {
         // default to LU factorization for the general case
         Array<int> ipiv(N);
         LUFactors lu(A.Data(), ipiv);

         if (!lu.Factor(N,TOL)) { return false; } // singular

         lu.Solve(N, 1, X);
      }

   } // END switch

   return true;
}

void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a)
{
   MFEM_ASSERT(a.Height() == b.Height() && a.Width() == c.Width() &&
               b.Width() == c.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'N';
   static real_t alpha = 1.0, beta = 0.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
                             c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   real_t *ad = a.Data();
   const real_t *bd = b.Data();
   const real_t *cd = c.Data();
   kernels::Mult(ah,aw,bw,bd,cd,ad);
#endif
}

void AddMult_a(real_t alpha, const DenseMatrix &b, const DenseMatrix &c,
               DenseMatrix &a)
{
   MFEM_ASSERT(a.Height() == b.Height() && a.Width() == c.Width() &&
               b.Width() == c.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'N';
   static real_t beta = 1.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
                             c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   real_t *ad = a.Data();
   const real_t *bd = b.Data();
   const real_t *cd = c.Data();
   for (int j = 0; j < aw; j++)
   {
      for (int k = 0; k < bw; k++)
      {
         for (int i = 0; i < ah; i++)
         {
            ad[i+j*ah] += alpha * bd[i+k*ah] * cd[k+j*bw];
         }
      }
   }
#endif
}

void AddMult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a)
{
   MFEM_ASSERT(a.Height() == b.Height() && a.Width() == c.Width() &&
               b.Width() == c.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'N';
   static real_t alpha = 1.0, beta = 1.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
                             c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   real_t *ad = a.Data();
   const real_t *bd = b.Data();
   const real_t *cd = c.Data();
   for (int j = 0; j < aw; j++)
   {
      for (int k = 0; k < bw; k++)
      {
         for (int i = 0; i < ah; i++)
         {
            ad[i+j*ah] += bd[i+k*ah] * cd[k+j*bw];
         }
      }
   }
#endif
}

void CalcAdjugate(const DenseMatrix &a, DenseMatrix &adja)
{
#ifdef MFEM_DEBUG
   if (a.Width() > a.Height() || a.Width() < 1 || a.Height() > 3)
   {
      mfem_error("CalcAdjugate(...): unsupported dimensions");
   }
   if (a.Width() != adja.Height() || a.Height() != adja.Width())
   {
      mfem_error("CalcAdjugate(...): dimension mismatch");
   }
#endif

   if (a.Width() < a.Height())
   {
      const real_t *d = a.Data();
      real_t *ad = adja.Data();
      if (a.Width() == 1)
      {
         // N x 1, N = 2,3
         ad[0] = d[0];
         ad[1] = d[1];
         if (a.Height() == 3)
         {
            ad[2] = d[2];
         }
      }
      else
      {
         // 3 x 2
         real_t e, g, f;
         e = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
         g = d[3]*d[3] + d[4]*d[4] + d[5]*d[5];
         f = d[0]*d[3] + d[1]*d[4] + d[2]*d[5];

         ad[0] = d[0]*g - d[3]*f;
         ad[1] = d[3]*e - d[0]*f;
         ad[2] = d[1]*g - d[4]*f;
         ad[3] = d[4]*e - d[1]*f;
         ad[4] = d[2]*g - d[5]*f;
         ad[5] = d[5]*e - d[2]*f;
      }
      return;
   }

   if (a.Width() == 1)
   {
      adja(0,0) = 1.0;
   }
   else if (a.Width() == 2)
   {
      adja(0,0) =  a(1,1);
      adja(0,1) = -a(0,1);
      adja(1,0) = -a(1,0);
      adja(1,1) =  a(0,0);
   }
   else
   {
      adja(0,0) = a(1,1)*a(2,2)-a(1,2)*a(2,1);
      adja(0,1) = a(0,2)*a(2,1)-a(0,1)*a(2,2);
      adja(0,2) = a(0,1)*a(1,2)-a(0,2)*a(1,1);

      adja(1,0) = a(1,2)*a(2,0)-a(1,0)*a(2,2);
      adja(1,1) = a(0,0)*a(2,2)-a(0,2)*a(2,0);
      adja(1,2) = a(0,2)*a(1,0)-a(0,0)*a(1,2);

      adja(2,0) = a(1,0)*a(2,1)-a(1,1)*a(2,0);
      adja(2,1) = a(0,1)*a(2,0)-a(0,0)*a(2,1);
      adja(2,2) = a(0,0)*a(1,1)-a(0,1)*a(1,0);
   }
}

void CalcAdjugateTranspose(const DenseMatrix &a, DenseMatrix &adjat)
{
#ifdef MFEM_DEBUG
   if (a.Height() != a.Width() || adjat.Height() != adjat.Width() ||
       a.Width() != adjat.Width() || a.Width() < 1 || a.Width() > 3)
   {
      mfem_error("CalcAdjugateTranspose(...): dimension mismatch");
   }
#endif
   if (a.Width() == 1)
   {
      adjat(0,0) = 1.0;
   }
   else if (a.Width() == 2)
   {
      adjat(0,0) =  a(1,1);
      adjat(1,0) = -a(0,1);
      adjat(0,1) = -a(1,0);
      adjat(1,1) =  a(0,0);
   }
   else
   {
      adjat(0,0) = a(1,1)*a(2,2)-a(1,2)*a(2,1);
      adjat(1,0) = a(0,2)*a(2,1)-a(0,1)*a(2,2);
      adjat(2,0) = a(0,1)*a(1,2)-a(0,2)*a(1,1);

      adjat(0,1) = a(1,2)*a(2,0)-a(1,0)*a(2,2);
      adjat(1,1) = a(0,0)*a(2,2)-a(0,2)*a(2,0);
      adjat(2,1) = a(0,2)*a(1,0)-a(0,0)*a(1,2);

      adjat(0,2) = a(1,0)*a(2,1)-a(1,1)*a(2,0);
      adjat(1,2) = a(0,1)*a(2,0)-a(0,0)*a(2,1);
      adjat(2,2) = a(0,0)*a(1,1)-a(0,1)*a(1,0);
   }
}

void CalcInverse(const DenseMatrix &a, DenseMatrix &inva)
{
   MFEM_ASSERT(a.Width() <= a.Height() && a.Width() >= 1 && a.Height() <= 3, "");
   MFEM_ASSERT(inva.Height() == a.Width(), "incorrect dimensions");
   MFEM_ASSERT(inva.Width() == a.Height(), "incorrect dimensions");

   if (a.Width() < a.Height())
   {
      const real_t *d = a.Data();
      real_t *id = inva.Data();
      if (a.Height() == 2)
      {
         kernels::CalcLeftInverse<2,1>(d, id);
      }
      else
      {
         if (a.Width() == 1)
         {
            kernels::CalcLeftInverse<3,1>(d, id);
         }
         else
         {
            kernels::CalcLeftInverse<3,2>(d, id);
         }
      }
      return;
   }

#ifdef MFEM_DEBUG
   const real_t t = a.Det();
   MFEM_ASSERT(std::abs(t) > 1.0e-14 * pow(a.FNorm()/a.Width(), a.Width()),
               "singular matrix!");
#endif

   switch (a.Height())
   {
      case 1:
         inva(0,0) = 1.0 / a.Det();
         break;
      case 2:
         kernels::CalcInverse<2>(a.Data(), inva.Data());
         break;
      case 3:
         kernels::CalcInverse<3>(a.Data(), inva.Data());
         break;
   }
}

void CalcInverseTranspose(const DenseMatrix &a, DenseMatrix &inva)
{
#ifdef MFEM_DEBUG
   if ( (a.Width() != a.Height()) || ( (a.Height()!= 1) && (a.Height()!= 2)
                                       && (a.Height()!= 3) ) )
   {
      mfem_error("CalcInverseTranspose(...): dimension mismatch");
   }
#endif

   real_t t = 1. / a.Det() ;

   switch (a.Height())
   {
      case 1:
         inva(0,0) = 1.0 / a(0,0);
         break;
      case 2:
         inva(0,0) = a(1,1) * t ;
         inva(1,0) = -a(0,1) * t ;
         inva(0,1) = -a(1,0) * t ;
         inva(1,1) = a(0,0) * t ;
         break;
      case 3:
         inva(0,0) = (a(1,1)*a(2,2)-a(1,2)*a(2,1))*t;
         inva(1,0) = (a(0,2)*a(2,1)-a(0,1)*a(2,2))*t;
         inva(2,0) = (a(0,1)*a(1,2)-a(0,2)*a(1,1))*t;

         inva(0,1) = (a(1,2)*a(2,0)-a(1,0)*a(2,2))*t;
         inva(1,1) = (a(0,0)*a(2,2)-a(0,2)*a(2,0))*t;
         inva(2,1) = (a(0,2)*a(1,0)-a(0,0)*a(1,2))*t;

         inva(0,2) = (a(1,0)*a(2,1)-a(1,1)*a(2,0))*t;
         inva(1,2) = (a(0,1)*a(2,0)-a(0,0)*a(2,1))*t;
         inva(2,2) = (a(0,0)*a(1,1)-a(0,1)*a(1,0))*t;
         break;
   }
}

void CalcOrtho(const DenseMatrix &J, Vector &n)
{
   MFEM_ASSERT( ((J.Height() == 2 && J.Width() == 1)
                 || (J.Height() == 3 && J.Width() == 2))
                && (J.Height() == n.Size()),
                "Matrix must be 3x2 or 2x1, "
                << "and the Vector must be sized with the rows. "
                << " J.Height() = " << J.Height()
                << ", J.Width() = " << J.Width()
                << ", n.Size() = " << n.Size()
              );

   const real_t *d = J.Data();
   if (J.Height() == 2)
   {
      n(0) =  d[1];
      n(1) = -d[0];
   }
   else
   {
      n(0) = d[1]*d[5] - d[2]*d[4];
      n(1) = d[2]*d[3] - d[0]*d[5];
      n(2) = d[0]*d[4] - d[1]*d[3];
   }
}

void MultAAt(const DenseMatrix &a, DenseMatrix &aat)
{
   const int height = a.Height();
   const int width = a.Width();
   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j <= i; j++)
      {
         real_t temp = 0.;
         for (int k = 0; k < width; k++)
         {
            temp += a(i,k) * a(j,k);
         }
         aat(j,i) = aat(i,j) = temp;
      }
   }
}

void AddMultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt)
{
   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = 0; j < i; j++)
      {
         real_t t = 0.;
         for (int k = 0; k < A.Width(); k++)
         {
            t += D(k) * A(i, k) * A(j, k);
         }
         ADAt(i, j) += t;
         ADAt(j, i) += t;
      }
   }

   // process diagonal
   for (int i = 0; i < A.Height(); i++)
   {
      real_t t = 0.;
      for (int k = 0; k < A.Width(); k++)
      {
         t += D(k) * A(i, k) * A(i, k);
      }
      ADAt(i, i) += t;
   }
}

void MultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt)
{
   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = 0; j <= i; j++)
      {
         real_t t = 0.;
         for (int k = 0; k < A.Width(); k++)
         {
            t += D(k) * A(i, k) * A(j, k);
         }
         ADAt(j, i) = ADAt(i, j) = t;
      }
   }
}

void MultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width() ||
       A.Width() != B.Width())
   {
      mfem_error("MultABt(...): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'T';
   static real_t alpha = 1.0, beta = 0.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
                             B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = ABt.Data();

   kernels::MultABt(ah, aw, bh, ad, bd, cd);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = ABt.Data();

   for (int j = 0; j < bh; j++)
      for (int i = 0; i < ah; i++)
      {
         real_t d = 0.0;
         const real_t *ap = ad + i;
         const real_t *bp = bd + j;
         for (int k = 0; k < aw; k++)
         {
            d += (*ap) * (*bp);
            ap += ah;
            bp += bh;
         }
         *(cd++) = d;
      }
#else
   int i, j, k;
   real_t d;

   for (i = 0; i < A.Height(); i++)
      for (j = 0; j < B.Height(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Width(); k++)
         {
            d += A(i, k) * B(j, k);
         }
         ABt(i, j) = d;
      }
#endif
}

void MultADBt(const DenseMatrix &A, const Vector &D,
              const DenseMatrix &B, DenseMatrix &ADBt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ADBt.Height() || B.Height() != ADBt.Width() ||
       A.Width() != B.Width() || A.Width() != D.Size())
   {
      mfem_error("MultADBt(...): dimension mismatch");
   }
#endif

   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   const real_t *dd = D.GetData();
   real_t *cd = ADBt.Data();

   for (int i = 0, s = ah*bh; i < s; i++)
   {
      cd[i] = 0.0;
   }
   for (int k = 0; k < aw; k++)
   {
      real_t *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const real_t dk_bjk = dd[k] * bd[j];
         for (int i = 0; i < ah; i++)
         {
            cp[i] += ad[i] * dk_bjk;
         }
         cp += ah;
      }
      ad += ah;
      bd += bh;
   }
}

void AddMultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width() ||
       A.Width() != B.Width())
   {
      mfem_error("AddMultABt(...): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'T';
   static real_t alpha = 1.0, beta = 1.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
                             B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = ABt.Data();

   for (int k = 0; k < aw; k++)
   {
      real_t *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const real_t bjk = bd[j];
         for (int i = 0; i < ah; i++)
         {
            cp[i] += ad[i] * bjk;
         }
         cp += ah;
      }
      ad += ah;
      bd += bh;
   }
#else
   int i, j, k;
   real_t d;

   for (i = 0; i < A.Height(); i++)
      for (j = 0; j < B.Height(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Width(); k++)
         {
            d += A(i, k) * B(j, k);
         }
         ABt(i, j) += d;
      }
#endif
}

void AddMultADBt(const DenseMatrix &A, const Vector &D,
                 const DenseMatrix &B, DenseMatrix &ADBt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ADBt.Height() || B.Height() != ADBt.Width() ||
       A.Width() != B.Width() || A.Width() != D.Size())
   {
      mfem_error("AddMultADBt(...): dimension mismatch");
   }
#endif

   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   const real_t *dd = D.GetData();
   real_t *cd = ADBt.Data();

   for (int k = 0; k < aw; k++)
   {
      real_t *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const real_t dk_bjk = dd[k] * bd[j];
         for (int i = 0; i < ah; i++)
         {
            cp[i] += ad[i] * dk_bjk;
         }
         cp += ah;
      }
      ad += ah;
      bd += bh;
   }
}

void AddMult_a_ABt(real_t a, const DenseMatrix &A, const DenseMatrix &B,
                   DenseMatrix &ABt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width() ||
       A.Width() != B.Width())
   {
      mfem_error("AddMult_a_ABt(...): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'T';
   real_t alpha = a;
   static real_t beta = 1.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
                             B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = ABt.Data();

   for (int k = 0; k < aw; k++)
   {
      real_t *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const real_t bjk = a * bd[j];
         for (int i = 0; i < ah; i++)
         {
            cp[i] += ad[i] * bjk;
         }
         cp += ah;
      }
      ad += ah;
      bd += bh;
   }
#else
   int i, j, k;
   real_t d;

   for (i = 0; i < A.Height(); i++)
      for (j = 0; j < B.Height(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Width(); k++)
         {
            d += A(i, k) * B(j, k);
         }
         ABt(i, j) += a * d;
      }
#endif
}

void MultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB)
{
#ifdef MFEM_DEBUG
   if (A.Width() != AtB.Height() || B.Width() != AtB.Width() ||
       A.Height() != B.Height())
   {
      mfem_error("MultAtB(...): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'T', transb = 'N';
   static real_t alpha = 1.0, beta = 0.0;
   int m = A.Width(), n = B.Width(), k = A.Height();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &k,
                             B.Data(), &k, &beta, AtB.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int aw = A.Width();
   const int bw = B.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = AtB.Data();

   for (int j = 0; j < bw; j++)
   {
      const real_t *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         real_t d = 0.0;
         for (int k = 0; k < ah; k++)
         {
            d += ap[k] * bd[k];
         }
         *(cd++) = d;
         ap += ah;
      }
      bd += ah;
   }
#else
   int i, j, k;
   real_t d;

   for (i = 0; i < A.Width(); i++)
      for (j = 0; j < B.Width(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Height(); k++)
         {
            d += A(k, i) * B(k, j);
         }
         AtB(i, j) = d;
      }
#endif
}

void AddMultAtB(const DenseMatrix &A, const DenseMatrix &B,
                DenseMatrix &AtB)
{
   MFEM_ASSERT(AtB.Height() == A.Width() && AtB.Width() == B.Width() &&
               A.Height() == B.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'T', transb = 'N';
   static real_t alpha = 1.0, beta = 1.0;
   int m = A.Width(), n = B.Width(), k = A.Height();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &k,
                             B.Data(), &k, &beta, AtB.Data(), &m);
#else
   const int ah = A.Height();
   const int aw = A.Width();
   const int bw = B.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = AtB.Data();

   for (int j = 0; j < bw; j++)
   {
      const real_t *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         real_t d = 0.0;
         for (int k = 0; k < ah; k++)
         {
            d += ap[k] * bd[k];
         }
         *(cd++) += d;
         ap += ah;
      }
      bd += ah;
   }
#endif
}

void AddMult_a_AtB(real_t a, const DenseMatrix &A, const DenseMatrix &B,
                   DenseMatrix &AtB)
{
   MFEM_ASSERT(AtB.Height() == A.Width() && AtB.Width() == B.Width() &&
               A.Height() == B.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'T', transb = 'N';
   real_t alpha = a;
   static real_t beta = 1.0;
   int m = A.Width(), n = B.Width(), k = A.Height();

   MFEM_LAPACK_PREFIX(gemm_)(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &k,
                             B.Data(), &k, &beta, AtB.Data(), &m);
#else
   const int ah = A.Height();
   const int aw = A.Width();
   const int bw = B.Width();
   const real_t *ad = A.Data();
   const real_t *bd = B.Data();
   real_t *cd = AtB.Data();

   for (int j = 0; j < bw; j++)
   {
      const real_t *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         real_t d = 0.0;
         for (int k = 0; k < ah; k++)
         {
            d += ap[k] * bd[k];
         }
         *(cd++) += a * d;
         ap += ah;
      }
      bd += ah;
   }
#endif
}

void AddMult_a_AAt(real_t a, const DenseMatrix &A, DenseMatrix &AAt)
{
   real_t d;

   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = 0; j < i; j++)
      {
         d = 0.;
         for (int k = 0; k < A.Width(); k++)
         {
            d += A(i,k) * A(j,k);
         }
         AAt(i, j) += (d *= a);
         AAt(j, i) += d;
      }
      d = 0.;
      for (int k = 0; k < A.Width(); k++)
      {
         d += A(i,k) * A(i,k);
      }
      AAt(i, i) += a * d;
   }
}

void Mult_a_AAt(real_t a, const DenseMatrix &A, DenseMatrix &AAt)
{
   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = 0; j <= i; j++)
      {
         real_t d = 0.;
         for (int k = 0; k < A.Width(); k++)
         {
            d += A(i,k) * A(j,k);
         }
         AAt(i, j) = AAt(j, i) = a * d;
      }
   }
}

void MultVVt(const Vector &v, DenseMatrix &vvt)
{
   for (int i = 0; i < v.Size(); i++)
   {
      for (int j = 0; j <= i; j++)
      {
         vvt(i,j) = vvt(j,i) = v(i) * v(j);
      }
   }
}

void MultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt)
{
#ifdef MFEM_DEBUG
   if (v.Size() != VWt.Height() || w.Size() != VWt.Width())
   {
      mfem_error("MultVWt(...): dimension mismatch");
   }
#endif

   for (int i = 0; i < v.Size(); i++)
   {
      const real_t vi = v(i);
      for (int j = 0; j < w.Size(); j++)
      {
         VWt(i, j) = vi * w(j);
      }
   }
}

void AddMultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt)
{
   const int m = v.Size(), n = w.Size();

#ifdef MFEM_DEBUG
   if (VWt.Height() != m || VWt.Width() != n)
   {
      mfem_error("AddMultVWt(...): dimension mismatch");
   }
#endif

   for (int i = 0; i < m; i++)
   {
      const real_t vi = v(i);
      for (int j = 0; j < n; j++)
      {
         VWt(i, j) += vi * w(j);
      }
   }
}

void AddMultVVt(const Vector &v, DenseMatrix &VVt)
{
   const int n = v.Size();

#ifdef MFEM_DEBUG
   if (VVt.Height() != n || VVt.Width() != n)
   {
      mfem_error("AddMultVVt(...): dimension mismatch");
   }
#endif

   for (int i = 0; i < n; i++)
   {
      const real_t vi = v(i);
      for (int j = 0; j < i; j++)
      {
         const real_t vivj = vi * v(j);
         VVt(i, j) += vivj;
         VVt(j, i) += vivj;
      }
      VVt(i, i) += vi * vi;
   }
}

void AddMult_a_VWt(const real_t a, const Vector &v, const Vector &w,
                   DenseMatrix &VWt)
{
   const int m = v.Size(), n = w.Size();

#ifdef MFEM_DEBUG
   if (VWt.Height() != m || VWt.Width() != n)
   {
      mfem_error("AddMult_a_VWt(...): dimension mismatch");
   }
#endif

   for (int j = 0; j < n; j++)
   {
      const real_t awj = a * w(j);
      for (int i = 0; i < m; i++)
      {
         VWt(i, j) += v(i) * awj;
      }
   }
}

void AddMult_a_VVt(const real_t a, const Vector &v, DenseMatrix &VVt)
{
   MFEM_ASSERT(VVt.Height() == v.Size() && VVt.Width() == v.Size(),
               "incompatible dimensions!");

   const int n = v.Size();
   for (int i = 0; i < n; i++)
   {
      real_t avi = a * v(i);
      for (int j = 0; j < i; j++)
      {
         const real_t avivj = avi * v(j);
         VVt(i, j) += avivj;
         VVt(j, i) += avivj;
      }
      VVt(i, i) += avi * v(i);
   }
}

void RAP(const DenseMatrix &A, const DenseMatrix &P, DenseMatrix & RAP)
{
   DenseMatrix RA(P.Width(),A.Width());
   MultAtB(P,A,RA);
   RAP.SetSize(RA.Height(), P.Width());
   Mult(RA,P, RAP);
}

void RAP(const DenseMatrix &Rt, const DenseMatrix &A,
         const DenseMatrix &P, DenseMatrix & RAP)
{
   DenseMatrix RA(Rt.Width(),A.Width());
   MultAtB(Rt,A,RA);
   RAP.SetSize(RA.Height(), P.Width());
   Mult(RA,P, RAP);
}

bool LUFactors::Factor(int m, real_t TOL)
{
#ifdef MFEM_USE_LAPACK
   int info = 0;
   if (m) { MFEM_LAPACK_PREFIX(getrf_)(&m, &m, data, &m, ipiv, &info); }
   return info == 0;
#else
   // compiling without LAPACK
   real_t *data_ptr = this->data;
   for (int i = 0; i < m; i++)
   {
      // pivoting
      {
         int piv = i;
         real_t a = std::abs(data_ptr[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const real_t b = std::abs(data_ptr[j+i*m]);
            if (b > a)
            {
               a = b;
               piv = j;
            }
         }
         ipiv[i] = piv + 1;
         if (piv != i)
         {
            // swap rows i and piv in both L and U parts
            for (int j = 0; j < m; j++)
            {
               mfem::Swap<real_t>(data_ptr[i+j*m], data_ptr[piv+j*m]);
            }
         }
      }

      if (abs(data_ptr[i + i*m]) <= TOL)
      {
         return false; // failed
      }

      const real_t a_ii_inv = 1.0 / data_ptr[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         data_ptr[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const real_t a_ik = data_ptr[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            data_ptr[j+k*m] -= a_ik * data_ptr[j+i*m];
         }
      }
   }
#endif

   return true; // success
}

real_t LUFactors::Det(int m) const
{
   real_t det = 1.0;
   for (int i=0; i<m; i++)
   {
      if (ipiv[i] != i - ipiv_base)
      {
         det *= -data[m * i + i];
      }
      else
      {
         det *=  data[m * i + i];
      }
   }
   return det;
}

void LUFactors::Mult(int m, int n, real_t *X) const
{
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- U X
      for (int i = 0; i < m; i++)
      {
         real_t x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- L X
      for (int i = m-1; i >= 0; i--)
      {
         real_t x_i = x[i];
         for (int j = 0; j < i; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- P^{-1} X
      for (int i = m-1; i >= 0; i--)
      {
         mfem::Swap<real_t>(x[i], x[ipiv[i]-ipiv_base]);
      }
      x += m;
   }
}

void LUFactors::LSolve(int m, int n, real_t *X) const
{
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      kernels::LSolve(data, m, ipiv, x);
      x += m;
   }
}

void LUFactors::USolve(int m, int n, real_t *X) const
{
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      kernels::USolve(data, m, x);
      x += m;
   }
}

void LUFactors::Solve(int m, int n, real_t *X) const
{
#ifdef MFEM_USE_LAPACK
   char trans = 'N';
   int  info = 0;
   if (m > 0 && n > 0)
   {
      MFEM_LAPACK_PREFIX(getrs_)(&trans, &m, &n, data, &m, ipiv, X, &m, &info);
   }
   MFEM_VERIFY(!info, "LAPACK: error in DGETRS");
#else
   // compiling without LAPACK
   LSolve(m, n, X);
   USolve(m, n, X);
#endif
}

void LUFactors::RightSolve(int m, int n, real_t *X) const
{
   real_t *x;
#ifdef MFEM_USE_LAPACK
   char n_ch = 'N', side = 'R', u_ch = 'U', l_ch = 'L';
   real_t alpha = 1.0;
   if (m > 0 && n > 0)
   {
      MFEM_LAPACK_PREFIX(trsm_)(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      MFEM_LAPACK_PREFIX(trsm_)(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
   }
#else
   // compiling without LAPACK
   // X <- X U^{-1}
   x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const real_t x_j = ( x[j*n] /= data[j+j*m]);
         for (int i = j+1; i < m; i++)
         {
            x[i*n] -= data[j + i*m] * x_j;
         }
      }
      ++x;
   }

   // X <- X L^{-1}
   x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const real_t x_j = x[j*n];
         for (int i = 0; i < j; i++)
         {
            x[i*n] -= data[j + i*m] * x_j;
         }
      }
      ++x;
   }
#endif
   // X <- X P
   x = X;
   for (int k = 0; k < n; k++)
   {
      for (int i = m-1; i >= 0; --i)
      {
         mfem::Swap<real_t>(x[i*n], x[(ipiv[i]-ipiv_base)*n]);
      }
      ++x;
   }
}

void LUFactors::GetInverseMatrix(int m, real_t *X) const
{
   // A^{-1} = U^{-1} L^{-1} P
   // X <- U^{-1} (set only the upper triangular part of X)
   real_t *x = X;
   for (int k = 0; k < m; k++)
   {
      const real_t minus_x_k = -( x[k] = 1.0/data[k+k*m] );
      for (int i = 0; i < k; i++)
      {
         x[i] = data[i+k*m] * minus_x_k;
      }
      for (int j = k-1; j >= 0; j--)
      {
         const real_t x_j = ( x[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
   // X <- X L^{-1} (use input only from the upper triangular part of X)
   {
      int k = m-1;
      for (int j = 0; j < k; j++)
      {
         const real_t minus_L_kj = -data[k+j*m];
         for (int i = 0; i <= j; i++)
         {
            X[i+j*m] += X[i+k*m] * minus_L_kj;
         }
         for (int i = j+1; i < m; i++)
         {
            X[i+j*m] = X[i+k*m] * minus_L_kj;
         }
      }
   }
   for (int k = m-2; k >= 0; k--)
   {
      for (int j = 0; j < k; j++)
      {
         const real_t L_kj = data[k+j*m];
         for (int i = 0; i < m; i++)
         {
            X[i+j*m] -= X[i+k*m] * L_kj;
         }
      }
   }
   // X <- X P
   for (int k = m-1; k >= 0; k--)
   {
      const int piv_k = ipiv[k]-ipiv_base;
      if (k != piv_k)
      {
         for (int i = 0; i < m; i++)
         {
            Swap<real_t>(X[i+k*m], X[i+piv_k*m]);
         }
      }
   }
}

void LUFactors::SubMult(int m, int n, int r, const real_t *A21,
                        const real_t *X1, real_t *X2)
{
   kernels::SubMult(m, n, r, A21, X1, X2);
}

void LUFactors::BlockFactor(
   int m, int n, real_t *A12, real_t *A21, real_t *A22) const
{
   kernels::BlockFactor(data, m, ipiv, n, A12, A21, A22);
}

void LUFactors::BlockForwSolve(int m, int n, int r, const real_t *L21,
                               real_t *B1, real_t *B2) const
{
   // B1 <- L^{-1} P B1
   LSolve(m, r, B1);
   // B2 <- B2 - L21 B1
   SubMult(m, n, r, L21, B1, B2);
}

void LUFactors::BlockBackSolve(int m, int n, int r, const real_t *U12,
                               const real_t *X2, real_t *Y1) const
{
   // Y1 <- Y1 - U12 X2
   SubMult(n, m, r, U12, X2, Y1);
   // Y1 <- U^{-1} Y1
   USolve(m, r, Y1);
}


bool CholeskyFactors::Factor(int m, real_t TOL)
{
#ifdef MFEM_USE_LAPACK
   int info = 0;
   char uplo = 'L';
   MFEM_VERIFY(data, "Matrix data not set");
   if (m) { MFEM_LAPACK_PREFIX(potrf_)(&uplo, &m, data, &m, &info); }
   return info == 0;
#else
   // Cholesky–Crout algorithm
   for (int j = 0; j<m; j++)
   {
      real_t a = 0.;
      for (int k = 0; k<j; k++)
      {
         a+=data[j+k*m]*data[j+k*m];
      }

      MFEM_VERIFY(data[j+j*m] - a > 0.,
                  "CholeskyFactors::Factor: The matrix is not SPD");

      data[j+j*m] = std::sqrt(data[j+j*m] - a);

      if (data[j + j*m] <= TOL)
      {
         return false; // failed
      }

      for (int i = j+1; i<m; i++)
      {
         a = 0.;
         for (int k = 0; k<j; k++)
         {
            a+= data[i+k*m]*data[j+k*m];
         }
         data[i+j*m] = 1./data[j+m*j]*(data[i+j*m] - a);
      }
   }
   return true; // success
#endif
}

real_t CholeskyFactors::Det(int m) const
{
   real_t det = 1.0;
   for (int i=0; i<m; i++)
   {
      det *=  data[i + i*m];
   }
   return det;
}

void CholeskyFactors::LMult(int m, int n, real_t * X) const
{
   // X <- L X
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         real_t x_j = x[j] * data[j+j*m];
         for (int i = 0; i < j; i++)
         {
            x_j += x[i] * data[j+i*m];
         }
         x[j] = x_j;
      }
      x += m;
   }
}

void CholeskyFactors::UMult(int m, int n, real_t * X) const
{
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int i = 0; i < m; i++)
      {
         real_t x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[j+i*m];
         }
         x[i] = x_i;
      }
      x += m;
   }
}

void CholeskyFactors::LSolve(int m, int n, real_t * X) const
{

#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   char trans = 'N';
   char diag = 'N';
   int info = 0;

   MFEM_LAPACK_PREFIX(trtrs_)(&uplo, &trans, &diag, &m, &n, data, &m, X, &m,
                              &info);
   MFEM_VERIFY(!info, "CholeskyFactors:LSolve:: info");

#else
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const real_t x_j = (x[j] /= data[j+j*m]);
         for (int i = j+1; i < m; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
#endif
}

void CholeskyFactors::USolve(int m, int n, real_t * X) const
{
#ifdef MFEM_USE_LAPACK

   char uplo = 'L';
   char trans = 'T';
   char diag = 'N';
   int info = 0;

   MFEM_LAPACK_PREFIX(trtrs_)(&uplo, &trans, &diag, &m, &n, data, &m, X, &m,
                              &info);
   MFEM_VERIFY(!info, "CholeskyFactors:USolve:: info");

#else
   // X <- L^{-t} X
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const real_t x_j = ( x[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            x[i] -= data[j+i*m] * x_j;
         }
      }
      x += m;
   }
#endif
}

void CholeskyFactors::Solve(int m, int n, real_t * X) const
{
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   int info = 0;
   MFEM_LAPACK_PREFIX(potrs_)(&uplo, &m, &n, data, &m, X, &m, &info);
   MFEM_VERIFY(!info, "CholeskyFactors:Solve:: info");

#else
   LSolve(m, n, X);
   USolve(m, n, X);
#endif
}

void CholeskyFactors::RightSolve(int m, int n, real_t * X) const
{
#ifdef MFEM_USE_LAPACK
   char side = 'R';
   char uplo = 'L';
   char transt = 'T';
   char trans = 'N';
   char diag = 'N';

   real_t alpha = 1.0;
   if (m > 0 && n > 0)
   {
      MFEM_LAPACK_PREFIX(trsm_)(&side,&uplo,&transt,&diag,&n,&m,&alpha,data,&m,X,&n);
      MFEM_LAPACK_PREFIX(trsm_)(&side,&uplo,&trans,&diag,&n,&m,&alpha,data,&m,X,&n);
   }
#else
   // X <- X L^{-t}
   real_t *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const real_t x_j = ( x[j*n] /= data[j+j*m]);
         for (int i = j+1; i < m; i++)
         {
            x[i*n] -= data[i + j*m] * x_j;
         }
      }
      ++x;
   }
   // X <- X L^{-1}
   x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const real_t x_j = (x[j*n] /= data[j+j*m]);
         for (int i = 0; i < j; i++)
         {
            x[i*n] -= data[j + i*m] * x_j;
         }
      }
      ++x;
   }
#endif
}

void CholeskyFactors::GetInverseMatrix(int m, real_t * X) const
{
   // A^{-1} = L^{-t} L^{-1}
#ifdef MFEM_USE_LAPACK
   // copy the lower triangular part of L to X
   for (int i = 0; i<m; i++)
   {
      for (int j = i; j<m; j++)
      {
         X[j+i*m] = data[j+i*m];
      }
   }
   char uplo = 'L';
   int info = 0;
   MFEM_LAPACK_PREFIX(potri_)(&uplo, &m, X, &m, &info);
   MFEM_VERIFY(!info, "CholeskyFactors:GetInverseMatrix:: info");
   // fill in the upper triangular part
   for (int i = 0; i<m; i++)
   {
      for (int j = i+1; j<m; j++)
      {
         X[i+j*m] = X[j+i*m];
      }
   }
#else
   // L^-t * L^-1 (in place)
   for (int k = 0; k<m; k++)
   {
      X[k+k*m] = 1./data[k+k*m];
      for (int i = k+1; i < m; i++)
      {
         real_t s=0.;
         for (int j=k; j<i; j++)
         {
            s -= data[i+j*m] * X[j+k*m]/data[i+i*m];
         }
         X[i+k*m] = s;
      }
   }
   for (int i = 0; i < m; i++)
   {
      for (int j = i; j < m; j++)
      {
         real_t s = 0.;
         for (int k=j; k<m; k++)
         {
            s += X[k+i*m] * X[k+j*m];
         }
         X[i+j*m] = X[j+i*m] = s;
      }
   }
#endif
}


void DenseMatrixInverse::Init(int m)
{
   if (spd)
   {
      factors = new CholeskyFactors();
   }
   else
   {
      factors = new LUFactors();
   }
   if (m>0)
   {
      factors->data = new real_t[m*m];
      if (!spd)
      {
         dynamic_cast<LUFactors *>(factors)->ipiv = new int[m];
      }
      own_data = true;
   }
}

DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix &mat, bool spd_)
   : MatrixInverse(mat), spd(spd_)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = &mat;
   Init(width);
   Factor();
}

DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix *mat, bool spd_)
   : MatrixInverse(*mat), spd(spd_)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = mat;
   Init(width);
}

void DenseMatrixInverse::Factor()
{
   MFEM_ASSERT(a, "DenseMatrix is not given");
   const real_t *adata = a->data;
   const int s = width*width;
   for (int i = 0; i < s; i++)
   {
      factors->data[i] = adata[i];
   }
   factors->Factor(width);
}

void DenseMatrixInverse::GetInverseMatrix(DenseMatrix &Ainv) const
{
   Ainv.SetSize(width);
   factors->GetInverseMatrix(width,Ainv.Data());
}

void DenseMatrixInverse::Factor(const DenseMatrix &mat)
{
   MFEM_VERIFY(mat.height == mat.width, "DenseMatrix is not square!");
   if (width != mat.width)
   {
      height = width = mat.width;
      if (own_data) { delete [] factors->data; }
      factors->data = new real_t[width*width];

      if (!spd)
      {
         LUFactors * lu = dynamic_cast<LUFactors *>(factors);
         if (own_data) { delete [] lu->ipiv; }
         lu->ipiv = new int[width];
      }
      own_data = true;
   }
   a = &mat;
   Factor();
}

void DenseMatrixInverse::SetOperator(const Operator &op)
{
   const DenseMatrix *p = dynamic_cast<const DenseMatrix*>(&op);
   MFEM_VERIFY(p != NULL, "Operator is not a DenseMatrix!");
   Factor(*p);
}

void DenseMatrixInverse::Mult(const real_t *x, real_t *y) const
{
   for (int row = 0; row < height; row++)
   {
      y[row] = x[row];
   }
   factors->Solve(width, 1, y);
}

void DenseMatrixInverse::Mult(const Vector &x, Vector &y) const
{
   y = x;
   factors->Solve(width, 1, y.GetData());
}

void DenseMatrixInverse::Mult(const DenseMatrix &B, DenseMatrix &X) const
{
   X = B;
   factors->Solve(width, X.Width(), X.Data());
}

void DenseMatrixInverse::TestInversion()
{
   DenseMatrix C(width);
   Mult(*a, C);
   for (int i = 0; i < width; i++)
   {
      C(i,i) -= 1.0;
   }
   mfem::out << "size = " << width << ", i_max = " << C.MaxMaxNorm() << endl;
}

DenseMatrixInverse::~DenseMatrixInverse()
{
   if (own_data)
   {
      delete [] factors->data;
      if (!spd)
      {
         delete [] dynamic_cast<LUFactors *>(factors)->ipiv;
      }
   }
   delete factors;
}

#ifdef MFEM_USE_LAPACK

DenseMatrixEigensystem::DenseMatrixEigensystem(DenseMatrix &m)
   : mat(m)
{
   n = mat.Width();
   EVal.SetSize(n);
   EVect.SetSize(n);
   ev.SetDataAndSize(NULL, n);

   jobz = 'V';
   uplo = 'U';
   lwork = -1;
   real_t qwork;
   MFEM_LAPACK_PREFIX(syev_)(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
                             &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new real_t[lwork];
}

DenseMatrixEigensystem::DenseMatrixEigensystem(
   const DenseMatrixEigensystem &other)
   : mat(other.mat), EVal(other.EVal), EVect(other.EVect), ev(NULL, other.n),
     n(other.n)
{
   jobz = other.jobz;
   uplo = other.uplo;
   lwork = other.lwork;

   work = new real_t[lwork];
}

void DenseMatrixEigensystem::Eval()
{
#ifdef MFEM_DEBUG
   if (mat.Width() != n)
   {
      mfem_error("DenseMatrixEigensystem::Eval(): dimension mismatch");
   }
#endif

   EVect = mat;
   MFEM_LAPACK_PREFIX(syev_)(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
                             work, &lwork, &info);

   if (info != 0)
   {
      mfem::err << "DenseMatrixEigensystem::Eval(): DSYEV error code: "
                << info << endl;
      mfem_error();
   }
}

DenseMatrixEigensystem::~DenseMatrixEigensystem()
{
   delete [] work;
}


DenseMatrixGeneralizedEigensystem::DenseMatrixGeneralizedEigensystem(
   DenseMatrix &a, DenseMatrix &b,
   bool left_eigen_vectors,
   bool right_eigen_vectors)
   : A(a), B(b)
{
   MFEM_VERIFY(A.Height() == A.Width(), "A has to be a square matrix");
   MFEM_VERIFY(B.Height() == B.Width(), "B has to be a square matrix");
   n = A.Width();
   MFEM_VERIFY(B.Height() == n, "A and B dimension mismatch");

   jobvl = 'N';
   jobvr = 'N';
   A_copy.SetSize(n);
   B_copy.SetSize(n);
   if (left_eigen_vectors)
   {
      jobvl = 'V';
      Vl.SetSize(n);
   }
   if (right_eigen_vectors)
   {
      jobvr = 'V';
      Vr.SetSize(n);
   }

   lwork = -1;
   real_t qwork;

   alphar = new real_t[n];
   alphai = new real_t[n];
   beta = new real_t[n];

   int nl = max(1,Vl.Height());
   int nr = max(1,Vr.Height());

   MFEM_LAPACK_PREFIX(ggev_)(&jobvl,&jobvr,&n,A_copy.Data(),&n,B_copy.Data(),&n,
                             alphar, alphai, beta, Vl.Data(), &nl, Vr.Data(),
                             &nr, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new real_t[lwork];
}

void DenseMatrixGeneralizedEigensystem::Eval()
{
   int nl = max(1,Vl.Height());
   int nr = max(1,Vr.Height());

   A_copy = A;
   B_copy = B;
   MFEM_LAPACK_PREFIX(ggev_)(&jobvl,&jobvr,&n,A_copy.Data(),&n,B_copy.Data(),&n,
                             alphar, alphai, beta, Vl.Data(), &nl, Vr.Data(),
                             &nr, work, &lwork, &info);
   if (info != 0)
   {
      mfem::err << "DenseMatrixGeneralizedEigensystem::Eval(): DGGEV error code: "
                << info << endl;
      mfem_error();
   }
   evalues_r.SetSize(n);
   evalues_i.SetSize(n);
   for (int i = 0; i<n; i++)
   {
      if (beta[i] != 0.)
      {
         evalues_r(i) = alphar[i]/beta[i];
         evalues_i(i) = alphai[i]/beta[i];
      }
      else
      {
         evalues_r(i) = infinity();
         evalues_i(i) = infinity();
      }
   }
}

DenseMatrixGeneralizedEigensystem::~DenseMatrixGeneralizedEigensystem()
{
   delete [] alphar;
   delete [] alphai;
   delete [] beta;
   delete [] work;
}

DenseMatrixSVD::DenseMatrixSVD(DenseMatrix &M,
                               bool left_singular_vectors,
                               bool right_singular_vectors)
{
   m = M.Height();
   n = M.Width();
   jobu = (left_singular_vectors)? 'S' : 'N';
   jobvt = (right_singular_vectors)? 'S' : 'N';
   Init();
}

DenseMatrixSVD::DenseMatrixSVD(int h, int w,
                               bool left_singular_vectors,
                               bool right_singular_vectors)
{
   m = h;
   n = w;
   jobu = (left_singular_vectors)? 'S' : 'N';
   jobvt = (right_singular_vectors)? 'S' : 'N';
   Init();
}

DenseMatrixSVD::DenseMatrixSVD(DenseMatrix &M,
                               char left_singular_vectors,
                               char right_singular_vectors)
{
   m = M.Height();
   n = M.Width();
   jobu = left_singular_vectors;
   jobvt = right_singular_vectors;
   Init();
}

DenseMatrixSVD::DenseMatrixSVD(int h, int w,
                               char left_singular_vectors,
                               char right_singular_vectors)
{
   m = h;
   n = w;
   jobu = left_singular_vectors;
   jobvt = right_singular_vectors;
   Init();
}

void DenseMatrixSVD::Init()
{
   sv.SetSize(min(m, n));
   real_t qwork;
   lwork = -1;
   MFEM_LAPACK_PREFIX(gesvd_)(&jobu, &jobvt, &m, &n, NULL, &m, sv.GetData(),
                              NULL, &m, NULL, &n, &qwork, &lwork, &info);
   lwork = (int) qwork;
   work = new real_t[lwork];
}

void DenseMatrixSVD::Eval(DenseMatrix &M)
{
#ifdef MFEM_DEBUG
   if (M.Height() != m || M.Width() != n)
   {
      mfem_error("DenseMatrixSVD::Eval()");
   }
#endif
   real_t * datau = nullptr;
   real_t * datavt = nullptr;
   if (jobu == 'A')
   {
      U.SetSize(m,m);
      datau = U.Data();
   }
   else if (jobu == 'S')
   {
      U.SetSize(m,min(m,n));
      datau = U.Data();
   }
   if (jobvt == 'A')
   {
      Vt.SetSize(n,n);
      datavt = Vt.Data();
   }
   else if (jobvt == 'S')
   {
      Vt.SetSize(min(m,n),n);
      datavt = Vt.Data();
   }
   Mc = M;
   MFEM_LAPACK_PREFIX(gesvd_)(&jobu, &jobvt, &m, &n, Mc.Data(), &m, sv.GetData(),
                              datau, &m, datavt, &n, work, &lwork, &info);

   if (info)
   {
      mfem::err << "DenseMatrixSVD::Eval() : info = " << info << endl;
      mfem_error();
   }
}

DenseMatrixSVD::~DenseMatrixSVD()
{
   delete [] work;
}

#endif // if MFEM_USE_LAPACK


void DenseTensor::AddMult(const Table &elem_dof, const Vector &x, Vector &y)
const
{
   int n = SizeI(), ne = SizeK();
   const int *I = elem_dof.GetI(), *J = elem_dof.GetJ(), *dofs;
   const real_t *d_col = tdata.HostRead();
   real_t *yp = y.HostReadWrite();
   real_t x_col;
   const real_t *xp = x.HostRead();
   // the '4' here can be tuned for given platform and compiler
   if (n <= 4)
   {
      for (int i = 0; i < ne; i++)
      {
         dofs = J + I[i];
         for (int col = 0; col < n; col++)
         {
            x_col = xp[dofs[col]];
            for (int row = 0; row < n; row++)
            {
               yp[dofs[row]] += x_col*d_col[row];
            }
            d_col += n;
         }
      }
   }
   else
   {
      Vector ye(n);
      for (int i = 0; i < ne; i++)
      {
         dofs = J + I[i];
         x_col = xp[dofs[0]];
         for (int row = 0; row < n; row++)
         {
            ye(row) = x_col*d_col[row];
         }
         d_col += n;
         for (int col = 1; col < n; col++)
         {
            x_col = xp[dofs[col]];
            for (int row = 0; row < n; row++)
            {
               ye(row) += x_col*d_col[row];
            }
            d_col += n;
         }
         for (int row = 0; row < n; row++)
         {
            yp[dofs[row]] += ye(row);
         }
      }
   }
}

DenseTensor &DenseTensor::operator=(real_t c)
{
   int s = SizeI() * SizeJ() * SizeK();
   for (int i=0; i<s; i++)
   {
      tdata[i] = c;
   }
   return *this;
}

void BatchLUFactor(DenseTensor &Mlu, Array<int> &P, const real_t TOL)
{
   BatchedLinAlg::LUFactor(Mlu, P);
}

void BatchLUSolve(const DenseTensor &Mlu, const Array<int> &P, Vector &X)
{
   BatchedLinAlg::LUSolve(Mlu, P, X);
}

#ifdef MFEM_USE_LAPACK
void BandedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                 Array<int> &ipiv)
{
   int LDAB = (2*KL) + KU + 1;
   int N = AB.NumCols();
   int NRHS = B.NumCols();
   int info;
   ipiv.SetSize(N);
   MFEM_LAPACK_PREFIX(gbsv_)(&N, &KL, &KU, &NRHS, AB.GetData(), &LDAB,
                             ipiv.GetData(), B.GetData(), &N, &info);
   MFEM_ASSERT(info == 0, "BandedSolve failed in LAPACK");
}

void BandedFactorizedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                           bool transpose, Array<int> &ipiv)
{
   int LDAB = (2*KL) + KU + 1;
   int N = AB.NumCols();
   int NRHS = B.NumCols();
   char trans = transpose ? 'T' : 'N';
   int info;
   MFEM_LAPACK_PREFIX(gbtrs_)(&trans, &N, &KL, &KU, &NRHS, AB.GetData(), &LDAB,
                              ipiv.GetData(), B.GetData(), &N, &info);
   MFEM_ASSERT(info == 0, "BandedFactorizedSolve failed in LAPACK");
}
#endif

} // namespace mfem
