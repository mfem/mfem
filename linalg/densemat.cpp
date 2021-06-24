// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "kernels.hpp"
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


#ifdef MFEM_USE_LAPACK
extern "C" void
dgemm_(char *, char *, int *, int *, int *, double *, double *,
       int *, double *, int *, double *, double *, int *);
extern "C" void
dgetrf_(int *, int *, double *, int *, int *, int *);
extern "C" void
dgetrs_(char *, int *, int *, double *, int *, int *, double *, int *, int *);
extern "C" void
dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK,
        int *LWORK, int *INFO);
extern "C" void
dsyevr_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
        double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M,
        double *W, double *Z, int *LDZ, int *ISUPPZ, double *WORK, int *LWORK,
        int *IWORK, int *LIWORK, int *INFO);
extern "C" void
dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W,
       double *WORK, int *LWORK, int *INFO);
extern "C" void
dsygv_ (int *ITYPE, char *JOBZ, char *UPLO, int * N, double *A, int *LDA,
        double *B, int *LDB, double *W,  double *WORK, int *LWORK, int *INFO);
extern "C" void
dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
        double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK,
        int *LWORK, int *INFO);
extern "C" void
dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
       double *alpha, double *a, int *lda, double *b, int *ldb);
#endif


namespace mfem
{

using namespace std;

DenseMatrix::DenseMatrix() : Matrix(0)
{
   data.Reset();
}

DenseMatrix::DenseMatrix(const DenseMatrix &m) : Matrix(m.height, m.width)
{
   const int hw = height * width;
   if (hw > 0)
   {
      MFEM_ASSERT(m.data, "invalid source matrix");
      data.New(hw);
      std::memcpy(data, m.data, sizeof(double)*hw);
   }
   else
   {
      data.Reset();
   }
}

DenseMatrix::DenseMatrix(int s) : Matrix(s)
{
   MFEM_ASSERT(s >= 0, "invalid DenseMatrix size: " << s);
   if (s > 0)
   {
      data.New(s*s);
      *this = 0.0; // init with zeroes
   }
   else
   {
      data.Reset();
   }
}

DenseMatrix::DenseMatrix(int m, int n) : Matrix(m, n)
{
   MFEM_ASSERT(m >= 0 && n >= 0,
               "invalid DenseMatrix size: " << m << " x " << n);
   const int capacity = m*n;
   if (capacity > 0)
   {
      data.New(capacity);
      *this = 0.0; // init with zeroes
   }
   else
   {
      data.Reset();
   }
}

DenseMatrix::DenseMatrix(const DenseMatrix &mat, char ch)
   : Matrix(mat.width, mat.height)
{
   MFEM_CONTRACT_VAR(ch);
   const int capacity = height*width;
   if (capacity > 0)
   {
      data.New(capacity);

      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < width; j++)
         {
            (*this)(i,j) = mat(j,i);
         }
      }
   }
   else
   {
      data.Reset();
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
   const int hw = h*w;
   if (hw > data.Capacity())
   {
      data.Delete();
      data.New(hw);
      *this = 0.0; // init with zeroes
   }
}

double &DenseMatrix::Elem(int i, int j)
{
   return (*this)(i,j);
}

const double &DenseMatrix::Elem(int i, int j) const
{
   return (*this)(i,j);
}

void DenseMatrix::Mult(const double *x, double *y) const
{
   kernels::Mult(height, width, Data(), x, y);
}

void DenseMatrix::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   Mult((const double *)x, (double *)y);
}

double DenseMatrix::operator *(const DenseMatrix &m) const
{
   MFEM_ASSERT(Height() == m.Height() && Width() == m.Width(),
               "incompatible dimensions");

   const int hw = height * width;
   double a = 0.0;
   for (int i = 0; i < hw; i++)
   {
      a += data[i] * m.data[i];
   }

   return a;
}

void DenseMatrix::MultTranspose(const double *x, double *y) const
{
   double *d_col = Data();
   for (int col = 0; col < width; col++)
   {
      double y_col = 0.0;
      for (int row = 0; row < height; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] = y_col;
      d_col += height;
   }
}

void DenseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   MultTranspose((const double *)x, (double *)y);
}

void DenseMatrix::AddMult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   const double *xp = x, *d_col = data;
   double *yp = y;
   for (int col = 0; col < width; col++)
   {
      double x_col = xp[col];
      for (int row = 0; row < height; row++)
      {
         yp[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

void DenseMatrix::AddMultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   const double *d_col = data;
   for (int col = 0; col < width; col++)
   {
      double y_col = 0.0;
      for (int row = 0; row < height; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] += y_col;
      d_col += height;
   }
}

void DenseMatrix::AddMult_a(double a, const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   const double *xp = x, *d_col = data;
   double *yp = y;
   for (int col = 0; col < width; col++)
   {
      const double x_col = a*xp[col];
      for (int row = 0; row < height; row++)
      {
         yp[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

void DenseMatrix::AddMultTranspose_a(double a, const Vector &x,
                                     Vector &y) const
{
   MFEM_ASSERT(height == x.Size() && width == y.Size(),
               "incompatible dimensions");

   const double *d_col = data;
   for (int col = 0; col < width; col++)
   {
      double y_col = 0.0;
      for (int row = 0; row < height; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] += a * y_col;
      d_col += height;
   }
}

double DenseMatrix::InnerProduct(const double *x, const double *y) const
{
   double prod = 0.0;

   for (int i = 0; i < height; i++)
   {
      double Axi = 0.0;
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
   double * it_data = data;
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
   double * it_data = data;
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
   double sj;
   double * it_data = data;
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
   double * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      const double sj = 1./s(j);
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

   double * ss = new double[width];
   double * it_s = s.GetData();
   double * it_ss = ss;
   for ( double * end_s = it_s + width; it_s != end_s; ++it_s)
   {
      *(it_ss++) = sqrt(*it_s);
   }

   double * it_data = data;
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

   double * ss = new double[width];
   double * it_s = s.GetData();
   double * it_ss = ss;
   for (double * end_s = it_s + width; it_s != end_s; ++it_s)
   {
      *(it_ss++) = 1./sqrt(*it_s);
   }

   double * it_data = data;
   for (int j = 0; j < width; ++j)
   {
      for (int i = 0; i < height; ++i)
      {
         *(it_data++) *= ss[i]*ss[j];
      }
   }

   delete[] ss;
}

double DenseMatrix::Trace() const
{
#ifdef MFEM_DEBUG
   if (Width() != Height())
   {
      mfem_error("DenseMatrix::Trace() : not a square matrix!");
   }
#endif

   double t = 0.0;

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

double DenseMatrix::Det() const
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
         const double *d = data;
         return
            d[0] * (d[4] * d[8] - d[5] * d[7]) +
            d[3] * (d[2] * d[7] - d[1] * d[8]) +
            d[6] * (d[1] * d[5] - d[2] * d[4]);
      }
      case 4:
      {
         const double *d = data;
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

double DenseMatrix::Weight() const
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
      const double *d = data;
      double E = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      double G = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
      double F = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
      return sqrt(E * G - F * F);
   }
   mfem_error("DenseMatrix::Weight(): mismatched or unsupported dimensions");
   return 0.0;
}

void DenseMatrix::Set(double alpha, const double *A)
{
   const int s = Width()*Height();
   for (int i = 0; i < s; i++)
   {
      data[i] = alpha*A[i];
   }
}

void DenseMatrix::Add(const double c, const DenseMatrix &A)
{
   for (int j = 0; j < Width(); j++)
   {
      for (int i = 0; i < Height(); i++)
      {
         (*this)(i,j) += c * A(i,j);
      }
   }
}

DenseMatrix &DenseMatrix::operator=(double c)
{
   const int s = Height()*Width();
   for (int i = 0; i < s; i++)
   {
      data[i] = c;
   }
   return *this;
}

DenseMatrix &DenseMatrix::operator=(const double *d)
{
   const int s = Height()*Width();
   for (int i = 0; i < s; i++)
   {
      data[i] = d[i];
   }
   return *this;
}

DenseMatrix &DenseMatrix::operator=(const DenseMatrix &m)
{
   SetSize(m.height, m.width);

   const int hw = height * width;
   for (int i = 0; i < hw; i++)
   {
      data[i] = m.data[i];
   }

   return *this;
}

DenseMatrix &DenseMatrix::operator+=(const double *m)
{
   kernels::Add(Height(), Width(), m, (double*)data);
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

DenseMatrix &DenseMatrix::operator*=(double c)
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
   double qwork, *work;
   int    info;

   dgetrf_(&width, &width, data, &width, ipiv, &info);

   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in DGETRF");
   }

   dgetri_(&width, data, &width, ipiv, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];

   dgetri_(&width, data, &width, ipiv, work, &lwork, &info);

   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in DGETRI");
   }

   delete [] work;
   delete [] ipiv;
#else
   int c, i, j, n = Width();
   double a, b;
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
         mfem::Swap<double>((*this)(c, j), (*this)(i, j));
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
         mfem::Swap<double>((*this)(i, c), (*this)(i, j));
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

void DenseMatrix::Norm2(double *v) const
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

double DenseMatrix::MaxMaxNorm() const
{
   int hw = Height()*Width();
   const double *d = data;
   double norm = 0.0, abs_entry;

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

void DenseMatrix::FNorm(double &scale_factor, double &scaled_fnorm2) const
{
   int i, hw = Height() * Width();
   double max_norm = 0.0, entry, fnorm2;

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
   double   *A        = new double[N*N];
   int       LDA      = N;
   double    VL       = 0.0;
   double    VU       = 1.0;
   int       IL       = 0;
   int       IU       = 1;
   double    ABSTOL   = 0.0;
   int       M;
   double   *W        = ev.GetData();
   double   *Z        = NULL;
   int       LDZ      = 1;
   int      *ISUPPZ   = new int[2*N];
   int       LWORK    = -1; // query optimal (double) workspace size
   double    QWORK;
   double   *WORK     = NULL;
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
   double *data = a.Data();

   for (int i = 0; i < hw; i++)
   {
      A[i] = data[i];
   }

   dsyevr_( &JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
            &ABSTOL, &M, W, Z, &LDZ, ISUPPZ, &QWORK, &LWORK,
            &QIWORK, &LIWORK, &INFO );

   LWORK  = (int) QWORK;
   LIWORK = QIWORK;

   WORK  = new double[LWORK];
   IWORK = new int[LIWORK];

   dsyevr_( &JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
            &ABSTOL, &M, W, Z, &LDZ, ISUPPZ, WORK, &LWORK,
            IWORK, &LIWORK, &INFO );

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

   double *A    = NULL;
   double *W    = ev.GetData();
   double *WORK = NULL;
   double  QWORK;

   if (evect)
   {
      JOBZ = 'V';
      evect->SetSize(N);
      A = evect->Data();
   }
   else
   {
      A = new double[N*N];
   }

   int hw = a.Height() * a.Width();
   double *data = a.Data();
   for (int i = 0; i < hw; i++)
   {
      A[i] = data[i];
   }

   dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, &QWORK, &LWORK, &INFO);

   LWORK = (int) QWORK;
   WORK = new double[LWORK];

   dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);

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

   double *A    = NULL;
   double *B    = new double[N*N];
   double *W    = ev.GetData();
   double *WORK = NULL;
   double  QWORK;

   if (evect)
   {
      JOBZ = 'V';
      evect->SetSize(N);
      A = evect->Data();
   }
   else
   {
      A = new double[N*N];
   }

   int hw = a.Height() * a.Width();
   double *a_data = a.Data();
   double *b_data = b.Data();
   for (int i = 0; i < hw; i++)
   {
      A[i] = a_data[i];
      B[i] = b_data[i];
   }

   dsygv_(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, &QWORK, &LWORK, &INFO);

   LWORK = (int) QWORK;
   WORK = new double[LWORK];

   dsygv_(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, WORK, &LWORK, &INFO);

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
   double      *a           = copy_of_this.data;
   sv.SetSize(min(m, n));
   double      *s           = sv;
   double      *u           = NULL;
   double      *vt          = NULL;
   double      *work        = NULL;
   int         lwork        = -1;
   int         info;
   double      qwork;

   dgesvd_(&jobu, &jobvt, &m, &n, a, &m,
           s, u, &m, vt, &n, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];

   dgesvd_(&jobu, &jobvt, &m, &n, a, &m,
           s, u, &m, vt, &n, work, &lwork, &info);

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

int DenseMatrix::Rank(double tol) const
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

double DenseMatrix::CalcSingularvalue(const int i) const
{
   MFEM_ASSERT(Height() == Width() && Height() > 0 && Height() < 4,
               "The matrix must be square and sized 1, 2, or 3 to compute the"
               " singular values."
               << "  Height() = " << Height()
               << ", Width() = " << Width());

   const int n = Height();
   const double *d = data;

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

void DenseMatrix::CalcEigenvalues(double *lambda, double *vec) const
{
#ifdef MFEM_DEBUG
   if (Height() != Width() || Height() < 2 || Height() > 3)
   {
      mfem_error("DenseMatrix::CalcEigenvalues");
   }
#endif

   const int n = Height();
   const double *d = data;

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

   const double* rp = data + r;
   double* vp = row.GetData();

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

   double *cp = Data() + c * m;
   double *vp = col.GetData();

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

void DenseMatrix::GetRowSums(Vector &l) const
{
   l.SetSize(height);
   for (int i = 0; i < height; i++)
   {
      double d = 0.0;
      for (int j = 0; j < width; j++)
      {
         d += operator()(i, j);
      }
      l(i) = d;
   }
}

void DenseMatrix::Diag(double c, int n)
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

void DenseMatrix::Diag(double *diag, int n)
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
   double t;

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
      double L = 0.0;
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
         double x = (*this)(i,0);
         double y = (*this)(i,1);

         int j = i+n;

         // curl of (Ui,0)
         curl(i,0) = -y;

         // curl of (0,Ui)
         curl(j,0) =  x;
      }
   }
   else
   {
      for (int i = 0; i < n; i++)
      {
         // (x,y,z) is grad of Ui
         double x = (*this)(i,0);
         double y = (*this)(i,1);
         double z = (*this)(i,2);

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

void DenseMatrix::GradToDiv(Vector &div)
{
   MFEM_ASSERT(Width()*Height() == div.Size(), "incompatible Vector 'div'!");

   // div(dof*j+i) <-- (*this)(i,j)

   const int n = height * width;
   double *ddata = div.GetData();

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
   double *v = A.Data();

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
   double *v = A.Data();

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
               "this DenseMatrix is too small to accomodate the submatrix.  "
               << "row_offset = " << row_offset
               << ", m = " << m
               << ", this->Height() = " << this->Height()
               << ", col_offset = " << col_offset
               << ", n = " << n
               << ", this->Width() = " << this->Width()
              );
   MFEM_VERIFY(Aro+m <= A.Height() && Aco+n <= A.Width(),
               "The A DenseMatrix is too small to accomodate the submatrix.  "
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

void DenseMatrix::CopyMNDiag(double c, int n, int row_offset, int col_offset)
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

void DenseMatrix::CopyMNDiag(double *diag, int n, int row_offset,
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
   double *p, *ap;

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

void DenseMatrix::AddMatrix(double a, const DenseMatrix &A, int ro, int co)
{
   int h, ah, aw;
   double *p, *ap;

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

void DenseMatrix::AddToVector(int offset, Vector &v) const
{
   const int n = height * width;
   double *vdata = v.GetData() + offset;

   for (int i = 0; i < n; i++)
   {
      vdata[i] += data[i];
   }
}

void DenseMatrix::GetFromVector(int offset, const Vector &v)
{
   const int n = height * width;
   const double *vdata = v.GetData() + offset;

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

void DenseMatrix::SetRow(int row, double value)
{
   for (int j = 0; j < Width(); j++)
   {
      (*this)(row, j) = value;
   }
}

void DenseMatrix::SetCol(int col, double value)
{
   for (int i = 0; i < Height(); i++)
   {
      (*this)(i, col) = value;
   }
}

void DenseMatrix::SetRow(int r, const double* row)
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

void DenseMatrix::SetCol(int c, const double* col)
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

void DenseMatrix::Threshold(double eps)
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

void DenseMatrix::Print(std::ostream &out, int width_) const
{
   // save current output flags
   ios::fmtflags old_flags = out.flags();
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      out << "[row " << i << "]\n";
      for (int j = 0; j < width; j++)
      {
         out << (*this)(i,j);
         if (j+1 == width || (j+1) % width_ == 0)
         {
            out << '\n';
         }
         else
         {
            out << ' ';
         }
      }
   }
   // reset output flags to original values
   out.flags(old_flags);
}

void DenseMatrix::PrintMatlab(std::ostream &out) const
{
   // save current output flags
   ios::fmtflags old_flags = out.flags();
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         out << (*this)(i,j);
         out << ' ';
      }
      out << "\n";
   }
   // reset output flags to original values
   out.flags(old_flags);
}

void DenseMatrix::PrintT(std::ostream &out, int width_) const
{
   // save current output flags
   ios::fmtflags old_flags = out.flags();
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int j = 0; j < width; j++)
   {
      out << "[col " << j << "]\n";
      for (int i = 0; i < height; i++)
      {
         out << (*this)(i,j);
         if (i+1 == height || (i+1) % width_ == 0)
         {
            out << '\n';
         }
         else
         {
            out << ' ';
         }
      }
   }
   // reset output flags to original values
   out.flags(old_flags);
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
   mfem::Swap(width, other.width);
   mfem::Swap(height, other.height);
   mfem::Swap(data, other.data);
}

DenseMatrix::~DenseMatrix()
{
   data.Delete();
}



void Add(const DenseMatrix &A, const DenseMatrix &B,
         double alpha, DenseMatrix &C)
{
   kernels::Add(C.Height(), C.Width(), alpha, A.Data(), B.Data(), C.Data());
}

void Add(double alpha, const double *A,
         double beta,  const double *B, DenseMatrix &C)
{
   kernels::Add(C.Height(), C.Width(), alpha, A, beta, B, C.Data());
}

void Add(double alpha, const DenseMatrix &A,
         double beta,  const DenseMatrix &B, DenseMatrix &C)
{
   MFEM_ASSERT(A.Height() == C.Height(), "");
   MFEM_ASSERT(B.Height() == C.Height(), "");
   MFEM_ASSERT(A.Width() == C.Width(), "");
   MFEM_ASSERT(B.Width() == C.Width(), "");
   Add(alpha, A.GetData(), beta, B.GetData(), C);
}

bool LinearSolve(DenseMatrix& A, double* X, double TOL)
{
   MFEM_VERIFY(A.IsSquare(), "A must be a square matrix!");
   MFEM_ASSERT(A.NumCols() > 0, "supplied matrix, A, is empty!");
   MFEM_ASSERT(X != nullptr, "supplied vector, X, is null!");

   int N = A.NumCols();

   switch (N)
   {
      case 1:
      {
         double det = A(0,0);
         if (std::abs(det) <= TOL) { return false; } // singular

         X[0] /= det;
         break;
      }
      case 2:
      {
         double det = A.Det();
         if (std::abs(det) <= TOL) { return false; } // singular

         double invdet = 1. / det;

         double b0 = X[0];
         double b1 = X[1];

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
   static double alpha = 1.0, beta = 0.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
          c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   double *ad = a.Data();
   const double *bd = b.Data();
   const double *cd = c.Data();
   kernels::Mult(ah,aw,bw,bd,cd,ad);
#endif
}

void AddMult_a(double alpha, const DenseMatrix &b, const DenseMatrix &c,
               DenseMatrix &a)
{
   MFEM_ASSERT(a.Height() == b.Height() && a.Width() == c.Width() &&
               b.Width() == c.Height(), "incompatible dimensions");

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'N';
   static double beta = 1.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
          c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   double *ad = a.Data();
   const double *bd = b.Data();
   const double *cd = c.Data();
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
   static double alpha = 1.0, beta = 1.0;
   int m = b.Height(), n = c.Width(), k = b.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, b.Data(), &m,
          c.Data(), &k, &beta, a.Data(), &m);
#else
   const int ah = a.Height();
   const int aw = a.Width();
   const int bw = b.Width();
   double *ad = a.Data();
   const double *bd = b.Data();
   const double *cd = c.Data();
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
      const double *d = a.Data();
      double *ad = adja.Data();
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
         double e, g, f;
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

   double t;

   if (a.Width() < a.Height())
   {
      const double *d = a.Data();
      double *id = inva.Data();
      if (a.Height() == 2)
      {
         t = 1.0 / (d[0]*d[0] + d[1]*d[1]);
         id[0] = d[0] * t;
         id[1] = d[1] * t;
      }
      else
      {
         if (a.Width() == 1)
         {
            t = 1.0 / (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            id[0] = d[0] * t;
            id[1] = d[1] * t;
            id[2] = d[2] * t;
         }
         else
         {
            double e, g, f;
            e = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            g = d[3]*d[3] + d[4]*d[4] + d[5]*d[5];
            f = d[0]*d[3] + d[1]*d[4] + d[2]*d[5];
            t = 1.0 / (e*g - f*f);
            e *= t; g *= t; f *= t;

            id[0] = d[0]*g - d[3]*f;
            id[1] = d[3]*e - d[0]*f;
            id[2] = d[1]*g - d[4]*f;
            id[3] = d[4]*e - d[1]*f;
            id[4] = d[2]*g - d[5]*f;
            id[5] = d[5]*e - d[2]*f;
         }
      }
      return;
   }

#ifdef MFEM_DEBUG
   t = a.Det();
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

   double t = 1. / a.Det() ;

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

   const double *d = J.Data();
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
         double temp = 0.;
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
         double t = 0.;
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
      double t = 0.;
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
         double t = 0.;
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
   static double alpha = 1.0, beta = 0.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
          B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const double *ad = A.Data();
   const double *bd = B.Data();
   double *cd = ABt.Data();

   kernels::MultABt(ah, aw, bh, ad, bd, cd);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const double *ad = A.Data();
   const double *bd = B.Data();
   double *cd = ABt.Data();

   for (int j = 0; j < bh; j++)
      for (int i = 0; i < ah; i++)
      {
         double d = 0.0;
         const double *ap = ad + i;
         const double *bp = bd + j;
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
   double d;

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
   const double *ad = A.Data();
   const double *bd = B.Data();
   const double *dd = D.GetData();
   double *cd = ADBt.Data();

   for (int i = 0, s = ah*bh; i < s; i++)
   {
      cd[i] = 0.0;
   }
   for (int k = 0; k < aw; k++)
   {
      double *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const double dk_bjk = dd[k] * bd[j];
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
   static double alpha = 1.0, beta = 1.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
          B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const double *ad = A.Data();
   const double *bd = B.Data();
   double *cd = ABt.Data();

   for (int k = 0; k < aw; k++)
   {
      double *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const double bjk = bd[j];
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
   double d;

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
   const double *ad = A.Data();
   const double *bd = B.Data();
   const double *dd = D.GetData();
   double *cd = ADBt.Data();

   for (int k = 0; k < aw; k++)
   {
      double *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const double dk_bjk = dd[k] * bd[j];
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

void AddMult_a_ABt(double a, const DenseMatrix &A, const DenseMatrix &B,
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
   double alpha = a;
   static double beta = 1.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
          B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int bh = B.Height();
   const int aw = A.Width();
   const double *ad = A.Data();
   const double *bd = B.Data();
   double *cd = ABt.Data();

   for (int k = 0; k < aw; k++)
   {
      double *cp = cd;
      for (int j = 0; j < bh; j++)
      {
         const double bjk = a * bd[j];
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
   double d;

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
   static double alpha = 1.0, beta = 0.0;
   int m = A.Width(), n = B.Width(), k = A.Height();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &k,
          B.Data(), &k, &beta, AtB.Data(), &m);
#elif 1
   const int ah = A.Height();
   const int aw = A.Width();
   const int bw = B.Width();
   const double *ad = A.Data();
   const double *bd = B.Data();
   double *cd = AtB.Data();

   for (int j = 0; j < bw; j++)
   {
      const double *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         double d = 0.0;
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
   double d;

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

void AddMult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt)
{
   double d;

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

void Mult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt)
{
   for (int i = 0; i < A.Height(); i++)
   {
      for (int j = 0; j <= i; j++)
      {
         double d = 0.;
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
      const double vi = v(i);
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
      const double vi = v(i);
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
      const double vi = v(i);
      for (int j = 0; j < i; j++)
      {
         const double vivj = vi * v(j);
         VVt(i, j) += vivj;
         VVt(j, i) += vivj;
      }
      VVt(i, i) += vi * vi;
   }
}

void AddMult_a_VWt(const double a, const Vector &v, const Vector &w,
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
      const double awj = a * w(j);
      for (int i = 0; i < m; i++)
      {
         VWt(i, j) += v(i) * awj;
      }
   }
}

void AddMult_a_VVt(const double a, const Vector &v, DenseMatrix &VVt)
{
   MFEM_ASSERT(VVt.Height() == v.Size() && VVt.Width() == v.Size(),
               "incompatible dimensions!");

   const int n = v.Size();
   for (int i = 0; i < n; i++)
   {
      double avi = a * v(i);
      for (int j = 0; j < i; j++)
      {
         const double avivj = avi * v(j);
         VVt(i, j) += avivj;
         VVt(j, i) += avivj;
      }
      VVt(i, i) += avi * v(i);
   }
}


bool LUFactors::Factor(int m, double TOL)
{
#ifdef MFEM_USE_LAPACK
   int info = 0;
   if (m) { dgetrf_(&m, &m, data, &m, ipiv, &info); }
   return info == 0;
#else
   // compiling without LAPACK
   double *data = this->data;
   for (int i = 0; i < m; i++)
   {
      // pivoting
      {
         int piv = i;
         double a = std::abs(data[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const double b = std::abs(data[j+i*m]);
            if (b > a)
            {
               a = b;
               piv = j;
            }
         }
         ipiv[i] = piv;
         if (piv != i)
         {
            // swap rows i and piv in both L and U parts
            for (int j = 0; j < m; j++)
            {
               mfem::Swap<double>(data[i+j*m], data[piv+j*m]);
            }
         }
      }

      if (abs(data[i + i*m]) <= TOL)
      {
         return false; // failed
      }

      const double a_ii_inv = 1.0 / data[i+i*m];
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
   }
#endif

   return true; // success
}

double LUFactors::Det(int m) const
{
   double det = 1.0;
   for (int i=0; i<m; i++)
   {
      if (ipiv[i] != i-ipiv_base)
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

void LUFactors::Mult(int m, int n, double *X) const
{
   const double *data = this->data;
   const int *ipiv = this->ipiv;
   double *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- U X
      for (int i = 0; i < m; i++)
      {
         double x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- L X
      for (int i = m-1; i >= 0; i--)
      {
         double x_i = x[i];
         for (int j = 0; j < i; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- P^{-1} X
      for (int i = m-1; i >= 0; i--)
      {
         mfem::Swap<double>(x[i], x[ipiv[i]-ipiv_base]);
      }
      x += m;
   }
}

void LUFactors::LSolve(int m, int n, double *X) const
{
   const double *data = this->data;
   const int *ipiv = this->ipiv;
   double *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         mfem::Swap<double>(x[i], x[ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const double x_j = x[j];
         for (int i = j+1; i < m; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
}

void LUFactors::USolve(int m, int n, double *X) const
{
   const double *data = this->data;
   double *x = X;
   // X <- U^{-1} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const double x_j = ( x[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
}

void LUFactors::Solve(int m, int n, double *X) const
{
#ifdef MFEM_USE_LAPACK
   char trans = 'N';
   int  info = 0;
   if (m > 0 && n > 0) { dgetrs_(&trans, &m, &n, data, &m, ipiv, X, &m, &info); }
   MFEM_VERIFY(!info, "LAPACK: error in DGETRS");
#else
   // compiling without LAPACK
   LSolve(m, n, X);
   USolve(m, n, X);
#endif
}

void LUFactors::RightSolve(int m, int n, double *X) const
{
   double *x;
#ifdef MFEM_USE_LAPACK
   char n_ch = 'N', side = 'R', u_ch = 'U', l_ch = 'L';
   double alpha = 1.0;
   if (m > 0 && n > 0)
   {
      dtrsm_(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      dtrsm_(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
   }
#else
   // compiling without LAPACK
   // X <- X U^{-1}
   x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const double x_j = ( x[j*n] /= data[j+j*m]);
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
         const double x_j = x[j*n];
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
         mfem::Swap<double>(x[i*n], x[(ipiv[i]-ipiv_base)*n]);
      }
      ++x;
   }
}

void LUFactors::GetInverseMatrix(int m, double *X) const
{
   // A^{-1} = U^{-1} L^{-1} P
   const double *data = this->data;
   const int *ipiv = this->ipiv;
   // X <- U^{-1} (set only the upper triangular part of X)
   double *x = X;
   for (int k = 0; k < m; k++)
   {
      const double minus_x_k = -( x[k] = 1.0/data[k+k*m] );
      for (int i = 0; i < k; i++)
      {
         x[i] = data[i+k*m] * minus_x_k;
      }
      for (int j = k-1; j >= 0; j--)
      {
         const double x_j = ( x[j] /= data[j+j*m] );
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
         const double minus_L_kj = -data[k+j*m];
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
         const double L_kj = data[k+j*m];
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
            Swap<double>(X[i+k*m], X[i+piv_k*m]);
         }
      }
   }
}

void LUFactors::SubMult(int m, int n, int r, const double *A21,
                        const double *X1, double *X2)
{
   // X2 <- X2 - A21 X1
   for (int k = 0; k < r; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const double x1_jk = X1[j+k*m];
         for (int i = 0; i < n; i++)
         {
            X2[i+k*n] -= A21[i+j*n] * x1_jk;
         }
      }
   }
}

void LUFactors::BlockFactor(
   int m, int n, double *A12, double *A21, double *A22) const
{
   const double *data = this->data;
   // A12 <- L^{-1} P A12
   LSolve(m, n, A12);
   // A21 <- A21 U^{-1}
   for (int j = 0; j < m; j++)
   {
      const double u_jj_inv = 1.0/data[j+j*m];
      for (int i = 0; i < n; i++)
      {
         A21[i+j*n] *= u_jj_inv;
      }
      for (int k = j+1; k < m; k++)
      {
         const double u_jk = data[j+k*m];
         for (int i = 0; i < n; i++)
         {
            A21[i+k*n] -= A21[i+j*n] * u_jk;
         }
      }
   }
   // A22 <- A22 - A21 A12
   SubMult(m, n, n, A21, A12, A22);
}

void LUFactors::BlockForwSolve(int m, int n, int r, const double *L21,
                               double *B1, double *B2) const
{
   // B1 <- L^{-1} P B1
   LSolve(m, r, B1);
   // B2 <- B2 - L21 B1
   SubMult(m, n, r, L21, B1, B2);
}

void LUFactors::BlockBackSolve(int m, int n, int r, const double *U12,
                               const double *X2, double *Y1) const
{
   // Y1 <- Y1 - U12 X2
   SubMult(n, m, r, U12, X2, Y1);
   // Y1 <- U^{-1} Y1
   USolve(m, r, Y1);
}


DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix &mat)
   : MatrixInverse(mat)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = &mat;
   lu.data = new double[width*width];
   lu.ipiv = new int[width];
   Factor();
}

DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix *mat)
   : MatrixInverse(*mat)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = mat;
   lu.data = new double[width*width];
   lu.ipiv = new int[width];
}

void DenseMatrixInverse::Factor()
{
   MFEM_ASSERT(a, "DenseMatrix is not given");
   const double *adata = a->data;
   const int s = width*width;
   for (int i = 0; i < s; i++)
   {
      lu.data[i] = adata[i];
   }
   lu.Factor(width);
}

void DenseMatrixInverse::GetInverseMatrix(DenseMatrix &Ainv) const
{
   Ainv.SetSize(width);
   lu.GetInverseMatrix(width, Ainv.Data());
}

void DenseMatrixInverse::Factor(const DenseMatrix &mat)
{
   MFEM_VERIFY(mat.height == mat.width, "DenseMatrix is not square!");
   if (width != mat.width)
   {
      height = width = mat.width;
      delete [] lu.data;
      lu.data = new double[width*width];
      delete [] lu.ipiv;
      lu.ipiv = new int[width];
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

void DenseMatrixInverse::Mult(const double *x, double *y) const
{
   for (int row = 0; row < height; row++)
   {
      y[row] = x[row];
   }
   lu.Solve(width, 1, y);
}

void DenseMatrixInverse::Mult(const Vector &x, Vector &y) const
{
   y = x;
   lu.Solve(width, 1, y.GetData());
}

void DenseMatrixInverse::Mult(const DenseMatrix &B, DenseMatrix &X) const
{
   X = B;
   lu.Solve(width, X.Width(), X.Data());
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
   delete [] lu.data;
   delete [] lu.ipiv;
}


DenseMatrixEigensystem::DenseMatrixEigensystem(DenseMatrix &m)
   : mat(m)
{
   n = mat.Width();
   EVal.SetSize(n);
   EVect.SetSize(n);
   ev.SetDataAndSize(NULL, n);

#ifdef MFEM_USE_LAPACK
   jobz = 'V';
   uplo = 'U';
   lwork = -1;
   double qwork;
   dsyev_(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
          &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];
#endif
}

DenseMatrixEigensystem::DenseMatrixEigensystem(
   const DenseMatrixEigensystem &other)
   : mat(other.mat), EVal(other.EVal), EVect(other.EVect), ev(NULL, other.n),
     n(other.n)
{
#ifdef MFEM_USE_LAPACK
   jobz = other.jobz;
   uplo = other.uplo;
   lwork = other.lwork;

   work = new double[lwork];
#endif
}

void DenseMatrixEigensystem::Eval()
{
#ifdef MFEM_DEBUG
   if (mat.Width() != n)
   {
      mfem_error("DenseMatrixEigensystem::Eval(): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   EVect = mat;
   dsyev_(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
          work, &lwork, &info);

   if (info != 0)
   {
      mfem::err << "DenseMatrixEigensystem::Eval(): DSYEV error code: "
                << info << endl;
      mfem_error();
   }
#else
   mfem_error("DenseMatrixEigensystem::Eval(): Compiled without LAPACK");
#endif
}

DenseMatrixEigensystem::~DenseMatrixEigensystem()
{
#ifdef MFEM_USE_LAPACK
   delete [] work;
#endif
}


DenseMatrixSVD::DenseMatrixSVD(DenseMatrix &M)
{
   m = M.Height();
   n = M.Width();
   Init();
}

DenseMatrixSVD::DenseMatrixSVD(int h, int w)
{
   m = h;
   n = w;
   Init();
}

void DenseMatrixSVD::Init()
{
#ifdef MFEM_USE_LAPACK
   sv.SetSize(min(m, n));

   jobu  = 'N';
   jobvt = 'N';

   double qwork;
   lwork = -1;
   dgesvd_(&jobu, &jobvt, &m, &n, NULL, &m, sv.GetData(), NULL, &m,
           NULL, &n, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];
#else
   mfem_error("DenseMatrixSVD::Init(): Compiled without LAPACK");
#endif
}

void DenseMatrixSVD::Eval(DenseMatrix &M)
{
#ifdef MFEM_DEBUG
   if (M.Height() != m || M.Width() != n)
   {
      mfem_error("DenseMatrixSVD::Eval()");
   }
#endif

#ifdef MFEM_USE_LAPACK
   dgesvd_(&jobu, &jobvt, &m, &n, M.Data(), &m, sv.GetData(), NULL, &m,
           NULL, &n, work, &lwork, &info);

   if (info)
   {
      mfem::err << "DenseMatrixSVD::Eval() : info = " << info << endl;
      mfem_error();
   }
#else
   mfem_error("DenseMatrixSVD::Eval(): Compiled without LAPACK");
#endif
}

DenseMatrixSVD::~DenseMatrixSVD()
{
#ifdef MFEM_USE_LAPACK
   delete [] work;
#endif
}


void DenseTensor::AddMult(const Table &elem_dof, const Vector &x, Vector &y)
const
{
   int n = SizeI(), ne = SizeK();
   const int *I = elem_dof.GetI(), *J = elem_dof.GetJ(), *dofs;
   const double *d_col = tdata;
   double *yp = y.HostReadWrite();
   double x_col;
   const double *xp = x;
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

DenseTensor &DenseTensor::operator=(double c)
{
   int s = SizeI() * SizeJ() * SizeK();
   for (int i=0; i<s; i++)
   {
      tdata[i] = c;
   }
   return *this;
}

DenseTensor &DenseTensor::operator=(const DenseTensor &other)
{
   DenseTensor new_tensor(other);
   Swap(new_tensor);
   return *this;
}

void BatchLUFactor(DenseTensor &Mlu, Array<int> &P, const double TOL)
{
   const int m = Mlu.SizeI();
   const int NE = Mlu.SizeK();
   P.SetSize(m*NE);

   auto data_all = mfem::Reshape(Mlu.ReadWrite(), m, m, NE);
   auto ipiv_all = mfem::Reshape(P.Write(), m, NE);
   Array<bool> pivot_flag(1);
   pivot_flag[0] = true;
   bool *d_pivot_flag = pivot_flag.ReadWrite();

   MFEM_FORALL(e, NE,
   {
      for (int i = 0; i < m; i++)
      {
         // pivoting
         {
            int piv = i;
            double a = fabs(data_all(piv,i,e));
            for (int j = i+1; j < m; j++)
            {
               const double b = fabs(data_all(j,i,e));
               if (b > a)
               {
                  a = b;
                  piv = j;
               }
            }
            ipiv_all(i,e) = piv;
            if (piv != i)
            {
               // swap rows i and piv in both L and U parts
               for (int j = 0; j < m; j++)
               {
                  mfem::kernels::internal::Swap<double>(data_all(i,j,e), data_all(piv,j,e));
               }
            }
         } // pivot end

         if (abs(data_all(i,i,e)) <= TOL)
         {
            d_pivot_flag[0] = false;
         }

         const double a_ii_inv = 1.0 / data_all(i,i,e);
         for (int j = i+1; j < m; j++)
         {
            data_all(j,i,e) *= a_ii_inv;
         }

         for (int k = i+1; k < m; k++)
         {
            const double a_ik = data_all(i,k,e);
            for (int j = i+1; j < m; j++)
            {
               data_all(j,k,e) -= a_ik * data_all(j,i,e);
            }
         }

      } // m loop

   });

   MFEM_ASSERT(pivot_flag.HostRead()[0], "Batch LU factorization failed \n");
}

void BatchLUSolve(const DenseTensor &Mlu, const Array<int> &P, Vector &X)
{

   const int m = Mlu.SizeI();
   const int NE = Mlu.SizeK();

   auto data_all = mfem::Reshape(Mlu.Read(), m, m, NE);
   auto piv_all = mfem::Reshape(P.Read(), m, NE);
   auto x_all = mfem::Reshape(X.ReadWrite(), m, NE);

   MFEM_FORALL(e, NE,
   {
      kernels::LUSolve(&data_all(0, 0,e), m, &piv_all(0, e), &x_all(0,e));
   });

}

} // namespace mfem
