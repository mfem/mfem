// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


// Implementation of data types dense matrix, inverse dense matrix


#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

#include "vector.hpp"
#include "matrix.hpp"
#include "densemat.hpp"


DenseMatrix::DenseMatrix() : Matrix(0)
{
   height = 0;
   data = NULL;
}

DenseMatrix::DenseMatrix(const DenseMatrix &m) : Matrix(m.size)
{
   height = m.height;
   int hw = size * height;
   data = new double[hw];
   for (int i = 0; i < hw; i++)
      data[i] = m.data[i];
}

DenseMatrix::DenseMatrix(int s) : Matrix(s)
{
   height = s;

   data = new double[s*s];

   for (int i = 0; i < s; i++)
      for (int j = 0; j < s; j++)
         (*this)(i,j) = 0;
}

DenseMatrix::DenseMatrix(int m, int n) : Matrix(n)
{
   height = m;

   data = new double[m*n];

   for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
         (*this)(i,j) = 0;
}

DenseMatrix::DenseMatrix(const DenseMatrix &mat, char ch) : Matrix(mat.height)
{
   if (ch == 't')
   {
      height = mat.size;

      data = new double[size*height];

      for (int i = 0; i < height; i++)
         for (int j = 0; j < size; j++)
            (*this)(i,j) = mat(j,i);
   }
}

void DenseMatrix::SetSize(int s)
{
   if (Size() == s && Height() == s)
      return;
   if (data != NULL)
      delete [] data;
   size = height = s;
   if (s > 0)
   {
      int ss = s*s;
      data = new double[ss];
      for (int i = 0; i < ss; i++)
         data[i] = 0.0;
   }
   else
      data = 0;
}

void DenseMatrix::SetSize(int h, int w)
{
   if (Size() == w && Height() == h)
      return;
   if (data != NULL)
      delete [] data;
   size = w;
   height = h;
   if (h > 0 && w > 0)
   {
      int hw = h*w;
      data = new double[hw];
      for (int i = 0; i < hw; i++)
         data[i] = 0.0;
   }
   else
      data = 0;
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
   for (int i = 0; i < height; i++)
   {
      double a = 0.;
      for (int j = 0; j < size; j++)
         a += (*this)(i,j) * x[j];
      y[i] = a;
   }
}

void DenseMatrix::Mult(const Vector &x, Vector &y) const
{
#ifdef MFEM_DEBUG
   if ( size != x.Size() || height != y.Size() )
      mfem_error("DenseMatrix::Mult");
#endif

   Mult((const double *)x, (double *)y);
}

double DenseMatrix::operator *(const DenseMatrix &m) const
{
#ifdef MFEM_DEBUG
   if (Height() != m.Height() || Width() != m.Width())
      mfem_error("DenseMatrix::operator *(...)");
#endif

   int hw = size * height;
   double a = 0.0;
   for (int i = 0; i < hw; i++)
      a += data[i] * m.data[i];

   return a;
}

void DenseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
#ifdef MFEM_DEBUG
   if ( height != x.Size() || size != y.Size() )
      mfem_error("DenseMatrix::MultTranspose");
#endif

   for (int i = 0; i < size; i++)
   {
      double d = 0.0;
      for (int j = 0; j < height; j++)
         d += (*this)(j,i) * x(j);
      y(i) = d;
   }
}

void DenseMatrix::AddMult(const Vector &x, Vector &y) const
{
#ifdef MFEM_DEBUG
   if ( size != x.Size() || height != y.Size() )
      mfem_error("DenseMatrix::AddMult");
#endif

   for (int i = 0; i < height; i++)
      for (int j = 0; j < size; j++)
         y(i) += (*this)(i,j) * x(j);
}

double DenseMatrix::InnerProduct(const double *x, const double *y) const
{
   double prod = 0.0;

   for (int i = 0; i < height; i++)
   {
      double Axi = 0.0;
      for (int j = 0; j < size; j++)
         Axi += (*this)(i,j) * x[j];
      prod += y[i] * Axi;
   }

   return prod;
}

double DenseMatrix::Trace() const
{
#ifdef MFEM_DEBUG
   if (Width() != Height())
      mfem_error("DenseMatrix::Trace() : not a square matrix!");
#endif

   double t = 0.0;

   for (int i = 0; i < size; i++)
      t += (*this)(i, i);

   return t;
}

MatrixInverse *DenseMatrix::Inverse() const
{
   return new DenseMatrixInverse(*this);
}

double DenseMatrix::Det() const
{
#ifdef MFEM_DEBUG
   if (Height() != Width() || Height() < 1 || Height() > 3)
      mfem_error("DenseMatrix::Det");
#endif

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
   }
   return 0.0;
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
   else if ((Height() == 3) && (Width() == 2))
   {
      const double *d = data;
      double E = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      double G = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
      double F = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
      return sqrt(E * G - F * F);
   }
   mfem_error("DenseMatrix::Weight()");
   return 0.0;
}

void DenseMatrix::Add(const double c, DenseMatrix &A)
{
   for (int i = 0; i < Height(); i++)
      for (int j = 0; j < Size(); j++)
         (*this)(i,j) += c * A(i,j);
}

DenseMatrix &DenseMatrix::operator=(double c)
{
   int s = Size()*Height();
   if (data != NULL)
      for (int i = 0; i < s; i++)
         data[i] = c;
   return *this;
}

DenseMatrix &DenseMatrix::operator=(const DenseMatrix &m)
{
   int i, hw;

   SetSize(m.height, m.size);

   hw = size * height;
   for (i = 0; i < hw; i++)
      data[i] = m.data[i];

   return *this;
}

DenseMatrix &DenseMatrix::operator+=(DenseMatrix &m)
{
   int i, j;

   for (i = 0; i < height; i++)
      for (j = 0; j < size; j++)
         (*this)(i, j) += m(i, j);

   return *this;
}

DenseMatrix &DenseMatrix::operator-=(DenseMatrix &m)
{
   int i, j;

   for (i = 0; i < height; i++)
      for (j = 0; j < size; j++)
         (*this)(i, j) -= m(i, j);

   return *this;
}

DenseMatrix &DenseMatrix::operator*=(double c)
{
   int s = Size()*Height();
   if (data != NULL)
      for (int i = 0; i < s; i++)
         data[i] *= c;
   return *this;
}

void DenseMatrix::Neg()
{
   int i, hw = Height() * Size();

   for (i = 0; i < hw; i++)
      data[i] = -data[i];
}

#ifdef MFEM_USE_LAPACK
extern "C" void
dgetrf_(int *, int *, double *, int *, int *, int *);
extern "C" void
dgetrs_(char *, int *, int *, double *, int *, int *, double *, int *, int *);
extern "C" void
dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK,
        int *LWORK, int *INFO);
#endif

void DenseMatrix::Invert()
{
#ifdef MFEM_DEBUG
   if (Size() <= 0 || Size() != Height())
      mfem_error("DenseMatrix::Invert()");
#endif

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[size];
   int    lwork = -1;
   double qwork, *work;
   int    info;

   dgetrf_(&size, &size, data, &size, ipiv, &info);

   if (info)
      mfem_error("DenseMatrix::Invert() : Error in DGETRF");

   dgetri_(&size, data, &size, ipiv, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];

   dgetri_(&size, data, &size, ipiv, work, &lwork, &info);

   if (info)
      mfem_error("DenseMatrix::Invert() : Error in DGETRI");

   delete [] work;
   delete [] ipiv;
#else
   int c, i, j, n = Size();
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
         mfem_error("DenseMatrix::Invert() : singular matrix");
      piv[c] = i;
      for (j = 0; j < n; j++)
         Swap<double>((*this)(c, j), (*this)(i, j));

      a = (*this)(c, c) = 1.0 / (*this)(c, c);
      for (j = 0; j < c; j++)
         (*this)(c, j) *= a;
      for (j++; j < n; j++)
         (*this)(c, j) *= a;
      for (i = 0; i < c; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
            (*this)(i, j) += b * (*this)(c, j);
         for (j++; j < n; j++)
            (*this)(i, j) += b * (*this)(c, j);
      }
      for (i++; i < n; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
            (*this)(i, j) += b * (*this)(c, j);
         for (j++; j < n; j++)
            (*this)(i, j) += b * (*this)(c, j);
      }
   }

   for (c = n - 1; c >= 0; c--)
   {
      j = piv[c];
      for (i = 0; i < n; i++)
         Swap<double>((*this)(i, c), (*this)(i, j));
   }
#endif
}

void DenseMatrix::Norm2(double *v) const
{
   for (int j = 0; j < Size(); j++)
   {
      v[j] = 0.0;
      for (int i = 0; i < Height(); i++)
         v[j] += (*this)(i,j)*(*this)(i,j);
      v[j] = sqrt(v[j]);
   }
}

double DenseMatrix::FNorm() const
{
   int i, hw = Height() * Size();
   double max_norm = 0.0, entry, fnorm2;

   for (i = 0; i < hw; i++)
   {
      entry = fabs(data[i]);
      if (entry > max_norm)
         max_norm = entry;
   }

   if (max_norm == 0.0)
      return 0.0;

   fnorm2 = 0.0;
   for (i = 0; i < hw; i++)
   {
      entry = data[i] / max_norm;
      fnorm2 += entry * entry;
   }

   return max_norm * sqrt(fnorm2);
}

#ifdef MFEM_USE_LAPACK
extern "C" void
dsyevr_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
        double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M,
        double *W, double *Z, int *LDZ, int *ISUPPZ, double *WORK, int *LWORK,
        int *IWORK, int *LIWORK, int *INFO);
extern "C" void
dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W,
       double *WORK, int *LWORK, int *INFO);
extern "C" void
dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
        double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK,
        int *LWORK, int *INFO);
#endif

void dsyevr_Eigensystem(DenseMatrix &a, Vector &ev, DenseMatrix *evect)
{

#ifdef MFEM_USE_LAPACK

   ev.SetSize(a.Size());

   char      JOBZ     = 'N';
   char      RANGE    = 'A';
   char      UPLO     = 'U';
   int       N        = a.Size();
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
      A[i] = data[i];

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
      cerr << "dsyevr_Eigensystem(...): DSYEVR error code: "
           << INFO << endl;
      mfem_error();
   }

#ifdef MFEM_DEBUG
   if (M < N)
   {
      cerr << "dsyevr_Eigensystem(...):\n"
           << " DSYEVR did not find all eigenvalues "
           << M << "/" << N << endl;
      mfem_error();
   }
   for (IL = 0; IL < N; IL++)
      if (!finite(W[IL]))
         mfem_error("dsyevr_Eigensystem(...): !finite value in W");
   for (IL = 0; IL < N*N; IL++)
      if (!finite(Z[IL]))
         mfem_error("dsyevr_Eigensystem(...): !finite value in Z");
   VU = 0.0;
   for (IL = 0; IL < N; IL++)
      for (IU = 0; IU <= IL; IU++)
      {
         VL = 0.0;
         for (M = 0; M < N; M++)
            VL += Z[M+IL*N] * Z[M+IU*N];
         if (IU < IL)
            VL = fabs(VL);
         else
            VL = fabs(VL-1.0);
         if (VL > VU)
            VU = VL;
         if (VU > 0.5)
         {
            cerr << "dsyevr_Eigensystem(...):"
                 << " Z^t Z - I deviation = " << VU
                 << "\n W[max] = " << W[N-1] << ", W[min] = "
                 << W[0] << ", N = " << N << endl;
            mfem_error();
         }
      }
   if (VU > 1e-9)
   {
      cerr << "dsyevr_Eigensystem(...):"
           << " Z^t Z - I deviation = " << VU
           << "\n W[max] = " << W[N-1] << ", W[min] = "
           << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
      mfem_error("dsyevr_Eigensystem(...): ERROR: ...");
   VU = 0.0;
   for (IL = 0; IL < N; IL++)
      for (IU = 0; IU < N; IU++)
      {
         VL = 0.0;
         for (M = 0; M < N; M++)
            VL += Z[IL+M*N] * W[M] * Z[IU+M*N];
         VL = fabs(VL-data[IL+N*IU]);
         if (VL > VU)
            VU = VL;
      }
   if (VU > 1e-9)
   {
      cerr << "dsyevr_Eigensystem(...):"
           << " max matrix deviation = " << VU
           << "\n W[max] = " << W[N-1] << ", W[min] = "
           << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
      mfem_error("dsyevr_Eigensystem(...): ERROR: ...");
#endif

   delete [] IWORK;
   delete [] WORK;
   delete [] ISUPPZ;
   delete [] A;

#endif
}

void dsyev_Eigensystem(DenseMatrix &a, Vector &ev, DenseMatrix *evect)
{

#ifdef MFEM_USE_LAPACK

   int   N      = a.Size();
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
      A[i] = data[i];

   dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, &QWORK, &LWORK, &INFO);

   LWORK = (int) QWORK;
   WORK = new double[LWORK];

   dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);

   if (INFO != 0)
   {
      cerr << "dsyev_Eigensystem: DSYEV error code: " << INFO << endl;
      mfem_error();
   }

   delete [] WORK;
   if (evect == NULL)  delete [] A;

#endif
}

void DenseMatrix::Eigensystem(Vector &ev, DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK

   // dsyevr_Eigensystem(*this, ev, evect);

   dsyev_Eigensystem(*this, ev, evect);

#else

   mfem_error("DenseMatrix::Eigensystem");

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
      cerr << "DenseMatrix::SingularValues : info = " << info << endl;
      mfem_error();
   }
#else
   // compiling without lapack
   mfem_error("DenseMatrix::SingularValues");
#endif
}

double DenseMatrix::CalcSingularvalue(const int i) const
{
#ifdef MFEM_DEBUG
   if (Height() != Width() || Height() < 2 || Height() > 3)
      mfem_error("DenseMatrix::CalcSingularvalue");
#endif

   const int n = Height();
   const double *d = data;

   if (n == 2)
   {
#if 0
      double b11 = d[0]*d[0] + d[1]*d[1];
      double b12 = d[0]*d[2] + d[1]*d[3];
      double b22 = d[2]*d[2] + d[3]*d[3];

      double tmp     = 0.5*(b11 - b22);
      double sqrtD_2 = sqrt(tmp*tmp + b12*b12);
      double mid     = 0.5*(b11 + b22);
      if (i == 0)
         return sqrt(mid + sqrtD_2);
      if ((mid -= sqrtD_2) <= 0.0)
         return 0.0;
      return sqrt(mid);
#else
      register double d0, d1, d2, d3;
      d0 = d[0];
      d1 = d[1];
      d2 = d[2];
      d3 = d[3];
      // double b11 = d[0]*d[0] + d[1]*d[1];
      // double b12 = d[0]*d[2] + d[1]*d[3];
      // double b22 = d[2]*d[2] + d[3]*d[3];
      // t = 0.5*(a+b).(a-b) = 0.5*(|a|^2-|b|^2)
      // with a,b - the columns of (*this)
      // double t = 0.5*(b11 - b22);
      double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
      // double s = sqrt(0.5*(b11 + b22) + sqrt(t*t + b12*b12));
      double s = d0*d2 + d1*d3;
      s = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));
      if (s == 0.0)
         return 0.0;
      t = fabs(d0*d3 - d1*d2) / s;
// #ifdef MFEM_DEBUG
      if (t > s)
      {
         if (i == 0)
            return t;
         return s;
         // mfem_error("DenseMatrix::CalcSingularvalue : 2x2");
      }
// #endif
      if (i == 0)
         return s;
      return t;
#endif
   }
   else
   {
      double b11 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
      double b12 = d[0]*d[3] + d[1]*d[4] + d[2]*d[5];
      double b13 = d[0]*d[6] + d[1]*d[7] + d[2]*d[8];
      double b22 = d[3]*d[3] + d[4]*d[4] + d[5]*d[5];
      double b23 = d[3]*d[6] + d[4]*d[7] + d[5]*d[8];
      double b33 = d[6]*d[6] + d[7]*d[7] + d[8]*d[8];

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
         double b11_b22 = ((d[0]-d[3])*(d[0]+d[3])+
                           (d[1]-d[4])*(d[1]+d[4])+
                           (d[2]-d[5])*(d[2]+d[5]));
         double b22_b33 = ((d[3]-d[6])*(d[3]+d[6])+
                           (d[4]-d[7])*(d[4]+d[7])+
                           (d[5]-d[8])*(d[5]+d[8]));
         double b33_b11 = ((d[6]-d[0])*(d[6]+d[0])+
                           (d[7]-d[1])*(d[7]+d[1])+
                           (d[8]-d[2])*(d[8]+d[2]));
         c1 = (b11_b22 - b33_b11)/3;
         c2 = (b22_b33 - b11_b22)/3;
         c3 = (b33_b11 - b22_b33)/3;
      }
      double Q = (2*(b12*b12 + b13*b13 + b23*b23) +
                  c1*c1 + c2*c2 + c3*c3)/6;
      double R = (c1*(b23*b23 - c2*c3)+ b12*(b12*c3 - 2*b13*b23) +
                  b13*b13*c2)/2;
      // R = (-1/2)*det(B-(tr(B)/3)*I)
      // Note: 54*(det(S))^2 <= |S|_F^6, when S^t=S and tr(S)=0, S is 3x3
      // Therefore: R^2 <= Q^3

      if (Q <= 0.)
      {
         ;
      }

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
               R = -1.;
               r = 2*sqrtQ;
            }
            else
            {
               R = 1.;
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
                  aa -= 2*sqrtQ*cos(acos(R)/3); // min
               else if (i == 0)
                  aa -= 2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
               else
                  aa -= 2*sqrtQ*cos((acos(R) - 2.0*M_PI)/3); // mid
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

         // QR factorization of
         //   c1  b12  b13
         //  b12   c2  b23
         //  b13  b23   c3
         // to find an eigenvector (z1,z2,z3) for [tr(B)/3 + r]
         double z1, z2, z3;
         double sigma = b12*b12 + b13*b13;
         double mu, gamma, u1;
         double c12, c13, c23, c32, w1, w2, w3;
         if (sigma == 0.)
         {
            z1 = 1.;
            z2 = z3 = 0.;
         }
         else
         {
            mu = copysign(sqrt(c1*c1 + sigma), c1);
            u1 = -sigma/(c1 + mu); // = c1 - mu
            gamma = 2./(sigma + u1*u1);
            // u = (u1, b12, b13),  gamma = 2/(u^t u)
            // Q = I - 2/(u^t u) u u^t,  Q (c1, b12, b13) = mu e_1
            c1 = mu;
            w2  = gamma*(b12*(u1 + c2) + b23*b13);
            w3  = gamma*(b13*(u1 + c3) + b23*b12);
            c12 = b12 - u1 *w2;
            c2  = c2  - b12*w2;
            c32 = b23 - b13*w2;
            c13 = b13 - u1 *w3;
            c23 = b23 - b12*w3;
            c3  = c3  - b13*w3;

            sigma = c32*c32;
            if (sigma == 0.)
            {
               ;
            }
            else
            {
               mu = copysign(sqrt(c2*c2 + sigma), c2);
               u1 = -sigma/(c2 + mu); // = c2 - mu
               gamma = 2./(sigma + u1*u1);
               // u = (0, u1, c32),  gamma = 2/(u^t u)
               // Q = I - 2/(u^t u) u u^t,  Q (c12, c2, c32) = (c12, mu, 0)
               c2 = mu;
               w3 = gamma*(u1*c23 + c32*c3);
               c23 = c23 - u1 *w3;
               c3  = c3  - c32*w3;
            }
            //     solve:
            // | c1 c12 c13 | | z1 |   | 0 |
            // | 0   c2 c23 | | z2 | = | 0 |
            // | 0   0   c3 | | z3 |   | 0 |
            //  either c2 or c3 must be 0
            if (fabs(c3) < fabs(c2))
            {
               // c3 ~ 0?  -->  set z3 = 1
               //           c2*z2 + c23 = 0  ==>  z2 = -c23/c2
               //  c1*z1 + c12*z2 + c13 = 0  ==>  z1 = (-c13 - c12*z2)/c1
               z3 = 1.;
               z2 = -c23/c2;
               z1 = -(c13 + c12*z2)/c1;
            }
            else
            {
               // c2 ~ 0?
               z3 = 0.;
               z2 = 1.;
               z1 = -c12/c1;
            }
         }

         // using the eigenvector z=(z1,z2,z3) transform B into
         //         | *   0   0 |
         // Q B Q = | 0  c2 c23 |
         //         | 0 c23  c3 |
         sigma = z2*z2 + z3*z3;
         if (sigma == 0.)
         {
            c2  = b22;
            c23 = b23;
            c3  = b33;
         }
         else
         {
            mu = copysign(sqrt(z1*z1 + sigma), z1);
            u1 = -sigma/(z1 + mu); // = z1 - mu
            gamma = 2./(sigma + u1*u1);
            // u = (u1, z2, z3),  gamma = 2/(u^t u)
            // Q = I - 2/(u^t u) u u^t,  Q (z1, z2, z3) = mu e_1
            // Compute Q B Q
            // w = gamma*B u
            w1 = gamma*(b11*u1 + b12*z2 + b13*z3);
            w2 = gamma*(b12*u1 + b22*z2 + b23*z3);
            w3 = gamma*(b13*u1 + b23*z2 + b33*z3);
            // w <-  w - (gamma*(u^t w)/2) u
            double gutw2 = gamma*(u1*w1 + z2*w2 + z3*w3)/2;
            w2 -= gutw2*z2;
            w3 -= gutw2*z3;
            c2  = b22 - 2*z2*w2;
            c23 = b23 - z2*w3 - z3*w2;
            c3  = b33 - 2*z3*w3;

#ifdef MFEM_DEBUG
            // for debugger testing
            // is z close to an eigenvector?
            w1 -= gutw2*u1;
            c1  = b11 - 2*u1*w1; // is c1 more accurate than (aa + r)?
            c12 = b12 - u1*w2 - z2*w1;
            c13 = b13 - u1*w3 - z3*w1;
#endif
         }

         // find the eigenvalues of
         //  |  c2 c23 |
         //  | c23  c3 |
         w1 = 0.5*(c3 - c2);
         w2 = 0.5*(c2 + c3) + sqrt(w1*w1 + c23*c23);
         if (w2 == 0.0)
         {
            w1 = 0.0;
         }
         else
         {
            w1 = (c2*c3 - c23*c23)/w2;
         }

         if (R < 0.)
         {
            // order is w1 <= w2 <= aa + r
            if (i == 2)
               aa = w1;
            else if (i == 0)
               aa += r;
            else
               aa = w2;
         }
         else
         {
            // order is aa + r <= w1 <= w2
            if (i == 2)
               aa += r;
            else if (i == 0)
               aa = w2;
            else
               aa = w1;
         }

         // double theta = acos(R / sqrtQ3);
         // double A = -2 * sqrt(Q);
         //
         // if (i == 2)
         // {
         //    aa += A * cos(theta / 3); // min
         // }
         // else if (i == 0)
         // {
         //    aa += A * cos((theta + 2.0 * M_PI) / 3); // max
         // }
         // else
         // {
         //    aa += A * cos((theta - 2.0 * M_PI) / 3); // mid
         // }
      }

   have_aa:
      if (aa < 0.0)
      {
         cerr << "DenseMatrix::CalcSingularvalue (3x3) : aa = " << aa << endl;
         mfem_error();
         return 0.0;
      }

      return sqrt(aa);
   }

//   return 0.0;
}

void DenseMatrix::CalcEigenvalues(double *lambda, double *vec) const
{
#ifdef MFEM_DEBUG
   if (Height() != Width() || Height() < 2 || Height() > 3)
      mfem_error("DenseMatrix::CalcEigenvalues");
#endif

   const int n = Height();
   const double *d = data;

   if (n == 2)
   {
      const double d0 = d[0];
      const double d2 = d[2]; // use the upper triangular entry
      const double d3 = d[3];

      double c, s, l0, l1;
      if (d2 == 0.)
      {
         c = 1.;
         s = 0.;
         l0 = d0;
         l1 = d3;
      }
      else
      {
         // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
         double zeta = (d3 - d0)/(2*d2);
         double t = copysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
         c = 1./sqrt(1. + t*t);
         s = c*t;
         l0 = d0 - d2*t;
         l1 = d3 + d2*t;
      }
      if (l0 <= l1)
      {
         lambda[0] = l0;
         lambda[1] = l1;
         vec[0] =  c;
         vec[1] = -s;
         vec[2] =  s;
         vec[3] =  c;
      }
      else
      {
         lambda[0] = l1;
         lambda[1] = l0;
         vec[0] =  s;
         vec[1] =  c;
         vec[2] =  c;
         vec[3] = -s;
      }
   }
   else
   {
      const double d11 = d[0];
      const double d22 = d[4];
      const double d33 = d[8];
      const double d12 = d[3]; // use the upper triangular entries
      const double d13 = d[6];
      const double d23 = d[7];

      double aa = (d11 + d22 + d33)/3;  // aa = tr(A)/3
      double c1 = d11 - aa;
      double c2 = d22 - aa;
      double c3 = d33 - aa;

      double Q = (2*(d12*d12 + d13*d13 + d23*d23) +
                  c1*c1 + c2*c2 + c3*c3)/6;
      double R = (c1*(d23*d23 - c2*c3)+ d12*(d12*c3 - 2*d13*d23) +
                  d13*d13*c2)/2;

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
               R = -1.;
               r = 2*sqrtQ;
            }
            else
            {
               R = 1.;
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

         c1 -= r;
         c2 -= r;
         c3 -= r;

         // QR factorization of
         //   c1  d12  d13
         //  d12   c2  d23
         //  d13  d23   c3
         // to find an eigenvector (z1,z2,z3) for [tr(A)/3 + r]
         double z1, z2, z3;
         double sigma = d12*d12 + d13*d13;
         double mu, gamma, u1;
         double c12, c13, c23, c32, w1, w2, w3;
         if (sigma == 0.)
         {
            z1 = 1.;
            z2 = z3 = 0.;
         }
         else
         {
            mu = copysign(sqrt(c1*c1 + sigma), c1);
            u1 = -sigma/(c1 + mu);
            gamma = 2./(sigma + u1*u1);
            c1 = mu;
            w2  = gamma*(d12*(u1 + c2) + d23*d13);
            w3  = gamma*(d13*(u1 + c3) + d23*d12);
            c12 = d12 - u1 *w2;
            c2  = c2  - d12*w2;
            c32 = d23 - d13*w2;
            c13 = d13 - u1 *w3;
            c23 = d23 - d12*w3;
            c3  = c3  - d13*w3;

            sigma = c32*c32;
            if (sigma == 0.)
            {
               ;
            }
            else
            {
               mu = copysign(sqrt(c2*c2 + sigma), c2);
               u1 = -sigma/(c2 + mu);
               gamma = 2./(sigma + u1*u1);
               c2 = mu;
               w3 = gamma*(u1*c23 + c32*c3);
               c23 = c23 - u1 *w3;
               c3  = c3  - c32*w3;
            }
            if (fabs(c3) < fabs(c2))
            {
               z3 = 1.;
               z2 = -c23/c2;
               z1 = -(c13 + c12*z2)/c1;
            }
            else
            {
               z3 = 0.;
               z2 = 1.;
               z1 = -c12/c1;
            }
         }

         // using the eigenvector z=(z1,z2,z3) transform A into
         //         | *   0   0 |
         // Q A Q = | 0  c2 c23 |
         //         | 0 c23  c3 |
         sigma = z2*z2 + z3*z3;
         if (sigma == 0.)
         {
            c2  = d22;
            c23 = d23;
            c3  = d33;
         }
         else
         {
            mu = copysign(sqrt(z1*z1 + sigma), z1);
            u1 = -sigma/(z1 + mu);
            gamma = 2./(sigma + u1*u1);
            w1 = gamma*(d11*u1 + d12*z2 + d13*z3);
            w2 = gamma*(d12*u1 + d22*z2 + d23*z3);
            w3 = gamma*(d13*u1 + d23*z2 + d33*z3);
            double gutw2 = gamma*(u1*w1 + z2*w2 + z3*w3)/2;
            w2 -= gutw2*z2;
            w3 -= gutw2*z3;
            c2  = d22 - 2*z2*w2;
            c23 = d23 - z2*w3 - z3*w2;
            c3  = d33 - 2*z3*w3;

#ifdef MFEM_DEBUG
            // for debugger testing
            // is z close to an eigenvector?
            w1 -= gutw2*u1;
            c1  = d11 - 2*u1*w1; // is c1 more accurate than (aa + r)?
            c12 = d12 - u1*w2 - z2*w1;
            c13 = d13 - u1*w3 - z3*w1;
#endif
         }

         // find the eigenvalues and eigenvectors for
         // |  c2 c23 |
         // | c23  c3 |
         double c, s;
         if (c23 == 0.)
         {
            c = 1.;
            s = 0.;
            w1 = c2;
            w2 = c3;
         }
         else
         {
            double zeta = (c3 - c2)/(2*c23);
            double t = copysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
            c = 1./sqrt(1. + t*t);
            s = c*t;
            w1 = c2 - c23*t;
            w2 = c3 + c23*t;
         }
         // w1 <-> Q (0, c, -s), w2 <-> Q (0, s, c)
         // Q = I, when sigma = 0
         // Q = I - gamma* u u^t,  u = (u1, z2, z3)

         double *vec_ar, *vec_w1, *vec_w2;
         if (R < 0.)
         {
            // (aa + r) is max
            lambda[2] = aa + r;
            vec_ar = vec + 6;
            if (w1 <= w2)
            {
               lambda[0] = w1;  vec_w1 = vec;
               lambda[1] = w2;  vec_w2 = vec + 3;
            }
            else
            {
               lambda[0] = w2;  vec_w2 = vec;
               lambda[1] = w1;  vec_w1 = vec + 3;
            }
         }
         else
         {
            // (aa + r) is min
            lambda[0] = aa + r;
            vec_ar = vec;
            if (w1 <= w2)
            {
               lambda[1] = w1;  vec_w1 = vec + 3;
               lambda[2] = w2;  vec_w2 = vec + 6;
            }
            else
            {
               lambda[1] = w2;  vec_w2 = vec + 3;
               lambda[2] = w1;  vec_w1 = vec + 6;
            }
         }

         if (sigma == 0.)
         {
            mu = fabs(z1);
            vec_w1[0] = 0.;  vec_w2[0] = 0.;
            vec_w1[1] =  c;  vec_w2[1] = s;
            vec_w1[2] = -s;  vec_w2[2] = c;
         }
         else
         {
            mu = fabs(mu);
            w1 = gamma*(z2*c - z3*s);
            w2 = gamma*(z2*s + z3*c);
            vec_w1[0] =    - u1*w1;  vec_w2[0] =   - u1*w2;
            vec_w1[1] =  c - z2*w1;  vec_w2[1] = s - z2*w2;
            vec_w1[2] = -s - z3*w1;  vec_w2[2] = c - z3*w2;
         }
         vec_ar[0] = z1/mu;
         vec_ar[1] = z2/mu;
         vec_ar[2] = z3/mu;
      }
   }
}

void DenseMatrix::GetColumn(int c, Vector &col)
{
   int n;
   double *cp, *vp;

   n = Height();
   col.SetSize(n);
   cp = data + c * n;
   vp = col.GetData();

   for (int i = 0; i < n; i++)
      vp[i] = cp[i];
}

void DenseMatrix::Diag(double c, int n)
{
   SetSize(n);

   int i, N = n*n;
   for (i = 0; i < N; i++)
      data[i] = 0.0;
   for (i = 0; i < n; i++)
      data[i*(n+1)] = c;
}

void DenseMatrix::Diag(double *diag, int n)
{
   SetSize(n);

   int i, N = n*n;
   for (i = 0; i < N; i++)
      data[i] = 0.0;
   for (i = 0; i < n; i++)
      data[i*(n+1)] = diag[i];
}

void DenseMatrix::Transpose()
{
   int i, j;
   double t;

   if (Size() == Height())
   {
      for (i = 0; i < Height(); i++)
         for (j = i+1; j < Size(); j++)
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

void DenseMatrix::Transpose(DenseMatrix &A)
{
   SetSize(A.Size(),A.Height());

   for (int i = 0; i < Height(); i++)
      for (int j = 0; j < Size(); j++)
         (*this)(i,j) = A(j,i);
}

void DenseMatrix::Symmetrize()
{
#ifdef MFEM_DEBUG
   if (Width() != Height())
      mfem_error("DenseMatrix::Symmetrize() : not a square matrix!");
#endif

   for (int i = 0; i < Height(); i++)
      for (int j = 0; j < i; j++)
      {
         double a = 0.5 * ((*this)(i,j) + (*this)(j,i));
         (*this)(j,i) = (*this)(i,j) = a;
      }
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
   if (Width() != 3 || curl.Width() != 3 || 3*n != curl.Height())
      mfem_error("DenseMatrix::GradToCurl(...)");
#endif

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

void DenseMatrix::GradToDiv(Vector &div)
{

#ifdef MFEM_DEBUG
   if (Width()*Height() != div.Size())
      mfem_error("DenseMatrix::GradToDiv(...)");
#endif

   // div(dof*j+i) <-- (*this)(i,j)

   int n = size * height;
   double *ddata = div.GetData();

   for (int i = 0; i < n; i++)
      ddata[i] = data[i];
}

void DenseMatrix::CopyRows(DenseMatrix &A, int row1, int row2)
{
   SetSize(row2 - row1 + 1, A.Size());

   for (int i = row1; i <= row2; i++)
      for (int j = 0; j < Size(); j++)
         (*this)(i-row1,j) = A(i,j);
}

void DenseMatrix::CopyCols(DenseMatrix &A, int col1, int col2)
{
   SetSize(A.Height(), col2 - col1 + 1);

   for (int i = 0; i < Height(); i++)
      for (int j = col1; j <= col2; j++)
         (*this)(i,j-col1) = A(i,j);
}

void DenseMatrix::CopyMN(DenseMatrix &A, int m, int n, int Aro, int Aco)
{
   int i, j;

   SetSize(m,n);

   for (j = 0; j < n; j++)
      for (i = 0; i < m; i++)
         (*this)(i,j) = A(Aro+i,Aco+j);
}

void DenseMatrix::CopyMN(DenseMatrix &A, int row_offset, int col_offset)
{
   int i, j;
   double *v = A.data;

   for (j = 0; j < A.Size(); j++)
      for (i = 0; i < A.Height(); i++)
         (*this)(row_offset+i,col_offset+j) = *(v++);
}

void DenseMatrix::CopyMNt(DenseMatrix &A, int row_offset, int col_offset)
{
   int i, j;
   double *v = A.data;

   for (i = 0; i < A.Size(); i++)
      for (j = 0; j < A.Height(); j++)
         (*this)(row_offset+i,col_offset+j) = *(v++);
}

void DenseMatrix::CopyMNDiag(double c, int n, int row_offset, int col_offset)
{
   int i, j;

   for (i = 0; i < n; i++)
      for (j = i+1; j < n; j++)
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;

   for (i = 0; i < n; i++)
      (*this)(row_offset+i,col_offset+i) = c;
}

void DenseMatrix::CopyMNDiag(double *diag, int n, int row_offset,
                             int col_offset)
{
   int i, j;

   for (i = 0; i < n; i++)
      for (j = i+1; j < n; j++)
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;

   for (i = 0; i < n; i++)
      (*this)(row_offset+i,col_offset+i) = diag[i];
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
      mfem_error("DenseMatrix::AddMatrix(...) 1");
#endif

   p  = data + ro + co * h;
   ap = A.data;

   for (int c = 0; c < aw; c++)
   {
      for (int r = 0; r < ah; r++)
         p[r] += ap[r];
      p  += h;
      ap += ah;
   }
}

void DenseMatrix::AddMatrix(double a, DenseMatrix &A, int ro, int co)
{
   int h, ah, aw;
   double *p, *ap;

   h  = Height();
   ah = A.Height();
   aw = A.Width();

#ifdef MFEM_DEBUG
   if (co+aw > Width() || ro+ah > h)
      mfem_error("DenseMatrix::AddMatrix(...) 2");
#endif

   p  = data + ro + co * h;
   ap = A.data;

   for (int c = 0; c < aw; c++)
   {
      for (int r = 0; r < ah; r++)
         p[r] += a * ap[r];
      p  += h;
      ap += ah;
   }
}

void DenseMatrix::AddToVector(int offset, Vector &v) const
{
   int i, n = size * height;
   double *vdata = v.GetData() + offset;

   for (i = 0; i < n; i++)
      vdata[i] += data[i];
}

void DenseMatrix::GetFromVector(int offset, const Vector &v)
{
   int i, n = size * height;
   const double *vdata = v.GetData() + offset;

   for (i = 0; i < n; i++)
      data[i] = vdata[i];
}

void DenseMatrix::AdjustDofDirection(Array<int> &dofs)
{
   int n = Height();

#ifdef MFEM_DEBUG
   if (dofs.Size() != n || Width() != n)
      mfem_error("DenseMatrix::AdjustDofDirection(...)");
#endif

   int *dof = dofs;
   for (int i = 0; i < n-1; i++)
   {
      int s = (dof[i] < 0) ? (-1) : (1);
      for (int j = i+1; j < n; j++)
      {
         int t = (dof[j] < 0) ? (-s) : (s);
         if (t < 0)
         {
            (*this)(i,j) = -(*this)(i,j);
            (*this)(j,i) = -(*this)(j,i);
         }
      }
   }
}


void DenseMatrix::Print(ostream &out, int width) const
{
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      out << "[row " << i << "]\n";
      for (int j = 0; j < size; j++)
      {
         out << (*this)(i,j);
         if (j+1 == size || (j+1) % width == 0)
            out << '\n';
         else
            out << ' ';
      }
   }
}

void DenseMatrix::PrintT(ostream &out, int width) const
{
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int j = 0; j < size; j++)
   {
      out << "[col " << j << "]\n";
      for (int i = 0; i < height; i++)
      {
         out << (*this)(i,j);
         if (i+1 == height || (i+1) % width == 0)
            out << '\n';
         else
            out << ' ';
      }
   }
}

void DenseMatrix::TestInversion()
{
   DenseMatrix copy(*this), C(size);
   Invert();
   ::Mult(*this, copy, C);

   double i_max = 0.;
   for (int j = 0; j < size; j++)
      for (int i = 0; i < size; i++)
      {
         if (i == j)
            C(i,j) -= 1.;
         i_max = fmax(i_max, fabs(C(i, j)));
      }
   cout << "size = " << size << ", i_max = " << i_max
        << ", cond_F = " << FNorm()*copy.FNorm() << endl;
}

DenseMatrix::~DenseMatrix()
{
   if (data != NULL)
      delete [] data;
}



void Add(const DenseMatrix &A, const DenseMatrix &B,
         double alpha, DenseMatrix &C)
{
   for (int i = 0; i < C.Height(); i++)
      for (int j = 0; j < C.Size(); j++)
         C(i,j) = A(i,j) + alpha * B(i,j);
}


void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a)
{
#ifdef MFEM_DEBUG
   if (a.height != b.height || a.size != c.size || b.size != c.height)
      mfem_error("Mult (product of DenseMatrices)");
#endif

   register int ah = a.height;
   register int as = a.size;
   register int bs = b.size;
   register double *ad = a.data;
   register double *bd = b.data;
   register double *cd = c.data;
   register int i, j, k;
   register double d, *bdd, *cdd;

   for (j = 0; j < as; j++, cd += bs)
   {
      for (i = 0; i < ah; i++, ad++)
      {
         d = 0.0;
         bdd = bd+i;
         cdd = cd;
         for (k = 0 ; k < bs; k++)
         {
            d += (*bdd) * (*cdd);
            cdd++;
            bdd += ah;
         }
         *ad = d;
      }
   }
}

void CalcAdjugate(const DenseMatrix &a, DenseMatrix &adja)
{
#ifdef MFEM_DEBUG
   if (a.Height() != a.Size() || adja.Height() != adja.Size() ||
       a.Size() != adja.Size() || a.Size() < 2 || a.Size() > 3)
      mfem_error("DenseMatrix::CalcAdjugate(...)");
#endif
   if (a.Size() == 2)
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
   if (a.Height() != a.Size() || adjat.Height() != adjat.Size() ||
       a.Size() != adjat.Size() || a.Size() < 2 || a.Size() > 3)
      mfem_error("DenseMatrix::CalcAdjugateTranspose(...)");
#endif
   if (a.Size() == 2)
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
#ifdef MFEM_DEBUG
   if (a.Width() > a.Height() || a.Width() < 1 || a.Height() > 3)
      mfem_error("DenseMatrix::CalcInverse(...)");
#endif

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
   if (fabs(t) < 1.0e-12 * a.FNorm())
      cerr << "DenseMatrix::CalcInverse(...) : singular matrix!"
           << endl;
   t = 1. / t;
#else
   t = 1.0 / a.Det();
#endif

   switch (a.Height())
   {
   case 1:
      inva(0,0) = t;
      break;
   case 2:
      inva(0,0) = a(1,1) * t ;
      inva(0,1) = -a(0,1) * t ;
      inva(1,0) = -a(1,0) * t ;
      inva(1,1) = a(0,0) * t ;
      break;
   case 3:
      inva(0,0) = (a(1,1)*a(2,2)-a(1,2)*a(2,1))*t;
      inva(0,1) = (a(0,2)*a(2,1)-a(0,1)*a(2,2))*t;
      inva(0,2) = (a(0,1)*a(1,2)-a(0,2)*a(1,1))*t;

      inva(1,0) = (a(1,2)*a(2,0)-a(1,0)*a(2,2))*t;
      inva(1,1) = (a(0,0)*a(2,2)-a(0,2)*a(2,0))*t;
      inva(1,2) = (a(0,2)*a(1,0)-a(0,0)*a(1,2))*t;

      inva(2,0) = (a(1,0)*a(2,1)-a(1,1)*a(2,0))*t;
      inva(2,1) = (a(0,1)*a(2,0)-a(0,0)*a(2,1))*t;
      inva(2,2) = (a(0,0)*a(1,1)-a(0,1)*a(1,0))*t;
      break;
   }
}

void CalcInverseTranspose(const DenseMatrix &a, DenseMatrix &inva)
{
#ifdef MFEM_DEBUG
   if ( (a.Size() != a.Height()) || ( (a.Height()!= 1) && (a.Height()!= 2)
                                      && (a.Height()!= 3) ) )
      mfem_error("DenseMatrix::CalcInverse(...)");
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

void MultAAt(const DenseMatrix &a, DenseMatrix &aat)
{
   for (int i = 0; i < a.Height(); i++)
      for (int j = 0; j <= i; j++)
      {
         double temp = 0.;
         for (int k = 0; k < a.Size(); k++)
            temp += a(i,k) * a(j,k);
         aat(j,i) = aat(i,j) = temp;
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
            t += D(k) * A(i, k) * A(j, k);
         ADAt(i, j) += t;
         ADAt(j, i) += t;
      }
   }

   // process diagonal
   for (int i = 0; i < A.Height(); i++)
   {
      double t = 0.;
      for (int k = 0; k < A.Width(); k++)
         t += D(k) * A(i, k) * A(i, k);
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
            t += D(k) * A(i, k) * A(j, k);
         ADAt(j, i) = ADAt(i, j) = t;
      }
   }
}

#ifdef MFEM_USE_LAPACK
extern "C" void
dgemm_(char *, char *, int *, int *, int *, double *, double *,
       int *, double *, int *, double *, double *, int *);
#endif

void MultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt)
{
#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width() ||
       A.Width() != B.Width())
      mfem_error("MultABt(...)");
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'N', transb = 'T';
   static double alpha = 1.0, beta = 0.0;
   int m = A.Height(), n = B.Height(), k = A.Width();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &m,
          B.Data(), &n, &beta, ABt.Data(), &m);
#elif 1
   register const int ah = A.Height();
   register const int bh = B.Height();
   register const int aw = A.Width();
   register const double *ad = A.Data();
   register const double *bd = B.Data();
   register double *cd = ABt.Data();
   register double *cp;

   cp = cd;
   for (register int i = 0, s = ah*bh; i < s; i++)
      *(cp++) = 0.0;
   for (register int k = 0; k < aw; k++)
   {
      cp = cd;
      for (register int j = 0; j < bh; j++)
      {
         register const double bjk = bd[j];
         for (register int i = 0; i < ah; i++)
         {
            *(cp++) += ad[i] * bjk;
         }
      }
      ad += ah;
      bd += bh;
   }
#elif 1
   register const int ah = A.Height();
   register const int bh = B.Height();
   register const int aw = A.Width();
   register const double *ad = A.Data();
   register const double *bd = B.Data();
   register double *cd = ABt.Data();

   for (register int j = 0; j < bh; j++)
      for (register int i = 0; i < ah; i++)
      {
         register double d = 0.0;
         register const double *ap = ad + i;
         register const double *bp = bd + j;
         for (register int k = 0; k < aw; k++)
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
            d += A(i, k) * B(j, k);
         ABt(i, j) = d;
      }
#endif
}

void AddMultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt)
{
   int i, j, k;
   double d;

#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width())
      mfem_error("AddMultABt(...)");
#endif

   for (i = 0; i < A.Height(); i++)
      for (j = 0; j < B.Height(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Width(); k++)
            d += A(i, k) * B(j, k);
         ABt(i, j) += d;
      }
}

void MultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB)
{
#ifdef MFEM_DEBUG
   if (A.Width() != AtB.Height() || B.Width() != AtB.Width() ||
       A.Height() != B.Height())
      mfem_error("MultAtB(...)");
#endif

#ifdef MFEM_USE_LAPACK
   static char transa = 'T', transb = 'N';
   static double alpha = 1.0, beta = 0.0;
   int m = A.Width(), n = B.Width(), k = A.Height();

   dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.Data(), &k,
          B.Data(), &k, &beta, AtB.Data(), &m);
#elif 1
   register const int ah = A.Height();
   register const int aw = A.Width();
   register const int bw = B.Width();
   register const double *ad = A.Data();
   register const double *bd = B.Data();
   register double *cd = AtB.Data();

   for (register int j = 0; j < bw; j++)
   {
      register const double *ap = ad;
      for (register int i = 0; i < aw; i++)
      {
         register double d = 0.0;
         for (register int k = 0; k < ah; k++)
         {
            d += *(ap++) * bd[k];
         }
         *(cd++) = d;
      }
      bd += ah;
   }
#else
   int i, j, k;
   double d;

   for (i = 0; i < A.Size(); i++)
      for (j = 0; j < B.Size(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Height(); k++)
            d += A(k, i) * B(k, j);
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
            d += A(i,k) * A(j,k);
         AAt(i, j) += (d *= a);
         AAt(j, i) += d;
      }
      d = 0.;
      for (int k = 0; k < A.Width(); k++)
         d += A(i,k) * A(i,k);
      AAt(i, i) += a * d;
   }
}

void Mult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt)
{
   for (int i = 0; i < A.Height(); i++)
      for (int j = 0; j <= i; j++)
      {
         double d = 0.;
         for (int k = 0; k < A.Width(); k++)
            d += A(i,k) * A(j,k);
         AAt(i, j) = AAt(j, i) = a * d;
      }
}

void MultVVt(const Vector &v, DenseMatrix &vvt)
{
   for (int i = 0; i < v.Size(); i++)
      for (int j = 0; j <= i; j++)
      {
         vvt(i,j) = vvt(j,i) = v(i) * v(j);
      }
}

void MultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt)
{
   int i, j;
   double vi;

#ifdef MFEM_DEBUG
   if (v.Size() != VWt.Height() || w.Size() != VWt.Size())
      mfem_error("MultVWt(...)");
#endif

   for (i = 0; i < v.Size(); i++)
   {
      vi = v(i);
      for (j = 0; j < w.Size(); j++)
         VWt(i, j) = vi * w(j);
   }
}

void AddMultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt)
{
   int m = v.Size(), n = w.Size();

#ifdef MFEM_DEBUG
   if (VWt.Height() != m || VWt.Width() != n)
      mfem_error("AddMultVWt(...)");
#endif

   for (int i = 0; i < m; i++)
   {
      double vi = v(i);
      for (int j = 0; j < n; j++)
         VWt(i, j) += vi * w(j);
   }
}

void AddMult_a_VVt(const double a, const Vector &v, DenseMatrix &VVt)
{
   int n = v.Size();

#ifdef MFEM_DEBUG
   if (VVt.Height() != n || VVt.Width() != n)
      mfem_error("AddMult_a_VVt(...)");
#endif

   for (int i = 0; i < n; i++)
   {
      double avi = a * v(i);
      for (int j = 0; j < i; j++)
      {
         double avivj = avi * v(j);
         VVt(i, j) += avivj;
         VVt(j, i) += avivj;
      }
      VVt(i, i) += avi * v(i);
   }
}


DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix &mat)
   : MatrixInverse(mat)
{
   data = new double[size*size];
#ifdef MFEM_USE_LAPACK
   ipiv = new int[size];
#endif
   Factor();
}

DenseMatrixInverse::DenseMatrixInverse(const DenseMatrix *mat)
   : MatrixInverse(*mat)
{
   data = new double[size*size];
#ifdef MFEM_USE_LAPACK
   ipiv = new int[size];
#endif
}

void DenseMatrixInverse::Factor()
{
   const double *adata = ((const DenseMatrix *)a)->data;

#ifdef MFEM_USE_LAPACK
   for (int i = 0; i < size*size; i++)
      data[i] = adata[i];

   int info;
   dgetrf_(&size, &size, data, &size, ipiv, &info);

   if (info)
      mfem_error("DenseMatrixInverse::Factor : Error in DGETRF");
#else
   // compiling without LAPACK
   int i, j, k;

   // perform LU factorization.
   for (i = 0; i < size; i++)
   {
#ifdef MFEM_DEBUG
      if (i > 0 && data[i-1+size*(i-1)] == 0.0)
         mfem_error("DenseMatrixInverse::Factor()");
#endif
      for (j = 0; j < i; j++)
      {
         data[i+size*j] = adata[i+size*j];
         for (k = 0; k < j; k++)
            data[i+size*j] -= data[i+size*k] * data[k+size*j];
         data[i+size*j] /= data[j+size*j];
      }

      for (j = i; j < size; j++)
      {
         data[i+size*j] = adata[i+size*j];
         for (k = 0; k < i; k++)
            data[i+size*j] -= data[i+size*k] * data[k+size*j];
      }
   }
#endif
}

void DenseMatrixInverse::Factor(const DenseMatrix &mat)
{
#ifdef MFEM_DEBUG
   if (mat.height != mat.size)
      mfem_error("DenseMatrixInverse::Factor #1");
   if (size != mat.size)
      mfem_error("DenseMatrixInverse::Factor #2");
#endif
   a = &mat;

   Factor();
}

void DenseMatrixInverse::Mult(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_LAPACK
   char trans = 'N';
   int  n     = size;
   int  nrhs  = 1;
   int  info;
   y = x;
   dgetrs_(&trans, &n, &nrhs, data, &n, ipiv, y.GetData(), &n, &info);

   if (info)
      mfem_error("DenseMatrixInverse::Mult #1");
#else
   // compiling without LAPACK
   int i, j;

   // y <- L^{-1} x
   for (i = 0; i < size; i++)
   {
      y(i) = x(i);
      for (j = 0; j < i; j++)
         y(i) -= data[i+size*j] * y(j);
   }

   // y <- U^{-1} y
   for (i = size-1; i >= 0; i--)
   {
      for (j = i+1; j < size; j++)
         y(i) -= data[i+size*j] * y(j);
#ifdef MFEM_DEBUG
      if ((data[i+size*i]) == 0.0)
         mfem_error("DenseMatrixInverse::Mult #2");
#endif
      y(i) /= data[i+size*i];
   }
#endif
}

DenseMatrixInverse::~DenseMatrixInverse()
{
   delete [] data;
#ifdef MFEM_USE_LAPACK
   delete [] ipiv;
#endif
}


DenseMatrixEigensystem::DenseMatrixEigensystem(DenseMatrix &m)
   : mat(m)
{
   n = m.Size();
   EVal.SetSize(n);
   EVect.SetSize(n);
   ev.SetDataAndSize(NULL, n);
   jobz = 'V';
   uplo = 'U';

#ifdef MFEM_USE_LAPACK
   lwork = -1;
   double qwork;
   dsyev_(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
          &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];
#endif
}

void DenseMatrixEigensystem::Eval()
{
#ifdef MFEM_DEBUG
   if (mat.Size() != n)
      mfem_error("DenseMatrixEigensystem::Eval()");
#endif

#ifdef MFEM_USE_LAPACK
   EVect = mat;
   dsyev_(&jobz, &uplo, &n, EVect.Data(), &n, EVal.GetData(),
          work, &lwork, &info);

   if (info != 0)
   {
      cerr << "DenseMatrixEigensystem::Eval(): DSYEV error code: "
           << info << endl;
      mfem_error();
   }
#else
   mfem_error("DenseMatrixEigensystem::Eval(): Compiled without LAPACK");
#endif
}

DenseMatrixEigensystem::~DenseMatrixEigensystem()
{
   delete [] work;
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
   work = NULL;
   mfem_error("DenseMatrixSVD::Init(): Compiled without LAPACK");
#endif
}

void DenseMatrixSVD::Eval(DenseMatrix &M)
{
#ifdef MFEM_DEBUG
   if (M.Height() != m || M.Width() != n)
      mfem_error("DenseMatrixSVD::Eval()");
#endif

#ifdef MFEM_USE_LAPACK
   dgesvd_(&jobu, &jobvt, &m, &n, M.Data(), &m, sv.GetData(), NULL, &m,
           NULL, &n, work, &lwork, &info);

   if (info)
   {
      cerr << "DenseMatrixSVD::Eval() : info = " << info << endl;
      mfem_error();
   }
#else
   mfem_error("DenseMatrixSVD::Eval(): Compiled without LAPACK");
#endif
}

DenseMatrixSVD::~DenseMatrixSVD()
{
   delete [] work;
}
