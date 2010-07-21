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


DenseMatrix::DenseMatrix () : Matrix(0)
{
   height = 0;
   data = NULL;
}

DenseMatrix::DenseMatrix (const DenseMatrix &m) : Matrix(m.size)
{
   height = m.height;
   int hw = size * height;
   data = new double[hw];
   for (int i = 0; i < hw; i++)
      data[i] = m.data[i];
}

DenseMatrix::DenseMatrix (int s) : Matrix(s)
{
   height = s;

   data = new double[s*s];

   for (int i = 0; i < s; i++)
      for (int j = 0; j < s; j++)
         (*this)(i,j) = 0;
}

DenseMatrix::DenseMatrix (int m, int n) : Matrix(n)
{
   height = m;

   data = new double[m*n];

   for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
         (*this)(i,j) = 0;
}

DenseMatrix::DenseMatrix (const DenseMatrix & mat, char ch) : Matrix(mat.height)
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

double & DenseMatrix::Elem (int i, int j)
{
   return (*this)(i,j);
}

const double & DenseMatrix::Elem (int i, int j) const
{
   return (*this)(i,j);
}

void DenseMatrix::Mult (const Vector & x, Vector & y) const
{
#ifdef MFEM_DEBUG
   if ( size != x.Size() || height != y.Size() )
      mfem_error ("DenseMatrix::Mult");
#endif

   for (int i = 0; i < height; i++)
   {
      double a = 0.;
      for (int j = 0; j < size; j++)
         a += (*this)(i,j) * x(j);
      y(i) = a;
   }
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

void DenseMatrix::MultTranspose (const Vector & x, Vector & y) const
{
#ifdef MFEM_DEBUG
   if ( height != x.Size() || size != y.Size() )
      mfem_error ("DenseMatrix::MultTranspose");
#endif

   for (int i = 0; i < size; i++)
   {
      double d = 0.0;
      for (int j = 0; j < height; j++)
         d += (*this)(j,i) * x(j);
      y(i) = d;
   }
}

void DenseMatrix::AddMult (const Vector & x, Vector & y) const
{
#ifdef MFEM_DEBUG
   if ( size != x.Size() || height != y.Size() )
      mfem_error ("DenseMatrix::AddMult");
#endif

   for (int i = 0; i < height; i++)
      for (int j = 0; j < size; j++)
         y(i) += (*this)(i,j) * x(j);
}

double DenseMatrix::InnerProduct (const Vector &x, const Vector &y) const
{
   double prod = 0.0;

   for (int i = 0; i < height; i++)
   {
      double Axi = 0.0;
      for (int j = 0; j < size; j++)
         Axi += (*this)(i,j) * x(j);
      prod += y(i) * Axi;
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
      mfem_error ("DenseMatrix::Det");
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
   mfem_error ("DenseMatrix::Weight()");
   return 0.0;
}

void DenseMatrix::Add(const double c, DenseMatrix & A)
{
   for(int i=0;i<Height();i++)
      for(int j=0;j<Size();j++)
         (*this)(i,j) += c * A(i,j);
}

DenseMatrix &DenseMatrix::operator=(double c)
{
   int s=Size()*Height();
   if (data != NULL)
      for(int i = 0; i < s; i++)
         data[i] = c;
   return *this;
}

DenseMatrix &DenseMatrix::operator=(const DenseMatrix &m)
{
   int i, hw;

   SetSize (m.height, m.size);

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
   int s=Size()*Height();
   if (data != NULL)
      for(int i = 0; i < s; i++)
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
      mfem_error ("DenseMatrix::Invert()");
#endif

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[size];
   int    lwork = -1;
   double qwork, *work;
   int    info;

   dgetrf_ (&size, &size, data, &size, ipiv, &info);

   if (info)
      mfem_error ("DenseMatrix::Invert() : Error in DGETRF");

   dgetri_(&size, data, &size, ipiv, &qwork, &lwork, &info);

   lwork = (int) qwork;
   work = new double[lwork];

   dgetri_(&size, data, &size, ipiv, work, &lwork, &info);

   if (info)
      mfem_error ("DenseMatrix::Invert() : Error in DGETRI");

   delete [] work;
   delete [] ipiv;
#else
   int c, i, j, n = Size();
   double a, b;

   for (c = 0; c < n; c++)
   {
#ifdef MFEM_DEBUG
      if ((*this)(c, c) == 0.0)
         mfem_error ("DenseMatrix::Invert() : division by zero");
#endif
      a = (*this)(c, c) = 1.0 / (*this)(c, c);
      for (j = 0; j < c; j++)
         (*this)(c, j) *= a;
      for (j = c+1; j < n; j++)
         (*this)(c, j) *= a;
      for (i = 0; i < c; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
            (*this)(i, j) += b * (*this)(c, j);
         for (j = c+1; j < n; j++)
            (*this)(i, j) += b * (*this)(c, j);
      }
      for (i = c+1; i < n; i++)
      {
         (*this)(i, c) = a * (b = -(*this)(i, c));
         for (j = 0; j < c; j++)
            (*this)(i, j) += b * (*this)(c, j);
         for (j = c+1; j < n; j++)
            (*this)(i, j) += b * (*this)(c, j);
      }
   }
#endif
}


void DenseMatrix::Norm2 (double * v)
{
   for (int j = 0; j < Size(); j++) {
      v[j] = 0.0;
      for (int i = 0; i < Height(); i++)
         v[j] += (*this)(i,j)*(*this)(i,j);
      v[j] = sqrt(v[j]);
   }
}

double DenseMatrix::FNorm() const
{
   int i, hw = Height() * Size();
   double a = 0.0;

   for (i = 0; i < hw; i++)
      a += data[i] * data[i];

   return sqrt(a);
}


#ifdef MFEM_USE_LAPACK
extern "C" void
dsyevr_ ( char *JOBZ, char *RANGE, char *UPLO, int *N, double *A,
          int *LDA, double *VL, double *VU, int *IL, int *IU,
          double *ABSTOL, int *M, double *W, double *Z, int *LDZ,
          int *ISUPPZ, double *WORK, int *LWORK, int *IWORK,
          int *LIWORK, int *INFO);
extern "C" void
dsyev_ ( char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W,
         double *WORK, int *LWORK, int *INFO );
extern "C" void
dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
        double *S, double *U, int *LDU, double *VT, int *LDVT,
        double *WORK, int *LWORK, int *INFO);
#endif

void dsyevr_Eigensystem (DenseMatrix &a,
                         Vector &ev, DenseMatrix *evect)
{

#ifdef MFEM_USE_LAPACK

   ev.SetSize (a.Size());

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
   double   *WORK     = new double[26*N];
   int       LWORK    = 26*N;
   int      *IWORK    = new int[10*N];
   int       LIWORK   = 10*N;
   int       INFO;

   if (evect) // Compute eigenvectors too
   {
      evect -> SetSize (N);

      JOBZ     = 'V';
      Z        = evect -> Data();
      LDZ      = N;
   }

   int hw = a.Height() * a.Width();
   double *data = a.Data();

   for (int i = 0; i < hw; i++)
      A[i] = data[i];


   dsyevr_( &JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
            &ABSTOL, &M, W, Z, &LDZ, ISUPPZ, WORK, &LWORK,
            IWORK, &LIWORK, &INFO );

   if (INFO != 0)
   {
      cerr << "dsyevr_Eigensystem (...): DSYEVR error code: "
           << INFO << endl;
      mfem_error();
   }

#ifdef MFEM_DEBUG
   if (M < N)
   {
      cerr << "dsyevr_Eigensystem (...):\n"
           << " DSYEVR did not find all eigenvalues "
           << M << "/" << N << endl;
      mfem_error();
   }
   for (IL = 0; IL < N; IL++)
      if (!finite(W[IL]))
         mfem_error ("dsyevr_Eigensystem (...): !finite value in W");
   for (IL = 0; IL < N*N; IL++)
      if (!finite(Z[IL]))
         mfem_error ("dsyevr_Eigensystem (...): !finite value in Z");
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
            cerr << "dsyevr_Eigensystem (...):"
                 << " Z^t Z - I deviation = " << VU
                 << "\n W[max] = " << W[N-1] << ", W[min] = "
                 << W[0] << ", N = " << N << endl;
            mfem_error();
         }
      }
   if (VU > 1e-9)
   {
      cerr << "dsyevr_Eigensystem (...):"
           << " Z^t Z - I deviation = " << VU
           << "\n W[max] = " << W[N-1] << ", W[min] = "
           << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
      mfem_error ("dsyevr_Eigensystem (...): ERROR: ...");
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
      cerr << "dsyevr_Eigensystem (...):"
           << " max matrix deviation = " << VU
           << "\n W[max] = " << W[N-1] << ", W[min] = "
           << W[0] << ", N = " << N << endl;
   }
   if (VU > 1e-5)
      mfem_error ("dsyevr_Eigensystem (...): ERROR: ...");
#endif

   delete [] IWORK;
   delete [] WORK;
   delete [] ISUPPZ;
   delete [] A;

#endif
}

void dsyev_Eigensystem (DenseMatrix &a,
                        Vector &ev, DenseMatrix *evect)
{

#ifdef MFEM_USE_LAPACK

   int   N      = a.Size();
   char  JOBZ   = 'N';
   char  UPLO   = 'U';
   int   LDA    = N;
   int   LWORK  = 3*N; /* max(1,3*N-1) */
   int   INFO;

   ev.SetSize (N);

   double *A    = NULL;
   double *W    = ev.GetData();
   double *WORK = NULL;

   if (evect)
   {
      JOBZ = 'V';
      evect -> SetSize (N);
      A = evect -> Data();
   }
   else
   {
      A = new double[N*N];
   }

   WORK = new double[3*N];

   int hw = a.Height() * a.Width();
   double *data = a.Data();
   for (int i = 0; i < hw; i++)
      A[i] = data[i];

   dsyev_ ( &JOBZ, &UPLO, &N, A, &LDA, W,
            WORK, &LWORK, &INFO );

   if (INFO != 0)
   {
      cerr << "dsyev_Eigensystem: DSYEV error code: " << INFO << endl;
      mfem_error();
   }

   delete [] WORK;
   if (evect == NULL)  delete [] A;

#endif
}

void DenseMatrix::Eigensystem (Vector &ev, DenseMatrix *evect)
{
#ifdef MFEM_USE_LAPACK

   // dsyevr_Eigensystem (*this, ev, evect);

   dsyev_Eigensystem (*this, ev, evect);

#else

   mfem_error ("DenseMatrix::Eigensystem");

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

void DenseMatrix::GetColumn (int c, Vector &col)
{
   int n;
   double *cp, *vp;

   n = Height();
   col.SetSize (n);
   cp = data + c * n;
   vp = col.GetData();

   for (int i = 0; i < n; i++)
      vp[i] = cp[i];
}

void DenseMatrix::Diag (double c, int n)
{
   SetSize(n);

   int i, N = n*n;
   for(i = 0; i < N; i++)
      data[i] = 0.0;
   for(i = 0; i < n; i++)
      data[i*(n+1)] = c;
}


void DenseMatrix::Diag (double * diag, int n)
{
   SetSize(n);

   int i, N = n*n;
   for(i = 0; i < N; i++)
      data[i] = 0.0;
   for(i = 0; i < n; i++)
      data[i*(n+1)] = diag[i];
}

void DenseMatrix::Transpose ()
{
   int i, j;
   double t;

   if (Size() == Height()) {
      for (i = 0; i < Height(); i++)
         for (j = i+1; j < Size(); j++) {
            t = (*this)(i,j);
            (*this)(i,j) = (*this)(j,i);
            (*this)(j,i) = t;
         }
   } else {
      DenseMatrix T(*this,'t');
      (*this) = T;
   }
}

void DenseMatrix::Transpose (DenseMatrix & A)
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

void DenseMatrix::GradToCurl (DenseMatrix &curl)
{
   int n = Height();

#ifdef MFEM_DEBUG
   if (Width() != 3 || curl.Width() != 3 || 3*n != curl.Height())
      mfem_error ("DenseMatrix::GradToCurl (...)");
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

void DenseMatrix::GradToDiv (Vector &div)
{

#ifdef MFEM_DEBUG
   if (Width()*Height() != div.Size())
      mfem_error ("DenseMatrix::GradToDiv (...)");
#endif

   // div(dof*j+i) <-- (*this)(i,j)

   int n = size * height;
   double *ddata = div.GetData();

   for (int i = 0; i < n; i++)
      ddata[i] = data[i];
}

void DenseMatrix::CopyRows (DenseMatrix & A, int row1, int row2)
{
   SetSize (row2 - row1 + 1, A.Size());

   for (int i = row1; i <= row2; i++)
      for (int j = 0; j < Size(); j++)
         (*this)(i-row1,j) = A(i,j);
}

void DenseMatrix::CopyCols (DenseMatrix & A, int col1, int col2)
{
   SetSize (A.Height(), col2 - col1 + 1);

   for (int i = 0; i < Height(); i++)
      for (int j = col1; j <= col2; j++)
         (*this)(i,j-col1) = A(i,j);
}

void DenseMatrix::CopyMN (DenseMatrix & A, int m, int n, int Aro, int Aco)
{
   int i,j;

   SetSize(m,n);

   for (j = 0; j < n; j++)
      for (i = 0; i < m; i++)
         (*this)(i,j) = A(Aro+i,Aco+j);
}

void DenseMatrix::CopyMN (DenseMatrix & A,
                          int row_offset, int col_offset)
{
   int i,j;
   double * v = A.data;

   for (j = 0; j < A.Size(); j++)
      for (i = 0; i < A.Height(); i++)
         (*this)(row_offset+i,col_offset+j) = *(v++);
}

void DenseMatrix::CopyMNt (DenseMatrix & A,
                           int row_offset, int col_offset)
{
   int i,j;
   double * v = A.data;

   for (i = 0; i < A.Size(); i++)
      for (j = 0; j < A.Height(); j++)
         (*this)(row_offset+i,col_offset+j) = *(v++);
}

void DenseMatrix::CopyMNDiag (double c, int n,
                              int row_offset, int col_offset)
{
   int i,j;

   for (i = 0; i < n; i++)
      for (j = i+1; j < n; j++)
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;

   for (i = 0; i < n; i++)
      (*this)(row_offset+i,col_offset+i) = c;
}

void DenseMatrix::CopyMNDiag (double * diag, int n,
                              int row_offset, int col_offset)
{
   int i,j;

   for (i = 0; i < n; i++)
      for (j = i+1; j < n; j++)
         (*this)(row_offset+i,col_offset+j) =
            (*this)(row_offset+j,col_offset+i) = 0.0;

   for (i = 0; i < n; i++)
      (*this)(row_offset+i,col_offset+i) = diag[i];
}

void DenseMatrix::AddMatrix (DenseMatrix &A, int ro, int co)
{
   int h, ah, aw;
   double *p, *ap;

   h  = Height();
   ah = A.Height();
   aw = A.Width();

#ifdef MFEM_DEBUG
   if (co+aw > Width() || ro+ah > h)
      mfem_error ("DenseMatrix::AddMatrix (...) 1");
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

void DenseMatrix::AddMatrix (double a, DenseMatrix &A,
                             int ro, int co)
{
   int h, ah, aw;
   double *p, *ap;

   h  = Height();
   ah = A.Height();
   aw = A.Width();

#ifdef MFEM_DEBUG
   if (co+aw > Width() || ro+ah > h)
      mfem_error ("DenseMatrix::AddMatrix (...) 2");
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

void DenseMatrix::AddToVector (int offset, Vector &v) const
{
   int i, n = size * height;
   double *vdata = v.GetData() + offset;

   for (i = 0; i < n; i++)
      vdata[i] += data[i];
}

void DenseMatrix::GetFromVector (int offset, const Vector &v)
{
   int i, n = size * height;
   const double *vdata = v.GetData() + offset;

   for (i = 0; i < n; i++)
      data[i] = vdata[i];
}

void DenseMatrix::AdjustDofDirection (Array<int> &dofs)
{
   int n = Height();

#ifdef MFEM_DEBUG
   if (dofs.Size() != n || Width() != n)
      mfem_error ("DenseMatrix::AdjustDofDirection (...)");
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


void DenseMatrix::Print (ostream & out, int width) const
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

void DenseMatrix::PrintT (ostream & out, int width) const
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

DenseMatrix::~DenseMatrix()
{
   if (data != NULL)
      delete [] data;
}



void Add (const DenseMatrix & A, const DenseMatrix & B,
          double alpha, DenseMatrix & C)
{
   for (int i = 0; i < C.Height(); i++)
      for (int j = 0; j < C.Size(); j++)
         C(i,j) = A(i,j) + alpha * B(i,j);
}


void Mult (const DenseMatrix & b,
           const DenseMatrix & c,
           DenseMatrix & a)
{
#ifdef MFEM_DEBUG
   if ( a.height != b.height || a.size != c.size || b.size != c.height )
      mfem_error ("Mult (product of DenseMatrices)");
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

void CalcAdjugate (const DenseMatrix & a, DenseMatrix & adja)
{
#ifdef MFEM_DEBUG
   if (a.Height() != a.Size() || adja.Height() != adja.Size() ||
       a.Size() != adja.Size() || a.Size() < 2 || a.Size() > 3)
      mfem_error ("DenseMatrix::CalcAdjugate (...)");
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

void CalcAdjugateTranspose (const DenseMatrix & a, DenseMatrix & adjat)
{
#ifdef MFEM_DEBUG
   if (a.Height() != a.Size() || adjat.Height() != adjat.Size() ||
       a.Size() != adjat.Size() || a.Size() < 2 || a.Size() > 3)
      mfem_error ("DenseMatrix::CalcAdjugateTranspose (...)");
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

void CalcInverse(const DenseMatrix & a, DenseMatrix & inva)
{
#ifdef MFEM_DEBUG
   if ( (a.Size() != a.Height()) || ( (a.Height()!= 1) && (a.Height()!= 2)
                                      && (a.Height()!= 3) ) )
      mfem_error ("DenseMatrix::CalcInverse(...)");
#endif

   double t;
#ifdef MFEM_DEBUG
   t = a.Det();
   if (fabs(t) < 1.0e-12 * a.FNorm())
      cerr << "DenseMatrix::CalcInverse(...) : singular matrix!"
           << endl;
   t = 1. / t;
#else
   t = 1.0 / a.Det();
#endif

   switch (a.Height()) {
   case 1:
      inva(0,0) = 1.0 / a(0,0);
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

void CalcInverseTranspose(const DenseMatrix & a, DenseMatrix & inva)
{
#ifdef MFEM_DEBUG
   if ( (a.Size() != a.Height()) || ( (a.Height()!= 1) && (a.Height()!= 2)
                                      && (a.Height()!= 3) ) )
      mfem_error ("DenseMatrix::CalcInverse(...)");
#endif

   double t = 1. / a.Det() ;

   switch (a.Height()) {
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

void MultAAt(DenseMatrix & a, DenseMatrix & aat)
{
   for(int i=0;i<a.Height();i++)
      for(int j=0;j<=i;j++)
      {
         double temp = 0. ;
         for(int k=0;k<a.Size();k++)
            temp += a(i,k) * a(j,k) ;
         aat(j,i) = aat(i,j) = temp;
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

void MultABt(const DenseMatrix &A, const DenseMatrix &B,
             DenseMatrix &ABt)
{
   int i, j, k;
   double d;

#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width() ||
       A.Width() != B.Width())
      mfem_error ("MultABt (...)");
#endif

   for (i = 0; i < A.Height(); i++)
      for (j = 0; j < B.Height(); j++)
      {
         d = 0.0;
         for (k = 0; k < A.Width(); k++)
            d += A(i, k) * B(j, k);
         ABt(i, j) = d;
      }
}

void AddMultABt ( DenseMatrix & A, DenseMatrix & B,
                  DenseMatrix & ABt )
{
   int i, j, k;
   double d;

#ifdef MFEM_DEBUG
   if (A.Height() != ABt.Height() || B.Height() != ABt.Width())
      mfem_error ("AddMultABt (...)");
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

void MultAtB(const DenseMatrix &A, const DenseMatrix &B,
             DenseMatrix &AtB)
{
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
}

void AddMult_a_AAt (double a, DenseMatrix &A, DenseMatrix &AAt)
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

void Mult_a_AAt (double a, DenseMatrix &A, DenseMatrix &AAt)
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
   for(int i = 0; i < v.Size(); i++)
      for(int j = 0; j <= i; j++)
      {
         vvt(i,j) = vvt(j,i) = v(i) * v(j);
      }
}

void MultVWt(Vector &v, Vector &w, DenseMatrix &VWt)
{
   int i, j;
   double vi;

#ifdef MFEM_DEBUG
   if (v.Size() != VWt.Height() || w.Size() != VWt.Size())
      mfem_error ("MultVWt (...)");
#endif

   for (i = 0; i < v.Size(); i++)
   {
      vi = v(i);
      for (j = 0; j < w.Size(); j++)
         VWt(i, j) = vi * w(j);
   }
}

void AddMultVWt(Vector &v, Vector &w, DenseMatrix &VWt)
{
   int m = v.Size(), n = w.Size();

#ifdef MFEM_DEBUG
   if (VWt.Height() != m || VWt.Width() != n)
      mfem_error ("AddMultVWt (...)");
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
      mfem_error ("AddMult_a_VVt (...)");
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
   dgetrf_ (&size, &size, data, &size, ipiv, &info);

   if (info)
      mfem_error ("DenseMatrixInverse::Factor : Error in DGETRF");
#else
   // compiling without LAPACK
   int i, j, k;

   // perform LU factorization.
   for (i = 0; i < size; i++)
   {
#ifdef MFEM_DEBUG
      if (i > 0 && data[i-1+size*(i-1)] == 0.0)
         mfem_error ("DenseMatrixInverse::Factor()");
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
   dgetrs_ (&trans, &n, &nrhs, data, &n, ipiv,
            y.GetData(), &n, &info);

   if (info)
      mfem_error ("DenseMatrixInverse::Mult #1");
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
         mfem_error ("DenseMatrixInverse::Mult #2");
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
   lwork = 3*n;
   work = new double[lwork];
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
