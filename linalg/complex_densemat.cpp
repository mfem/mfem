// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "complex_densemat.hpp"
#include <complex>

#ifdef MFEM_USE_LAPACK
extern "C" void
zgetrf_(int *, int *, std::complex<double> *, int *, int *, int *);
extern "C" void
zgetrs_(char *, int *, int *, std::complex<double> *, int *, int *,
        std::complex<double> *, int *, int *);
extern "C" void
zgetri_(int *, std::complex<double> *, int *, int *,
        std::complex<double> *, int *, int *);
extern "C" void
ztrsm_(char *, char *, char *, char *, int *, int *, std::complex<double> *,
       std::complex<double> *, int *, std::complex<double> *, int *);
extern "C" void
zpotrf_(char *, int *, std::complex<double> *, int *, int *);

extern "C" void
ztrtrs_(char *, char*, char *, int *, int *, std::complex<double> *, int *,
        std::complex<double> *, int *, int *);
extern "C" void
zpotri_(char *, int *, std::complex<double> *, int*, int *);

extern "C" void
zpotrs_(char *, int *, int *, std::complex<double> *, int *,
        std::complex<double> *, int *, int *);
#endif

namespace mfem
{

DenseMatrix & ComplexDenseMatrix::real()
{
   MFEM_ASSERT(Op_Real_, "ComplexDenseMatrix has no real part!");
   return dynamic_cast<DenseMatrix &>(*Op_Real_);
}

DenseMatrix & ComplexDenseMatrix::imag()
{
   MFEM_ASSERT(Op_Imag_, "ComplexDenseMatrix has no imaginary part!");
   return dynamic_cast<DenseMatrix &>(*Op_Imag_);
}

const DenseMatrix & ComplexDenseMatrix::real() const
{
   MFEM_ASSERT(Op_Real_, "ComplexDenseMatrix has no real part!");
   return dynamic_cast<const DenseMatrix &>(*Op_Real_);
}

const DenseMatrix & ComplexDenseMatrix::imag() const
{
   MFEM_ASSERT(Op_Imag_, "ComplexDenseMatrix has no imaginary part!");
   return dynamic_cast<const DenseMatrix &>(*Op_Imag_);
}

DenseMatrix * ComplexDenseMatrix::GetSystemMatrix() const
{
   int h = height/2;
   int w = width/2;
   DenseMatrix * A = new DenseMatrix(2*h,2*w);
   double * data = A->Data();
   double * data_r = nullptr;
   double * data_i = nullptr;

   const double factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;
   *A = 0.;
   if (hasRealPart())
   {
      data_r = real().Data();
      for (int j = 0; j<w; j++)
      {
         for (int i = 0; i<h; i++)
         {
            data[i+j*height] = data_r[i+j*h];
            data[i+h+(j+h)*height] = factor*data_r[i+j*h];
         }
      }
   }
   if (hasImagPart())
   {
      data_i = imag().Data();
      for (int j = 0; j<w; j++)
      {
         for (int i = 0; i<h; i++)
         {
            data[i+h+j*height] = factor*data_i[i+j*h];
            data[i+(j+h)*height] = -data_i[i+j*h];
         }
      }
   }
   return A;
}

ComplexDenseMatrix * ComplexDenseMatrix::ComputeInverse()
{
   MFEM_VERIFY(height == width, "Matrix has to be square");

   // complex data
   int h = height/2;
   int w = width/2;
   std::complex<double> * data = new std::complex<double>[h*w];

   // copy data
   if (hasRealPart() && hasImagPart())
   {
      double * data_r = real().Data();
      double * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<double> (data_r[i], data_i[i]);
      }
   }
   else if (hasRealPart())
   {
      double * data_r = real().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<double> (data_r[i], 0.);
      }
   }
   else if (hasImagPart())
   {
      double * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<double> (0., data_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexDenseMatrix has neither only a real nor an imag part");
   }

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[w];
   int    lwork = -1;
   std::complex<double> qwork, *work;
   int    info;

   zgetrf_(&w, &w, data, &w, ipiv, &info);
   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRF");
   }

   zgetri_(&w, data, &w, ipiv, &qwork, &lwork, &info);

   lwork = (int) qwork.real();
   work = new std::complex<double>[lwork];

   zgetri_(&w, data, &w, ipiv, work, &lwork, &info);

   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRI");
   }

   delete [] work;
   delete [] ipiv;

#else
   // compiling without LAPACK
   int c, i, j, n = w;
   double a, b;
   Array<int> piv(n);
   std::complex<double> ac,bc;

   for (c = 0; c < n; c++)
   {
      a = std::abs(data[c+c*h]);
      i = c;
      for (j = c + 1; j < n; j++)
      {
         b = std::abs(data[j+c*h]);
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
         mfem::Swap<std::complex<double>>(data[c+j*h], data[i+j*h]);
      }

      ac = data[c+c*h] = 1.0 / data[c+c*h];
      for (j = 0; j < c; j++)
      {
         data[c+j*h] *= ac;
      }
      for (j++; j < n; j++)
      {
         data[c+j*h] *= ac;
      }
      for (i = 0; i < c; i++)
      {
         data[i+c*h] = ac * (bc = -data[i+c*h]);
         for (j = 0; j < c; j++)
         {
            data[i+j*h] += bc * data[c+j*h];
         }
         for (j++; j < n; j++)
         {
            data[i+j*h] += bc * data[c+j*h];
         }
      }
      for (i++; i < n; i++)
      {
         data[i+c*h] = ac * (bc = -data[i+c*h]);
         for (j = 0; j < c; j++)
         {
            data[i+j*h] += bc * data[c+j*h];
         }
         for (j++; j < n; j++)
         {
            data[i+j*h] += bc * data[c+j*h];
         }
      }
   }

   for (c = n - 1; c >= 0; c--)
   {
      j = piv[c];
      for (i = 0; i < n; i++)
      {
         mfem::Swap<std::complex<double>>(data[i+c*h], data[i+j*h]);
      }
   }

#endif

   DenseMatrix * C_r = new DenseMatrix(h);
   DenseMatrix * C_i = new DenseMatrix(h);

   double * datac_r = C_r->Data();
   double * datac_i = C_i->Data();

   for (int k = 0; k < h*w; k++)
   {
      datac_r[k] = data[k].real();
      datac_i[k] = data[k].imag();
   }
   delete [] data;
   return new ComplexDenseMatrix(C_r,C_i,true,true);

}

ComplexDenseMatrix * Mult(const ComplexDenseMatrix &A,
                          const ComplexDenseMatrix &B)
{
   // C = C_r + i C_i = (A_r + i * A_i) * (B_r + i * B_i)
   //                 = A_r * B_r - A_i B_i + i (A_r * B_i + A_i * B_r)

   int h = A.Height()/2;
   int w = B.Width()/2;

   MFEM_VERIFY(A.Width() == B.Height(), "Incompatible matrix dimensions");

   //only real case (imag is null)
   DenseMatrix * C_r = nullptr;
   DenseMatrix * C_i = nullptr;
   if ((A.hasRealPart() && B.hasRealPart()) ||
       (A.hasImagPart() && B.hasImagPart()))
   {
      C_r = new DenseMatrix(h,w);
   }
   if ((A.hasRealPart() && B.hasImagPart()) ||
       (A.hasImagPart() && B.hasRealPart()))
   {
      C_i = new DenseMatrix(h,w);
   }

   MFEM_VERIFY(C_r || C_i, "Both real and imag parts are null");

   if (A.hasRealPart() && B.hasRealPart())
   {
      Mult(A.real(), B.real(),*C_r);
   }
   if (A.hasImagPart() && B.hasImagPart())
   {
      if (A.hasRealPart() && B.hasRealPart())
      {
         AddMult_a(-1.,A.imag(), B.imag(),*C_r);
      }
      else
      {
         Mult(A.imag(), B.imag(),*C_r);
      }
   }

   if (A.hasRealPart() && B.hasImagPart())
   {
      Mult(A.real(), B.imag(),*C_i);
   }

   if (A.hasImagPart() && B.hasRealPart())
   {
      if (A.hasRealPart() && B.hasImagPart())
      {
         AddMult(A.imag(), B.real(),*C_i);
      }
      else
      {
         Mult(A.imag(), B.real(),*C_i);
      }
   }

   return new ComplexDenseMatrix(C_r,C_i,true,true);
}

ComplexDenseMatrix * MultAtB(const ComplexDenseMatrix &A,
                             const ComplexDenseMatrix &B)
{
   // C = C_r + i C_i = (A_r^t - i * A_i^t) * (B_r + i * B_i)
   //                     = A_r^t * B_r + A_i^t * B_i + i (A_r^t * B_i - A_i^t * B_r)

   int h = A.Width()/2;
   int w = B.Width()/2;

   MFEM_VERIFY(A.Height() == B.Height(), "Incompatible matrix dimensions");

   //only real case (imag is null)
   DenseMatrix * C_r = nullptr;
   DenseMatrix * C_i = nullptr;
   if ((A.hasRealPart() && B.hasRealPart()) ||
       (A.hasImagPart() && B.hasImagPart()))
   {
      C_r = new DenseMatrix(h,w);
   }
   if ((A.hasRealPart() && B.hasImagPart()) ||
       (A.hasImagPart() && B.hasRealPart()))
   {
      C_i = new DenseMatrix(h,w);
   }

   MFEM_VERIFY(C_r || C_i, "Both real and imag parts are null");

   if (A.hasRealPart() && B.hasRealPart())
   {
      MultAtB(A.real(), B.real(),*C_r);
   }
   if (A.hasImagPart() && B.hasImagPart())
   {
      if (A.hasRealPart() && B.hasRealPart())
      {
         DenseMatrix tempC_r(h,w);
         MultAtB(A.imag(), B.imag(),tempC_r);
         (*C_r) += tempC_r;
      }
      else
      {
         MultAtB(A.imag(), B.imag(),*C_r);
      }
   }

   if (A.hasRealPart() && B.hasImagPart())
   {
      MultAtB(A.real(), B.imag(),*C_i);
   }

   if (A.hasImagPart() && B.hasRealPart())
   {
      if (A.hasRealPart() && B.hasImagPart())
      {
         DenseMatrix tempC_i(h,w);
         MultAtB(A.imag(), B.real(),tempC_i);
         (*C_i) -= tempC_i;
      }
      else
      {
         MultAtB(A.imag(), B.real(),*C_i);
      }
   }

   return new ComplexDenseMatrix(C_r,C_i,true,true);
}

std::complex<double> * ComplexFactors::RealToComplex
(int m, const double * x_r, const double * x_i) const
{
   std::complex<double> * x = new std::complex<double>[m];
   if (x_r && x_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<double>(x_r[i], x_i[i]);
      }
   }
   else if (data_r)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<double>(x_r[i], 0.);
      }
   }
   else if (data_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<double>(0., x_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexFactors::RealToComplex:both real and imag part are null");
      return nullptr;
   }
   return x;
}

void ComplexFactors::ComplexToReal(int m, const std::complex<double> * x,
                                   double * x_r, double * x_i) const
{
   for (int i = 0; i<m; i++)
   {
      x_r[i] = x[i].real();
      x_i[i] = x[i].imag();
   }
}


void ComplexFactors::SetComplexData(int m)
{
   if (data) { return; }
   MFEM_VERIFY(data_r || data_i, "ComplexFactors data not set");
   data = RealToComplex(m,data_r,data_i);
}


bool ComplexLUFactors::Factor(int m, double TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   MFEM_VERIFY(data, "Matrix data not set");
   if (m) { zgetrf_(&m, &m, data, &m, ipiv, &info); }
   return info == 0;
#else
   // compiling without LAPACK
   std::complex<double> *data_ptr = this->data;
   for (int i = 0; i < m; i++)
   {
      // pivoting
      {
         int piv = i;
         double a = std::abs(data_ptr[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const double b = std::abs(data_ptr[j+i*m]);
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
               mfem::Swap<std::complex<double>>(data_ptr[i+j*m], data_ptr[piv+j*m]);
            }
         }
      }

      if (abs(data_ptr[i + i*m]) <= TOL)
      {
         return false; // failed
      }

      const std::complex<double> a_ii_inv = 1.0 / data_ptr[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         data_ptr[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const std::complex<double> a_ik = data_ptr[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            data_ptr[j+k*m] -= a_ik * data_ptr[j+i*m];
         }
      }
   }
#endif

   return true; // success
}

std::complex<double> ComplexLUFactors::Det(int m) const
{
   std::complex<double> det(1.0,0.);
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

void ComplexLUFactors::Mult(int m, int n, double *X_r, double * X_i) const
{
   std::complex<double> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   for (int k = 0; k < n; k++)
   {
      // X <- U X
      for (int i = 0; i < m; i++)
      {
         std::complex<double> x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- L X
      for (int i = m-1; i >= 0; i--)
      {
         std::complex<double> x_i = x[i];
         for (int j = 0; j < i; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- P^{-1} X
      for (int i = m-1; i >= 0; i--)
      {
         mfem::Swap<std::complex<double>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
}

void ComplexLUFactors::LSolve(int m, int n, double *X_r, double * X_i) const
{
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         mfem::Swap<std::complex<double>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<double> x_j = x[j];
         for (int i = j+1; i < m; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexLUFactors::USolve(int m, int n, double *X_r, double * X_i) const
{
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
   // X <- U^{-1} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<double> x_j = ( x[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexLUFactors::Solve(int m, int n, double *X_r, double * X_i) const
{
#ifdef MFEM_USE_LAPACK
   std::complex<double> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   char trans = 'N';
   int  info = 0;
   if (m > 0 && n > 0) { zgetrs_(&trans, &m, &n, data, &m, ipiv, x, &m, &info); }
   MFEM_VERIFY(!info, "LAPACK: error in ZGETRS");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
#else
   // compiling without LAPACK
   LSolve(m, n, X_r, X_i);
   USolve(m, n, X_r, X_i);
#endif
}

void ComplexLUFactors::RightSolve(int m, int n, double *X_r, double * X_i) const
{
   std::complex<double> * X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> * x = X;
#ifdef MFEM_USE_LAPACK
   char n_ch = 'N', side = 'R', u_ch = 'U', l_ch = 'L';
   if (m > 0 && n > 0)
   {
      std::complex<double> alpha(1.0,0.0);
      ztrsm_(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      ztrsm_(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
   }
#else
   // compiling without LAPACK
   // X <- X U^{-1}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<double> x_j = ( x[j*n] /= data[j+j*m]);
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
         const std::complex<double> x_j = x[j*n];
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
         mfem::Swap<std::complex<double>>(x[i*n], x[(ipiv[i]-ipiv_base)*n]);
      }
      ++x;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexLUFactors::GetInverseMatrix(int m, double *X_r, double *X_i) const
{
   // A^{-1} = U^{-1} L^{-1} P
   // X <- U^{-1} (set only the upper triangular part of X)
   std::complex<double> * X = ComplexFactors::RealToComplex(m*m,X_r,X_i);
   std::complex<double> * x = X;
   for (int k = 0; k < m; k++)
   {
      const std::complex<double> minus_x_k = -( x[k] = 1.0/data[k+k*m] );
      for (int i = 0; i < k; i++)
      {
         x[i] = data[i+k*m] * minus_x_k;
      }
      for (int j = k-1; j >= 0; j--)
      {
         const std::complex<double> x_j = ( x[j] /= data[j+j*m] );
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
         const std::complex<double> minus_L_kj = -data[k+j*m];
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
         const std::complex<double> L_kj = data[k+j*m];
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
            Swap<std::complex<double>>(X[i+k*m], X[i+piv_k*m]);
         }
      }
   }
   ComplexFactors::ComplexToReal(m*m,X,X_r,X_i);
   delete [] X;
}


bool ComplexCholeskyFactors::Factor(int m, double TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   char uplo = 'L';
   MFEM_VERIFY(data, "Matrix data not set");
   if (m) {zpotrf_(&uplo, &m, data, &m, &info);}
   return info == 0;
#else
   // Cholesky–Crout algorithm
   for (int j = 0; j<m; j++)
   {
      std::complex<double> a(0.,0.);
      for (int k = 0; k<j; k++)
      {
         a+=data[j+k*m]*std::conj(data[j+k*m]);
      }

      MFEM_VERIFY((data[j+j*m] - a).real() > 0.,
                  "CholeskyFactors::Factor: The matrix is not SPD");

      data[j+j*m] = std::sqrt((data[j+j*m] - a).real());

      if (data[j + j*m].real() <= TOL) { return false; }

      for (int i = j+1; i<m; i++)
      {
         a = std::complex<double>(0.,0.);
         for (int k = 0; k<j; k++)
         {
            a+= data[i+k*m]*std::conj(data[j+k*m]);
         }
         data[i+j*m] = 1./data[j+m*j]*(data[i+j*m] - a);
      }
   }
   return true; // success
#endif
}

std::complex<double> ComplexCholeskyFactors::Det(int m) const
{
   std::complex<double> det(1.0,0.0);
   for (int i=0; i<m; i++)
   {
      det *=  data[i + i*m];
   }
   return det;
}

void ComplexCholeskyFactors::LMult(int m, int n, double * X_r,
                                   double * X_i) const
{
   // X <- L X
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         std::complex<double> x_j = x[j] * data[j+j*m];
         for (int i = 0; i < j; i++)
         {
            x_j += x[i] * data[j+i*m];
         }
         x[j] = x_j;
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexCholeskyFactors::UMult(int m, int n, double * X_r,
                                   double * X_i) const
{
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int i = 0; i < m; i++)
      {
         std::complex<double> x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * std::conj(data[j+i*m]);
         }
         x[i] = x_i;
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexCholeskyFactors::LSolve(int m, int n, double * X_r,
                                    double * X_i) const
{
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   char trans = 'N';
   char diag = 'N';
   int info = 0;

   ztrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:LSolve:: info");
#else
   for (int k = 0; k < n; k++)
   {
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<double> x_j = (x[j] /= data[j+j*m]);
         for (int i = j+1; i < m; i++)
         {
            x[i] -= data[i+j*m] * x_j;
         }
      }
      x += m;
   }
#endif
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexCholeskyFactors::USolve(int m, int n, double * X_r,
                                    double * X_i) const
{
   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
#ifdef MFEM_USE_LAPACK

   char uplo = 'L';
   char trans = 'C';
   char diag = 'N';
   int info = 0;

   ztrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:USolve:: info");
#else
   // X <- L^{-t} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<double> x_j = ( x[j] /= data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            x[i] -= std::conj(data[j+i*m]) * x_j;
         }
      }
      x += m;
   }
#endif
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexCholeskyFactors::Solve(int m, int n, double * X_r,
                                   double * X_i) const
{
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   int info = 0;
   std::complex<double> *x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   zpotrs_(&uplo, &m, &n, data, &m, x, &m, &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:Solve:: info");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete x;
#else
   LSolve(m, n, X_r,X_i);
   USolve(m, n, X_r,X_i);
#endif
}

void ComplexCholeskyFactors::RightSolve(int m, int n, double * X_r,
                                        double * X_i) const
{

   std::complex<double> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<double> *x = X;
#ifdef MFEM_USE_LAPACK
   char side = 'R';
   char uplo = 'L';
   char transt = 'C';
   char trans = 'N';
   char diag = 'N';

   std::complex<double> alpha(1.0,0.0);
   if (m > 0 && n > 0)
   {
      ztrsm_(&side,&uplo,&transt,&diag,&n,&m,&alpha,data,&m,x,&n);
      ztrsm_(&side,&uplo,&trans,&diag,&n,&m,&alpha,data,&m,x,&n);
   }
#else
   // X <- X L^{-H}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<double> x_j = ( x[j*n] /= data[j+j*m]);
         for (int i = j+1; i < m; i++)
         {
            x[i*n] -= std::conj(data[i + j*m]) * x_j;
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
         const std::complex<double> x_j = (x[j*n] /= data[j+j*m]);
         for (int i = 0; i < j; i++)
         {
            x[i*n] -= data[j + i*m] * x_j;
         }
      }
      ++x;
   }
#endif
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexCholeskyFactors::GetInverseMatrix(int m, double * X_r,
                                              double * X_i) const
{
   // A^{-1} = L^{-t} L^{-1}
   std::complex<double> * X = new std::complex<double>[m*m];
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
   zpotri_(&uplo, &m, X, &m, &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:GetInverseMatrix:: info");
   // fill in the upper triangular part
   for (int i = 0; i<m; i++)
   {
      for (int j = i+1; j<m; j++)
      {
         X[i+j*m] = std::conj(X[j+i*m]);
      }
   }
#else
   // L^-t * L^-1 (in place)
   for (int k = 0; k<m; k++)
   {
      X[k+k*m] = 1./data[k+k*m];
      for (int i = k+1; i < m; i++)
      {
         std::complex<double> s(0.,0.);
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
         std::complex<double> s(0.,0.);
         for (int k=j; k<m; k++)
         {
            s += X[k+i*m] * std::conj(X[k+j*m]);
         }
         X[j+i*m] = s;
         X[i+j*m] = std::conj(s);
      }
   }
#endif
   ComplexFactors::ComplexToReal(m*m,X,X_r,X_i);
   delete [] X;
}

} // mfem namespace
