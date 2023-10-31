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
#ifdef MFEM_USE_FLOAT
extern "C" void
cgetrf_(int *, int *, std::complex<float> *, int *, int *, int *);
extern "C" void
cgetrs_(char *, int *, int *, std::complex<float> *, int *, int *,
        std::complex<float> *, int *, int *);
extern "C" void
cgetri_(int *, std::complex<float> *, int *, int *,
        std::complex<float> *, int *, int *);
extern "C" void
ctrsm_(char *, char *, char *, char *, int *, int *, std::complex<float> *,
       std::complex<float> *, int *, std::complex<float> *, int *);
extern "C" void
cpotrf_(char *, int *, std::complex<float> *, int *, int *);

extern "C" void
ctrtrs_(char *, char*, char *, int *, int *, std::complex<float> *, int *,
        std::complex<float> *, int *, int *);
extern "C" void
cpotri_(char *, int *, std::complex<float> *, int*, int *);

extern "C" void
cpotrs_(char *, int *, int *, std::complex<float> *, int *,
        std::complex<float> *, int *, int *);
#else // Double-precision
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
   fptype * data = A->Data();
   fptype * data_r = nullptr;
   fptype * data_i = nullptr;

   const fptype factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;
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
   std::complex<fptype> * data = new std::complex<fptype>[h*w];

   // copy data
   if (hasRealPart() && hasImagPart())
   {
      fptype * data_r = real().Data();
      fptype * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<fptype> (data_r[i], data_i[i]);
      }
   }
   else if (hasRealPart())
   {
      fptype * data_r = real().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<fptype> (data_r[i], 0.);
      }
   }
   else if (hasImagPart())
   {
      fptype * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<fptype> (0., data_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexDenseMatrix has neither only a real nor an imag part");
   }

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[w];
   int    lwork = -1;
   std::complex<fptype> qwork, *work;
   int    info;

#ifdef MFEM_USE_FLOAT
   cgetrf_(&w, &w, data, &w, ipiv, &info);
#else
   zgetrf_(&w, &w, data, &w, ipiv, &info);
#endif
   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRF");
   }

#ifdef MFEM_USE_FLOAT
   cgetri_(&w, data, &w, ipiv, &qwork, &lwork, &info);
#else
   zgetri_(&w, data, &w, ipiv, &qwork, &lwork, &info);
#endif
   lwork = (int) qwork.real();
   work = new std::complex<fptype>[lwork];

#ifdef MFEM_USE_FLOAT
   cgetri_(&w, data, &w, ipiv, work, &lwork, &info);
#else
   zgetri_(&w, data, &w, ipiv, work, &lwork, &info);
#endif
   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRI");
   }

   delete [] work;
   delete [] ipiv;

#else
   // compiling without LAPACK
   int c, i, j, n = w;
   fptype a, b;
   Array<int> piv(n);
   std::complex<fptype> ac,bc;

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
         mfem::Swap<std::complex<fptype>>(data[c+j*h], data[i+j*h]);
      }

      fptype fone = 1.0;  // TODO: why?
      ac = data[c+c*h] = fone / data[c+c*h];
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
         mfem::Swap<std::complex<fptype>>(data[i+c*h], data[i+j*h]);
      }
   }

#endif

   DenseMatrix * C_r = new DenseMatrix(h);
   DenseMatrix * C_i = new DenseMatrix(h);

   fptype * datac_r = C_r->Data();
   fptype * datac_i = C_i->Data();

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

std::complex<fptype> * ComplexFactors::RealToComplex
(int m, const fptype * x_r, const fptype * x_i) const
{
   std::complex<fptype> * x = new std::complex<fptype>[m];
   if (x_r && x_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<fptype>(x_r[i], x_i[i]);
      }
   }
   else if (data_r)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<fptype>(x_r[i], 0.);
      }
   }
   else if (data_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<fptype>(0., x_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexFactors::RealToComplex:both real and imag part are null");
      return nullptr;
   }
   return x;
}

void ComplexFactors::ComplexToReal(int m, const std::complex<fptype> * x,
                                   fptype * x_r, fptype * x_i) const
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


bool ComplexLUFactors::Factor(int m, fptype TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   MFEM_VERIFY(data, "Matrix data not set");
#ifdef MFEM_USE_FLOAT
   if (m) { cgetrf_(&m, &m, data, &m, ipiv, &info); }
#else
   if (m) { zgetrf_(&m, &m, data, &m, ipiv, &info); }
#endif
   return info == 0;
#else
   // compiling without LAPACK
   std::complex<fptype> *data_ptr = this->data;
   for (int i = 0; i < m; i++)
   {
      // pivoting
      {
         int piv = i;
         fptype a = std::abs(data_ptr[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const fptype b = std::abs(data_ptr[j+i*m]);
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
               mfem::Swap<std::complex<fptype>>(data_ptr[i+j*m], data_ptr[piv+j*m]);
            }
         }
      }

      if (abs(data_ptr[i + i*m]) <= TOL)
      {
         return false; // failed
      }

      fptype fone = 1.0;  // TODO: why?
      const std::complex<fptype> a_ii_inv = fone / data_ptr[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         data_ptr[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const std::complex<fptype> a_ik = data_ptr[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            data_ptr[j+k*m] -= a_ik * data_ptr[j+i*m];
         }
      }
   }
#endif

   return true; // success
}

std::complex<fptype> ComplexLUFactors::Det(int m) const
{
   std::complex<fptype> det(1.0,0.);
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

void ComplexLUFactors::Mult(int m, int n, fptype *X_r, fptype * X_i) const
{
   std::complex<fptype> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   for (int k = 0; k < n; k++)
   {
      // X <- U X
      for (int i = 0; i < m; i++)
      {
         std::complex<fptype> x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- L X
      for (int i = m-1; i >= 0; i--)
      {
         std::complex<fptype> x_i = x[i];
         for (int j = 0; j < i; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- P^{-1} X
      for (int i = m-1; i >= 0; i--)
      {
         mfem::Swap<std::complex<fptype>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
}

void ComplexLUFactors::LSolve(int m, int n, fptype *X_r, fptype * X_i) const
{
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         mfem::Swap<std::complex<fptype>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<fptype> x_j = x[j];
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

void ComplexLUFactors::USolve(int m, int n, fptype *X_r, fptype * X_i) const
{
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
   // X <- U^{-1} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<fptype> x_j = ( x[j] /= data[j+j*m] );
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

void ComplexLUFactors::Solve(int m, int n, fptype *X_r, fptype * X_i) const
{
#ifdef MFEM_USE_LAPACK
   std::complex<fptype> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   char trans = 'N';
   int  info = 0;
#ifdef MFEM_USE_FLOAT
   if (m > 0 && n > 0) { cgetrs_(&trans, &m, &n, data, &m, ipiv, x, &m, &info); }
#else
   if (m > 0 && n > 0) { zgetrs_(&trans, &m, &n, data, &m, ipiv, x, &m, &info); }
#endif
   MFEM_VERIFY(!info, "LAPACK: error in ZGETRS");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
#else
   // compiling without LAPACK
   LSolve(m, n, X_r, X_i);
   USolve(m, n, X_r, X_i);
#endif
}

void ComplexLUFactors::RightSolve(int m, int n, fptype *X_r, fptype * X_i) const
{
   std::complex<fptype> * X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> * x = X;
#ifdef MFEM_USE_LAPACK
   char n_ch = 'N', side = 'R', u_ch = 'U', l_ch = 'L';
   if (m > 0 && n > 0)
   {
      std::complex<fptype> alpha(1.0,0.0);
#ifdef MFEM_USE_FLOAT
      ctrsm_(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      ctrsm_(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
#else
      ztrsm_(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      ztrsm_(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
#endif
   }
#else
   // compiling without LAPACK
   // X <- X U^{-1}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<fptype> x_j = ( x[j*n] /= data[j+j*m]);
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
         const std::complex<fptype> x_j = x[j*n];
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
         mfem::Swap<std::complex<fptype>>(x[i*n], x[(ipiv[i]-ipiv_base)*n]);
      }
      ++x;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexLUFactors::GetInverseMatrix(int m, fptype *X_r, fptype *X_i) const
{
   // A^{-1} = U^{-1} L^{-1} P
   // X <- U^{-1} (set only the upper triangular part of X)
   std::complex<fptype> * X = ComplexFactors::RealToComplex(m*m,X_r,X_i);
   std::complex<fptype> * x = X;
   fptype fone = 1.0;  // TODO: why?
   for (int k = 0; k < m; k++)
   {
      const std::complex<fptype> minus_x_k = -( x[k] = fone/data[k+k*m] );
      for (int i = 0; i < k; i++)
      {
         x[i] = data[i+k*m] * minus_x_k;
      }
      for (int j = k-1; j >= 0; j--)
      {
         const std::complex<fptype> x_j = ( x[j] /= data[j+j*m] );
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
         const std::complex<fptype> minus_L_kj = -data[k+j*m];
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
         const std::complex<fptype> L_kj = data[k+j*m];
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
            Swap<std::complex<fptype>>(X[i+k*m], X[i+piv_k*m]);
         }
      }
   }
   ComplexFactors::ComplexToReal(m*m,X,X_r,X_i);
   delete [] X;
}


bool ComplexCholeskyFactors::Factor(int m, fptype TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   char uplo = 'L';
   MFEM_VERIFY(data, "Matrix data not set");
#ifdef MFEM_USE_FLOAT
   if (m) {cpotrf_(&uplo, &m, data, &m, &info);}
#else
   if (m) {zpotrf_(&uplo, &m, data, &m, &info);}
#endif
   return info == 0;
#else
   // Cholesky–Crout algorithm
   fptype fone = 1.0;  // TODO: why?
   for (int j = 0; j<m; j++)
   {
      std::complex<fptype> a(0.,0.);
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
         a = std::complex<fptype>(0.,0.);
         for (int k = 0; k<j; k++)
         {
            a+= data[i+k*m]*std::conj(data[j+k*m]);
         }
         data[i+j*m] = fone/data[j+m*j]*(data[i+j*m] - a);
      }
   }
   return true; // success
#endif
}

std::complex<fptype> ComplexCholeskyFactors::Det(int m) const
{
   std::complex<fptype> det(1.0,0.0);
   for (int i=0; i<m; i++)
   {
      det *=  data[i + i*m];
   }
   return det;
}

void ComplexCholeskyFactors::LMult(int m, int n, fptype * X_r,
                                   fptype * X_i) const
{
   // X <- L X
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         std::complex<fptype> x_j = x[j] * data[j+j*m];
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

void ComplexCholeskyFactors::UMult(int m, int n, fptype * X_r,
                                   fptype * X_i) const
{
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int i = 0; i < m; i++)
      {
         std::complex<fptype> x_i = x[i] * data[i+i*m];
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

void ComplexCholeskyFactors::LSolve(int m, int n, fptype * X_r,
                                    fptype * X_i) const
{
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   char trans = 'N';
   char diag = 'N';
   int info = 0;

#ifdef MFEM_USE_FLOAT
   ctrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
#else
   ztrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
#endif
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:LSolve:: info");
#else
   for (int k = 0; k < n; k++)
   {
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<fptype> x_j = (x[j] /= data[j+j*m]);
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

void ComplexCholeskyFactors::USolve(int m, int n, fptype * X_r,
                                    fptype * X_i) const
{
   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
#ifdef MFEM_USE_LAPACK

   char uplo = 'L';
   char trans = 'C';
   char diag = 'N';
   int info = 0;

#ifdef MFEM_USE_FLOAT
   ctrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
#else
   ztrtrs_(&uplo, &trans, &diag, &m, &n, data, &m, x, &m, &info);
#endif
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:USolve:: info");
#else
   // X <- L^{-t} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<fptype> x_j = ( x[j] /= data[j+j*m] );
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

void ComplexCholeskyFactors::Solve(int m, int n, fptype * X_r,
                                   fptype * X_i) const
{
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   int info = 0;
   std::complex<fptype> *x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
#ifdef MFEM_USE_FLOAT
   cpotrs_(&uplo, &m, &n, data, &m, x, &m, &info);
#else
   zpotrs_(&uplo, &m, &n, data, &m, x, &m, &info);
#endif
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:Solve:: info");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete x;
#else
   LSolve(m, n, X_r,X_i);
   USolve(m, n, X_r,X_i);
#endif
}

void ComplexCholeskyFactors::RightSolve(int m, int n, fptype * X_r,
                                        fptype * X_i) const
{

   std::complex<fptype> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<fptype> *x = X;
#ifdef MFEM_USE_LAPACK
   char side = 'R';
   char uplo = 'L';
   char transt = 'C';
   char trans = 'N';
   char diag = 'N';

   std::complex<fptype> alpha(1.0,0.0);
   if (m > 0 && n > 0)
   {
#ifdef MFEM_USE_FLOAT
      ctrsm_(&side,&uplo,&transt,&diag,&n,&m,&alpha,data,&m,x,&n);
      ctrsm_(&side,&uplo,&trans,&diag,&n,&m,&alpha,data,&m,x,&n);
#else
      ztrsm_(&side,&uplo,&transt,&diag,&n,&m,&alpha,data,&m,x,&n);
      ztrsm_(&side,&uplo,&trans,&diag,&n,&m,&alpha,data,&m,x,&n);
#endif
   }
#else
   // X <- X L^{-H}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<fptype> x_j = ( x[j*n] /= data[j+j*m]);
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
         const std::complex<fptype> x_j = (x[j*n] /= data[j+j*m]);
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

void ComplexCholeskyFactors::GetInverseMatrix(int m, fptype * X_r,
                                              fptype * X_i) const
{
   // A^{-1} = L^{-t} L^{-1}
   std::complex<fptype> * X = new std::complex<fptype>[m*m];
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
#ifdef MFEM_USE_FLOAT
   cpotri_(&uplo, &m, X, &m, &info);
#else
   zpotri_(&uplo, &m, X, &m, &info);
#endif
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
   fptype fone = 1.0;  // TODO: why?
   for (int k = 0; k<m; k++)
   {
      X[k+k*m] = fone/data[k+k*m];
      for (int i = k+1; i < m; i++)
      {
         std::complex<fptype> s(0.,0.);
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
         std::complex<fptype> s(0.,0.);
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
