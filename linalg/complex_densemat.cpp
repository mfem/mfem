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

#include "complex_densemat.hpp"
#include "lapack.hpp"
#include <complex>

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
   real_t * data = A->Data();
   real_t * data_r = nullptr;
   real_t * data_i = nullptr;

   const real_t factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;
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
   std::complex<real_t> * data = new std::complex<real_t>[h*w];

   // copy data
   if (hasRealPart() && hasImagPart())
   {
      real_t * data_r = real().Data();
      real_t * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<real_t> (data_r[i], data_i[i]);
      }
   }
   else if (hasRealPart())
   {
      real_t * data_r = real().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<real_t> (data_r[i], 0.);
      }
   }
   else if (hasImagPart())
   {
      real_t * data_i = imag().Data();
      for (int i = 0; i < h*w; i++)
      {
         data[i] = std::complex<real_t> (0., data_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexDenseMatrix has neither only a real nor an imag part");
   }

#ifdef MFEM_USE_LAPACK
   int   *ipiv = new int[w];
   int    lwork = -1;
   std::complex<real_t> qwork, *work;
   int    info;

   MFEM_LAPACK_COMPLEX(getrf_)(&w, &w, data, &w, ipiv, &info);
   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRF");
   }

   MFEM_LAPACK_COMPLEX(getri_)(&w, data, &w, ipiv, &qwork, &lwork, &info);
   lwork = (int) qwork.real();
   work = new std::complex<real_t>[lwork];

   MFEM_LAPACK_COMPLEX(getri_)(&w, data, &w, ipiv, work, &lwork, &info);
   if (info)
   {
      mfem_error("DenseMatrix::Invert() : Error in ZGETRI");
   }

   delete [] work;
   delete [] ipiv;

#else
   // compiling without LAPACK
   int c, i, j, n = w;
   real_t a, b;
   Array<int> piv(n);
   std::complex<real_t> ac,bc;

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
         mfem::Swap<std::complex<real_t>>(data[c+j*h], data[i+j*h]);
      }

      real_t fone = 1.0;  // TODO: why?
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
         mfem::Swap<std::complex<real_t>>(data[i+c*h], data[i+j*h]);
      }
   }

#endif

   DenseMatrix * C_r = new DenseMatrix(h);
   DenseMatrix * C_i = new DenseMatrix(h);

   real_t * datac_r = C_r->Data();
   real_t * datac_i = C_i->Data();

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

std::complex<real_t> * ComplexFactors::RealToComplex
(int m, const real_t * x_r, const real_t * x_i) const
{
   std::complex<real_t> * x = new std::complex<real_t>[m];
   if (x_r && x_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<real_t>(x_r[i], x_i[i]);
      }
   }
   else if (data_r)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<real_t>(x_r[i], 0.);
      }
   }
   else if (data_i)
   {
      for (int i = 0; i<m; i++)
      {
         x[i] = std::complex<real_t>(0., x_i[i]);
      }
   }
   else
   {
      MFEM_ABORT("ComplexFactors::RealToComplex:both real and imag part are null");
      return nullptr;
   }
   return x;
}

void ComplexFactors::ComplexToReal(int m, const std::complex<real_t> * x,
                                   real_t * x_r, real_t * x_i) const
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


bool ComplexLUFactors::Factor(int m, real_t TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   MFEM_VERIFY(data, "Matrix data not set");
   if (m) { MFEM_LAPACK_COMPLEX(getrf_)(&m, &m, data, &m, ipiv, &info); }
   return info == 0;
#else
   // compiling without LAPACK
   std::complex<real_t> *data_ptr = this->data;
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
               mfem::Swap<std::complex<real_t>>(data_ptr[i+j*m], data_ptr[piv+j*m]);
            }
         }
      }

      if (abs(data_ptr[i + i*m]) <= TOL)
      {
         return false; // failed
      }

      real_t fone = 1.0;  // TODO: why?
      const std::complex<real_t> a_ii_inv = fone / data_ptr[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         data_ptr[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const std::complex<real_t> a_ik = data_ptr[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            data_ptr[j+k*m] -= a_ik * data_ptr[j+i*m];
         }
      }
   }
#endif

   return true; // success
}

std::complex<real_t> ComplexLUFactors::Det(int m) const
{
   std::complex<real_t> det(1.0,0.);
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

void ComplexLUFactors::Mult(int m, int n, real_t *X_r, real_t * X_i) const
{
   std::complex<real_t> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   for (int k = 0; k < n; k++)
   {
      // X <- U X
      for (int i = 0; i < m; i++)
      {
         std::complex<real_t> x_i = x[i] * data[i+i*m];
         for (int j = i+1; j < m; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- L X
      for (int i = m-1; i >= 0; i--)
      {
         std::complex<real_t> x_i = x[i];
         for (int j = 0; j < i; j++)
         {
            x_i += x[j] * data[i+j*m];
         }
         x[i] = x_i;
      }
      // X <- P^{-1} X
      for (int i = m-1; i >= 0; i--)
      {
         mfem::Swap<std::complex<real_t>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      x += m;
   }
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
}

void ComplexLUFactors::LSolve(int m, int n, real_t *X_r, real_t * X_i) const
{
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
   for (int k = 0; k < n; k++)
   {
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         mfem::Swap<std::complex<real_t>>(x[i], x[ipiv[i]-ipiv_base]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<real_t> x_j = x[j];
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

void ComplexLUFactors::USolve(int m, int n, real_t *X_r, real_t * X_i) const
{
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
   // X <- U^{-1} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<real_t> x_j = ( x[j] /= data[j+j*m] );
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

void ComplexLUFactors::Solve(int m, int n, real_t *X_r, real_t * X_i) const
{
#ifdef MFEM_USE_LAPACK
   std::complex<real_t> * x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   char trans = 'N';
   int  info = 0;
   if (m > 0 && n > 0)
   {
      MFEM_LAPACK_COMPLEX(getrs_)(&trans, &m, &n, data, &m, ipiv, x, &m, &info);
   }
   MFEM_VERIFY(!info, "LAPACK: error in ZGETRS");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete [] x;
#else
   // compiling without LAPACK
   LSolve(m, n, X_r, X_i);
   USolve(m, n, X_r, X_i);
#endif
}

void ComplexLUFactors::RightSolve(int m, int n, real_t *X_r, real_t * X_i) const
{
   std::complex<real_t> * X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> * x = X;
#ifdef MFEM_USE_LAPACK
   char n_ch = 'N', side = 'R', u_ch = 'U', l_ch = 'L';
   if (m > 0 && n > 0)
   {
      std::complex<real_t> alpha(1.0,0.0);
      MFEM_LAPACK_COMPLEX(trsm_)(&side,&u_ch,&n_ch,&n_ch,&n,&m,&alpha,data,&m,X,&n);
      MFEM_LAPACK_COMPLEX(trsm_)(&side,&l_ch,&n_ch,&u_ch,&n,&m,&alpha,data,&m,X,&n);
   }
#else
   // compiling without LAPACK
   // X <- X U^{-1}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<real_t> x_j = ( x[j*n] /= data[j+j*m]);
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
         const std::complex<real_t> x_j = x[j*n];
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
         mfem::Swap<std::complex<real_t>>(x[i*n], x[(ipiv[i]-ipiv_base)*n]);
      }
      ++x;
   }
   ComplexFactors::ComplexToReal(m*n,X,X_r,X_i);
   delete [] X;
}

void ComplexLUFactors::GetInverseMatrix(int m, real_t *X_r, real_t *X_i) const
{
   // A^{-1} = U^{-1} L^{-1} P
   // X <- U^{-1} (set only the upper triangular part of X)
   std::complex<real_t> * X = ComplexFactors::RealToComplex(m*m,X_r,X_i);
   std::complex<real_t> * x = X;
   real_t fone = 1.0;  // TODO: why?
   for (int k = 0; k < m; k++)
   {
      const std::complex<real_t> minus_x_k = -( x[k] = fone/data[k+k*m] );
      for (int i = 0; i < k; i++)
      {
         x[i] = data[i+k*m] * minus_x_k;
      }
      for (int j = k-1; j >= 0; j--)
      {
         const std::complex<real_t> x_j = ( x[j] /= data[j+j*m] );
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
         const std::complex<real_t> minus_L_kj = -data[k+j*m];
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
         const std::complex<real_t> L_kj = data[k+j*m];
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
            Swap<std::complex<real_t>>(X[i+k*m], X[i+piv_k*m]);
         }
      }
   }
   ComplexFactors::ComplexToReal(m*m,X,X_r,X_i);
   delete [] X;
}


bool ComplexCholeskyFactors::Factor(int m, real_t TOL)
{
   SetComplexData(m*m);
#ifdef MFEM_USE_LAPACK
   int info = 0;
   char uplo = 'L';
   MFEM_VERIFY(data, "Matrix data not set");
   if (m) { MFEM_LAPACK_COMPLEX(potrf_)(&uplo, &m, data, &m, &info); }
   return info == 0;
#else
   // Cholesky–Crout algorithm
   real_t fone = 1.0;  // TODO: why?
   for (int j = 0; j<m; j++)
   {
      std::complex<real_t> a(0.,0.);
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
         a = std::complex<real_t>(0.,0.);
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

std::complex<real_t> ComplexCholeskyFactors::Det(int m) const
{
   std::complex<real_t> det(1.0,0.0);
   for (int i=0; i<m; i++)
   {
      det *=  data[i + i*m];
   }
   return det;
}

void ComplexCholeskyFactors::LMult(int m, int n, real_t * X_r,
                                   real_t * X_i) const
{
   // X <- L X
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         std::complex<real_t> x_j = x[j] * data[j+j*m];
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

void ComplexCholeskyFactors::UMult(int m, int n, real_t * X_r,
                                   real_t * X_i) const
{
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
   for (int k = 0; k < n; k++)
   {
      for (int i = 0; i < m; i++)
      {
         std::complex<real_t> x_i = x[i] * data[i+i*m];
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

void ComplexCholeskyFactors::LSolve(int m, int n, real_t * X_r,
                                    real_t * X_i) const
{
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   char trans = 'N';
   char diag = 'N';
   int info = 0;

   MFEM_LAPACK_COMPLEX(trtrs_)(&uplo, &trans, &diag, &m, &n, data, &m, x, &m,
                               &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:LSolve:: info");
#else
   for (int k = 0; k < n; k++)
   {
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const std::complex<real_t> x_j = (x[j] /= data[j+j*m]);
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

void ComplexCholeskyFactors::USolve(int m, int n, real_t * X_r,
                                    real_t * X_i) const
{
   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
#ifdef MFEM_USE_LAPACK

   char uplo = 'L';
   char trans = 'C';
   char diag = 'N';
   int info = 0;

   MFEM_LAPACK_COMPLEX(trtrs_)(&uplo, &trans, &diag, &m, &n, data, &m, x, &m,
                               &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:USolve:: info");
#else
   // X <- L^{-t} X
   for (int k = 0; k < n; k++)
   {
      for (int j = m-1; j >= 0; j--)
      {
         const std::complex<real_t> x_j = ( x[j] /= data[j+j*m] );
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

void ComplexCholeskyFactors::Solve(int m, int n, real_t * X_r,
                                   real_t * X_i) const
{
#ifdef MFEM_USE_LAPACK
   char uplo = 'L';
   int info = 0;
   std::complex<real_t> *x = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   MFEM_LAPACK_COMPLEX(potrs_)(&uplo, &m, &n, data, &m, x, &m, &info);
   MFEM_VERIFY(!info, "ComplexCholeskyFactors:Solve:: info");
   ComplexFactors::ComplexToReal(m*n,x,X_r,X_i);
   delete x;
#else
   LSolve(m, n, X_r,X_i);
   USolve(m, n, X_r,X_i);
#endif
}

void ComplexCholeskyFactors::RightSolve(int m, int n, real_t * X_r,
                                        real_t * X_i) const
{

   std::complex<real_t> *X = ComplexFactors::RealToComplex(m*n,X_r,X_i);
   std::complex<real_t> *x = X;
#ifdef MFEM_USE_LAPACK
   char side = 'R';
   char uplo = 'L';
   char transt = 'C';
   char trans = 'N';
   char diag = 'N';

   std::complex<real_t> alpha(1.0,0.0);
   if (m > 0 && n > 0)
   {
      MFEM_LAPACK_COMPLEX(trsm_)(&side,&uplo,&transt,&diag,&n,&m,&alpha,data,&m,x,&n);
      MFEM_LAPACK_COMPLEX(trsm_)(&side,&uplo,&trans,&diag,&n,&m,&alpha,data,&m,x,&n);
   }
#else
   // X <- X L^{-H}
   for (int k = 0; k < n; k++)
   {
      for (int j = 0; j < m; j++)
      {
         const std::complex<real_t> x_j = ( x[j*n] /= data[j+j*m]);
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
         const std::complex<real_t> x_j = (x[j*n] /= data[j+j*m]);
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

void ComplexCholeskyFactors::GetInverseMatrix(int m, real_t * X_r,
                                              real_t * X_i) const
{
   // A^{-1} = L^{-t} L^{-1}
   std::complex<real_t> * X = new std::complex<real_t>[m*m];
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
   MFEM_LAPACK_COMPLEX(potri_)(&uplo, &m, X, &m, &info);
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
   real_t fone = 1.0;  // TODO: why?
   for (int k = 0; k<m; k++)
   {
      X[k+k*m] = fone/data[k+k*m];
      for (int i = k+1; i < m; i++)
      {
         std::complex<real_t> s(0.,0.);
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
         std::complex<real_t> s(0.,0.);
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
