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

#include "complex_densemat.hpp"
#include <complex>

#ifdef MFEM_USE_LAPACK
extern "C" void
zgetrf_(int *, int *, std::complex<double> *, int *, int *, int *);
extern "C" void
zgetrs_(char *, int *, int *, std::complex<double> *, int *, int *,
        std::complex<double> *, int *, int *);
extern "C" void
zgetri_(int *N, std::complex<double> *A, int *LDA, int *IPIV,
        std::complex<double> *WORK,
        int *LWORK, int *INFO);
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

   // assuming Hermitian convension
   *A = 0.;
   if (hasRealPart())
   {
      data_r = real().Data();
      for (int j = 0; j<w; j++)
      {
         for (int i = 0; i<h; i++)
         {
            data[i+j*height] = data_r[i+j*h];
            data[i+h+(j+h)*height] = data_r[i+j*h];
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
            data[i+h+j*height] = data_i[i+j*h];
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
   else
   {
      MFEM_ABORT("ComplexDenseMatrix has either only real or imag part");
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

   for (int i = 0; i < h*w; i++)
   {
      datac_r[i] = data[i].real();
      datac_i[i] = data[i].imag();
   }

   return new ComplexDenseMatrix(C_r,C_i,true,true);

}

ComplexDenseMatrix * Mult(const ComplexDenseMatrix &A,
                          const ComplexDenseMatrix &B)
{
   // C = C_r + i C_i = (A_r + i * A_i) * (B_r + i * B_i)
   //                 = A_r * B_r - A_i B_i + i (A_r * B_i + A_i * B_r)

   int h = A.Height()/2;
   int w = B.Width()/2;

   MFEM_VERIFY(A.Width() == B.Height(), "Incompatible matrix dimenions");

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

   MFEM_VERIFY(A.Height() == B.Height(), "Incompatible matrix dimenions");

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


} // mfem namespace
