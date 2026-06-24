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


// Implementation of data type BandMatrix
#include "densemat.hpp"
#include "bandmatrix.hpp"
#include "kernels.hpp"
#include "lapack.hpp"
namespace mfem
{

BandMatrix::BandMatrix() : Matrix(0) { bandwidth = stride = 0;}

BandMatrix::BandMatrix(int s, int bw) : Matrix(s)
{
   MFEM_ASSERT(s >= 0, "invalid BandMatrix size: " << s);
   MFEM_ASSERT(s > bw, "invalid bandwidth size: " << bw);
   MFEM_ASSERT(bw >= 0, "invalid bandwidth size: " << bw);
   bandwidth = bw; stride = 2*bw + 1;
   if (s > 0)
   {
      data.SetSize(s*stride);
      *this = 0.0; // init with zeroes
   }
}

BandMatrix::BandMatrix(int h, int w, int bw) : Matrix(h,w)
{
   MFEM_ASSERT(h >= 0, "invalid BandMatrix size: " << h);
   MFEM_ASSERT(w >= 0, "invalid BandMatrix size: " << w);
   MFEM_ASSERT(w > bw, "invalid bandwidth size: " << bw);
   MFEM_ASSERT(bw >= 0, "invalid bandwidth size: " << bw);
   bandwidth = bw; stride = 2*bw + 1;
   if (h > 0)
   {
      data.SetSize(h*stride);
      *this = 0.0; // init with zeroes
   }
}

BandMatrix::BandMatrix(const DenseMatrix &dm, int bw) : Matrix(dm.Height(),
                                                                  dm.Width())
{
   Reset(dm, bw);
}

void BandMatrix::Reset(const DenseMatrix &dm, int bw)
{
   height = dm.Height();
   width = dm.Width();
   if (bw < 0)
   {
      bandwidth = 0;
      for (int i = 0; i < dm.Height(); i++)
      {
         for (int j = 0; j < dm.Width(); j++)
         {
            if (dm(i,j)*dm(i,j) > 1e-16)
            {
               bandwidth = std::max(i - j, bandwidth);
               bandwidth = std::max(j - i, bandwidth);
            }
         }
      }
   }
   else
   {
      bandwidth = bw;
   }

   stride = 2*bandwidth + 1;
   if (Height() > 0)
   {
      data.SetSize(Height()*stride);
   }

   for (int i = 0; i < dm.Height(); i++)
   {
      for (int j  = std::max(i - bandwidth, 0);
           j <= std::min(i + bandwidth, width-1) ; j++)
      {
         Elem(i,j) = dm(i,j);
      }
   }
}


void BandMatrix::SetSize(int s, int bw)
{
   MFEM_ASSERT(s >= 0, "invalid BandMatrix size: " << s);
   MFEM_ASSERT(s > bw, "invalid bandwidth size: " << bw);
   MFEM_ASSERT(bw >= 0, "invalid bandwidth size: " << bw);
   if (Height() == s&& bandwidth == bw)
   {
      return;
   }
   height = width = s; bandwidth = bw; stride = 2*bw + 1;
   data.SetSize(height*stride);
   *this = 0.0; // init with zeroes
}

void BandMatrix::SetSize(int h, int w, int bw)
{
   MFEM_ASSERT(h >= 0, "invalid BandMatrix size: " << h);
   MFEM_ASSERT(w >= 0, "invalid BandMatrix size: " << w);
   MFEM_ASSERT(w > bw, "invalid bandwidth size: " << bw);
   MFEM_ASSERT(bw >= 0, "invalid bandwidth size: " << bw);
   if (Height() == h && bandwidth == bw)
   {
      return;
   }
   height = h; width = w; bandwidth = bw; stride = 2*bw + 1;
   data.SetSize(height*stride);
   *this = 0.0; // init with zeroes
}

BandMatrix &BandMatrix::operator=(real_t c)
{
   const int s = GetStoredSize();
   for (int i = 0; i < s; i++)
   {
      data[i] = c;
   }
   return *this;
}

real_t &BandMatrix::Elem(int i, int j)
{
   return (*this)(i,j);
}

const real_t &BandMatrix::Elem(int i, int j) const
{
   return (*this)(i,j);
}

BandMatrix &BandMatrix::operator*=(real_t c)
{
   int s = GetStoredSize();
   for (int i = 0; i < s; i++)
   {
      data[i] *= c;
   }
   return *this;
}

void BandMatrix::Mult(const real_t *x, real_t *y) const
{
   kernels::BandMult(height, width, bandwidth, data.HostRead(), x, y);
}

void BandMatrix::Mult(const real_t *x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size(), "incompatible dimensions");

   Mult(x, y.HostWrite());
}

void BandMatrix::Mult(const Vector &x, real_t *y) const
{
   MFEM_ASSERT(width == x.Size(), "incompatible dimensions");

   Mult(x.HostRead(), y);
}

void BandMatrix::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   Mult(x.HostRead(), y.HostWrite());
}

void BandMatrix::Solve(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(height == y.Size() && width == x.Size(),
               "incompatible dimensions");

   MatrixInverse *inv = Inverse();
   inv->Mult(x, y);
   delete inv;
}

MatrixInverse *BandMatrix::Inverse() const
{
#ifdef MFEM_USE_LAPACK
   return new BandMatrixInverse(*this);
#else
   MFEM_WARNING("LAPACK not linked. Converting BandMatrix to DenseMatrix.");
   return new DenseMatrixInverse(ToDenseMatrix());
#endif
}

void BandMatrix::Inverse(DenseMatrix &dm)
{
#ifdef MFEM_DEBUG
   if (Height() <= 0 || Height() != Width())
   {
      mfem_error("DenseMatrix::Invert(): dimension mismatch");
   }
#endif

#ifdef MFEM_USE_LAPACK
   BandMatrixInverse bmi(*this);
   bmi.Mult(DenseMatrix::Identity(height), dm);
#else
   MFEM_WARNING("LAPACK not linked. Converting BandMatrix to DenseMatrix.");
   dm = ToDenseMatrix();
   dm.invert();
#endif
}

void BandMatrix::Invert(real_t tol, int bw)
{
#ifdef MFEM_DEBUG
   if (Height() <= 0 || Height() != Width())
   {
      mfem_error("DenseMatrix::Invert(): dimension mismatch");
   }
#endif

   DenseMatrix inv(height);
   Inverse(inv);
   if (tol < 0.0)
   {
      Reset(inv);
   }
   else
   {
      DenseMatrix I = DenseMatrix::Identity(height);
      DenseMatrix diff(height);
      if (bw < 0) // Find bandwidth
      {
         int i;
         for (i = 0; i < height - 1; i++)
         {
            BandMatrix binv(inv, i);
            DenseMatrix ans(height);
            mfem::Mult(binv, *this, ans);
            Add(ans, I, -1.0, diff);
            if (diff.FNorm()  < tol) { break; }
         }
         bw = i;
      }
      else  // Use given bandwidth
      {
         BandMatrix binv(inv, bw);
         DenseMatrix ans(height);
         mfem::Mult(binv, *this, ans);
         Add(ans, I, -1.0, diff);
         MFEM_VERIFY(diff.FNorm() > tol,
                     "Specified bandwidth does not achieve accuracy");
      }
      Reset(inv, bw);
   }
}


DenseMatrix BandMatrix::ToDenseMatrix() const
{
   DenseMatrix dm(height, width);
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         dm(i,j) = Elem(i,j);
      }
   }
   return dm;
}

void BandMatrix::Print(std::ostream &os, int width33_) const
{
   int width_ = 999;
   // save current output flags
   std::ios::fmtflags old_flags = os.flags();
   // output flags = scientific + show sign
   // os << setiosflags(std::ios::scientific | std::ios::showpos);
   for (int i = 0; i < height; i++)
   {
      // os << "[row " << i << "]\n";
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
   //  os.flags(old_flags);
}

void Mult(const BandMatrix &b,  const DenseMatrix &c, DenseMatrix &a)
{
   Mult(b.ToDenseMatrix(),c,a);
}

void Mult(const DenseMatrix &b, const BandMatrix &c,  DenseMatrix &a)
{
   Mult(b,c.ToDenseMatrix(),a);
}

void Mult(const BandMatrix &b,  const BandMatrix &c,  DenseMatrix &a)
{
   Mult(b.ToDenseMatrix(),c.ToDenseMatrix(),a);
}


/// C = A + beta*B
void Add(const DenseMatrix &A, const BandMatrix &B,
         real_t alpha, DenseMatrix &C)
{
   Add(1.0, A, alpha, B, C);
}

/// C = A + beta*B
void Add(const BandMatrix &A, const DenseMatrix &B,
         real_t alpha, DenseMatrix &C)
{
   Add(1.0, A, alpha, B, C);
}

/// C = A + alpha*A + beta*B
void Add(real_t alpha, const DenseMatrix &A,
         real_t beta,  const BandMatrix &B, DenseMatrix &C)
{
   MFEM_ASSERT(A.Height() == B.Height() && B.Height() == C.Height() &&
               A.Width() == B.Width() && B.Width() == C.Width(),
               "incompatible dimensions");
   for (int i = 0; i < C.Height(); i++)
   {
      for (int j = 0; j < C.Width(); j++)
      {
         C(i,j) = alpha*A(i,j) + beta*B(i,j);
      }
   }
}

/// C = alpha*A + beta*B
void Add(real_t alpha, const BandMatrix &A,
         real_t beta,  const DenseMatrix &B, DenseMatrix &C)
{
   MFEM_ASSERT(A.Height() == B.Height() && B.Height() == C.Height() &&
               A.Width() == B.Width() && B.Width() == C.Width(),
               "incompatible dimensions");
   for (int i = 0; i < C.Height(); i++)
   {
      for (int j = 0; j < C.Width(); j++)
      {
         C(i,j) = alpha*A(i,j) + beta*B(i,j);
      }
   }
}

bool BandLUFactors::Factor(int m, real_t TOL)
{
#ifdef MFEM_USE_LAPACK
   int ldab = 3*bw + 1;
   int info;
   MFEM_LAPACK_PREFIX(gbtrf_)(&m, &m, &bw, &bw, data, &ldab,
                              ipiv, &info);
   MFEM_ASSERT(info == 0, "BandedFactorizedSolve failed in LAPACK");
#else
   MFEM_ERROR("BandLUFactors::Factor requires lapack");
#endif
   return true;
}

real_t BandLUFactors::Det(int m) const
{
   int ldab = 3*bw + 1;
   real_t det = 1.0;
   for (int i = 0; i < m; i++)
   {
      if (ipiv[i] != i + 1)
      {
         det *= -data[2*bw + i * ldab];
      }
      else
      {
         det *= data[2*bw + i * ldab];
      }
   }
   return det;
}

void BandLUFactors::Solve(int m, int n, real_t *X) const
{
#ifdef MFEM_USE_LAPACK
   int bw_ = bw; // avoid const error
   int ldab = 3*bw + 1;
   bool transpose = false;
   char trans = transpose ? 'T' : 'N';
   int info;
   MFEM_LAPACK_PREFIX(gbtrs_)(&trans, &m, &bw_, &bw_, &n, data, &ldab,
                              ipiv, X, &m, &info);
   MFEM_ASSERT(info == 0, "BandedFactorizedSolve failed in LAPACK");
#else
   MFEM_ERROR("BandLUFactors::Solve requires lapack");
#endif
}

void BandLUFactors::GetInverseMatrix(int m, real_t *X) const
{
   for (int i = 0; i < m*m; i++) { X[i] = 0.0; }
   for (int i = 0; i < m; i++) { X[i + i*m] = 1.0; }
   Solve(m, m, X);
}

void BandMatrixInverse::Init(int m, int bw)
{
   factors = new BandLUFactors();

   if (m>0)
   {
      factors->data = new real_t[m*(3*bw + 1)];
      factors->ipiv = new int[m];
      own_data = true;
   }
}

BandMatrixInverse::BandMatrixInverse(const BandMatrix &mat)
   : MatrixInverse(mat)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = &mat;
   Init(width, a->bandwidth);
   Factor();
}

BandMatrixInverse::BandMatrixInverse(const BandMatrix *mat)
   : MatrixInverse(*mat)
{
   MFEM_ASSERT(height == width, "not a square matrix");
   a = mat;
   Init(width, a->bandwidth);
}

void BandMatrixInverse::Factor()
{
   MFEM_ASSERT(a, "DenseMatrix is not given");
   int bw = a->bandwidth;
   factors->bw = bw;
   int ldab = 3*bw + 1;
#ifdef MFEM_DEBUG
   for (int i = 0; i < height*ldab; ++i) { factors->data[i] = -0.0; }
#endif
   for (int j = 0; j < height; ++j)
   {
      int i_min = std::max(0, j - bw);
      int i_max = std::min(height - 1, j + bw);

      for (int i = i_min; i <= i_max; ++i)
      {
         int band_row = 2*bw + i - j;
         factors->data[band_row + j * ldab] = a->Elem(i, j);
      }
   }
   factors->Factor(width);
}

void BandMatrixInverse::GetInverseMatrix(DenseMatrix &Ainv) const
{
   Ainv.SetSize(width);
   factors->GetInverseMatrix(width,Ainv.Data());
}

void BandMatrixInverse::Factor(const BandMatrix &mat)
{
   MFEM_VERIFY(mat.height == mat.width, "DenseMatrix is not square!");
   if (width != mat.width)
   {
      height = width = mat.width;
      if (own_data)
      {
         delete [] factors->data;
         delete [] factors->ipiv;
      }
      factors->data = new real_t[width*(3*mat.bandwidth + 1)];
      factors->ipiv = new int[width];
      own_data = true;
   }
   a = &mat;
   Factor();
}

void BandMatrixInverse::SetOperator(const Operator &op)
{
   const BandMatrix *p = dynamic_cast<const BandMatrix*>(&op);
   MFEM_VERIFY(p != NULL, "Operator is not a BandMatrix!");
   Factor(*p);
}

void BandMatrixInverse::Mult(const real_t *x, real_t *y) const
{
   for (int row = 0; row < height; row++)
   {
      y[row] = x[row];
   }
   factors->Solve(width, 1, y);
}

void BandMatrixInverse::Mult(const Vector &x, Vector &y) const
{
   y = x;
   factors->Solve(width, 1, y.GetData());
}

void BandMatrixInverse::Mult(const DenseMatrix &B, DenseMatrix &X) const
{
   X = B;
   factors->Solve(width, X.Width(), X.Data());
}

void BandMatrixInverse::TestInversion()
{
   DenseMatrix C(width);
   Mult(a->ToDenseMatrix(), C);
   for (int i = 0; i < width; i++)
   {
      C(i,i) -= 1.0;
   }
   mfem::out << "size = " << width << ", i_max = " << C.MaxMaxNorm() << std::endl;
}

BandMatrixInverse::~BandMatrixInverse()
{
   if (own_data)
   {
      delete [] factors->data;
      delete [] factors->ipiv;
   }
   delete factors;
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

}
