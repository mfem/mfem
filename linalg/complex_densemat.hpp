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

#ifndef MFEM_COMPLEX_DENSEMAT
#define MFEM_COMPLEX_DENSEMAT

#include "complex_operator.hpp"
#include <complex>

namespace mfem
{

/** @brief Specialization of the ComplexOperator built from a pair of
    Dense Matrices.  The purpose of this specialization is to support
    the inverse of a ComplexDenseMatrix and various MatMat operations
    See ComplexOperator documentation for more information.
 */
class ComplexDenseMatrix : public ComplexOperator
{

public:
   ComplexDenseMatrix(DenseMatrix * A_Real, DenseMatrix * A_Imag,
                      bool ownReal, bool ownImag, Convention convention = HERMITIAN)
      : ComplexOperator(A_Real, A_Imag, ownReal, ownImag, convention)
   { }

   DenseMatrix & real() override;
   DenseMatrix & imag() override;

   const DenseMatrix & real() const override;
   const DenseMatrix & imag() const override;

   /** Combine the blocks making up this complex operator into a single
       DenseMatrix. Note that this combined operator requires roughly
       twice the memory of the block structured operator. */
   DenseMatrix * GetSystemMatrix() const;

   Type GetType() const override { return Complex_DenseMat; }

   ComplexDenseMatrix * ComputeInverse();

};

/// Matrix matrix multiplication.  A = B * C.
ComplexDenseMatrix * Mult(const ComplexDenseMatrix &B,
                          const ComplexDenseMatrix &C);

/// Multiply the complex conjugate transpose of a matrix A with a matrix B. A^H*B
ComplexDenseMatrix * MultAtB(const ComplexDenseMatrix &A,
                             const ComplexDenseMatrix &B);


/** Abstract class that can compute factorization of external data and perform various
    operations with the factored data. */
class ComplexFactors
{
protected:

   // returns a new complex array
   std::complex<real_t> * RealToComplex(int m, const real_t * x_r,
                                        const real_t * x_i) const;
   // copies the given complex array to real and imag arrays
   void ComplexToReal(int m, const std::complex<real_t> * x, real_t * x_r,
                      real_t * x_i) const;

public:

   real_t *data_r = nullptr;
   real_t *data_i = nullptr;
   std::complex<real_t> * data = nullptr;

   ComplexFactors() { }

   ComplexFactors(real_t *data_r_, real_t *data_i_)
      : data_r(data_r_), data_i(data_i_) { }

   void SetComplexData(int m);

   void ResetComplexData(int m)
   {
      delete [] data; data = nullptr;
      SetComplexData(m);
   }

   virtual bool Factor(int m, real_t TOL = 0.0)
   {
      mfem_error("ComplexFactors::ComplexFactors(...)");
      return false;
   }

   virtual std::complex<real_t> Det(int m) const
   {
      mfem_error("Factors::Det(...)");
      return 0.;
   }

   virtual void Solve(int m, int n, real_t *X_r, real_t * X_i) const
   {
      mfem_error("Factors::Solve(...)");
   }

   virtual void GetInverseMatrix(int m, real_t *X_r, real_t * X_i) const
   {
      mfem_error("Factors::GetInverseMatrix(...)");
   }

   virtual ~ComplexFactors()
   {
      delete [] data; data = nullptr;
   }
};

/** Class that computes factorization of external data and perform various
    operations with the factored data. */
class ComplexLUFactors : public ComplexFactors
{
public:
   int *ipiv;
   static constexpr int ipiv_base = 1;

   /** With this constructor, the (public) data and ipiv members should be set
       explicitly before calling class methods. */
   ComplexLUFactors(): ComplexFactors() { }

   ComplexLUFactors(real_t *data_r_,real_t * data_i, int *ipiv_)
      : ComplexFactors(data_r_, data_i), ipiv(ipiv_) { }
   /**
    * @brief Compute the LU factorization of the current matrix
    *
    * Factorize the current matrix of size (m x m) overwriting it with the
    * LU factors. The factorization is such that L.U = P.A, where A is the
    * original matrix and P is a permutation matrix represented by ipiv.
    *
    * @param [in] m size of the square matrix
    * @param [in] TOL optional fuzzy comparison tolerance. Defaults to 0.0.
    *
    * @return status set to true if successful, otherwise, false.
    */
   bool Factor(int m, real_t TOL = 0.0) override;

   /** Assuming L.U = P.A factored data of size (m x m), compute |A|
       from the diagonal values of U and the permutation information. */
   std::complex<real_t> Det(int m) const override;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A X,
       for a matrix X of size (m x n). */
   void Mult(int m, int n, real_t *X_r, real_t * X_i) const;

   void Mult(int m, int n, std::complex<real_t> *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- L^{-1} P X, for a matrix X of size (m x n). */
   void LSolve(int m, int n, real_t *X_r, real_t *X_i) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- U^{-1} X, for a matrix X of size (m x n). */
   void USolve(int m, int n, real_t *X_r, real_t *X_i) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, real_t *X_r, real_t *X_i) const override;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- X A^{-1},
       for a matrix X of size (n x m). */
   void RightSolve(int m, int n, real_t *X_r, real_t *X_i) const;

   /// Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, real_t *X_r, real_t * X_i) const override;
};


/** Class that can compute Cholesky factorizations of external data of an
    Hermitian positive matrix and perform various operations with the factored data. */
class ComplexCholeskyFactors : public ComplexFactors
{
public:

   /** With this constructor, the (public) data should be set
       explicitly before calling class methods. */
   ComplexCholeskyFactors() : ComplexFactors() { }

   ComplexCholeskyFactors(real_t *data_r_, real_t * data_i_)
      : ComplexFactors(data_r_, data_i_) { }

   /**
    * @brief Compute the Cholesky factorization of the current matrix
    *
    * Factorize the current matrix of size (m x m) overwriting it with the
    * Cholesky factors. The factorization is such that LL^H = A, where A is the
    * original matrix
    *
    * @param [in] m size of the square matrix
    * @param [in] TOL optional fuzzy comparison tolerance. Defaults to 0.0.
    *
    * @return status set to true if successful, otherwise, false.
    */
   bool Factor(int m, real_t TOL = 0.0) override;

   /** Assuming LL^H = A factored data of size (m x m), compute |A|
       from the diagonal values of L */
   std::complex<real_t> Det(int m) const override;

   /** Assuming L.L^H = A factored data of size (m x m), compute X <- L X,
       for a matrix X of size (m x n). */
   void LMult(int m, int n, real_t *X_r, real_t * X_i) const;

   /** Assuming L.L^H = A factored data of size (m x m), compute X <- L^t X,
       for a matrix X of size (m x n). */
   void UMult(int m, int n, real_t *X_r, real_t *X_i) const;

   /** Assuming L L^H = A factored data of size (m x m), compute
       X <- L^{-1} X, for a matrix X of size (m x n). */
   void LSolve(int m, int n, real_t *X_r, real_t * X_i) const;

   /** Assuming L L^H = A factored data of size (m x m), compute
       X <- L^{-t} X, for a matrix X of size (m x n). */
   void USolve(int m, int n, real_t *X_r, real_t *X_i) const;

   /** Assuming L.L^H = A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, real_t *X_r, real_t * X_i) const override;

   /** Assuming L.L^H = A factored data of size (m x m), compute X <- X A^{-1},
       for a matrix X of size (n x m). */
   void RightSolve(int m, int n, real_t *X_r, real_t *X_i) const;

   /// Assuming L.L^H = A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, real_t *X_r, real_t * X_i) const override;

};

} // namespace mfem

#endif // MFEM_COMPLEX_DENSEMAT
