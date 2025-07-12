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
#ifdef MFEM_USE_LAPACK
   static const int ipiv_base = 1;
#else
   static const int ipiv_base = 0;
#endif

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

class StdComplexDenseMatrix
{
protected:
   int height; ///< Dimension of the output / number of rows in the matrix.
   int width;  ///< Dimension of the input / number of columns in the matrix.

private:
   Memory<std::complex<real_t> > data;

   mutable DenseMatrix re_part;
   mutable DenseMatrix im_part;

public:
   /** Default constructor for DenseMatrix.
       Sets data = NULL and height = width = 0. */
   StdComplexDenseMatrix();

   /// Copy constructor
   StdComplexDenseMatrix(const StdComplexDenseMatrix &);

   /// Creates square matrix of size s.
   explicit StdComplexDenseMatrix(int s);

   /// Creates rectangular matrix of size m x n.
   StdComplexDenseMatrix(int m, int n);

   /// Construct a StdComplexDenseMatrix using an existing data array.
   /** The StdComplexDenseMatrix does not assume ownership of the data array,
       i.e. it will not delete the array. */
   StdComplexDenseMatrix(std::complex<real_t> *d, int h, int w)
      : height(h), width(w) { UseExternalData(d, h, w); }

   /// Create a dense matrix using a braced initializer list
   /// The inner lists correspond to rows of the matrix
   template <int M, int N, typename T = real_t>
   explicit StdComplexDenseMatrix(const T (&values)[M][N]) : StdComplexDenseMatrix(
         M, N)
   {
      // DenseMatrix is column-major so copies have to be element-wise
      for (int i = 0; i < M; i++)
      {
         for (int j = 0; j < N; j++)
         {
            (*this)(i,j) = values[i][j];
         }
      }
   }

   /// Change the data array and the size of the DenseMatrix.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the data array @a d. This method should not be used with
       DenseMatrix that owns its current data array. */
   void UseExternalData(std::complex<real_t> *d, int h, int w)
   {
      data.Wrap(d, h*w, false);
      height = h; width = w;
   }

   /// Change the data array and the size of the DenseMatrix.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the new array @a d. This method will delete the current data
       array, if owned. */
   void Reset(std::complex<real_t> *d, int h, int w)
   { if (OwnsData()) { data.Delete(); } UseExternalData(d, h, w); }

   /** Clear the data array and the dimensions of the DenseMatrix. This method
       should not be used with DenseMatrix that owns its current data array. */
   void ClearExternalData() { data.Reset(); height = width = 0; }

   /// Delete the matrix data array (if owned) and reset the matrix state.
   void Clear()
   { if (OwnsData()) { data.Delete(); } ClearExternalData(); }

   /// Get the height (size of output) of the Operator. Synonym with NumRows().
   inline int Height() const { return height; }
   /** @brief Get the number of rows (size of output) of the Operator. Synonym
       with Height(). */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols().
   inline int Width() const { return width; }
   /** @brief Get the number of columns (size of input) of the Operator. Synonym
       with Width(). */
   inline int NumCols() const { return width; }

   /// For backward compatibility define Size to be synonym of Width()
   int Size() const { return Width(); }

   // Total size = width*height
   int TotalSize() const { return width*height; }

   /// Change the size of the DenseMatrix to s x s.
   void SetSize(int s) { SetSize(s, s); }

   /// Change the size of the DenseMatrix to h x w.
   void SetSize(int h, int w);

   /// Returns the matrix data array.
   inline std::complex<real_t> *Data() const
   {
      return const_cast<std::complex<real_t>*>
             ((const std::complex<real_t>*)data);
   }

   /// Returns the matrix data array.
   inline std::complex<real_t> *GetData() const { return Data(); }

   Memory<std::complex<real_t> > &GetMemory() { return data; }
   const Memory<std::complex<real_t> > &GetMemory() const { return data; }

   /// Return the DenseMatrix data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Returns reference to a_{ij}.
   inline std::complex<real_t> &operator()(int i, int j);

   /// Returns constant reference to a_{ij}.
   inline const std::complex<real_t> &operator()(int i, int j) const;

   /// Returns reference to a_{ij}.
   std::complex<real_t> &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
   const std::complex<real_t> &Elem(int i, int j) const;

   /// Sets the matrix elements equal to constant c
   StdComplexDenseMatrix &operator=(real_t c);
   StdComplexDenseMatrix &operator=(std::complex<real_t> c);

   /// Copy the matrix entries from the given array
   StdComplexDenseMatrix &operator=(const real_t *d);
   StdComplexDenseMatrix &operator=(const std::complex<real_t> *d);

   /// Sets the matrix size and elements equal to those of m
   StdComplexDenseMatrix &operator=(const DenseMatrix &m);
   StdComplexDenseMatrix &operator=(const StdComplexDenseMatrix &m);

   StdComplexDenseMatrix &operator+=(const real_t *m);
   StdComplexDenseMatrix &operator+=(const std::complex<real_t> *m);
   StdComplexDenseMatrix &operator+=(const DenseMatrix &m);
   StdComplexDenseMatrix &operator+=(const StdComplexDenseMatrix &m);

   StdComplexDenseMatrix &operator-=(const DenseMatrix &m);
   StdComplexDenseMatrix &operator-=(const StdComplexDenseMatrix &m);

   StdComplexDenseMatrix &operator*=(real_t c);
   StdComplexDenseMatrix &operator*=(std::complex<real_t> c);

   /// (*this) = x + i * y
   StdComplexDenseMatrix &Set(const DenseMatrix &x, const DenseMatrix &y);

   std::size_t MemoryUsage() const
   { return data.Capacity() * sizeof(std::complex<real_t>); }

   /// Shortcut for mfem::Read( GetMemory(), TotalSize(), on_dev).
   const std::complex<real_t> *Read(bool on_dev = true) const
   { return mfem::Read(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), TotalSize(), false).
   const std::complex<real_t> *HostRead() const
   { return mfem::Read(data, Height()*Width(), false); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), on_dev).
   std::complex<real_t> *Write(bool on_dev = true)
   { return mfem::Write(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), false).
   std::complex<real_t> *HostWrite()
   { return mfem::Write(data, Height()*Width(), false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), on_dev).
   std::complex<real_t> *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), false).
   std::complex<real_t> *HostReadWrite()
   { return mfem::ReadWrite(data, Height()*Width(), false); }

   void Swap(StdComplexDenseMatrix &other);

   /// Return a reference to the real part of this matrix
   const DenseMatrix &real() const;

   /// Return a reference to the imaginary part of this matrix
   const DenseMatrix &imag() const;

   /// Destroys dense matrix.
   virtual ~StdComplexDenseMatrix();
};

/// Specialization of the template function Swap<> for class StdComplexDenseMatrix
template<> inline void Swap<StdComplexDenseMatrix>(StdComplexDenseMatrix &a,
                                                   StdComplexDenseMatrix &b)
{
   a.Swap(b);
}

// Inline methods

inline std::complex<real_t> &StdComplexDenseMatrix::operator()(int i, int j)
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

inline const std::complex<real_t> &StdComplexDenseMatrix::operator()
(int i, int j) const
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

} // namespace mfem

#endif // MFEM_COMPLEX_DENSEMAT
