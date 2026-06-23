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

#ifndef MFEM_BANDMATRIX
#define MFEM_BANDMATRIX

#include "../config/config.hpp"
#include "../general/globals.hpp"
#include "matrix.hpp"
#include "densemat.hpp"

namespace mfem
{

/// Dense symmetric matrix storing the upper triangular part. This class so far
/// has little functionality beyond storage.
class BandMatrix : public Matrix
{
   friend class BandMatrixInverse;

private:
   Array<real_t> data;
   int bandwidth, stride;
   real_t zero = -0.0;

public:

   /** Default constructor forBandMatrix.
       Sets data = NULL and height = width = 0. */
   BandMatrix();

   /// Creates square matrix of size s and bandwidth bw.
   explicit BandMatrix(int s, int bw);

   /// Creates a matrix of the given height, width and bandwidth bw.
   explicit BandMatrix(int h, int w, int bw);

   /// Creates a band matrix from a given DenseMartix.
   /** If @a bw is not specified (or negative) then the bandwidth is determined.
       If @a bw is specified this bandwidth is used.*/
   explicit BandMatrix(const DenseMatrix &dm, int bw = -1);

   /// Copy constructor: default
   BandMatrix(const BandMatrix &) = default;

   /// Move constructor: default
   BandMatrix(BandMatrix &&) = default;

   /// Copy assignment (deep copy).
   BandMatrix &operator=(const BandMatrix &) = default;

   /// Move assignment.
   BandMatrix &operator=(BandMatrix &&) = default;

   /// Construct aBandMatrix using an existing data array.
   /** The BandMatrix does not assume ownership of the data array,
       i.e. it will not delete the array. */
   BandMatrix(real_t *d, int s, int bw)
      : Matrix(s, s) { UseExternalData(d, s, s, bw); }

   BandMatrix(real_t *d, int h, int w, int bw)
      : Matrix(h, w) { UseExternalData(d, h, w, bw); }

   /// Change the data array and the size of the BandMatrix.
   /** TheBandMatrix does not assume ownership of the data array,
       i.e. it will not delete the data array @a d. This method should not be
       used withBandMatrix that owns its current data array. */
   void UseExternalData(real_t *d, int h, int w, int bw)
   {
      height = h; width = w; bandwidth = bw; stride = 2*bw + 1;
      data.MakeRef(d, h*stride);
   }

   /// Creates a band matrix from a given DenseMartix.
   /** If @a bw is not specified (or negative) then the bandwidth is determined.
       If @a bw is specified this bandwidth is used.*/
   void Reset(const DenseMatrix &dm, int bw = -1);

   /// Change the data array and the size of the BandMatrix.
   /** The BandMatrix does not assume ownership of the data array,
       i.e. it will not delete the new array @a d. This method will delete the
       current data array, if owned. */
   void Reset(real_t *d, int s, int bw)
   { UseExternalData(d, s, s, bw); }

   void Reset(real_t *d, int h, int w, int bw)
   { UseExternalData(d, h, w, bw); }

   /** Clear the data array and the dimensions of the BandMatrix. This
       method should not be used withBandMatrix that owns its current
       data array. */
   void ClearExternalData() { data.LoseData();
                              height = width = bandwidth = stride = 0; }

   /// Delete the matrix data array (if owned) and reset the matrix state.
   void Clear() { data.DeleteAll();
                  height = width = bandwidth = stride = 0; }


   /// Change the size of the BandMatrix to s x s.
   int GetBandWidth() { return bandwidth; };

   /// Change the size of the BandMatrix to s x s.
   void SetSize(int s, int bw);
   void SetSize(int h, int w, int bw);

   /// Return the number of stored nonzeros in the matrix.
   int GetStoredSize() const { return Height()*stride; }

   /// Returns the matrix data array.
   inline real_t *Data() const
   { return const_cast<real_t*>((const real_t*)data);}

   /// Returns the matrix data array.
   inline real_t *GetData() const { return Data(); }

   Memory<real_t> &GetMemory() { return data.GetMemory(); }
   const Memory<real_t> &GetMemory() const { return data.GetMemory(); }

   /// Return theBandMatrix data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsData(); }

   /// Returns reference to a_{ij}.
   inline real_t &operator()(int i, int j);

   /// Returns constant reference to a_{ij}.
   inline const real_t &operator()(int i, int j) const;

   /// Returns reference to a_{ij}.
   real_t &Elem(int i, int j) override;

   /// Returns constant reference to a_{ij}.
   const real_t &Elem(int i, int j) const override;

   /// Sets the matrix elements equal to constant c
   BandMatrix &operator=(real_t c);

   BandMatrix &operator*=(real_t c);

   std::size_t MemoryUsage() const { return data.Capacity() * sizeof(real_t); }

   /// Shortcut for mfem::Read(GetMemory(), GetStoredSize(), on_dev).
   const real_t *Read(bool on_dev = true) const { return data.Read(on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), GetStoredSize(), false).
   const real_t *HostRead() const { return data.Read(false); }

   /// Shortcut for mfem::Write(GetMemory(), GetStoredSize(), on_dev).
   real_t *Write(bool on_dev = true) { return data.Write(on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), GetStoredSize(), false).
   real_t *HostWrite() { return data.Write(false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), GetStoredSize(), on_dev).
   real_t *ReadWrite(bool on_dev = true) { return data.ReadWrite(on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), GetStoredSize(), false).
   real_t *HostReadWrite() { return data.ReadWrite(false); }

   /// Matrix vector multiplication.
   void Mult(const real_t *x, real_t *y) const;

   /// Matrix vector multiplication.
   void Mult(const real_t *x, Vector &y) const;

   /// Matrix vector multiplication.
   void Mult(const Vector &x, real_t *y) const;

   /// Matrix vector multiplication.
   void Mult(const Vector &x, Vector &y) const override;

   /// Returns a pointer to (an approximation) of the matrix inverse.
   MatrixInverse *Inverse() const override;

   /// Returns a reference to BandMatrix as DenseMatrix.
   DenseMatrix ToDenseMatrix() const;

   /// Replaces the current matrix with its inverse
   /** If @a tol is not specified (or is negative) the exact inverse is
       computed.  If @a tol is specified the reuslting matrix has the minimum
       bandwidth to achieve tolerance, when computing the Frobenius norm of 
       approc_inv(A)*A - I
       If @a bw is this bandwidth is used, the result is compared with the tolerance.*/
   void Invert(real_t tol = -1.0, int bw = -1);


   /// Replaces the given DenseMatrix @ dm with the inverse
   void Inverse(DenseMatrix &dm);

   /// Prints matrix to stream os.
   void Print(std::ostream & os = mfem::out, int width_ = 4) const override;

};

/// Matrix matrix multiplication.  A = B * C.
void Mult(const BandMatrix &b,  const DenseMatrix &c, DenseMatrix &a);
void Mult(const DenseMatrix &b, const BandMatrix &c,  DenseMatrix &a);
void Mult(const BandMatrix &b,  const BandMatrix &c,  DenseMatrix &a);


/// C = A + beta*B
void Add(const DenseMatrix &A, const BandMatrix &B,
         real_t alpha, DenseMatrix &C);
void Add(const BandMatrix &A, const DenseMatrix &B,
         real_t alpha, DenseMatrix &C);

/// C = A + alpha*A + beta*B
void Add(real_t alpha, const DenseMatrix &A,
         real_t beta,  const BandMatrix &B, DenseMatrix &C);
void Add(real_t alpha, const BandMatrix &A,
         real_t beta,  const DenseMatrix &B, DenseMatrix &C);

/** Class that can compute LU factorization of external data and perform various
    operations with the factored data. */
class BandLUFactors : public Factors
{
public:
   int *ipiv;
   static constexpr int ipiv_base = 1;
   int bw;

   /** With this constructor, the (public) data and ipiv members should be set
       explicitly before calling class methods. */
   BandLUFactors(): Factors() { }

   BandLUFactors(real_t *data_, int *ipiv_) : Factors(data_), ipiv(ipiv_) { }

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
   real_t Det(int m) const override;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, real_t *X) const override;

   /// Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, real_t *X) const override;
};


/** Data type for inverse of square band matrix.
    Stores LU matrix factors */
class BandMatrixInverse : public MatrixInverse
{
private:
   const BandMatrix *a;
   BandLUFactors * factors = nullptr;

   void Init(int m, int bw);
   bool own_data = false;
public:
   /// Default constructor.
   BandMatrixInverse(bool spd_=false) : a(NULL){ Init(0, 0); }

   /** Creates square dense matrix. Computes factorization of mat
       and stores its factors. */
   BandMatrixInverse(const BandMatrix &mat);

   /// Same as above but does not factorize the matrix.
   BandMatrixInverse(const BandMatrix *mat);

   ///  Get the size of the inverse matrix
   int Size() const { return Width(); }

   /// Factor the current BandMatrix, *a
   void Factor();

   /// Factor a new BandMatrix of the same size
   void Factor(const BandMatrix &mat);

   void SetOperator(const Operator &op) override;

   /// Matrix vector multiplication with the inverse of dense matrix.
   void Mult(const real_t *x, real_t *y) const;

   /// Matrix vector multiplication with the inverse of dense matrix.
   void Mult(const Vector &x, Vector &y) const override;

   /// Multiply the inverse matrix by another matrix: X = A^{-1} B.
   void Mult(const DenseMatrix &B, DenseMatrix &X) const;

   /// Multiply the inverse matrix by another matrix: X <- A^{-1} X.
   void Mult(DenseMatrix &X) const {factors->Solve(width, X.Width(), X.Data());}

   using Operator::Mult;

   /// Compute and return the inverse matrix in Ainv.
   void GetInverseMatrix(DenseMatrix &Ainv) const;

   /// Compute the determinant of the original DenseMatrix using the LU factors.
   real_t Det() const override { return factors->Det(width); }

   /// Print the numerical conditioning of the inversion: ||A^{-1} A - I||.
   void TestInversion();

   /// Destroys dense inverse matrix.
   virtual ~BandMatrixInverse();
};



#ifdef MFEM_USE_LAPACK
void BandedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                 Array<int> &ipiv);
void BandedFactorizedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                           bool transpose, Array<int> &ipiv);
#endif

// Inline methods

inline real_t &BandMatrix ::operator()(int i, int j)
{
   MFEM_ASSERT(data && i >= 0 && i < height
                    && j >= 0 && j < width
                    && j >= i - bandwidth && j <= i + bandwidth, "");
   return data[j - i + bandwidth + i*stride];
}

inline const real_t &BandMatrix ::operator()(int i, int j) const
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   if (j >= i - bandwidth && j <= i + bandwidth)
   {
      return data[j - i + bandwidth + i*stride];
   }
   else
   {
      return zero;
   }
}

} // namespace mfem

#endif
