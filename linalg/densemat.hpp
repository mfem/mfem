// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DENSEMAT
#define MFEM_DENSEMAT

#include "../config/config.hpp"
#include "../general/globals.hpp"
#include "matrix.hpp"

namespace mfem
{

/// Data type dense matrix using column-major storage
class DenseMatrix : public Matrix
{
   friend class DenseTensor;
   friend class DenseMatrixInverse;

private:
   Memory<double> data;

   void Eigensystem(Vector &ev, DenseMatrix *evect = NULL);

   void Eigensystem(DenseMatrix &b, Vector &ev, DenseMatrix *evect = NULL);

   // Auxiliary method used in FNorm2() and FNorm()
   void FNorm(double &scale_factor, double &scaled_fnorm2) const;

public:
   /** Default constructor for DenseMatrix.
       Sets data = NULL and height = width = 0. */
   DenseMatrix();

   /// Copy constructor
   DenseMatrix(const DenseMatrix &);

   /// Creates square matrix of size s.
   explicit DenseMatrix(int s);

   /// Creates rectangular matrix of size m x n.
   DenseMatrix(int m, int n);

   /// Creates rectangular matrix equal to the transpose of mat.
   DenseMatrix(const DenseMatrix &mat, char ch);

   /// Construct a DenseMatrix using an existing data array.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the array. */
   DenseMatrix(double *d, int h, int w)
      : Matrix(h, w) { UseExternalData(d, h, w); }

   /// Change the data array and the size of the DenseMatrix.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the data array @a d. This method should not be used with
       DenseMatrix that owns its current data array. */
   void UseExternalData(double *d, int h, int w)
   {
      data.Wrap(d, h*w, false);
      height = h; width = w;
   }

   /// Change the data array and the size of the DenseMatrix.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the new array @a d. This method will delete the current data
       array, if owned. */
   void Reset(double *d, int h, int w)
   { if (OwnsData()) { data.Delete(); } UseExternalData(d, h, w); }

   /** Clear the data array and the dimensions of the DenseMatrix. This method
       should not be used with DenseMatrix that owns its current data array. */
   void ClearExternalData() { data.Reset(); height = width = 0; }

   /// Delete the matrix data array (if owned) and reset the matrix state.
   void Clear()
   { if (OwnsData()) { data.Delete(); } ClearExternalData(); }

   /// For backward compatibility define Size to be synonym of Width()
   int Size() const { return Width(); }

   /// Change the size of the DenseMatrix to s x s.
   void SetSize(int s) { SetSize(s, s); }

   /// Change the size of the DenseMatrix to h x w.
   void SetSize(int h, int w);

   /// Returns the matrix data array.
   inline double *Data() const
   { return const_cast<double*>((const double*)data);}

   /// Returns the matrix data array.
   inline double *GetData() const { return Data(); }

   Memory<double> &GetMemory() { return data; }
   const Memory<double> &GetMemory() const { return data; }

   /// Return the DenseMatrix data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Returns reference to a_{ij}.
   inline double &operator()(int i, int j);

   /// Returns constant reference to a_{ij}.
   inline const double &operator()(int i, int j) const;

   /// Matrix inner product: tr(A^t B)
   double operator*(const DenseMatrix &m) const;

   /// Trace of a square matrix
   double Trace() const;

   /// Returns reference to a_{ij}.
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   void Mult(const double *x, double *y) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Multiply a vector with the transpose matrix.
   void MultTranspose(const double *x, double *y) const;

   /// Multiply a vector with the transpose matrix.
   virtual void MultTranspose(const Vector &x, Vector &y) const;

   /// y += A.x
   void AddMult(const Vector &x, Vector &y) const;

   /// y += A^t x
   void AddMultTranspose(const Vector &x, Vector &y) const;

   /// y += a * A.x
   void AddMult_a(double a, const Vector &x, Vector &y) const;

   /// y += a * A^t x
   void AddMultTranspose_a(double a, const Vector &x, Vector &y) const;

   /// Compute y^t A x
   double InnerProduct(const double *x, const double *y) const;

   /// LeftScaling this = diag(s) * this
   void LeftScaling(const Vector & s);
   /// InvLeftScaling this = diag(1./s) * this
   void InvLeftScaling(const Vector & s);
   /// RightScaling: this = this * diag(s);
   void RightScaling(const Vector & s);
   /// InvRightScaling: this = this * diag(1./s);
   void InvRightScaling(const Vector & s);
   /// SymmetricScaling this = diag(sqrt(s)) * this * diag(sqrt(s))
   void SymmetricScaling(const Vector & s);
   /// InvSymmetricScaling this = diag(sqrt(1./s)) * this * diag(sqrt(1./s))
   void InvSymmetricScaling(const Vector & s);

   /// Compute y^t A x
   double InnerProduct(const Vector &x, const Vector &y) const
   { return InnerProduct((const double *)x, (const double *)y); }

   /// Returns a pointer to the inverse matrix.
   virtual MatrixInverse *Inverse() const;

   /// Replaces the current matrix with its inverse
   void Invert();

   /// Replaces the current matrix with its square root inverse
   void SquareRootInverse();

   /// Calculates the determinant of the matrix
   /// (optimized for 2x2, 3x3, and 4x4 matrices)
   double Det() const;

   double Weight() const;

   /** @brief Set the matrix to alpha * A, assuming that A has the same
       dimensions as the matrix and uses column-major layout. */
   void Set(double alpha, const double *A);
   /// Set the matrix to alpha * A.
   void Set(double alpha, const DenseMatrix &A)
   {
      SetSize(A.Height(), A.Width());
      Set(alpha, A.GetData());
   }

   /// Adds the matrix A multiplied by the number c to the matrix
   void Add(const double c, const DenseMatrix &A);

   /// Sets the matrix elements equal to constant c
   DenseMatrix &operator=(double c);

   /// Copy the matrix entries from the given array
   DenseMatrix &operator=(const double *d);

   /// Sets the matrix size and elements equal to those of m
   DenseMatrix &operator=(const DenseMatrix &m);

   DenseMatrix &operator+=(const double *m);
   DenseMatrix &operator+=(const DenseMatrix &m);

   DenseMatrix &operator-=(const DenseMatrix &m);

   DenseMatrix &operator*=(double c);

   /// (*this) = -(*this)
   void Neg();

   /// Take the 2-norm of the columns of A and store in v
   void Norm2(double *v) const;

   /// Compute the norm ||A|| = max_{ij} |A_{ij}|
   double MaxMaxNorm() const;

   /// Compute the Frobenius norm of the matrix
   double FNorm() const { double s, n2; FNorm(s, n2); return s*sqrt(n2); }

   /// Compute the square of the Frobenius norm of the matrix
   double FNorm2() const { double s, n2; FNorm(s, n2); return s*s*n2; }

   /// Compute eigenvalues of A x = ev x where A = *this
   void Eigenvalues(Vector &ev)
   { Eigensystem(ev); }

   /// Compute eigenvalues and eigenvectors of A x = ev x where A = *this
   void Eigenvalues(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }

   /// Compute eigenvalues and eigenvectors of A x = ev x where A = *this
   void Eigensystem(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }

   /** Compute generalized eigenvalues and eigenvectors of A x = ev B x,
       where A = *this */
   void Eigenvalues(DenseMatrix &b, Vector &ev)
   { Eigensystem(b, ev); }

   /// Compute generalized eigenvalues of A x = ev B x, where A = *this
   void Eigenvalues(DenseMatrix &b, Vector &ev, DenseMatrix &evect)
   { Eigensystem(b, ev, &evect); }

   /** Compute generalized eigenvalues and eigenvectors of A x = ev B x,
       where A = *this */
   void Eigensystem(DenseMatrix &b, Vector &ev, DenseMatrix &evect)
   { Eigensystem(b, ev, &evect); }

   void SingularValues(Vector &sv) const;
   int Rank(double tol) const;

   /// Return the i-th singular value (decreasing order) of NxN matrix, N=1,2,3.
   double CalcSingularvalue(const int i) const;

   /** Return the eigenvalues (in increasing order) and eigenvectors of a
       2x2 or 3x3 symmetric matrix. */
   void CalcEigenvalues(double *lambda, double *vec) const;

   void GetRow(int r, Vector &row) const;
   void GetColumn(int c, Vector &col) const;
   double *GetColumn(int col) { return data + col*height; }
   const double *GetColumn(int col) const { return data + col*height; }

   void GetColumnReference(int c, Vector &col)
   { col.SetDataAndSize(data + c * height, height); }

   void SetRow(int r, const double* row);
   void SetRow(int r, const Vector &row);

   void SetCol(int c, const double* col);
   void SetCol(int c, const Vector &col);


   /// Set all entries of a row to the specified value.
   void SetRow(int row, double value);
   /// Set all entries of a column to the specified value.
   void SetCol(int col, double value);

   /// Returns the diagonal of the matrix
   void GetDiag(Vector &d) const;
   /// Returns the l1 norm of the rows of the matrix v_i = sum_j |a_ij|
   void Getl1Diag(Vector &l) const;
   /// Compute the row sums of the DenseMatrix
   void GetRowSums(Vector &l) const;

   /// Creates n x n diagonal matrix with diagonal elements c
   void Diag(double c, int n);
   /// Creates n x n diagonal matrix with diagonal given by diag
   void Diag(double *diag, int n);

   /// (*this) = (*this)^t
   void Transpose();
   /// (*this) = A^t
   void Transpose(const DenseMatrix &A);
   /// (*this) = 1/2 ((*this) + (*this)^t)
   void Symmetrize();

   void Lump();

   /** Given a DShape matrix (from a scalar FE), stored in *this, returns the
       CurlShape matrix. If *this is a N by D matrix, then curl is a D*N by
       D*(D-1)/2 matrix. The size of curl must be set outside. The dimension D
       can be either 2 or 3. */
   void GradToCurl(DenseMatrix &curl);
   /** Given a DShape matrix (from a scalar FE), stored in *this,
       returns the DivShape vector. If *this is a N by dim matrix,
       then div is a dim*N vector. The size of div must be set
       outside.  */
   void GradToDiv(Vector &div);

   /// Copy rows row1 through row2 from A to *this
   void CopyRows(const DenseMatrix &A, int row1, int row2);
   /// Copy columns col1 through col2 from A to *this
   void CopyCols(const DenseMatrix &A, int col1, int col2);
   /// Copy the m x n submatrix of A at row/col offsets Aro/Aco to *this
   void CopyMN(const DenseMatrix &A, int m, int n, int Aro, int Aco);
   /// Copy matrix A to the location in *this at row_offset, col_offset
   void CopyMN(const DenseMatrix &A, int row_offset, int col_offset);
   /// Copy matrix A^t to the location in *this at row_offset, col_offset
   void CopyMNt(const DenseMatrix &A, int row_offset, int col_offset);
   /** Copy the m x n submatrix of A at row/col offsets Aro/Aco to *this at
       row_offset, col_offset */
   void CopyMN(const DenseMatrix &A, int m, int n, int Aro, int Aco,
               int row_offset, int col_offset);
   /// Copy c on the diagonal of size n to *this at row_offset, col_offset
   void CopyMNDiag(double c, int n, int row_offset, int col_offset);
   /// Copy diag on the diagonal of size n to *this at row_offset, col_offset
   void CopyMNDiag(double *diag, int n, int row_offset, int col_offset);
   /// Copy All rows and columns except m and n from A
   void CopyExceptMN(const DenseMatrix &A, int m, int n);

   /// Perform (ro+i,co+j)+=A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(DenseMatrix &A, int ro, int co);
   /// Perform (ro+i,co+j)+=a*A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(double a, const DenseMatrix &A, int ro, int co);

   /// Add the matrix 'data' to the Vector 'v' at the given 'offset'
   void AddToVector(int offset, Vector &v) const;
   /// Get the matrix 'data' from the Vector 'v' at the given 'offset'
   void GetFromVector(int offset, const Vector &v);
   /** If (dofs[i] < 0 and dofs[j] >= 0) or (dofs[i] >= 0 and dofs[j] < 0)
       then (*this)(i,j) = -(*this)(i,j).  */
   void AdjustDofDirection(Array<int> &dofs);

   /// Replace small entries, abs(a_ij) <= eps, with zero.
   void Threshold(double eps);

   /** Count the number of entries in the matrix for which isfinite
       is false, i.e. the entry is a NaN or +/-Inf. */
   int CheckFinite() const { return mfem::CheckFinite(data, height*width); }

   /// Prints matrix to stream out.
   virtual void Print(std::ostream &out = mfem::out, int width_ = 4) const;
   virtual void PrintMatlab(std::ostream &out = mfem::out) const;
   /// Prints the transpose matrix to stream out.
   virtual void PrintT(std::ostream &out = mfem::out, int width_ = 4) const;

   /// Invert and print the numerical conditioning of the inversion.
   void TestInversion();

   long MemoryUsage() const { return data.Capacity() * sizeof(double); }

   /// Shortcut for mfem::Read( GetMemory(), TotalSize(), on_dev).
   const double *Read(bool on_dev = true) const
   { return mfem::Read(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), TotalSize(), false).
   const double *HostRead() const
   { return mfem::Read(data, Height()*Width(), false); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), on_dev).
   double *Write(bool on_dev = true)
   { return mfem::Write(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), false).
   double *HostWrite()
   { return mfem::Write(data, Height()*Width(), false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), on_dev).
   double *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), false).
   double *HostReadWrite()
   { return mfem::ReadWrite(data, Height()*Width(), false); }

   /// Destroys dense matrix.
   virtual ~DenseMatrix();
};

/// C = A + alpha*B
void Add(const DenseMatrix &A, const DenseMatrix &B,
         double alpha, DenseMatrix &C);

/// C = alpha*A + beta*B
void Add(double alpha, const double *A,
         double beta,  const double *B, DenseMatrix &C);

/// C = alpha*A + beta*B
void Add(double alpha, const DenseMatrix &A,
         double beta,  const DenseMatrix &B, DenseMatrix &C);

/// @brief Solves the dense linear system, `A * X = B` for `X`
///
/// @param [in,out] A the square matrix for the linear system
/// @param [in,out] X the rhs vector, B, on input, the solution, X, on output.
/// @param [in] TOL optional fuzzy comparison tolerance. Defaults to 1e-9.
///
/// @return status set to true if successful, otherwise, false.
///
/// @note This routine may replace the contents of the input Matrix, A, with the
///       corresponding LU factorization of the matrix. Matrices of size 1x1 and
///       2x2 are handled explicitly.
///
/// @pre A.IsSquare() == true
/// @pre X != nullptr
bool LinearSolve(DenseMatrix& A, double* X, double TOL = 1.e-9);

/// Matrix matrix multiplication.  A = B * C.
void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);

/// Matrix matrix multiplication.  A += B * C.
void AddMult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);

/// Matrix matrix multiplication.  A += alpha * B * C.
void AddMult_a(double alpha, const DenseMatrix &b, const DenseMatrix &c,
               DenseMatrix &a);

/** Calculate the adjugate of a matrix (for NxN matrices, N=1,2,3) or the matrix
    adj(A^t.A).A^t for rectangular matrices (2x1, 3x1, or 3x2). This operation
    is well defined even when the matrix is not full rank. */
void CalcAdjugate(const DenseMatrix &a, DenseMatrix &adja);

/// Calculate the transposed adjugate of a matrix (for NxN matrices, N=1,2,3)
void CalcAdjugateTranspose(const DenseMatrix &a, DenseMatrix &adjat);

/** Calculate the inverse of a matrix (for NxN matrices, N=1,2,3) or the
    left inverse (A^t.A)^{-1}.A^t (for 2x1, 3x1, or 3x2 matrices) */
void CalcInverse(const DenseMatrix &a, DenseMatrix &inva);

/// Calculate the inverse transpose of a matrix (for NxN matrices, N=1,2,3)
void CalcInverseTranspose(const DenseMatrix &a, DenseMatrix &inva);

/** For a given Nx(N-1) (N=2,3) matrix J, compute a vector n such that
    n_k = (-1)^{k+1} det(J_k), k=1,..,N, where J_k is the matrix J with the
    k-th row removed. Note: J^t.n = 0, det([n|J])=|n|^2=det(J^t.J). */
void CalcOrtho(const DenseMatrix &J, Vector &n);

/// Calculate the matrix A.At
void MultAAt(const DenseMatrix &a, DenseMatrix &aat);

/// ADAt = A D A^t, where D is diagonal
void MultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);

/// ADAt += A D A^t, where D is diagonal
void AddMultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);

/// Multiply a matrix A with the transpose of a matrix B:   A*Bt
void MultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);

/// ADBt = A D B^t, where D is diagonal
void MultADBt(const DenseMatrix &A, const Vector &D,
              const DenseMatrix &B, DenseMatrix &ADBt);

/// ABt += A * B^t
void AddMultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);

/// ADBt = A D B^t, where D is diagonal
void AddMultADBt(const DenseMatrix &A, const Vector &D,
                 const DenseMatrix &B, DenseMatrix &ADBt);

/// ABt += a * A * B^t
void AddMult_a_ABt(double a, const DenseMatrix &A, const DenseMatrix &B,
                   DenseMatrix &ABt);

/// Multiply the transpose of a matrix A with a matrix B:   At*B
void MultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB);

/// AAt += a * A * A^t
void AddMult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt);

/// AAt = a * A * A^t
void Mult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt);

/// Make a matrix from a vector V.Vt
void MultVVt(const Vector &v, DenseMatrix &vvt);

void MultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);

/// VWt += v w^t
void AddMultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);

/// VVt += v v^t
void AddMultVVt(const Vector &v, DenseMatrix &VWt);

/// VWt += a * v w^t
void AddMult_a_VWt(const double a, const Vector &v, const Vector &w,
                   DenseMatrix &VWt);

/// VVt += a * v v^t
void AddMult_a_VVt(const double a, const Vector &v, DenseMatrix &VVt);


/** Class that can compute LU factorization of external data and perform various
    operations with the factored data. */
class LUFactors
{
public:
   double *data;
   int *ipiv;
#ifdef MFEM_USE_LAPACK
   static const int ipiv_base = 1;
#else
   static const int ipiv_base = 0;
#endif

   /** With this constructor, the (public) data and ipiv members should be set
       explicitly before calling class methods. */
   LUFactors() { }

   LUFactors(double *data_, int *ipiv_) : data(data_), ipiv(ipiv_) { }

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
   bool Factor(int m, double TOL = 0.0);

   /** Assuming L.U = P.A factored data of size (m x m), compute |A|
       from the diagonal values of U and the permutation information. */
   double Det(int m) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A X,
       for a matrix X of size (m x n). */
   void Mult(int m, int n, double *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- L^{-1} P X, for a matrix X of size (m x n). */
   void LSolve(int m, int n, double *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- U^{-1} X, for a matrix X of size (m x n). */
   void USolve(int m, int n, double *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, double *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- X A^{-1},
       for a matrix X of size (n x m). */
   void RightSolve(int m, int n, double *X) const;

   /// Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, double *X) const;

   /** Given an (n x m) matrix A21, compute X2 <- X2 - A21 X1, for matrices X1,
       and X2 of size (m x r) and (n x r), respectively. */
   static void SubMult(int m, int n, int r, const double *A21,
                       const double *X1, double *X2);

   /** Assuming P.A = L.U factored data of size (m x m), compute the 2x2 block
       decomposition:
          | P 0 | |  A  A12 | = |  L  0 | | U U12 |
          | 0 I | | A21 A22 |   | L21 I | | 0 S22 |
       where A12, A21, and A22 are matrices of size (m x n), (n x m), and
       (n x n), respectively. The blocks are overwritten as follows:
          A12 <- U12 = L^{-1} P A12
          A21 <- L21 = A21 U^{-1}
          A22 <- S22 = A22 - L21 U12.
       The block S22 is the Schur complement. */
   void BlockFactor(int m, int n, double *A12, double *A21, double *A22) const;

   /** Given BlockFactor()'d data, perform the forward block solve for the
       linear system:
          |  A  A12 | | X1 | = | B1 |
          | A21 A22 | | X2 |   | B2 |
       written in the factored form:
          |  L  0 | | U U12 | | X1 | = | P 0 | | B1 |
          | L21 I | | 0 S22 | | X2 |   | 0 I | | B2 |.
       The resulting blocks Y1, Y2 solve the system:
          |  L  0 | | Y1 | = | P 0 | | B1 |
          | L21 I | | Y2 |   | 0 I | | B2 |
       The blocks are overwritten as follows:
          B1 <- Y1 = L^{-1} P B1
          B2 <- Y2 = B2 - L21 Y1 = B2 - A21 A^{-1} B1
       The blocks B1/Y1 and B2/Y2 are of size (m x r) and (n x r), respectively.
       The Schur complement system is given by: S22 X2 = Y2. */
   void BlockForwSolve(int m, int n, int r, const double *L21,
                       double *B1, double *B2) const;

   /** Given BlockFactor()'d data, perform the backward block solve in
          | U U12 | | X1 | = | Y1 |
          | 0 S22 | | X2 |   | Y2 |.
       The input is the solution block X2 and the block Y1 resulting from
       BlockForwSolve(). The result block X1 overwrites input block Y1:
          Y1 <- X1 = U^{-1} (Y1 - U12 X2). */
   void BlockBackSolve(int m, int n, int r, const double *U12,
                       const double *X2, double *Y1) const;
};


/** Data type for inverse of square dense matrix.
    Stores LU factors */
class DenseMatrixInverse : public MatrixInverse
{
private:
   const DenseMatrix *a;
   LUFactors lu;

public:
   /// Default constructor.
   DenseMatrixInverse() : a(NULL), lu(NULL, NULL) { }

   /** Creates square dense matrix. Computes factorization of mat
       and stores LU factors. */
   DenseMatrixInverse(const DenseMatrix &mat);

   /// Same as above but does not factorize the matrix.
   DenseMatrixInverse(const DenseMatrix *mat);

   ///  Get the size of the inverse matrix
   int Size() const { return Width(); }

   /// Factor the current DenseMatrix, *a
   void Factor();

   /// Factor a new DenseMatrix of the same size
   void Factor(const DenseMatrix &mat);

   virtual void SetOperator(const Operator &op);

   /// Matrix vector multiplication with the inverse of dense matrix.
   void Mult(const double *x, double *y) const;

   /// Matrix vector multiplication with the inverse of dense matrix.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Multiply the inverse matrix by another matrix: X = A^{-1} B.
   void Mult(const DenseMatrix &B, DenseMatrix &X) const;

   /// Multiply the inverse matrix by another matrix: X <- A^{-1} X.
   void Mult(DenseMatrix &X) const { lu.Solve(width, X.Width(), X.Data()); }

   /// Compute and return the inverse matrix in Ainv.
   void GetInverseMatrix(DenseMatrix &Ainv) const;

   /// Compute the determinant of the original DenseMatrix using the LU factors.
   double Det() const { return lu.Det(width); }

   /// Print the numerical conditioning of the inversion: ||A^{-1} A - I||.
   void TestInversion();

   /// Destroys dense inverse matrix.
   virtual ~DenseMatrixInverse();
};


class DenseMatrixEigensystem
{
   DenseMatrix &mat;
   Vector      EVal;
   DenseMatrix EVect;
   Vector ev;
   int n;

#ifdef MFEM_USE_LAPACK
   double *work;
   char jobz, uplo;
   int lwork, info;
#endif

public:

   DenseMatrixEigensystem(DenseMatrix &m);
   DenseMatrixEigensystem(const DenseMatrixEigensystem &other);
   void Eval();
   Vector &Eigenvalues() { return EVal; }
   DenseMatrix &Eigenvectors() { return EVect; }
   double Eigenvalue(int i) { return EVal(i); }
   const Vector &Eigenvector(int i)
   {
      ev.SetData(EVect.Data() + i * EVect.Height());
      return ev;
   }
   ~DenseMatrixEigensystem();
};


class DenseMatrixSVD
{
   Vector sv;
   int m, n;

#ifdef MFEM_USE_LAPACK
   double *work;
   char jobu, jobvt;
   int lwork, info;
#endif

   void Init();
public:

   DenseMatrixSVD(DenseMatrix &M);
   DenseMatrixSVD(int h, int w);
   void Eval(DenseMatrix &M);
   Vector &Singularvalues() { return sv; }
   double Singularvalue(int i) { return sv(i); }
   ~DenseMatrixSVD();
};

class Table;

/// Rank 3 tensor (array of matrices)
class DenseTensor
{
private:
   DenseMatrix Mk;
   Memory<double> tdata;
   int nk;

public:
   DenseTensor()
   {
      nk = 0;
      tdata.Reset();
   }

   DenseTensor(int i, int j, int k)
      : Mk(NULL, i, j)
   {
      nk = k;
      tdata.New(i*j*k);
   }

   /// Copy constructor: deep copy
   DenseTensor(const DenseTensor &other)
      : Mk(NULL, other.Mk.height, other.Mk.width), nk(other.nk)
   {
      const int size = Mk.Height()*Mk.Width()*nk;
      if (size > 0)
      {
         tdata.New(size, other.tdata.GetMemoryType());
         tdata.CopyFrom(other.tdata, size);
      }
      else
      {
         tdata.Reset();
      }
   }

   int SizeI() const { return Mk.Height(); }
   int SizeJ() const { return Mk.Width(); }
   int SizeK() const { return nk; }

   int TotalSize() const { return SizeI()*SizeJ()*SizeK(); }

   void SetSize(int i, int j, int k)
   {
      const MemoryType mt = tdata.GetMemoryType();
      tdata.Delete();
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      tdata.New(i*j*k, mt);
   }

   void UseExternalData(double *ext_data, int i, int j, int k)
   {
      tdata.Delete();
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      tdata.Wrap(ext_data, i*j*k, false);
   }

   /// Sets the tensor elements equal to constant c
   DenseTensor &operator=(double c);

   DenseMatrix &operator()(int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      Mk.data = Memory<double>(GetData(k), SizeI()*SizeJ(), false);
      return Mk;
   }
   const DenseMatrix &operator()(int k) const
   { return const_cast<DenseTensor&>(*this)(k); }

   double &operator()(int i, int j, int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(i, 0, SizeI());
      MFEM_ASSERT_INDEX_IN_RANGE(j, 0, SizeJ());
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata[i+SizeI()*(j+SizeJ()*k)];
   }

   const double &operator()(int i, int j, int k) const
   {
      MFEM_ASSERT_INDEX_IN_RANGE(i, 0, SizeI());
      MFEM_ASSERT_INDEX_IN_RANGE(j, 0, SizeJ());
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata[i+SizeI()*(j+SizeJ()*k)];
   }

   double *GetData(int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata+k*Mk.Height()*Mk.Width();
   }

   double *Data() { return tdata; }

   const double *Data() const { return tdata; }

   Memory<double> &GetMemory() { return tdata; }
   const Memory<double> &GetMemory() const { return tdata; }

   /** Matrix-vector product from unassembled element matrices, assuming both
       'x' and 'y' use the same elem_dof table. */
   void AddMult(const Table &elem_dof, const Vector &x, Vector &y) const;

   void Clear()
   { UseExternalData(NULL, 0, 0, 0); }

   long MemoryUsage() const { return nk*Mk.MemoryUsage(); }

   /// Shortcut for mfem::Read( GetMemory(), TotalSize(), on_dev).
   const double *Read(bool on_dev = true) const
   { return mfem::Read(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), TotalSize(), false).
   const double *HostRead() const
   { return mfem::Read(tdata, Mk.Height()*Mk.Width()*nk, false); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), on_dev).
   double *Write(bool on_dev = true)
   { return mfem::Write(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), false).
   double *HostWrite()
   { return mfem::Write(tdata, Mk.Height()*Mk.Width()*nk, false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), on_dev).
   double *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), false).
   double *HostReadWrite()
   { return mfem::ReadWrite(tdata, Mk.Height()*Mk.Width()*nk, false); }

   ~DenseTensor() { tdata.Delete(); }
};

/** @brief Compute the LU factorization of a batch of matrices

    Factorize n matrices of size (m x m) stored in a dense tensor overwriting it
    with the LU factors. The factorization is such that L.U = Piv.A, where A is
    the original matrix and Piv is a permutation matrix represented by P.

    @param [in, out] Mlu batch of square matrices - dimension m x m x n.
    @param [out] P array storing pivot information - dimension m x n.
    @param [in] TOL optional fuzzy comparison tolerance. Defaults to 0.0. */
void BatchLUFactor(DenseTensor &Mlu, Array<int> &P, const double TOL = 0.0);

/** @brief Solve batch linear systems

    Assuming L.U = P.A for n factored matrices (m x m), compute x <- A x, for n
    companion vectors.

    @param [in] Mlu batch of LU factors for matrix M - dimension m x m x n.
    @param [in] P array storing pivot information - dimension m x n.
    @param [in, out] X vector storing right-hand side and then solution -
    dimension m x n. */
void BatchLUSolve(const DenseTensor &Mlu, const Array<int> &P, Vector &X);


// Inline methods

inline double &DenseMatrix::operator()(int i, int j)
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

inline const double &DenseMatrix::operator()(int i, int j) const
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

} // namespace mfem

#endif
