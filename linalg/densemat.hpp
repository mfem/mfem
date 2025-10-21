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
   Memory<real_t> data;

   void Eigensystem(Vector &ev, DenseMatrix *evect = NULL);

   void Eigensystem(DenseMatrix &b, Vector &ev, DenseMatrix *evect = NULL);

   // Auxiliary method used in FNorm2() and FNorm()
   void FNorm(real_t &scale_factor, real_t &scaled_fnorm2) const;

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
   DenseMatrix(real_t *d, int h, int w)
      : Matrix(h, w) { UseExternalData(d, h, w); }

   /// Create a dense matrix using a braced initializer list
   /// The inner lists correspond to rows of the matrix
   template <int M, int N, typename T = real_t>
   explicit DenseMatrix(const T (&values)[M][N]) : DenseMatrix(M, N)
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
   void UseExternalData(real_t *d, int h, int w)
   {
      data.Wrap(d, h*w, false);
      height = h; width = w;
   }

   /// Change the data array and the size of the DenseMatrix.
   /** The DenseMatrix does not assume ownership of the data array, i.e. it will
       not delete the new array @a d. This method will delete the current data
       array, if owned. */
   void Reset(real_t *d, int h, int w)
   { if (OwnsData()) { data.Delete(); } UseExternalData(d, h, w); }

   /** Clear the data array and the dimensions of the DenseMatrix. This method
       should not be used with DenseMatrix that owns its current data array. */
   void ClearExternalData() { data.Reset(); height = width = 0; }

   /// Delete the matrix data array (if owned) and reset the matrix state.
   void Clear()
   { if (OwnsData()) { data.Delete(); } ClearExternalData(); }

   /// For backward compatibility define Size to be synonym of Width()
   int Size() const { return Width(); }

   // Total size = width*height
   int TotalSize() const { return width*height; }

   /// Change the size of the DenseMatrix to s x s.
   void SetSize(int s) { SetSize(s, s); }

   /// Change the size of the DenseMatrix to h x w.
   void SetSize(int h, int w);

   /// Returns the matrix data array.
   inline real_t *Data() const
   { return const_cast<real_t*>((const real_t*)data);}

   /// Returns the matrix data array.
   inline real_t *GetData() const { return Data(); }

   Memory<real_t> &GetMemory() { return data; }
   const Memory<real_t> &GetMemory() const { return data; }

   /// Return the DenseMatrix data (host pointer) ownership flag.
   inline bool OwnsData() const { return data.OwnsHostPtr(); }

   /// Returns reference to a_{ij}.
   inline real_t &operator()(int i, int j);

   /// Returns constant reference to a_{ij}.
   inline const real_t &operator()(int i, int j) const;

   /// Matrix inner product: tr(A^t B)
   real_t operator*(const DenseMatrix &m) const;

   /// Trace of a square matrix
   real_t Trace() const;

   /// Returns reference to a_{ij}.
   real_t &Elem(int i, int j) override;

   /// Returns constant reference to a_{ij}.
   const real_t &Elem(int i, int j) const override;

   /// Matrix vector multiplication.
   void Mult(const real_t *x, real_t *y) const;

   /// Matrix vector multiplication.
   void Mult(const real_t *x, Vector &y) const;

   /// Matrix vector multiplication.
   void Mult(const Vector &x, real_t *y) const;

   /// Matrix vector multiplication.
   void Mult(const Vector &x, Vector &y) const override;

   /// Absolute-value matrix vector multiplication.
   void AbsMult(const Vector &x, Vector &y) const override;

   /// Multiply a vector with the transpose matrix.
   void MultTranspose(const real_t *x, real_t *y) const;

   /// Multiply a vector with the transpose matrix.
   void MultTranspose(const real_t *x, Vector &y) const;

   /// Multiply a vector with the transpose matrix.
   void MultTranspose(const Vector &x, real_t *y) const;

   /// Multiply a vector with the transpose matrix.
   void MultTranspose(const Vector &x, Vector &y) const override;

   /// Multiply a vector with the absolute-value transpose matrix.
   void AbsMultTranspose(const Vector &x, Vector &y) const override;

   using Operator::Mult;
   using Operator::MultTranspose;

   /// y += a * A.x
   void AddMult(const Vector &x, Vector &y, const real_t a = 1.0) const override;

   /// y += a * A^t x
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t a = 1.0) const override;

   /// y += a * A.x
   void AddMult_a(real_t a, const Vector &x, Vector &y) const;

   /// y += a * A^t x
   void AddMultTranspose_a(real_t a, const Vector &x, Vector &y) const;

   /// Compute y^t A x
   real_t InnerProduct(const real_t *x, const real_t *y) const;

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
   real_t InnerProduct(const Vector &x, const Vector &y) const
   { return InnerProduct(x.GetData(), y.GetData()); }

   /// Returns a pointer to the inverse matrix.
   MatrixInverse *Inverse() const override;

   /// Replaces the current matrix with its inverse
   void Invert();

   /// Replaces the current matrix with its square root inverse
   void SquareRootInverse();

   /// Replaces the current matrix with its exponential
   /// (currently only supports 2x2 matrices)
   void Exponential();

   /// Calculates the determinant of the matrix
   /// (optimized for 2x2, 3x3, and 4x4 matrices)
   real_t Det() const;

   real_t Weight() const;

   /** @brief Set the matrix to alpha * A, assuming that A has the same
       dimensions as the matrix and uses column-major layout. */
   void Set(real_t alpha, const real_t *A);
   /// Set the matrix to alpha * A.
   void Set(real_t alpha, const DenseMatrix &A)
   {
      SetSize(A.Height(), A.Width());
      Set(alpha, A.GetData());
   }

   /// Adds the matrix A multiplied by the number c to the matrix.
   void Add(const real_t c, const DenseMatrix &A);

   /// Adds the matrix A multiplied by the number c to the matrix,
   /// assuming A has the same dimensions and uses column-major layout.
   void Add(const real_t c, const real_t *A);

   /// Sets the matrix elements equal to constant c
   DenseMatrix &operator=(real_t c);

   /// Copy the matrix entries from the given array
   DenseMatrix &operator=(const real_t *d);

   /// Sets the matrix size and elements equal to those of m
   DenseMatrix &operator=(const DenseMatrix &m);

   DenseMatrix &operator+=(const real_t *m);
   DenseMatrix &operator+=(const DenseMatrix &m);

   DenseMatrix &operator-=(const DenseMatrix &m);

   DenseMatrix &operator*=(real_t c);

   /// (*this) = -(*this)
   void Neg();

   /// Take the 2-norm of the columns of A and store in v
   void Norm2(real_t *v) const;

   /// Take the 2-norm of the columns of A and store in v
   void Norm2(Vector &v) const
   {
      MFEM_ASSERT(v.Size() == Width(), "incompatible Vector size!");
      Norm2(v.GetData());
   }

   /// Compute the norm ||A|| = max_{ij} |A_{ij}|
   real_t MaxMaxNorm() const;

   /// Compute the Frobenius norm of the matrix
   real_t FNorm() const { real_t s, n2; FNorm(s, n2); return s*sqrt(n2); }

   /// Compute the square of the Frobenius norm of the matrix
   real_t FNorm2() const { real_t s, n2; FNorm(s, n2); return s*s*n2; }

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
   int Rank(real_t tol) const;

   /// Return the i-th singular value (decreasing order) of NxN matrix, N=1,2,3.
   real_t CalcSingularvalue(const int i) const;

   /** Return the eigenvalues (in increasing order) and eigenvectors of a
       2x2 or 3x3 symmetric matrix. */
   void CalcEigenvalues(real_t *lambda, real_t *vec) const;

   void GetRow(int r, Vector &row) const;
   void GetColumn(int c, Vector &col) const;
   real_t *GetColumn(int col) { return data + col*height; }
   const real_t *GetColumn(int col) const { return data + col*height; }

   void GetColumnReference(int c, Vector &col)
   { col.SetDataAndSize(data + c * height, height); }

   void SetRow(int r, const real_t* row);
   void SetRow(int r, const Vector &row);

   void SetCol(int c, const real_t* col);
   void SetCol(int c, const Vector &col);


   /// Set all entries of a row to the specified value.
   void SetRow(int row, real_t value);
   /// Set all entries of a column to the specified value.
   void SetCol(int col, real_t value);

   /// Returns the diagonal of the matrix
   void GetDiag(Vector &d) const;
   /// Returns the l1 norm of the rows of the matrix v_i = sum_j |a_ij|
   void Getl1Diag(Vector &l) const;
   /// Compute the row sums of the DenseMatrix
   void GetRowSums(Vector &l) const;

   /// Creates n x n diagonal matrix with diagonal elements c
   void Diag(real_t c, int n);
   /// Creates n x n diagonal matrix with diagonal given by diag
   void Diag(real_t *diag, int n);

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
       can be either 2 or 3. In 2D this computes the scalar-valued curl of a
       2D vector field */
   void GradToCurl(DenseMatrix &curl);
   /** Given a DShape matrix (from a scalar FE), stored in *this, returns the
       CurlShape matrix. This computes the vector-valued curl of a scalar field.
       *this is N by 2 matrix and curl is N by 2 matrix as well. */
   void GradToVectorCurl2D(DenseMatrix &curl);
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
   void CopyMNDiag(real_t c, int n, int row_offset, int col_offset);
   /// Copy diag on the diagonal of size n to *this at row_offset, col_offset
   void CopyMNDiag(real_t *diag, int n, int row_offset, int col_offset);
   /// Copy All rows and columns except m and n from A
   void CopyExceptMN(const DenseMatrix &A, int m, int n);

   /// Perform (ro+i,co+j)+=A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(DenseMatrix &A, int ro, int co);
   /// Perform (ro+i,co+j)+=a*A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(real_t a, const DenseMatrix &A, int ro, int co);

   /** Get the square submatrix which corresponds to the given indices @a idx.
       Note: the @a A matrix will be resized to accommodate the data */
   void GetSubMatrix(const Array<int> & idx, DenseMatrix & A) const;

   /** Get the rectangular submatrix which corresponds to the given indices
      @a idx_i and @a idx_j. Note: the @a A matrix will be resized to
      accommodate the data */
   void GetSubMatrix(const Array<int> & idx_i, const Array<int> & idx_j,
                     DenseMatrix & A) const;

   /** Get the square submatrix which corresponds to the range
       [ @a ibeg, @a iend ). Note: the @a A matrix will be resized
        to accommodate the data */
   void GetSubMatrix(int ibeg, int iend, DenseMatrix & A);

   /** Get the square submatrix which corresponds to the range
      i ∈ [ @a ibeg, @a iend ) and j ∈ [ @a jbeg, @a jend )
      Note: the @a A matrix will be resized to accommodate the data */
   void GetSubMatrix(int ibeg, int iend, int jbeg, int jend, DenseMatrix & A);

   /// Set (*this)(idx[i],idx[j]) = A(i,j)
   void SetSubMatrix(const Array<int> & idx, const DenseMatrix & A);

   /// Set (*this)(idx_i[i],idx_j[j]) = A(i,j)
   void SetSubMatrix(const Array<int> & idx_i, const Array<int> & idx_j,
                     const DenseMatrix & A);

   /** Set a submatrix of (this) to the given matrix @a A
       with row and column offset @a ibeg */
   void SetSubMatrix(int ibeg, const DenseMatrix & A);

   /** Set a submatrix of (this) to the given matrix @a A
       with row and column offset @a ibeg and @a jbeg respectively */
   void SetSubMatrix(int ibeg, int jbeg, const DenseMatrix & A);

   /// (*this)(idx[i],idx[j]) += A(i,j)
   void AddSubMatrix(const Array<int> & idx, const DenseMatrix & A);

   /// (*this)(idx_i[i],idx_j[j]) += A(i,j)
   void AddSubMatrix(const Array<int> & idx_i, const Array<int> & idx_j,
                     const DenseMatrix & A);

   /** Add the submatrix @a A to this with row and column offset @a ibeg */
   void AddSubMatrix(int ibeg, const DenseMatrix & A);

   /** Add the submatrix @a A to this with row and column offsets
       @a ibeg and @a jbeg respectively */
   void AddSubMatrix(int ibeg, int jbeg, const DenseMatrix & A);

   /// Add the matrix 'data' to the Vector 'v' at the given 'offset'
   void AddToVector(int offset, Vector &v) const;
   /// Get the matrix 'data' from the Vector 'v' at the given 'offset'
   void GetFromVector(int offset, const Vector &v);
   /** If (dofs[i] < 0 and dofs[j] >= 0) or (dofs[i] >= 0 and dofs[j] < 0)
       then (*this)(i,j) = -(*this)(i,j).  */
   void AdjustDofDirection(Array<int> &dofs);

   /// Replace small entries, abs(a_ij) <= eps, with zero.
   void Threshold(real_t eps);

   /** Count the number of entries in the matrix for which isfinite
       is false, i.e. the entry is a NaN or +/-Inf. */
   int CheckFinite() const { return mfem::CheckFinite(HostRead(), height*width); }

   /// Prints matrix to stream out.
   void Print(std::ostream &out = mfem::out, int width_ = 4) const override;
   void PrintMatlab(std::ostream &out = mfem::out) const override;
   virtual void PrintMathematica(std::ostream &out = mfem::out) const;
   /// Prints the transpose matrix to stream out.
   virtual void PrintT(std::ostream &out = mfem::out, int width_ = 4) const;

   /// Invert and print the numerical conditioning of the inversion.
   void TestInversion();

   std::size_t MemoryUsage() const { return data.Capacity() * sizeof(real_t); }

   /// Shortcut for mfem::Read( GetMemory(), TotalSize(), on_dev).
   const real_t *Read(bool on_dev = true) const
   { return mfem::Read(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), TotalSize(), false).
   const real_t *HostRead() const
   { return mfem::Read(data, Height()*Width(), false); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), on_dev).
   real_t *Write(bool on_dev = true)
   { return mfem::Write(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), false).
   real_t *HostWrite()
   { return mfem::Write(data, Height()*Width(), false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), on_dev).
   real_t *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(data, Height()*Width(), on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), false).
   real_t *HostReadWrite()
   { return mfem::ReadWrite(data, Height()*Width(), false); }

   void Swap(DenseMatrix &other);

   /// Destroys dense matrix.
   virtual ~DenseMatrix();
};

/// C = A + alpha*B
void Add(const DenseMatrix &A, const DenseMatrix &B,
         real_t alpha, DenseMatrix &C);

/// C = alpha*A + beta*B
void Add(real_t alpha, const real_t *A,
         real_t beta,  const real_t *B, DenseMatrix &C);

/// C = alpha*A + beta*B
void Add(real_t alpha, const DenseMatrix &A,
         real_t beta,  const DenseMatrix &B, DenseMatrix &C);

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
bool LinearSolve(DenseMatrix& A, real_t* X, real_t TOL = 1.e-9);

/// Matrix matrix multiplication.  A = B * C.
void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);

/// Matrix matrix multiplication.  A += B * C.
void AddMult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);

/// Matrix matrix multiplication.  A += alpha * B * C.
void AddMult_a(real_t alpha, const DenseMatrix &b, const DenseMatrix &c,
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
void AddMult_a_ABt(real_t a, const DenseMatrix &A, const DenseMatrix &B,
                   DenseMatrix &ABt);

/// Multiply the transpose of a matrix A with a matrix B:   At*B
void MultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB);

/// AtB += A^t * B
void AddMultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB);

/// AtB += a * A^t * B
void AddMult_a_AtB(real_t a, const DenseMatrix &A, const DenseMatrix &B,
                   DenseMatrix &AtB);

/// AAt += a * A * A^t
void AddMult_a_AAt(real_t a, const DenseMatrix &A, DenseMatrix &AAt);

/// AAt = a * A * A^t
void Mult_a_AAt(real_t a, const DenseMatrix &A, DenseMatrix &AAt);

/// Make a matrix from a vector V.Vt
void MultVVt(const Vector &v, DenseMatrix &vvt);

void MultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);

/// VWt += v w^t
void AddMultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);

/// VVt += v v^t
void AddMultVVt(const Vector &v, DenseMatrix &VWt);

/// VWt += a * v w^t
void AddMult_a_VWt(const real_t a, const Vector &v, const Vector &w,
                   DenseMatrix &VWt);

/// VVt += a * v v^t
void AddMult_a_VVt(const real_t a, const Vector &v, DenseMatrix &VVt);

/** Computes matrix P^t * A * P. Note: The @a RAP matrix will be resized
    to accommodate the data */
void RAP(const DenseMatrix &A, const DenseMatrix &P, DenseMatrix & RAP);

/** Computes the matrix Rt^t * A * P. Note: The @a RAP matrix will be resized
    to accommodate the data */
void RAP(const DenseMatrix &Rt, const DenseMatrix &A,
         const DenseMatrix &P, DenseMatrix & RAP);

/** Abstract class that can compute factorization of external data and perform various
    operations with the factored data. */
class Factors
{
public:

   real_t *data;

   Factors() { }

   Factors(real_t *data_) : data(data_) { }

   virtual bool Factor(int m, real_t TOL = 0.0)
   {
      mfem_error("Factors::Factors(...)");
      return false;
   }

   virtual real_t Det(int m) const
   {
      mfem_error("Factors::Det(...)");
      return 0.;
   }

   virtual void Solve(int m, int n, real_t *X) const
   {
      mfem_error("Factors::Solve(...)");
   }

   virtual void GetInverseMatrix(int m, real_t *X) const
   {
      mfem_error("Factors::GetInverseMatrix(...)");
   }

   virtual ~Factors() {}
};


/** Class that can compute LU factorization of external data and perform various
    operations with the factored data. */
class LUFactors : public Factors
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
   LUFactors(): Factors() { }

   LUFactors(real_t *data_, int *ipiv_) : Factors(data_), ipiv(ipiv_) { }

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

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A X,
       for a matrix X of size (m x n). */
   void Mult(int m, int n, real_t *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- L^{-1} P X, for a matrix X of size (m x n). */
   void LSolve(int m, int n, real_t *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute
       X <- U^{-1} X, for a matrix X of size (m x n). */
   void USolve(int m, int n, real_t *X) const;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, real_t *X) const override;

   /** Assuming L.U = P.A factored data of size (m x m), compute X <- X A^{-1},
       for a matrix X of size (n x m). */
   void RightSolve(int m, int n, real_t *X) const;

   /// Assuming L.U = P.A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, real_t *X) const override;

   /** Given an (n x m) matrix A21, compute X2 <- X2 - A21 X1, for matrices X1,
       and X2 of size (m x r) and (n x r), respectively. */
   static void SubMult(int m, int n, int r, const real_t *A21,
                       const real_t *X1, real_t *X2);

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
   void BlockFactor(int m, int n, real_t *A12, real_t *A21, real_t *A22) const;

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
   void BlockForwSolve(int m, int n, int r, const real_t *L21,
                       real_t *B1, real_t *B2) const;

   /** Given BlockFactor()'d data, perform the backward block solve in
          | U U12 | | X1 | = | Y1 |
          | 0 S22 | | X2 |   | Y2 |.
       The input is the solution block X2 and the block Y1 resulting from
       BlockForwSolve(). The result block X1 overwrites input block Y1:
          Y1 <- X1 = U^{-1} (Y1 - U12 X2). */
   void BlockBackSolve(int m, int n, int r, const real_t *U12,
                       const real_t *X2, real_t *Y1) const;
};


/** Class that can compute Cholesky factorizations of external data of an
    SPD matrix and perform various operations with the factored data. */
class CholeskyFactors : public Factors
{
public:

   /** With this constructor, the (public) data should be set
       explicitly before calling class methods. */
   CholeskyFactors() : Factors() { }

   CholeskyFactors(real_t *data_) : Factors(data_) { }

   /**
    * @brief Compute the Cholesky factorization of the current matrix
    *
    * Factorize the current matrix of size (m x m) overwriting it with the
    * Cholesky factors. The factorization is such that LL^t = A, where A is the
    * original matrix
    *
    * @param [in] m size of the square matrix
    * @param [in] TOL optional fuzzy comparison tolerance. Defaults to 0.0.
    *
    * @return status set to true if successful, otherwise, false.
    */
   bool Factor(int m, real_t TOL = 0.0) override;

   /** Assuming LL^t = A factored data of size (m x m), compute |A|
       from the diagonal values of L */
   real_t Det(int m) const override;

   /** Assuming L.L^t = A factored data of size (m x m), compute X <- L X,
       for a matrix X of size (m x n). */
   void LMult(int m, int n, real_t *X) const;

   /** Assuming L.L^t = A factored data of size (m x m), compute X <- L^t X,
       for a matrix X of size (m x n). */
   void UMult(int m, int n, real_t *X) const;

   /** Assuming L L^t = A factored data of size (m x m), compute
       X <- L^{-1} X, for a matrix X of size (m x n). */
   void LSolve(int m, int n, real_t *X) const;

   /** Assuming L L^t = A factored data of size (m x m), compute
       X <- L^{-t} X, for a matrix X of size (m x n). */
   void USolve(int m, int n, real_t *X) const;

   /** Assuming L.L^t = A factored data of size (m x m), compute X <- A^{-1} X,
       for a matrix X of size (m x n). */
   void Solve(int m, int n, real_t *X) const override;

   /** Assuming L.L^t = A factored data of size (m x m), compute X <- X A^{-1},
       for a matrix X of size (n x m). */
   void RightSolve(int m, int n, real_t *X) const;

   /// Assuming L.L^t = A factored data of size (m x m), compute X <- A^{-1}.
   void GetInverseMatrix(int m, real_t *X) const override;

};


/** Data type for inverse of square dense matrix.
    Stores matrix factors, i.e.,  Cholesky factors if the matrix is SPD,
    LU otherwise. */
class DenseMatrixInverse : public MatrixInverse
{
private:
   const DenseMatrix *a;
   Factors * factors = nullptr;
   bool spd = false;

   void Init(int m);
   bool own_data = false;
public:
   /// Default constructor.
   DenseMatrixInverse(bool spd_=false) : a(NULL), spd(spd_) { Init(0); }

   /** Creates square dense matrix. Computes factorization of mat
       and stores its factors. */
   DenseMatrixInverse(const DenseMatrix &mat, bool spd_ = false);

   /// Same as above but does not factorize the matrix.
   DenseMatrixInverse(const DenseMatrix *mat, bool spd_ = false);

   ///  Get the size of the inverse matrix
   int Size() const { return Width(); }

   /// Factor the current DenseMatrix, *a
   void Factor();

   /// Factor a new DenseMatrix of the same size
   void Factor(const DenseMatrix &mat);

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
   real_t Det() const { return factors->Det(width); }

   /// Print the numerical conditioning of the inversion: ||A^{-1} A - I||.
   void TestInversion();

   /// Destroys dense inverse matrix.
   virtual ~DenseMatrixInverse();
};

#ifdef MFEM_USE_LAPACK

class DenseMatrixEigensystem
{
   DenseMatrix &mat;
   Vector      EVal;
   DenseMatrix EVect;
   Vector ev;
   int n;
   real_t *work;
   char jobz, uplo;
   int lwork, info;
public:

   DenseMatrixEigensystem(DenseMatrix &m);
   DenseMatrixEigensystem(const DenseMatrixEigensystem &other);
   void Eval();
   Vector &Eigenvalues() { return EVal; }
   DenseMatrix &Eigenvectors() { return EVect; }
   real_t Eigenvalue(int i) { return EVal(i); }
   const Vector &Eigenvector(int i)
   {
      ev.SetData(EVect.Data() + i * EVect.Height());
      return ev;
   }
   ~DenseMatrixEigensystem();
};

class DenseMatrixGeneralizedEigensystem
{
   DenseMatrix &A;
   DenseMatrix &B;
   DenseMatrix A_copy;
   DenseMatrix B_copy;
   Vector evalues_r;
   Vector evalues_i;
   DenseMatrix Vr;
   DenseMatrix Vl;
   int n;

   real_t *alphar;
   real_t *alphai;
   real_t *beta;
   real_t *work;
   char jobvl, jobvr;
   int lwork, info;

public:

   DenseMatrixGeneralizedEigensystem(DenseMatrix &a, DenseMatrix &b,
                                     bool left_eigen_vectors = false,
                                     bool right_eigen_vectors = false);
   void Eval();
   Vector &EigenvaluesRealPart() { return evalues_r; }
   Vector &EigenvaluesImagPart() { return evalues_i; }
   real_t EigenvalueRealPart(int i) { return evalues_r(i); }
   real_t EigenvalueImagPart(int i) { return evalues_i(i); }
   DenseMatrix &LeftEigenvectors() { return Vl; }
   DenseMatrix &RightEigenvectors() { return Vr; }
   ~DenseMatrixGeneralizedEigensystem();
};

/**
 @brief Class for Singular Value Decomposition of a DenseMatrix

 Singular Value Decomposition (SVD) of a DenseMatrix with the use of the DGESVD
 driver from LAPACK.
 */
class DenseMatrixSVD
{
   DenseMatrix Mc;
   Vector sv;
   DenseMatrix U,Vt;
   int m, n;

#ifdef MFEM_USE_LAPACK
   real_t *work;
   char jobu, jobvt;
   int lwork, info;
#endif

   void Init();
public:

   /**
    @brief Constructor for the DenseMatrixSVD

    Constructor for the DenseMatrixSVD with LAPACK. The parameters for the left
    and right singular vectors can be choosen according to the parameters for
    the LAPACK DGESVD.

    @param [in] M matrix to set the size to n=M.Height(), m=M.Width()
    @param [in] left_singular_vectors optional parameter to define if first
    left singular vectors should be computed
    @param [in] right_singular_vectors optional parameter to define if first
    right singular vectors should be computed
    */
   MFEM_DEPRECATED DenseMatrixSVD(DenseMatrix &M,
                                  bool left_singular_vectors=false,
                                  bool right_singular_vectors=false);

   /**
    @brief Constructor for the DenseMatrixSVD

    Constructor for the DenseMatrixSVD with LAPACK. The parameters for the left
    and right singular
    vectors can be choosen according to the parameters for the LAPACK DGESVD.

    @param [in] h height of the matrix
    @param [in] w width of the matrix
    @param [in] left_singular_vectors optional parameter to define if first
    left singular vectors should be computed
    @param [in] right_singular_vectors optional parameter to define if first
    right singular vectors should be computed
    */
   MFEM_DEPRECATED DenseMatrixSVD(int h, int w,
                                  bool left_singular_vectors=false,
                                  bool right_singular_vectors=false);

   /**
    @brief Constructor for the DenseMatrixSVD

    Constructor for the DenseMatrixSVD with LAPACK. The parameters for the left
    and right singular vectors can be choosen according to the parameters for
    the LAPACK DGESVD.

    @param [in] M matrix to set the size to n=M.Height(), m=M.Width()
    @param [in] left_singular_vectors optional parameter to define which left
    singular vectors should be computed
    @param [in] right_singular_vectors optional parameter to define which right
    singular vectors should be computed

    Options for computation of singular vectors:

    'A': All singular vectors are computed (default)

    'S': The first min(n,m) singular vectors are computed

    'N': No singular vectors are computed
    */
   DenseMatrixSVD(DenseMatrix &M,
                  char left_singular_vectors='A',
                  char right_singular_vectors='A');

   /**
    @brief Constructor for the DenseMatrixSVD

    Constructor for the DenseMatrixSVD with LAPACK. The parameters for the left
    and right singular vectors can be choosen according to the
    parameters for the LAPACK DGESVD.

    @param [in] h height of the matrix
    @param [in] w width of the matrix
    @param [in] left_singular_vectors optional parameter to define which left
    singular vectors should be computed
    @param [in] right_singular_vectors optional parameter to define which right
    singular vectors should be computed

    Options for computation of singular vectors:

    'A': All singular vectors are computed (default)

    'S': The first min(n,m) singular vectors are computed

    'N': No singular vectors are computed
    */
   DenseMatrixSVD(int h, int w,
                  char left_singular_vectors='A',
                  char right_singular_vectors='A');

   /**
    @brief Evaluate the SVD

    Call of the DGESVD driver from LAPACK for the DenseMatrix M. The singular
    vectors are computed according to the setup in the call of the constructor.

    @param [in] M DenseMatrix the SVD should be evaluated for
    */
   void Eval(DenseMatrix &M);

   /**
    @brief Return singular values

    @return sv Vector containing all singular values
    */
   Vector &Singularvalues() { return sv; }

   /**
    @brief Return specific singular value

    @return sv(i) i-th singular value
    */
   real_t Singularvalue(int i) { return sv(i); }

   /**
    @brief Return left singular vectors

    @return U DenseMatrix containing left singular vectors
    */
   DenseMatrix &LeftSingularvectors() { return U; }

   /**
    @brief Return right singular vectors

    @return Vt DenseMatrix containing right singular vectors
    */
   DenseMatrix &RightSingularvectors() { return Vt; }
   ~DenseMatrixSVD();
};

#endif // if MFEM_USE_LAPACK


class Table;

/// Rank 3 tensor (array of matrices)
class DenseTensor
{
private:
   mutable DenseMatrix Mk;
   Memory<real_t> tdata;
   int nk;

public:
   DenseTensor()
   {
      nk = 0;
   }

   DenseTensor(int i, int j, int k)
      : Mk(NULL, i, j)
   {
      nk = k;
      tdata.New(i*j*k);
   }

   DenseTensor(real_t *d, int i, int j, int k)
      : Mk(NULL, i, j)
   {
      nk = k;
      tdata.Wrap(d, i*j*k, false);
   }

   DenseTensor(int i, int j, int k, MemoryType mt)
      : Mk(NULL, i, j)
   {
      nk = k;
      tdata.New(i*j*k, mt);
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
   }

   int SizeI() const { return Mk.Height(); }
   int SizeJ() const { return Mk.Width(); }
   int SizeK() const { return nk; }

   int TotalSize() const { return SizeI()*SizeJ()*SizeK(); }

   void SetSize(int i, int j, int k, MemoryType mt_ = MemoryType::PRESERVE)
   {
      const MemoryType mt = mt_ == MemoryType::PRESERVE ? tdata.GetMemoryType() : mt_;
      tdata.Delete();
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      tdata.New(i*j*k, mt);
   }

   void UseExternalData(real_t *ext_data, int i, int j, int k)
   {
      tdata.Delete();
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      tdata.Wrap(ext_data, i*j*k, false);
   }

   /// @brief Reset the DenseTensor to use the given external Memory @a mem and
   /// dimensions @a i, @a j, and @a k.
   ///
   /// If @a own_mem is false, the DenseTensor will not own any of the pointers
   /// of @a mem.
   ///
   /// Note that when @a own_mem is true, the @a mem object can be destroyed
   /// immediately by the caller but `mem.Delete()` should NOT be called since
   /// the DenseTensor object takes ownership of all pointers owned by @a mem.
   void NewMemoryAndSize(const Memory<real_t> &mem, int i, int j, int k,
                         bool own_mem)
   {
      tdata.Delete();
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      if (own_mem)
      {
         tdata = mem;
      }
      else
      {
         tdata.MakeAlias(mem, 0, i*j*k);
      }
   }

   /// Sets the tensor elements equal to constant c
   DenseTensor &operator=(real_t c);

   /// Copy assignment operator (performs a deep copy)
   DenseTensor &operator=(const DenseTensor &other);

   DenseMatrix &operator()(int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      Mk.data = Memory<real_t>(GetData(k), SizeI()*SizeJ(), false);
      return Mk;
   }
   const DenseMatrix &operator()(int k) const
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      Mk.data = Memory<real_t>(const_cast<real_t*>(GetData(k)), SizeI()*SizeJ(),
                               false);
      return Mk;
   }

   real_t &operator()(int i, int j, int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(i, 0, SizeI());
      MFEM_ASSERT_INDEX_IN_RANGE(j, 0, SizeJ());
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata[i+SizeI()*(j+SizeJ()*k)];
   }

   const real_t &operator()(int i, int j, int k) const
   {
      MFEM_ASSERT_INDEX_IN_RANGE(i, 0, SizeI());
      MFEM_ASSERT_INDEX_IN_RANGE(j, 0, SizeJ());
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata[i+SizeI()*(j+SizeJ()*k)];
   }

   real_t *GetData(int k)
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata+k*Mk.Height()*Mk.Width();
   }

   const real_t *GetData(int k) const
   {
      MFEM_ASSERT_INDEX_IN_RANGE(k, 0, SizeK());
      return tdata+k*Mk.Height()*Mk.Width();
   }

   real_t *Data() { return tdata; }

   const real_t *Data() const { return tdata; }

   Memory<real_t> &GetMemory() { return tdata; }
   const Memory<real_t> &GetMemory() const { return tdata; }

   /** Matrix-vector product from unassembled element matrices, assuming both
       'x' and 'y' use the same elem_dof table. */
   void AddMult(const Table &elem_dof, const Vector &x, Vector &y) const;

   void Clear()
   { UseExternalData(NULL, 0, 0, 0); }

   std::size_t MemoryUsage() const { return nk*Mk.MemoryUsage(); }

   /// Shortcut for mfem::Read( GetMemory(), TotalSize(), on_dev).
   const real_t *Read(bool on_dev = true) const
   { return mfem::Read(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::Read(GetMemory(), TotalSize(), false).
   const real_t *HostRead() const
   { return mfem::Read(tdata, Mk.Height()*Mk.Width()*nk, false); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), on_dev).
   real_t *Write(bool on_dev = true)
   { return mfem::Write(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::Write(GetMemory(), TotalSize(), false).
   real_t *HostWrite()
   { return mfem::Write(tdata, Mk.Height()*Mk.Width()*nk, false); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), on_dev).
   real_t *ReadWrite(bool on_dev = true)
   { return mfem::ReadWrite(tdata, Mk.Height()*Mk.Width()*nk, on_dev); }

   /// Shortcut for mfem::ReadWrite(GetMemory(), TotalSize(), false).
   real_t *HostReadWrite()
   { return mfem::ReadWrite(tdata, Mk.Height()*Mk.Width()*nk, false); }

   void Swap(DenseTensor &t)
   {
      mfem::Swap(tdata, t.tdata);
      mfem::Swap(nk, t.nk);
      Mk.Swap(t.Mk);
   }

   ~DenseTensor() { tdata.Delete(); }
};

/** @brief Compute the LU factorization of a batch of matrices. Calls
    BatchedLinAlg::LUFactor.

    Factorize n matrices of size (m x m) stored in a dense tensor overwriting it
    with the LU factors. The factorization is such that L.U = Piv.A, where A is
    the original matrix and Piv is a permutation matrix represented by P.

    @param [in, out] Mlu batch of square matrices - dimension m x m x n.
    @param [out] P array storing pivot information - dimension m x n.
    @param [in] TOL optional fuzzy comparison tolerance. Defaults to 0.0. */
void BatchLUFactor(DenseTensor &Mlu, Array<int> &P, const real_t TOL = 0.0);

/** @brief Solve batch linear systems. Calls BatchedLinAlg::LUSolve.

    Assuming L.U = P.A for n factored matrices (m x m), compute x <- A x, for n
    companion vectors.

    @param [in] Mlu batch of LU factors for matrix M - dimension m x m x n.
    @param [in] P array storing pivot information - dimension m x n.
    @param [in, out] X vector storing right-hand side and then solution -
    dimension m x n. */
void BatchLUSolve(const DenseTensor &Mlu, const Array<int> &P, Vector &X);

#ifdef MFEM_USE_LAPACK
void BandedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                 Array<int> &ipiv);
void BandedFactorizedSolve(int KL, int KU, DenseMatrix &AB, DenseMatrix &B,
                           bool transpose, Array<int> &ipiv);
#endif

// Inline methods

inline real_t &DenseMatrix::operator()(int i, int j)
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

inline const real_t &DenseMatrix::operator()(int i, int j) const
{
   MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

} // namespace mfem

#endif
