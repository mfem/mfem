// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_DENSEMAT
#define MFEM_DENSEMAT

/// Data type dense matrix
class DenseMatrix : public Matrix
{
   friend class DenseTensor;

private:
   int height;
   double *data;

   friend class DenseMatrixInverse;
   friend void Mult(const DenseMatrix &b,
                    const DenseMatrix &c,
                    DenseMatrix &a);

   void Eigensystem(Vector &ev, DenseMatrix *evect = NULL);

public:
   /** Default constructor for DenseMatrix.
       Sets data = NULL size = height = 0 */
   DenseMatrix();

   /// Copy constructor
   DenseMatrix(const DenseMatrix &);

   /// Creates square matrix of size s.
   explicit DenseMatrix(int s);

   /// Creates rectangular matrix of size n and height m.
   DenseMatrix(int m, int n);

   /// Creates rectangular matrix equal to the transpose of mat.
   DenseMatrix(const DenseMatrix &mat, char ch);

   DenseMatrix(double *d, int h, int w) : Matrix(w) { data = d; height = h; }
   void UseExternalData(double *d, int h, int w)
   { data = d; height = h; size = w; }

   void ClearExternalData() { data = NULL; height = size = 0; }

   /// If the matrix is not a square matrix of size s then recreate it
   void SetSize(int s);

   /// If the matrix is not a matrix of size (h x w) then recreate it
   void SetSize(int h, int w);

   /// Returns the height of the matrix.
   inline int Height() const { return height; }

   /// Returns the width of the matrix.
   inline int Width() const { return size; }

   /// Returns vector of the elements.
   inline double *Data() const { return data; }

   /// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
   inline double &operator()(int i, int j);

   /// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
   inline const double &operator()(int i, int j) const;

   /// Matrix inner product: tr(A^t B)
   double operator*(const DenseMatrix &m) const;

   /// Trace of a square matrix
   double Trace() const;

   /// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   void Mult(const double *x, double *y) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Multiply a vector with the transpose matrix.
   virtual void MultTranspose(const Vector &x, Vector &y) const;

   /// y += A.x
   void AddMult(const Vector &x, Vector &y) const;

   /// Compute y^t A x
   double InnerProduct(const double *x, const double *y) const;

   /// Compute y^t A x
   double InnerProduct(const Vector &x, const Vector &y) const
   { return InnerProduct((const double *)x, (const double *)y); }

   /// Returns a pointer to the inverse matrix.
   virtual MatrixInverse *Inverse() const;

   /// Replaces the current matrix with its inverse
   void Invert();

   /// Calculates the determinant of the matrix (for 2x2 or 3x3 matrices)
   double Det() const;

   double Weight() const;

   /// Adds the matrix A multiplied by the number c to the matrix
   void Add(const double c, DenseMatrix &A);

   /// Sets the matrix elements equal to constant c
   DenseMatrix &operator=(double c);

   /// Sets the matrix size and elements equal to those of m
   DenseMatrix &operator=(const DenseMatrix &m);

   DenseMatrix &operator+=(DenseMatrix &m);

   DenseMatrix &operator-=(DenseMatrix &m);

   DenseMatrix &operator*=(double c);

   /// (*this) = -(*this)
   void Neg();

   /// Take the 2-norm of the columns of A and store in v
   void Norm2(double *v) const;

   double FNorm() const;

   void Eigenvalues(Vector &ev)
   { Eigensystem(ev); }

   void Eigenvalues(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }

   void Eigensystem(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }

   void SingularValues(Vector &sv) const;

   /// Return the i-th singular value (decreasing order) of a 2x2 or 3x3 matrix
   double CalcSingularvalue(const int i) const;

   /** Return the eigenvalues (in increasing order) and eigenvectors of a
       2x2 or 3x3 symmetric matrix. */
   void CalcEigenvalues(double *lambda, double *vec) const;

   void GetColumn(int c, Vector &col);

   void GetColumnReference(int c, Vector &col)
   { col.SetDataAndSize(data + c * height, height); }

   /// Creates n x n diagonal matrix with diagonal elements c
   void Diag(double c, int n);
   /// Creates n x n diagonal matrix with diagonal given by diag
   void Diag(double *diag, int n);

   /// (*this) = (*this)^t
   void Transpose();
   /// (*this) = A^t
   void Transpose(DenseMatrix &A);
   /// (*this) = 1/2 ((*this) + (*this)^t)
   void Symmetrize();

   void Lump();

   /** Given a DShape matrix (from a scalar FE), stored in *this,
       returns the CurlShape matrix. If *this is a N by 3 matrix,
       then curl is a 3*N by 3 matrix. The size of curl must be
       set outside.  */
   void GradToCurl(DenseMatrix &curl);
   /** Given a DShape matrix (from a scalar FE), stored in *this,
       returns the DivShape vector. If *this is a N by dim matrix,
       then div is a dim*N vector. The size of div must be set
       outside.  */
   void GradToDiv(Vector &div);

   /// Copy rows row1 through row2 from A to *this
   void CopyRows(DenseMatrix &A, int row1, int row2);
   /// Copy columns col1 through col2 from A to *this
   void CopyCols(DenseMatrix &A, int col1, int col2);
   /// Copy the m x n submatrix of A at (Aro)ffset, (Aco)loffset to *this
   void CopyMN(DenseMatrix &A, int m, int n, int Aro, int Aco);
   /// Copy matrix A to the location in *this at row_offset, col_offset
   void CopyMN(DenseMatrix &A, int row_offset, int col_offset);
   /// Copy matrix A^t to the location in *this at row_offset, col_offset
   void CopyMNt(DenseMatrix &A, int row_offset, int col_offset);
   /// Copy c on the diagonal of size n to *this at row_offset, col_offset
   void CopyMNDiag(double c, int n, int row_offset, int col_offset);
   /// Copy diag on the diagonal of size n to *this at row_offset, col_offset
   void CopyMNDiag(double *diag, int n, int row_offset, int col_offset);

   /// Perform (ro+i,co+j)+=A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(DenseMatrix &A, int ro, int co);
   /// Perform (ro+i,co+j)+=a*A(i,j) for 0<=i<A.Height, 0<=j<A.Width
   void AddMatrix(double a, DenseMatrix &A, int ro, int co);

   /// Add the matrix 'data' to the Vector 'v' at the given 'offset'
   void AddToVector(int offset, Vector &v) const;
   /// Get the matrix 'data' from the Vector 'v' at the given 'offset'
   void GetFromVector(int offset, const Vector &v);
   /** If (dofs[i] < 0 and dofs[j] >= 0) or (dofs[i] >= 0 and dofs[j] < 0)
       then (*this)(i,j) = -(*this)(i,j).  */
   void AdjustDofDirection(Array<int> &dofs);

   /// Prints matrix to stream out.
   virtual void Print(ostream &out = cout, int width = 4) const;
   /// Prints the transpose matrix to stream out.
   virtual void PrintT(ostream &out = cout, int width = 4) const;

   /// Invert and print the numerical conditioning of the inversion.
   void TestInversion();

   /// Destroys dense matrix.
   virtual ~DenseMatrix();
};

/// C = A + alpha*B
void Add(const DenseMatrix &A, const DenseMatrix &B,
         double alpha, DenseMatrix &C);

/// Matrix matrix multiplication.  A = B * C.
void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);

/// Calculate the adjugate of a matrix (for 2x2 or 3x3 matrices)
void CalcAdjugate(const DenseMatrix &a, DenseMatrix &adja);

/// Calculate the transposed adjugate of a matrix (for 2x2 or 3x3 matrices)
void CalcAdjugateTranspose(const DenseMatrix &a, DenseMatrix &adjat);

/** Calculate the inverse of a matrix (for 2x2 or 3x3 matrices) or the
    left inverse (A^t.A)^{-1}.A^t (for 2x1, 3x1, or 3x2 matrices) */
void CalcInverse(const DenseMatrix &a, DenseMatrix &inva);

/// Calculate the inverse transpose of a matrix (for 2x2 or 3x3 matrices)
void CalcInverseTranspose(const DenseMatrix &a, DenseMatrix &inva);

/// Calculate the matrix A.At
void MultAAt(const DenseMatrix &a, DenseMatrix &aat);

/// ADAt = A D A^t, where D is diagonal
void MultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);

/// ADAt += A D A^t, where D is diagonal
void AddMultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);

/// Multiply a matrix A with the transpose of a matrix B:   A*Bt
void MultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);

/// ABt += A * B^t
void AddMultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);

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

/// VVt += a * v v^t
void AddMult_a_VVt(const double a, const Vector &v, DenseMatrix &VVt);


/** Data type for inverse of square dense matrix.
    Stores LU factors */
class DenseMatrixInverse : public MatrixInverse
{
private:
   double *data;
#ifdef MFEM_USE_LAPACK
   int *ipiv;
#endif

public:
   /** Creates square dense matrix. Computes factorization of mat
       and stores LU factors. */
   DenseMatrixInverse(const DenseMatrix &mat);

   /// Same as above but does not factorize the matrix.
   DenseMatrixInverse(const DenseMatrix *mat);

   /// Factor the current DenseMatrix, *a
   void Factor();

   /// Factor a new DenseMatrix of the same size
   void Factor(const DenseMatrix &mat);

   /// Matrix vector multiplication with inverse of dense matrix.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Destroys dense inverse matrix.
   virtual ~DenseMatrixInverse();
};


class DenseMatrixEigensystem
{
   DenseMatrix &mat;

   Vector      EVal;
   DenseMatrix EVect;

   Vector ev;

   double *work;
   char jobz, uplo;
   int n, lwork, info;

public:

   DenseMatrixEigensystem(DenseMatrix &m);
   void Eval();
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
   char jobu, jobvt;
   int lwork, info;
   double *work;

   void Init();
public:

   DenseMatrixSVD(DenseMatrix &M);
   DenseMatrixSVD(int h, int w);
   void Eval(DenseMatrix &M);
   double Singularvalue(int i) { return sv(i); }
   ~DenseMatrixSVD();
};


/// Rank 3 tensor (array of matrices)
class DenseTensor
{
private:
   DenseMatrix Mk;
   double *tdata;
   int nk;

public:
   DenseTensor() { nk = 0; tdata = NULL; }

   DenseTensor(int i, int j, int k)
      : Mk(NULL, i, j)
   { nk = k; tdata = new double[i*j*k]; }

   int SizeI() const { return Mk.Height(); }
   int SizeJ() const { return Mk.Width(); }
   int SizeK() const { return nk; }

   void SetSize(int i, int j, int k)
   {
      delete [] tdata;
      Mk.UseExternalData(NULL, i, j);
      nk = k;
      tdata = new double[i*j*k];
   }

   DenseMatrix &operator()(int k) { Mk.data = GetData(k); return Mk; }

   double *GetData(int k) { return tdata+k*Mk.Height()*Mk.Width(); }

   double *Data() { return tdata; }

   ~DenseTensor() { delete [] tdata; Mk.ClearExternalData(); }
};


// Inline methods

inline double &DenseMatrix::operator()(int i, int j)
{
#ifdef MFEM_DEBUG
   if ( data == 0 || i < 0 || i >= height || j < 0 || j >= size )
      mfem_error("DenseMatrix::operator()");
#endif

   return data[i+j*height];
}

inline const double &DenseMatrix::operator()(int i, int j) const
{
#ifdef MFEM_DEBUG
   if ( data == 0 || i < 0 || i >= height || j < 0 || j >= size )
      mfem_error("DenseMatrix::operator() const");
#endif

   return data[i+j*height];
}

#endif
