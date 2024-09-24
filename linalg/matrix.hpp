// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MATRIX
#define MFEM_MATRIX

#include "../general/array.hpp"
#include "../general/globals.hpp"
#include "operator.hpp"

namespace mfem
{

// Abstract data types matrix, inverse matrix

class MatrixInverse;

/// Abstract data type matrix
class Matrix : public Operator
{
   friend class MatrixInverse;
public:

   /// Creates a square matrix of size s.
   explicit Matrix(int s) : Operator(s) { }

   /// Creates a matrix of the given height and width.
   explicit Matrix(int h, int w) : Operator(h, w) { }

   /// Returns whether the matrix is a square matrix.
   bool IsSquare() const { return (height == width); }

   /// Returns reference to a_{ij}.
   virtual real_t &Elem(int i, int j) = 0;

   /// Returns constant reference to a_{ij}.
   virtual const real_t &Elem(int i, int j) const = 0;

   /// Returns a pointer to (an approximation) of the matrix inverse.
   virtual MatrixInverse *Inverse() const = 0;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int) { }

   /// Prints matrix to stream out.
   virtual void Print(std::ostream & out = mfem::out, int width_ = 4) const;

   /// Destroys matrix.
   virtual ~Matrix() { }
};


/// Abstract data type for matrix inverse
class MatrixInverse : public Solver
{
public:
   MatrixInverse() { }

   /// Creates approximation of the inverse of square matrix
   MatrixInverse(const Matrix &mat)
      : Solver(mat.height, mat.width) { }
};

/// Abstract data type for sparse matrices
class AbstractSparseMatrix : public Matrix
{
public:
   /// Creates a square matrix of the given size.
   explicit AbstractSparseMatrix(int s = 0) : Matrix(s) { }

   /// Creates a matrix of the given height and width.
   explicit AbstractSparseMatrix(int h, int w) : Matrix(h, w) { }

   /// Returns the number of non-zeros in a matrix
   virtual int NumNonZeroElems() const = 0;

   /// Gets the columns indexes and values for row *row*.
   /** Returns:
       - 0 if @a cols and @a srow are copies of the values in the matrix.
       - 1 if @a cols and @a srow are views of the values in the matrix. */
   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const = 0;

   /** @brief If the matrix is square, this method will place 1 on the diagonal
       (i,i) if row i has "almost" zero l1-norm.

       If entry (i,i) does not belong to the sparsity pattern of A, then an
       error will occur. */
   virtual void EliminateZeroRows(const real_t threshold = 1e-12) = 0;

   /// Matrix-Vector Multiplication y = A*x
   void Mult(const Vector &x, Vector &y) const override = 0;
   /// Matrix-Vector Multiplication y = y + val*A*x
   void AddMult(const Vector &x, Vector &y,
                const real_t val = 1.) const override = 0;
   /// MatrixTranspose-Vector Multiplication y = A'*x
   void MultTranspose(const Vector &x, Vector &y) const override = 0;
   /// MatrixTranspose-Vector Multiplication y = y + val*A'*x
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t val = 1.) const override = 0;

   /// Destroys AbstractSparseMatrix.
   virtual ~AbstractSparseMatrix() { }
};

}

#endif
