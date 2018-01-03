// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BLOCKMATRIX
#define MFEM_BLOCKMATRIX

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../general/globals.hpp"
#include "vector.hpp"
#include "sparsemat.hpp"

namespace mfem
{

class BlockMatrix : public AbstractSparseMatrix
{
public:
   //! Constructor for square block matrices
   /**
    *  offsets: offsets that mark the start of each row/column block (size nRowBlocks+1).
    *  Note: BlockMatrix will not own/copy the data contained in offsets.
    */
   BlockMatrix(const Array<int> & offsets);
   //! Constructor for rectangular block matrices
   /**
    *  row_offsets: offsets that mark the start of each row block (size nRowBlocks+1).
    *  col_offsets: offsets that mark the start of each column block (size nColBlocks+1).
    *  Note: BlockMatrix will not own/copy the data contained in offsets.
    */
   BlockMatrix(const Array<int> & row_offsets, const Array<int> & col_offsets);
   //! Set A(i,j) = mat
   void SetBlock(int i, int j, SparseMatrix * mat);
   //! Return the number of row blocks
   int NumRowBlocks() const {return nRowBlocks; }
   //! Return the number of column blocks
   int NumColBlocks() const {return nColBlocks; }
   //! Return a reference to block (i,j). Reference may be invalid if Aij(i,j) == NULL
   SparseMatrix & GetBlock(int i, int j);
   //! Return a reference to block (i,j). Reference may be invalid if Aij(i,j)
   //! == NULL. (const version)
   const SparseMatrix & GetBlock(int i, int j) const;
   //! Check if block (i,j) is a zero block.
   int IsZeroBlock(int i, int j) const {return (Aij(i,j)==NULL) ? 1 : 0; }
   //! Return the row offsets for block starts
   Array<int> & RowOffsets() { return row_offsets; }
   //! Return the columns offsets for block starts
   Array<int> & ColOffsets() { return col_offsets; }
   //! Return the row offsets for block starts (const version)
   const Array<int> & RowOffsets() const { return row_offsets; }
   //! Return the row offsets for block starts (const version)
   const Array<int> & ColOffsets() const { return col_offsets; }
   //! Return the number of non zeros in row i
   int RowSize(const int i) const;
   /** Eliminates the column and row 'rc', replacing the element (rc,rc) with
    *  1.0. Assumes that element (i,rc) is assembled if and only if the element
    *  (rc,i) is assembled. If d != 0, the element (rc,rc) is kept the same. */
   void EliminateRowCol(int rc, int d = 0);
   //! Symmetric elimination of the marked degree of freedom.
   /**
    * ess_bc_dofs: marker of the degree of freedom to be eliminated
    *              dof i is eliminated if ess_bc_dofs[i] = 1.
    * sol: vector that stores the values of the degree of freedom that need to
    *       be eliminated
    * rhs: vector that stores the rhs of the system.
    */
   void EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol, Vector & rhs);

   ///  Finalize all the submatrices
   virtual void Finalize(int skip_zeros = 1) { Finalize(skip_zeros, false); }
   /// A slightly more general version of the Finalize(int) method.
   void Finalize(int skip_zeros, bool fix_empty_rows);

   //! Returns a monolithic CSR matrix that represents this operator.
   SparseMatrix * CreateMonolithic() const;
   //! Export the monolithic matrix to file.
   void PrintMatlab(std::ostream & os = mfem::out) const;

   //@name Matrix interface
   //@{
   /// Returns reference to a_{ij}.
   virtual double& Elem (int i, int j);
   /// Returns constant reference to a_{ij}.
   virtual const double& Elem (int i, int j) const;
   /// Returns a pointer to (approximation) of the matrix inverse.
   virtual MatrixInverse * Inverse() const
   {
      mfem_error("BlockMatrix::Inverse not implemented \n");
      return static_cast<MatrixInverse*>(NULL);
   }
   //@}

   //@name AbstractSparseMatrix interface
   //@{
   //! Returns the total number of non zeros in the matrix.
   virtual int NumNonZeroElems() const;
   /// Gets the columns indexes and values for row *row*.
   /// The return value is always 0 since cols and srow are copies of the values in the matrix.
   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const;
   //! If the matrix is square, it will place 1 on the diagonal (i,i) if row i
   //! has "almost" zero l1-norm.
   /**
    * If entry (i,i) does not belong to the sparsity pattern of A, then a error will occur.
    */
   virtual void EliminateZeroRows();

   /// Matrix-Vector Multiplication y = A*x
   virtual void Mult(const Vector & x, Vector & y) const;
   /// Matrix-Vector Multiplication y = y + val*A*x
   virtual void AddMult(const Vector & x, Vector & y, const double val = 1.) const;
   /// MatrixTranspose-Vector Multiplication y = A'*x
   virtual void MultTranspose(const Vector & x, Vector & y) const;
   /// MatrixTranspose-Vector Multiplication y = y + val*A'*x
   virtual void AddMultTranspose(const Vector & x, Vector & y,
                                 const double val = 1.) const;
   //@}

   //! Destructor
   virtual ~BlockMatrix();
   //! if owns_blocks the SparseMatrix objects Aij will be deallocated.
   int owns_blocks;

private:
   //! Given a global row iglobal finds to which row iloc in block iblock belongs to.
   inline void findGlobalRow(int iglobal, int & iblock, int & iloc) const;
   //! Given a global column jglobal finds to which column jloc in block jblock belongs to.
   inline void findGlobalCol(int jglobal, int & jblock, int & jloc) const;

   //! Number of row blocks
   int nRowBlocks;
   //! Number of columns blocks
   int nColBlocks;
   //! row offsets for each block start (length nRowBlocks+1).
   Array<int> row_offsets;
   //! column offsets for each block start (length nColBlocks+1).
   Array<int> col_offsets;
   //! 2D array that stores each block of the BlockMatrix. Aij(iblock, jblock)
   //! == NULL if block (iblock, jblock) is all zeros.
   Array2D<SparseMatrix *> Aij;
};

//! Transpose a BlockMatrix: result = A'
BlockMatrix * Transpose(const BlockMatrix & A);
//! Multiply BlockMatrix matrices: result = A*B
BlockMatrix * Mult(const BlockMatrix & A, const BlockMatrix & B);

inline void BlockMatrix::findGlobalRow(int iglobal, int & iblock,
                                       int & iloc) const
{
   if (iglobal > row_offsets[nRowBlocks])
   {
      mfem_error("BlockMatrix::findGlobalRow");
   }

   for (iblock = 0; iblock < nRowBlocks; ++iblock)
      if (row_offsets[iblock+1] > iglobal)
      {
         break;
      }

   iloc = iglobal - row_offsets[iblock];
}

inline void BlockMatrix::findGlobalCol(int jglobal, int & jblock,
                                       int & jloc) const
{
   if (jglobal > col_offsets[nColBlocks])
   {
      mfem_error("BlockMatrix::findGlobalCol");
   }

   for (jblock = 0; jblock < nColBlocks; ++jblock)
      if (col_offsets[jblock+1] > jglobal)
      {
         break;
      }

   jloc = jglobal - col_offsets[jblock];
}

}

#endif /* MFEM_BLOCKMATRIX */
