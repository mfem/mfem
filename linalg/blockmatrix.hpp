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
     @param offsets  offsets that mark the start of each row/column block (size
                     nRowBlocks+1).
     @note BlockMatrix will not own/copy the data contained in offsets.
    */
   BlockMatrix(const Array<int> & offsets);
   //! Constructor for rectangular block matrices
   /**
     @param row_offsets  offsets that mark the start of each row block (size
                         nRowBlocks+1).
     @param col_offsets  offsets that mark the start of each column block (size
                         nColBlocks+1).
     @note BlockMatrix will not own/copy the data contained in offsets.
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

   /// Eliminate the row and column @a rc from the matrix.
   /** Eliminates the column and row @a rc, replacing the element (rc,rc) with
       1.0. Assumes that element (i,rc) is assembled if and only if the element
       (rc,i) is assembled. If @a dpolicy is specified, the element (rc,rc) is
       treated according to that policy. */
   void EliminateRowCol(int rc, DiagonalPolicy dpolicy = DIAG_ONE);

   /** @brief Eliminate the rows and columns corresponding to the entries
       in @a vdofs + save the eliminated entries into
       @a Ae so that (*this) + Ae is equal to the original matrix. */
   void EliminateRowCols(const Array<int> & vdofs, BlockMatrix *Ae,
                         DiagonalPolicy dpolicy = DIAG_ONE);

   //! Symmetric elimination of the marked degree of freedom.
   /**
     @param ess_bc_dofs  marker of the degree of freedom to be eliminated
                         dof i is eliminated if @a ess_bc_dofs[i] = 1.
     @param sol          vector that stores the values of the degree of freedom
                         that need to be eliminated
     @param rhs          vector that stores the rhs of the system.
   */
   void EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol, Vector & rhs);

   ///  Finalize all the submatrices
   void Finalize(int skip_zeros = 1) override { Finalize(skip_zeros, false); }
   /// A slightly more general version of the Finalize(int) method.
   void Finalize(int skip_zeros, bool fix_empty_rows);

   //! Returns a monolithic CSR matrix that represents this operator.
   SparseMatrix * CreateMonolithic() const;
   //! Export the monolithic matrix to file.
   void PrintMatlab(std::ostream & os = mfem::out) const override;

   /// @name Matrix interface
   ///@{

   /// Returns reference to a_{ij}.
   real_t& Elem (int i, int j) override;
   /// Returns constant reference to a_{ij}.
   const real_t& Elem (int i, int j) const override;
   /// Returns a pointer to (approximation) of the matrix inverse.
   MatrixInverse * Inverse() const override
   {
      mfem_error("BlockMatrix::Inverse not implemented \n");
      return static_cast<MatrixInverse*>(NULL);
   }
   ///@}

   ///@name AbstractSparseMatrix interface
   ///@{

   //! Returns the total number of non zeros in the matrix.
   int NumNonZeroElems() const override;
   /// Gets the columns indexes and values for row *row*.
   /** The return value is always 0 since @a cols and @a srow are copies of the
       values in the matrix. */
   int GetRow(const int row, Array<int> &cols, Vector &srow) const override;
   /** @brief If the matrix is square, this method will place 1 on the diagonal
       (i,i) if row i has "almost" zero l1-norm.

       If entry (i,i) does not belong to the sparsity pattern of A, then a error
       will occur. */
   void EliminateZeroRows(const real_t threshold = 1e-12) override;

   /// Matrix-Vector Multiplication y = A*x
   void Mult(const Vector & x, Vector & y) const override;
   /// Matrix-Vector Multiplication y = y + val*A*x
   void AddMult(const Vector & x, Vector & y,
                const real_t val = 1.) const override;
   /// MatrixTranspose-Vector Multiplication y = A'*x
   void MultTranspose(const Vector & x, Vector & y) const override;
   /// MatrixTranspose-Vector Multiplication y = y + val*A'*x
   void AddMultTranspose(const Vector & x, Vector & y,
                         const real_t val = 1.) const override;
   ///@}

   /** @brief Partial matrix vector multiplication of (*this) with @a x
       involving only the rows given by @a rows. The result is given in @a y */
   void PartMult(const Array<int> &rows, const Vector &x, Vector &y) const;
   /** @brief Partial matrix vector multiplication of (*this) with @a x
       involving only the rows given by @a rows. The result is multiplied by
       @a a and added to @a y */
   void PartAddMult(const Array<int> &rows, const Vector &x, Vector &y,
                    const real_t a=1.0) const;

   //! Destructor
   virtual ~BlockMatrix();
   //! If owns_blocks the SparseMatrix objects Aij will be deallocated.
   int owns_blocks;

   virtual Type GetType() const { return MFEM_Block_Matrix; }

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
   {
      if (row_offsets[iblock+1] > iglobal) { break; }
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
   {
      if (col_offsets[jblock+1] > jglobal) { break; }
   }

   jloc = jglobal - col_offsets[jblock];
}

}

#endif /* MFEM_BLOCKMATRIX */
