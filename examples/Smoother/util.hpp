
#pragma once

#include "mfem.hpp"

mfem::SparseMatrix* GetSparseMatrixFromOperator(mfem::Operator *op);
mfem::SparseMatrix* GetSparseMatrixFromBlockMatrix(mfem::BlockMatrix * blk_mat);
hypre_CSRMatrix* GetHypreParMatrixData(const mfem::HypreParMatrix & hypParMat);
mfem::HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, mfem::Array<int> & offsets, 
         mfem::Array2D<mfem::HypreParMatrix*> & blocks, mfem::Array2D<double> & coefficient);
   