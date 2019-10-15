
#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


SparseMatrix* GetSparseMatrixFromOperator(Operator *op);
SparseMatrix* GetSparseMatrixFromBlockMatrix(BlockMatrix * blk_mat );
hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat);
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, 
         Array2D<HypreParMatrix*> const& blocks, Array2D<double> const& coefficient);