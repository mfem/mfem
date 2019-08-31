
#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


SparseMatrix* GetSparseMatrixFromOperator(Operator *op);
SparseMatrix* GetSparseMatrixFromBlockMatrix(BlockMatrix * blk_mat );
