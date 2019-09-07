
#include "mfem.hpp"
#include "util.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// This is very costly
SparseMatrix *GetSparseMatrixFromOperator(Operator *op)
{
    const int n = op->Height();
    MFEM_VERIFY(n == op->Width(), "");

    SparseMatrix *S = new SparseMatrix(n);
    Vector x(n);
    Vector y(n);

    for (int j = 0; j < n; ++j)
    {
        x = 0.0;
        x[j] = 1.0;
        op->Mult(x, y);
        for (int i = 0; i < n; ++i)
        {
            //if (y[i] != 0.0)
            if (fabs(y[i]) > 1.0e-15)
            {
                S->Set(i, j, y[i]);
            }
        }
    }
    S->Finalize();
    return S;
}
  

SparseMatrix *GetSparseMatrixFromBlockMatrix(BlockMatrix * blk_mat)
{
    int n = blk_mat->Height();
    int m = blk_mat->Width();
    SparseMatrix * S = new SparseMatrix(n,m);

    int numRowblocks = blk_mat->NumRowBlocks();
    int numColblocks = blk_mat->NumColBlocks();

    Array<int> offsets_i(numRowblocks); offsets_i[0] = 0;
    Array<int> offsets_j(numColblocks); offsets_j[0] = 0;

    for (int i=0; i<numRowblocks-1; i++) {offsets_i[i+1] = blk_mat->GetBlock(i,0).Height();}
    for (int j=0; j<numColblocks-1; j++) {offsets_j[j+1] = blk_mat->GetBlock(0,j).Width();}
    offsets_i.PartialSum(); offsets_j.PartialSum();

    for (int i = 0; i<numRowblocks; i++)
    {
        for (int j=0; j<numColblocks; j++)
        {
            SparseMatrix * block = &blk_mat->GetBlock(i,j);
            int nrows = block->NumRows();
            for (int k=0; k<nrows; k++)
            {
                int * col = block->GetRowColumns(k);
                int ncols = block->RowSize(k);
                double *data = block->GetRowEntries(k);
                for(int l=0; l<ncols; l++)
                {
                    S->Set(offsets_i[i]+k,offsets_j[j]+col[l], data[l]);
                }
            }    
        }
    }
    S->Finalize();
    return S;
}
  