
#include "mfem.hpp"

using namespace mfem;
using namespace std;


hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat)
{
  // First cast the parameter to a hypre_ParCSRMatrix
  hypre_ParCSRMatrix * parcsr_op =
    (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat);

  MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

  // Create the CSRMatrixMPI A_ by borrowing the internal data from a hypre_CSRMatrix.
  return hypre_MergeDiagAndOffd(parcsr_op);
}


// Row and column offsets are assumed to be the same.
// Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
                      Array2D<double> const& coefficient)
{
  const int numBlocks = offsets.Size() - 1;
  
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> all_num_loc_rows(nprocs);
  std::vector<int> procOffsets(nprocs);
  std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
  std::vector<std::vector<int> > blockProcOffsets(numBlocks);
  
  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows.data(), 1, MPI_INT, comm);

  for (int j=0; j<numBlocks; ++j)
    {
      all_block_num_loc_rows[j].resize(nprocs);
      blockProcOffsets[j].resize(nprocs);
      
      const int blockNumRows = offsets[j+1] - offsets[j];
      MPI_Allgather(&blockNumRows, 1, MPI_INT, all_block_num_loc_rows[j].data(), 1, MPI_INT, comm);

      blockProcOffsets[j][0] = 0;
      for (int i=0; i<nprocs-1; ++i)
   blockProcOffsets[j][i+1] = blockProcOffsets[j][i] + all_block_num_loc_rows[j][i];
    }
  
  int first_loc_row = 0;
  int glob_nrows = 0;
  procOffsets[0] = 0;
  for (int i=0; i<nprocs; ++i)
    {
      glob_nrows += all_num_loc_rows[i];
      if (i < rank)
   first_loc_row += all_num_loc_rows[i];

      if (i < nprocs-1)
   procOffsets[i+1] = procOffsets[i] + all_num_loc_rows[i];
    }
    
  const int glob_ncols = glob_nrows;

  std::vector<int> opI(num_loc_rows+1);
  std::vector<int> cnt(num_loc_rows);

  for (int i=0; i<num_loc_rows; ++i)
    {
      opI[i] = 0;
      cnt[i] = 0;
    }

  opI[num_loc_rows] = 0;
  
  Array2D<hypre_CSRMatrix*> csr_blocks(numBlocks, numBlocks);
  
  // Loop over all blocks, to determine nnz for each row.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
   {
     if (blocks(i, j) == NULL)
       {
         csr_blocks(i, j) = NULL;
       }
     else
       {
         csr_blocks(i, j) = GetHypreParMatrixData(*(blocks(i, j)));

         const int nrows = csr_blocks(i, j)->num_rows;

         for (int k=0; k<nrows; ++k)
      {
        const int rowg = offsets[i] + k;
        //(*(leftInjection(i, j)))[k]
        opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
      }
       }
   }
    }

  // Now opI[i] is nnz for row i-1. Do a partial sum to get offsets.
  for (int i=0; i<num_loc_rows; ++i)
    opI[i+1] += opI[i];

  const int nnz = opI[num_loc_rows];

  std::vector<HYPRE_Int> opJ(nnz);
  std::vector<double> data(nnz);

  // Loop over all blocks, to set matrix data.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
   {
     if (csr_blocks(i, j) != NULL)
       {
         const int nrows = csr_blocks(i, j)->num_rows;
         const double coef = coefficient(i, j);

        //  const bool failure = (nrows != offsets[i+1] - offsets[i]);
         
         MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
         
         for (int k=0; k<nrows; ++k)
      {
        const int rowg = offsets[i] + k;  // process-local row
        const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
        const int osk = csr_blocks(i, j)->i[k];
        
        for (int l=0; l<nnz_k; ++l)
          {
            // Find the column process offset for the block.
            const int bcol = csr_blocks(i, j)->j[osk + l];
            int bcolproc = 0;

            for (int p=1; p<nprocs; ++p)
         {
           if (blockProcOffsets[j][p] > bcol)
             {
               bcolproc = p-1;
               break;
             }
         }

            if (blockProcOffsets[j][nprocs - 1] <= bcol)
         bcolproc = nprocs - 1;

            const int colg = procOffsets[bcolproc] + offsets[j] + (bcol - blockProcOffsets[j][bcolproc]);

            if (colg < 0)
         cout << "BUG, negative global column index" << endl;
            
            opJ[opI[rowg] + cnt[rowg]] = colg;
            data[opI[rowg] + cnt[rowg]] = coef * csr_blocks(i, j)->data[osk + l];
            cnt[rowg]++;
          }
      }
       }
   }
    }

  bool cntCheck = true;
  for (int i=0; i<num_loc_rows; ++i)
    {
      if (cnt[i] != opI[i+1] - opI[i])
   cntCheck = false;
    }

  MFEM_VERIFY(cntCheck, "");

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
   {
     if (csr_blocks(i, j) != NULL)
       {
         hypre_CSRMatrixDestroy(csr_blocks(i, j));
       }
   }
    }
  
  std::vector<HYPRE_Int> rowStarts2(2);
  rowStarts2[0] = first_loc_row;
  rowStarts2[1] = first_loc_row + all_num_loc_rows[rank];

  HYPRE_Int minJ = opJ[0];
  HYPRE_Int maxJ = opJ[0];
  for (int i=0; i<nnz; ++i)
    {
      minJ = std::min(minJ, opJ[i]);
      maxJ = std::max(maxJ, opJ[i]);
    }
  
  HypreParMatrix *hmat = new HypreParMatrix(comm, num_loc_rows, glob_nrows, glob_ncols, (int*) opI.data(), (HYPRE_Int*) opJ.data(), (double*) data.data(),
                   (HYPRE_Int*) rowStarts2.data(), (HYPRE_Int*) rowStarts2.data());
  
  return hmat;
}