// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PASTIX

#include "pastix.hpp"

// #include <unistd.h>

namespace mfem {

PastixSparseMatrix::PastixSparseMatrix(const HypreParMatrix & hypParMat)
{
  hypre_ParCSRMatrix* const par_csr = hypParMat;
  hypre_CSRMatrix * const csr = hypre_MergeDiagAndOffd(par_csr);

  // int m         = parcsr_op->global_num_rows;
  //  int n         = parcsr_op->global_num_cols;
  //  int fst_row   = parcsr_op->first_row_index;
  //  int nnz_loc   = csr_op->num_nonzeros;
  //  int m_loc     = csr_op->num_rows;


  spmInit(&matrix_);

  matrix_.mtxtype = SpmGeneral;
  matrix_.flttype = SpmDouble;
  matrix_.fmttype = SpmCSR;

  matrix_.gN      = par_csr->global_num_rows;
  matrix_.n       = csr->num_rows;
  matrix_.gnnz = hypParMat.NNZ();
  matrix_.nnz = csr->num_nonzeros;


  matrix_.dof = 1;
  matrix_.dofs = nullptr;
  matrix_.loc2glob= nullptr;
  matrix_.layout = SpmColMajor;

  spmUpdateComputedFields(&matrix_);
  spmAlloc(&matrix_);

  // TODO: memcpy?
  for (int i = 0; i < csr->num_rows + 1; i++) {
    matrix_.rowptr[i] = (csr->i)[i];
  }

  #if MFEM_HYPRE_VERSION >= 21600
   // For now, this method assumes that HYPRE_Int is int. Also, csr_op->num_cols
   // is of type HYPRE_Int, so if we want to check for big indices in
   // csr_op->big_j, we'll have to check all entries and that check will only be
   // necessary in HYPRE_MIXEDINT mode which is not supported at the moment.
   hypre_CSRMatrixBigJtoJ(csr);
  #endif
  for (int i = 0; i < csr->num_nonzeros; i++) {
    matrix_.colptr[i] = (csr->j)[i];
    (static_cast<double*>(matrix_.values))[i] = (csr->data)[i];
  }

  matrix_.comm = hypParMat.GetComm();
  MPI_Comm_size( MPI_COMM_WORLD, &matrix_.clustnbr ); // Number of MPI nodes
  MPI_Comm_rank( MPI_COMM_WORLD, &matrix_.clustnum ); // Rank of MPI node

  // {
  //   volatile int i = 0;
  //   printf("sudo gdb -p %d <- RUN THIS\n", getpid());
  //   fflush(stdout);
  //   while (0 == i)
  //       sleep(5);
  // }
  matrix_.loc2glob = (spm_int_t*)malloc( matrix_.n * sizeof(spm_int_t) );
  const int start = par_csr->row_starts[matrix_.clustnum];
  for (int i = 0; i < matrix_.n; i++)
  {
    printf("Setting loc2glob[%d] to %d on rank %d\n", i, i + start, matrix_.clustnum);
    matrix_.loc2glob[i] = start + i;
  }

  spmatrix_t* spmg = spmGather( &matrix_, 0 );

  // if (matrix_.clustnum == 0)
  // {
  //   printf("Attempting to open file\n");
  //   FILE* fp = fopen("test.matrix", "w");
  //   printf("Could open file\n");
  //   spmPrint(&matrix_, fp);
  //   fclose(fp);
  // }

  if (matrix_.clustnum == 0 && spmg)
  {
    spmExit(spmg);
    free(spmg);
  }
  const_cast<HypreParMatrix*>(&hypParMat)->Print("hp.matrix");
}

PastixSparseMatrix::~PastixSparseMatrix()
{
  spmExit(&matrix_);
}

void PastixSparseMatrix::Mult(const Vector &x, Vector &y) const
{
  spmMatVec(SpmNoTrans, 1.0, &matrix_, static_cast<const double*>(x), 0.0, static_cast<double*>(y));
}

} // namespace mfem


#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
