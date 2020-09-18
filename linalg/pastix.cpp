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

namespace mfem {

static inline spm_int_t
spm_scatter_create_loc2glob( spmatrix_t *spm, spm_int_t baseval )
{
    spm_int_t i, size, begin, end, *loc2glob;
    int       clustnum, clustnbr;

    clustnum = spm->clustnum;
    clustnbr = spm->clustnbr;

    size  = spm->gN / clustnbr;
    begin = size *  clustnum    + spm_imin( clustnum,   spm->gN % clustnbr );
    end   = size * (clustnum+1) + spm_imin( clustnum+1, spm->gN % clustnbr );
    size  = end - begin;

    spm->n        = size;
    spm->loc2glob = (spm_int_t*)malloc( size * sizeof(spm_int_t) );
    loc2glob = spm->loc2glob;

    for ( i=begin; i<end; i++, loc2glob++ )
    {
        *loc2glob = i+baseval;
    }

    return size;
}

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
  for (int i = 0; i <csr->num_rows; i++) {
    matrix_.rowptr[i] = (csr->i)[i];
  }

  for (int i = 0; i < csr->num_nonzeros; i++) {
    matrix_.colptr[i] = (csr->j)[i];
    (static_cast<double*>(matrix_.values))[i] = (csr->data)[i];
  }

  matrix_.comm = hypParMat.GetComm();
  MPI_Comm_size( MPI_COMM_WORLD, &matrix_.clustnbr ); // Number of MPI nodes
  MPI_Comm_rank( MPI_COMM_WORLD, &matrix_.clustnum ); // Rank of MPI node

  spm_scatter_create_loc2glob(&matrix_, 0);

  spmatrix_t* spmg = spmGather( &matrix_, 0 );
  
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
