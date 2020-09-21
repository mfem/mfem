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
  const hypre_ParCSRMatrix* par_csr = hypParMat;

  // This is a non-modifying operation
  hypre_CSRMatrix* csr = hypre_MergeDiagAndOffd(const_cast<hypre_ParCSRMatrix*>(par_csr));

  width = par_csr->global_num_cols;
  height = par_csr->global_num_rows;

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
  int* jptr = csr->big_j;
  #else
  int* jptr = csr->j;
  #endif
  for (int i = 0; i < csr->num_nonzeros; i++) {
    matrix_.colptr[i] = static_cast<int>(jptr[i]);
    (static_cast<double*>(matrix_.values))[i] = (csr->data)[i];
  }

  matrix_.comm = hypParMat.GetComm();
  MPI_Comm_size( MPI_COMM_WORLD, &matrix_.clustnbr ); // Number of MPI nodes
  MPI_Comm_rank( MPI_COMM_WORLD, &matrix_.clustnum ); // Rank of MPI node

  matrix_.loc2glob = (spm_int_t*)malloc( matrix_.n * sizeof(spm_int_t) );
  const int start = par_csr->row_starts[matrix_.clustnum];
  for (int i = 0; i < matrix_.n; i++)
  {
    matrix_.loc2glob[i] = start + i;
  }

  hypre_CSRMatrixDestroy(csr);
}

PastixSparseMatrix::~PastixSparseMatrix()
{
  spmExit(&matrix_);
}

void PastixSparseMatrix::Mult(const Vector &x, Vector &y) const
{
  spmMatVec(SpmNoTrans, 1.0, &matrix_, static_cast<const double*>(x), 0.0, static_cast<double*>(y));
}

PastixSolver::PastixSolver(MPI_Comm comm) : comm_(comm)
{
  pastixInitParam(integer_params_, double_params_);
  // Hold off on initializing until parameters have been set by the user
}

PastixSolver::~PastixSolver()
{
  if(pastix_data_ != nullptr) {
    pastixFinalize( &pastix_data_ );
  }
}

void PastixSolver::Mult( const Vector& x, Vector& y) const
{
  MFEM_ASSERT(matrix_ != nullptr,
               "PastixSolver Error: The operator must be set before"
               " the system can be solved.");
  const spmatrix_t& spm = matrix_->InternalData();
  // Only solving for one RHS vector
  const int nrhs = 1;
  // The solution is stored in place of the right-hand-side vector
  y = x;
  pastix_task_solve(pastix_data_, nrhs, y, spm.n);
}

void PastixSolver::SetOperator(const Operator& op)
{
  // We assume the options have been set, so now we can initialize
  pastixInit( &pastix_data_, comm_, integer_params_, double_params_);

  matrix_ = dynamic_cast<const PastixSparseMatrix*>(&op);
  if (matrix_ == nullptr)
  {
    MFEM_ABORT("PastixSolver::SetOperator: Must be PastixSparseMatrix");
  }
  width = op.Width();
  height = op.Height();
  const spmatrix_t& spm = matrix_->InternalData();
  pastix_task_analyze(pastix_data_, &spm);
  // Non-modifying
  pastix_task_numfact(pastix_data_, const_cast<spmatrix_t*>(&spm));
}

} // namespace mfem


#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
