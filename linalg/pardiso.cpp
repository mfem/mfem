// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "pardiso.hpp"

#ifdef MFEM_USE_MKL_PARDISO

#include "sparsemat.hpp"

namespace mfem
{

PardisoSolver::PardisoSolver()
{
   // Indicate that default parameters are changed
   iparm[0] = 1;
   // Use METIS for fill-in reordering
   iparm[1] = 2;
   // Write the solution into the x vector data
   iparm[5] = 0;
   // Maximum number of iterative refinement steps
   iparm[7] = 2;
   // Perturb the pivot elements with 1E-13
   iparm[9] = 13;
   // Use nonsymmetric permutation
   iparm[10] = 1;
   // Perform a check on the input data
   iparm[26] = 1;
#ifdef MFEM_USE_SINGLE
   // Single precision
   iparm[27] = 1;
#endif
   // 0-based indexing in CSR data structure
   iparm[34] = 1;
   // Maximum number of numerical factorizations
   maxfct = 1;
   // Which factorization to use. This parameter is ignored and always assumed
   // to be equal to 1. See MKL documentation.
   mnum = 1;
   // Print statistical information in file
   msglvl = 0;
   // Initialize error flag
   error = 0;
   // Real nonsymmetric matrix
   mtype = MatType::REAL_NONSYMMETRIC;
   // Number of right hand sides
   nrhs = 1;
}

void PardisoSolver::SetOperator(const Operator &op)
{
   auto mat = const_cast<SparseMatrix *>(dynamic_cast<const SparseMatrix *>(&op));

   MFEM_ASSERT(mat, "Must pass SparseMatrix as Operator");

   height = op.Height();

   width = op.Width();

   m = mat->Size();

   nnz = mat->NumNonZeroElems();

   const int *Ap = mat->HostReadI();
   const int *Ai = mat->HostReadJ();
   const real_t *Ax = mat->HostReadData();

   csr_rowptr = new int[m + 1];
   reordered_csr_colind = new int[nnz];
   reordered_csr_nzval = new real_t[nnz];

   for (int i = 0; i <= m; i++)
   {
      csr_rowptr[i] = Ap[i];
   }

   // Pardiso expects the column indices to be sorted for each row
   mat->SortColumnIndices();

   for (int i = 0; i < nnz; i++)
   {
      reordered_csr_colind[i] = Ai[i];
      reordered_csr_nzval[i] = Ax[i];
   }

   // Analyze inputs
   phase = 11;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error);

   MFEM_ASSERT(error == 0, "Pardiso symbolic factorization error");

   // Numerical factorization
   phase = 22;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error);

   MFEM_ASSERT(error == 0, "Pardiso numerical factorization error");
}

void PardisoSolver::Mult(const Vector &b, Vector &x) const
{
   // Solve
   phase = 33;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, b.GetData(), x.GetData(), &error);

   MFEM_ASSERT(error == 0, "Pardiso solve error");
}

void PardisoSolver::SetPrintLevel(int print_level)
{
   msglvl = print_level;
}

void PardisoSolver::SetMatrixType(MatType mat_type)
{
   mtype = mat_type;
}

PardisoSolver::~PardisoSolver()
{
   // Release all internal memory
   phase = -1;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error);

   MFEM_ASSERT(error == 0, "Pardiso free error");

   delete[] csr_rowptr;
   delete[] reordered_csr_colind;
   delete[] reordered_csr_nzval;
}

} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO
