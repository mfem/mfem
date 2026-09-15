// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <algorithm>

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
   // One column, stated here rather than assumed: ArrayMult() sets nrhs to
   // the number of columns it was given, and a stale value would have
   // PARDISO read and write past the ends of b and x.
   nrhs = 1;

   // Solve
   phase = 33;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, b.GetData(), x.GetData(), &error);

   MFEM_ASSERT(error == 0, "Pardiso solve error");
}

void PardisoSolver::ArrayMult(const Array<const Vector *> &X,
                              Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in PardisoSolver::Mult!");

   const int ncols = X.Size();
   if (ncols == 0) { return; }

   if (ncols == 1)
   {
      MFEM_ASSERT(X[0] && Y[0], "Missing Vector in PardisoSolver::Mult!");
      nrhs = 1;
      Mult(*X[0], *Y[0]);
      return;
   }

   // PARDISO wants the columns contiguous and consecutive, stride m. The
   // packing buffer is ours rather than the caller's because MKL is
   // documented to use the right-hand side as workspace in some
   // configurations, and the caller's vectors are const.
   rhs_buf.SetSize(ncols * m);
   sol_buf.SetSize(ncols * m);
   real_t *rhs_data = rhs_buf.HostWrite();
   real_t *sol_data = sol_buf.HostWrite();

   for (int i = 0; i < ncols; i++)
   {
      MFEM_ASSERT(X[i] && X[i]->Size() == m,
                  "Missing or wrongly sized RHS Vector in PardisoSolver::Mult!");
      // HostRead() and not GetData(): a device-resident right-hand side has
      // to come down before it is copied. See UMFPackSolver::Mult().
      const real_t *xi = X[i]->HostRead();
      std::copy(xi, xi + m, rhs_data + i * m);
   }

   nrhs = ncols;
   phase = 33;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, rhs_data, sol_data, &error);

   MFEM_ASSERT(error == 0, "Pardiso solve error");

   for (int i = 0; i < ncols; i++)
   {
      MFEM_ASSERT(Y[i] && Y[i]->Size() == m,
                  "Missing or wrongly sized solution Vector in "
                  "PardisoSolver::Mult!");
      // HostWrite() invalidates the device copy, which a raw GetData() does
      // not; anything downstream on the device must see this solve.
      std::copy(sol_data + i * m, sol_data + (i + 1) * m, Y[i]->HostWrite());
   }
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
