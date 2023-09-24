// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "pardisocomp.hpp"

#ifdef MFEM_USE_MKL_PARDISO

#include "complex_operator.hpp"

namespace mfem
{

PardisoCompSolver::PardisoCompSolver()
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
   // Maximum weighted matching algorithm is switched-on (default for non-symmetric)
   iparm[12] = 1;
   // Perform a check on the input data
   iparm[26] = 1;
   // 0-based indexing in CSR data structure
   iparm[34] = 1;
   // in-core (0, 1) or out-of-core (2) https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html
   // remember to set MKL_PARDISO_OOC_MAX_CORE_SIZE=XXXXXXX in the environment variable
   iparm[59] = 1;
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
   mtype = MatType::COMPLEX_UNSYMMETRIC;
   // Number of right hand sides
   nrhs = 1;
}

void PardisoCompSolver::CSRRealToComplex(const SparseMatrix* x_r, const SparseMatrix* x_i)
{
    MFEM_ASSERT(x_r->Height() == x_i->Height() &&
        x_r->Width() == x_i->Width() && 
        x_r->NumNonZeroElems()== x_i->NumNonZeroElems(),
        "Matrices must have the same dimensions and NNZ.");

    complexCSR = new ComplexCSRMatrix();

    int numRows = x_r->Height();
    complexCSR->numNonZeros = x_r->NumNonZeroElems();

    complexCSR->row_offsets.assign(x_r->GetI(), x_r->GetI() + numRows + 1);

    // Allocate space for values and cols
    complexCSR->values.resize(complexCSR->numNonZeros);
    complexCSR->cols.resize(complexCSR->numNonZeros);

    // Combine real and imaginary parts into complex values and copy cols
    for (int i = 0; i < complexCSR->numNonZeros; ++i)
    {
        complexCSR->values[i] = std::complex<double>(x_r->GetData()[i], x_i->GetData()[i]);
        complexCSR->cols[i] = x_r->GetJ()[i];
    }
}

void PardisoCompSolver::SetOperator(const Operator &op)
{
   auto mat = const_cast<ComplexSparseMatrix *>(dynamic_cast<const ComplexSparseMatrix *>(&op));

   MFEM_ASSERT(mat, "Must pass ComplexSparseMatrix as Operator");

   height = op.Height();

   width = op.Width();

   MFEM_ASSERT(height==width, "Must pass ComplexSparseMatrix as a square matrix");

   // returns a new complex array
   CSRRealToComplex(&mat->real(), &mat->imag());

   nnz = complexCSR->numNonZeros;

   const int *Ap = complexCSR->row_offsets.data();
   const int *Ai = complexCSR->cols.data();
   const std::complex<double> *Ax = complexCSR->values.data();

   csr_rowptr = new int[m + 1];
   reordered_csr_colind = new int[nnz];
   reordered_csr_nzval = new std::complex<double>[nnz];

   for (int i = 0; i <= m; i++)
   {
      csr_rowptr[i] = Ap[i];
   }

   // Pardiso expects the column indices to be sorted for each row
   //mat->SortColumnIndices();

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

void PardisoCompSolver::Mult(const Vector &b, Vector &x) const
{
   // Solve
   phase = 33;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, b.GetData(), x.GetData(), &error);

   MFEM_ASSERT(error == 0, "Pardiso solve error");
}

void PardisoCompSolver::GetResidual(const Vector& b, const Vector& x) const // to refine
{
    // to be implemented
    MFEM_ABORT("Function PardisoCompSolver is not yet implemented.");
}

void PardisoCompSolver::SetPrintLevel(int print_level)
{
   msglvl = print_level;
}

void PardisoCompSolver::SetMatrixType(MatType mat_type)
{
   mtype = mat_type;
}

PardisoCompSolver::~PardisoCompSolver()
{
   // Release all internal memory
   phase = -1;
   PARDISO(pt, &maxfct, &mnum, &mtype, &phase, 
           &m, reordered_csr_nzval, csr_rowptr,
           reordered_csr_colind, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error);

   MFEM_ASSERT(error == 0, "Pardiso free error");

   delete[] csr_rowptr;
   delete[] reordered_csr_colind;
   delete[] reordered_csr_nzval;
   delete[] complexCSR;
}

} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO
