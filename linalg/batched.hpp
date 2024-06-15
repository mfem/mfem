// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BATCHED_LINALG
#define MFEM_BATCHED_LINALG

#include "../config/config.hpp"
#include "densemat.hpp"

namespace mfem
{

/** @brief Compute the LU factorization of a batch of matrices

    Factorize n matrices of size (m x m) stored in a dense tensor overwriting it
    with the LU factors. The factorization is such that L.U = Piv.A, where A is
    the original matrix and Piv is a permutation matrix represented by P.

    @param [in, out] A batch of square matrices - dimension m x m x n.
    @param [out] P array storing pivot information - dimension m x n.
    @param [in] tol optional fuzzy comparison tolerance. Defaults to 0.0. */
void BatchLUFactor(DenseTensor &A, Array<int> &P, const real_t tol = 0.0);

/** @brief Solve batch linear systems

    Assuming L.U = P.A for n factored matrices (m x m), compute x <- A x, for n
    companion vectors.

    @param [in] A batch of LU factors for matrix M - dimension m x m x n.
    @param [in] P array storing pivot information - dimension m x n.
    @param [in, out] X vector storing right-hand side and then solution -
                     dimension m x nrhs x n.
    @param [in] nrhs Number of right-hand sides. */
void BatchLUSolve(const DenseTensor &A, const Array<int> &P, Vector &X,
                  int nrhs = 1);

// void BatchMult(const DenseTensor &A);

void BatchSetup();

void BatchInverse(DenseTensor &A);

} // namespace mfem

#endif
