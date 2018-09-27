// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SPARSEMAT_KERNELS
#define MFEM_SPARSEMAT_KERNELS

MFEM_NAMESPACE

// *****************************************************************************
void kAddMult(const size_t height,
              const int *I, const int *J, const double *A,
              const double *x, double *y);

// *****************************************************************************
void kGauss_Seidel_forw_A_NULL(const size_t s,
                               RowNode **R,
                               const double *xp,
                               double *yp);
// *****************************************************************************
void kGauss_Seidel_forw(const size_t s,
                        const int *I, const int *J, const double *A,
                        const double *xp,
                        double *yp);

// *****************************************************************************
void kGauss_Seidel_back(const size_t s,
                        const int *I, const int *J, const double *A,
                        const double *xp,
                        double *yp);


MFEM_NAMESPACE_END

#endif // MFEM_SPARSEMAT_KERNELS
