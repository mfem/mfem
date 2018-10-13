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

#ifndef MFEM_DENSEMAT_KERNELS
#define MFEM_DENSEMAT_KERNELS

MFEM_NAMESPACE

void kLSolve(const int m, const int n, const double *data, const int *ipiv,
             double *X);

void kUSolve(const int m, const int n, const double *data, double *X);

void kFactorPrint(const int s, const double *data);

void kFactorSet(const int s, const double *adata, double *data);

void kFactor(const int m, int *ipiv, double *data);

void kMult(const int ah, const int aw, const int bw,
           const double *b, const double *c, double *a);

void DenseMatrixSet(const double, const size_t, double*);

void DenseMatrixTranspose(const size_t, const size_t, double*, const double*);

void kMultAAt(const size_t height, const size_t width,
              const double *a, double *aat);

void kGradToDiv(const size_t n, const double *data, double *ddata);

void kAddMult_a_VVt(const size_t n, const double a, const double *v,
                    const size_t height,double *VVt);

void kMult0(const size_t height, double *y);

void kMult(const size_t height, const size_t width,
           const double *data, const double *x, double *y);

MFEM_NAMESPACE_END

#endif // MFEM_DENSEMAT_KERNELS
