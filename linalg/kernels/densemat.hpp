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

namespace mfem
{
namespace kernels
{
namespace densemat
{

void GetInverseMatrix(const int m, const int *ipiv,
                      const double *data, double *x);

void LSolve(const int m, const int n,
            const double *data, const int *ipiv, double *x);

void USolve(const int m, const int n, const double *data, double *x);

void FactorPrint(const int s, const double *data);

void FactorSet(const int s, const double *adata, double *ludata);

void Factor(const int m, int *ipiv, double *data);

void Set(const double d, const int size, double *data);

void Transpose(const int height, const int width,
               double *data, const double *mdata);

void MultAAt(const int height, const int width,
             const double *a, double *aat);

void GradToDiv(const int n, const double *data, double *ddata);

void AddMult_a_VVt(const int n, const double a, const double *v,
                   const int height, double *VVt);

void MultWidth0(const int height, double *y);

void Mult(const int height, const int width,
          const double *data, const double *x, double *y);

void Mult(const int ah, const int aw, const int bw,
          const double *bd, const double *cd, double *ad);

void Diag(const int n, const int N, const double c, double *data);

void OpEQ(const int hw, const double *m, double *data);

double Det2(const double *data);

double Det3(const double *data);

double FNormMax(const int hw, const double *data);

double FNorm2(const int hw, const double max_norm, const double *data);

void CalcInverse2D(const double t, const double *a, double *inva);

void CalcInverse3D(const double t, const double *a, double *inva);

} // namespace densemat
} // namespace kernels
} // namespace mfem

#endif // MFEM_DENSEMAT_KERNELS
