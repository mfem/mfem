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

void Set(const double d, const size_t size, double *data);

void Transpose(const size_t height, const size_t width,
               double *data, const double *mdata);

void MultAAt(const size_t height, const size_t width,
              const double *a, double *aat);

void GradToDiv(const size_t n, const double *data, double *ddata);

void AddMult_a_VVt(const size_t n, const double a, const double *v,
                   const size_t height, double *VVt);

void MultWidth0(const size_t height, double *y);

void Mult(const size_t height, const size_t width,
          const double *data, const double *x, double *y);

void Mult(const size_t ah, const size_t aw, const size_t bw,
          const double *bd, const double *cd, double *ad);

void Diag(const size_t n, const size_t N, const double c, double *data);

void OpEQ(const size_t hw, const double *m, double *data);

double Det2(const double *data);

double Det3(const double *data);

double FNormMax(const size_t hw, const double *data);

double FNorm2(const size_t hw, const double max_norm, const double *data);

void CalcInverse2D(const double t, const double *a, double *inva);

void CalcInverse3D(const double t, const double *a, double *inva);

} // namespace densemat
} // namespace kernels
} // namespace mfem

#endif // MFEM_DENSEMAT_KERNELS
