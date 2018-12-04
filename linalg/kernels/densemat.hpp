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

void kGetInverseMatrix(const int, const int*, const double*, double*);

void kLSolve(const int, const int, const double*, const int*, double*);

void kUSolve(const int, const int, const double*, double*);

void kFactorPrint(const int, const double*);

void kFactorSet(const int, const double*, double*);

void kFactor(const int, int*, double*);


void DenseMatrixSet(const double, const size_t, double*);

void DenseMatrixTranspose(const size_t, const size_t, double*, const double*);

void kMultAAt(const size_t, const size_t, const double*, double*);

void kGradToDiv(const size_t, const double*, double*);

void kAddMult_a_VVt(const size_t, const double, const double*, const size_t, double*);

void kMultWidth0(const size_t, double*);

void kMult(const size_t, const size_t, const size_t, const double*, const double*, double*);

void kMult(const size_t, const size_t, const double*, const double*, double*);

void kDiag(const size_t, const size_t, const double, double*);

void kOpEq(const size_t, const double*, double*);

double kDet2(const double*);

double kDet3(const double*);

double kFNormMax(const size_t, const double*);

double kFNorm2(const size_t, const double, const double*);

void kCalcInverse2D(const double, const double*, double*);

void kCalcInverse3D(const double, const double*, double*);

} // mfem namespace

#endif // MFEM_DENSEMAT_KERNELS
