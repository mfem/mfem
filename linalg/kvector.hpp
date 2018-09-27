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

#ifndef MFEM_VECTOR_KERNELS
#define MFEM_VECTOR_KERNELS

MFEM_NAMESPACE
void kVectorAlphaAdd(double *vp, const double* v1p,
                     const double alpha, const double *v2p, const size_t N);

void kVectorSet(const size_t N, const double value, double *data);

void kVectorAssign(const size_t N, const double* v, double *data);

void kVectorMultOp(const size_t N, const double value, double *data);

void kVectorSubtract(double *zp, const double *xp, const double *yp, const size_t N);

double kVectorDot(const size_t N, const double *x, const double *y);

MFEM_NAMESPACE_END

#endif // MFEM_VECTOR_KERNELS
