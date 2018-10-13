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

void kVectorMapDof(const int, double*, const double*, const int*);
void kVectorMapDof(double*, const double*, const int, const int);

void kVectorSetDof(const int, double*, const double, const int*);
void kVectorSetDof(double*, const double, const int);

void kVectorGetSubvector(const int, double*, const double*, const int*);
void kVectorSetSubvector(const int, double*, const double*, const int*);

void kVectorAlphaAdd(double *vp, const double* v1p,
                     const double alpha, const double *v2p, const size_t N);

void kVectorPrint(const size_t N, const double *data);

void kVectorSet(const size_t N, const double value, double *data);

void kVectorAssign(const size_t N, const double* v, double *data);

void kVectorMultOp(const size_t N, const double value, double *data);

void kVectorSubtract(double *zp, const double *xp, const double *yp,
                     const size_t N);

double kVectorDot(const size_t N, const double *x, const double *y);

void kVectorDotOpPlusEQ(const size_t size, const double *v, double *data);

//void kSetSubVector(const size_t, const int*, const double*, double*);

void kVectorOpSubtract(const size_t, const double*, double*);

void kAddElementVector(const size_t, const int*, const double*, double*);

MFEM_NAMESPACE_END

#endif // MFEM_VECTOR_KERNELS
