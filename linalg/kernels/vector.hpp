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

namespace mfem
{
namespace kernels
{
namespace vector
{

double Dot(const size_t N, const double *x, const double *y);
double Min(const size_t N, const double *x);

void MapDof(const int N, double *y, const double *x, const int *dofs);
void MapDof(double *y, const double *x, const int dof, const int j);

void SetDof(double *y, const double alpha, const int dof);
void SetDof(const int N, double *y, const double alpha, const int *dofs);

void GetSubvector(const int N, double *y, const double *x, const int* dofs);
void SetSubvector(const int N, double *y, const double *x, const int* dofs);
void SetSubvector(const int N, double *y, const double d, const int* dofs);

void AlphaAdd(double *z, const double *x,
              const double a, const double *y, const size_t N);

void Subtract(double *z, const double *x, const double *y, const size_t N);

void Print(const size_t N, const double *x);

void Set(const size_t N, const double d, double *y);

void Assign(const size_t N, const double *x, double *y);

void OpMultEQ(const size_t N, const double d, double *y);

void OpPlusEQ(const size_t size, const double *x, double *y);

void OpAddEQ(const size_t, const double, const double*, double*);

void OpSubtractEQ(const size_t size, const double *x, double *y);

void AddElement(const size_t N, const int *dofs, const double *x, double *y);
void AddElementAlpha(const size_t, const int*, const double*, double*, const double);


} // namespace vector
} // namespace kernels
} // namespace mfem

#endif // MFEM_VECTOR_KERNELS
