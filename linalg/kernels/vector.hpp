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

void MapDof(const int, double*, const double*, const int*);
void MapDof(double*, const double*, const int, const int);

void SetDof(const int, double*, const double, const int*);
void SetDof(double*, const double, const int);

void GetSubvector(const int, double*, const double*, const int*);
void SetSubvector(const int, double*, const double*, const int*);

void AlphaAdd(double *, const double*, const double, const double *, const size_t);

void Print(const size_t, const double *);

void Set(const size_t, const double, double *);

void Assign(const size_t, const double *, double *);

void MultOp(const size_t, const double, double *);

void Subtract(double *, const double *, const double *, const size_t);

double Dot(const size_t, const double *, const double *);

void DotOpPlusEQ(const size_t, const double *, double *);

// void kSetSubVector(const size_t, const int*, const double*, double*);

void OpSubtract(const size_t, const double*, double*);

void AddElement(const size_t, const int*, const double*, double*);

} // namespace vector
} // namespace kernels


}

#endif // MFEM_VECTOR_KERNELS
