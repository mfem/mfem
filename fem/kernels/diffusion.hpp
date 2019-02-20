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

#ifndef MFEM_KERNELS_DIFFUSION
#define MFEM_KERNELS_DIFFUSION

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
void DiffusionAssemble(const int dim,
                       const int NQ1d,
                       const int NE,
                       const double* __restrict W,
                       const double* __restrict J,
                       const double COEFF,
                       double* __restrict oper);

// *****************************************************************************
void DiffusionMultAssembled(const int dim,
                            const int ND1d,
                            const int NQ1d,
                            const int NE,
                            const double* __restrict B,
                            const double* __restrict G,
                            const double* __restrict Bt,
                            const double* __restrict Gt,
                            const double* __restrict op,
                            const double* __restrict x,
                            double* __restrict y);

} // namespace fem
} // namespace kernels
} // namespace mfem

#endif // MFEM_KERNELS_DIFFUSION
