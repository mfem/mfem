// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file bilininteg_vecdiffusion_pa_simplices_mma.hpp
    Simplex VectorDiffusion PA MMA — QFn + registration only.

    form::Apply<DiffusionMetric<DIM,true>, …>(…, vdim) block-diagonal.
    Shared SYM PA (one metric pack for all components).
*/

#include "../bilininteg.hpp"
#include "bilininteg_diffusion_pa_simplices_mma.hpp" // DiffusionMetric
#include "mma/form/simplex.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template <int DIM, int D1D, int QND>
inline void MmaVectorDiffusionApplySimplex(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::DiffusionMetric;
   Apply<DiffusionMetric<DIM, true>, DIM, D1D, QND>(NE, G, d, x, y, vdim);
}

inline void MmaVectorDiffusionApplySimplex2D(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::DiffusionMetric;
   Apply<DiffusionMetric<2, true>, 2>(NE, G, d, x, y, vdim);
}

inline void MmaVectorDiffusionApplySimplex3D(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::DiffusionMetric;
   Apply<DiffusionMetric<3, true>, 3>(NE, G, d, x, y, vdim);
}

} // namespace internal

template <int DIM, int D1D, int QND>
VectorDiffusionIntegrator::ApplySimplexMmaKernelType
VectorDiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   return internal::MmaVectorDiffusionApplySimplex<DIM, D1D, QND>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
