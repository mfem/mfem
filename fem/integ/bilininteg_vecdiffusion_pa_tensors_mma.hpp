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

/** @file bilininteg_vecdiffusion_pa_tensors_mma.hpp
    Tensor-product VectorDiffusion PA MMA — QFn + registration only.

    form::ApplyTensor<DiffusionMetric, …, vdim> → block-diagonal multi-comp.
    Requires scalar coeff_vdim==1 PA from AssemblePA (VectorDiffusion layout).
*/

#include "../bilininteg.hpp"
#include "bilininteg_diffusion_pa_simplices_mma.hpp" // DiffusionMetric
#include "mma/form/tensors.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaVectorDiffusionApplyTensors(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::DiffusionMetric;
   // SYM metric pack (2D vector PA remapped full→SYM inside Grad shell).
   ApplyTensor<DiffusionMetric<DIM, true>, DIM, T_D1D, T_Q1D>(
      NE, b, g, bt, gt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorDiffusionApplyTensors2D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::DiffusionMetric;
   ApplyTensor<DiffusionMetric<2, true>, 2>(
      NE, b, g, bt, gt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorDiffusionApplyTensors3D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::DiffusionMetric;
   ApplyTensor<DiffusionMetric<3, true>, 3>(
      NE, b, g, bt, gt, d, x, y, d1d, q1d, vdim);
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
VectorDiffusionIntegrator::ApplyTensorsMmaKernelType
VectorDiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaVectorDiffusionApplyTensors<DIM, T_D1D, T_Q1D>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
