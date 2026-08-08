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

/** @file bilininteg_diffusion_pa_tensors_mma.hpp
    Tensor-product (quad/hex) diffusion PA MMA — QFn + registration only.

    form::ApplyTensor<DiffusionMetric, …> → generic Grad sum-fact engine.
*/

#include "../bilininteg.hpp"
#include "bilininteg_diffusion_pa_simplices_mma.hpp" // DiffusionMetric
#include "mma/form/tensors.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Runtime SYM dispatch: pick DiffusionMetric SYM and call ApplyTensor. */
template <int DIM, int T_D1D, int T_Q1D>
inline void MmaDiffusionApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::DiffusionMetric;
   // Runtime flag only selects QFn type; SYM is encoded in DiffusionMetric.
   if (symmetric)
   {
      ApplyTensor<DiffusionMetric<DIM, true>, DIM, T_D1D, T_Q1D>(
         NE, b, g, bt, gt, d, x, y, d1d, q1d);
   }
   else
   {
      ApplyTensor<DiffusionMetric<DIM, false>, DIM, T_D1D, T_Q1D>(
         NE, b, g, bt, gt, d, x, y, d1d, q1d);
   }
}

inline void MmaDiffusionApplyTensors2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors<2, 0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

inline void MmaDiffusionApplyTensors3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors<3, 0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaDiffusionApplyTensors<DIM, T_D1D, T_Q1D>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
