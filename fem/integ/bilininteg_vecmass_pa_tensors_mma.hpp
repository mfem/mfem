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

/** @file bilininteg_vecmass_pa_tensors_mma.hpp
    Tensor-product VectorMass PA MMA — QFn + registration only.

    form::ApplyTensor<MassScale, …, vdim> → block-diagonal multi-component mass.
    Requires scalar (coeff_vdim==1) PA data from AssemblePA.
*/

#include "../bilininteg.hpp"
#include "bilininteg_mass_pa_simplices_mma.hpp" // MassScale
#include "mma/form/tensors.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaVectorMassApplyTensors(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::MassScale;
   ApplyTensor<MassScale, DIM, T_D1D, T_Q1D>(
      NE, b, bt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorMassApplyTensors2D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::MassScale;
   ApplyTensor<MassScale, 2>(NE, b, bt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorMassApplyTensors3D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::MassScale;
   ApplyTensor<MassScale, 3>(NE, b, bt, d, x, y, d1d, q1d, vdim);
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
VectorMassIntegrator::ApplyTensorsMmaKernelType
VectorMassIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaVectorMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
