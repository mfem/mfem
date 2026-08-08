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

/** @file bilininteg_vecmass_pa_simplices_mma.hpp
    Simplex VectorMass PA MMA — QFn + registration only.

    form::Apply<MassScale, …>(…, vdim) → block-diagonal multi-component mass.
    Requires scalar (coeff_vdim==1) PA data from AssembleSimplexMmaPA.
*/

#include "../bilininteg.hpp"
#include "bilininteg_mass_pa_simplices_mma.hpp" // MassScale
#include "mma/form/simplex.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template <int DIM, int D1D, int QND>
inline void MmaVectorMassApplySimplex(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::MassScale;
   Apply<MassScale, DIM, D1D, QND>(NE, P, d, x, y, vdim);
}

inline void MmaVectorMassApplySimplex2D(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::MassScale;
   Apply<MassScale, 2>(NE, P, d, x, y, vdim);
}

inline void MmaVectorMassApplySimplex3D(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Apply;
   using mma::form::MassScale;
   Apply<MassScale, 3>(NE, P, d, x, y, vdim);
}

} // namespace internal

template <int DIM, int D1D, int QND>
VectorMassIntegrator::ApplySimplexMmaKernelType
VectorMassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   return internal::MmaVectorMassApplySimplex<DIM, D1D, QND>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
