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

#include "../quadinterpolator.hpp"
#include "grad.hpp"

namespace mfem
{
namespace internal
{
namespace quadrature_interpolator
{

template <bool P>
void InitGradByVDimKernels()
{
   constexpr auto L = QVectorLayout::byVDIM;
   // 2D
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,3,4,8>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,4,6,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,5,8,2>();

   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,3,3,8>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,3,4,8>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,6,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,5,8,2>();
   // 3D
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,3,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,4,6>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,5,8>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,3,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,4,6>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,5,8>();

   // 2D
   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,1,2,16>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,1,3,16>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,1,4,16>();

   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,2,2,16>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,2,3,4>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<2,L,P,2,4,2>();

   // 3D
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,1,2>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,1,3>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,1,4>();

   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,2,2>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,2,3>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,2,4>();

   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,3,2>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,3,3>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<3,L,P,3,4>();
}

template void InitGradByVDimKernels<true>();
template void InitGradByVDimKernels<false>();

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
