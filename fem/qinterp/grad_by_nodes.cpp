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
void InitGradByNodesKernels()
{
   constexpr auto L = QVectorLayout::byNODES;
   // 2D
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,3,3,16>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,3,4,16>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,4,3,16>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,1,4,4,16>();

   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,2,2,16>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,2,3,8>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,2,4,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,2,5,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,2,6,2>();

   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,3,3,2>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,3,4,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,3,4>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,3,6,2>();

   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,4,2>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,5,2>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,6,2>();
   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,4,7,2>();

   QuadratureInterpolator::AddGradSpecializations<2,L,P,2,5,6,2>();
   // 3D
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,2,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,3,3>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,3,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,3,6>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,4,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,1,4,8>();

   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,2,3>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,2,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,2,5>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,2,6>();

   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,3,3>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,3,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,3,5>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,3,6>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,4,4>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,4,6>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,4,7>();
   QuadratureInterpolator::AddGradSpecializations<3,L,P,3,4,8>();

   using k2 = QuadratureInterpolator::CollocatedGradKernels;

   // 2D
   k2::Specialization<2,L,P,1,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,3>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,4>::template Opt<16>::Add();
   k2::Specialization<2,L,P,2,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,2,3>::template Opt<4>::Add();
   k2::Specialization<2,L,P,2,4>::template Opt<2>::Add();

   k2::Specialization<3,L,P,1,2>::Add();
   k2::Specialization<3,L,P,1,3>::Add();
   k2::Specialization<3,L,P,1,4>::Add();

   k2::Specialization<3,L,P,2,2>::Add();
   k2::Specialization<3,L,P,2,3>::Add();
   k2::Specialization<3,L,P,2,4>::Add();

   k2::Specialization<3,L,P,3,2>::Add();
   k2::Specialization<3,L,P,3,3>::Add();
   k2::Specialization<3,L,P,3,4>::Add();
}

template void InitGradByNodesKernels<true>();
template void InitGradByNodesKernels<false>();

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
