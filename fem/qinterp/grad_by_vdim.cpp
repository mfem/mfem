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
   using k = QuadratureInterpolator::GradKernels;
   constexpr auto L = QVectorLayout::byVDIM;
   // 2D
   k::Specialization<2,L,P,1,3,4>::template Opt<8>::Add();
   k::Specialization<2,L,P,1,4,6>::template Opt<4>::Add();
   k::Specialization<2,L,P,1,5,8>::template Opt<2>::Add();

   k::Specialization<2,L,P,2,3,3>::template Opt<8>::Add();
   k::Specialization<2,L,P,2,3,4>::template Opt<8>::Add();
   k::Specialization<2,L,P,2,4,6>::template Opt<4>::Add();
   k::Specialization<2,L,P,2,5,8>::template Opt<2>::Add();
   // 3D
   k::Specialization<3,L,P,1,3,4>::Add();
   k::Specialization<3,L,P,1,4,6>::Add();
   k::Specialization<3,L,P,1,5,8>::Add();
   k::Specialization<3,L,P,3,3,4>::Add();
   k::Specialization<3,L,P,3,4,6>::Add();
   k::Specialization<3,L,P,3,5,8>::Add();

   using k2 = QuadratureInterpolator::CollocatedGradKernels;
   // 2D
   k2::Specialization<2,L,P,1,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,3>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,4>::template Opt<16>::Add();

   k2::Specialization<2,L,P,2,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,2,3>::template Opt<4>::Add();
   k2::Specialization<2,L,P,2,4>::template Opt<2>::Add();

   // 3D
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

template void InitGradByVDimKernels<true>();
template void InitGradByVDimKernels<false>();

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
