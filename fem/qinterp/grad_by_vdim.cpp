// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   // 2D
   k::Specialization<2,QVectorLayout::byVDIM,P,1,3,4>::template Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,P,1,4,6>::template Opt<4>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,P,1,5,8>::template Opt<2>::Add();

   k::Specialization<2,QVectorLayout::byVDIM,P,2,3,3>::template Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,P,2,3,4>::template Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,P,2,4,6>::template Opt<4>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,P,2,5,8>::template Opt<2>::Add();
   // 3D
   k::Specialization<3,QVectorLayout::byVDIM,P,1,3,4>::template Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,P,1,4,6>::template Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,P,1,5,8>::template Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,P,3,3,4>::template Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,P,3,4,6>::template Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,P,3,5,8>::template Opt<1>::Add();
}

template void InitGradByVDimKernels<true>();
template void InitGradByVDimKernels<false>();

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
