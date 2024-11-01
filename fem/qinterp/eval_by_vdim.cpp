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
#include "eval.hpp"

namespace mfem
{
namespace internal
{
namespace quadrature_interpolator
{

void InitEvalByVDimKernels()
{
   using k = QuadratureInterpolator::TensorEvalKernels;
   // 2D
   k::Specialization<2,QVectorLayout::byVDIM,1,2,4>::Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,1,3,6>::Opt<4>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,1,4,8>::Opt<2>::Add();

   k::Specialization<2,QVectorLayout::byVDIM,2,2,4>::Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,2,3,4>::Opt<8>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,2,3,6>::Opt<4>::Add();
   k::Specialization<2,QVectorLayout::byVDIM,2,4,8>::Opt<2>::Add();
   // 3D
   k::Specialization<3,QVectorLayout::byVDIM,1,2,4>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,1,3,6>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,1,4,8>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,2,4>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,3,6>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,4,8>::Opt<1>::Add();

   k::Specialization<3,QVectorLayout::byVDIM,3,2,2>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,3,3>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,4,4>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,5,5>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,6,6>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,7,7>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,8,8>::Opt<1>::Add();
   k::Specialization<3,QVectorLayout::byVDIM,3,9,9>::Opt<1>::Add();
}

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
