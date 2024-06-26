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
   auto &k = QuadratureInterpolator::GradKernels::Get();
   // 2D
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,3,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,4,6,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,5,8,2>();

   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,3,3,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,3,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,4,6,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,5,8,2>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,4,6,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,5,8,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,4,6,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,5,8,1>();
}

template void InitGradByVDimKernels<true>();
template void InitGradByVDimKernels<false>();

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
