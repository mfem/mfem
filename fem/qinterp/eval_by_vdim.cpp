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
   auto &k = QuadratureInterpolator::TensorEvalKernels::Get();
   // 2D
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,2,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,3,6,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,4,8,2>();

   k.AddSpecialization<2,QVectorLayout::byVDIM,2,2,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,3,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,3,6,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,4,8,2>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,3,6,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,4,8,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,3,6,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,4,8,1>();

   k.AddSpecialization<3,QVectorLayout::byVDIM,3,2,2,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,3,3,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,4,4,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,5,5,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,6,6,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,7,7,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,8,8,1>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,9,9,1>();
}

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
