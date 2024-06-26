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

void InitEvalByNodesKernels()
{
   auto &k = QuadratureInterpolator::TensorEvalKernels::Get();

   // 2D
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,3,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,2,4,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,2,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,4,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,4,3,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,4,4,1>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,2,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,3,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,4,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,5,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,6,1>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,3,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,4,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,6,1>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,3,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,4,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,5,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,6,1>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,7,1>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,5,6,1>();

   // 3D
   k.AddSpecialization<3,QVectorLayout::byNODES,1,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,6,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,8,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,2,2,2,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,2,2,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,2,3,4,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,5,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,6,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,5,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,6,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,6,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,7,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,8,1>();
}

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
