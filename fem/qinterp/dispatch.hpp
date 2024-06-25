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

// Internal header, included only by .cpp files

#include "../quadinterpolator.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

template <bool P>
void InitDerivativeKernels()
{
   auto &k = QuadratureInterpolator::GradKernels::Get();
   // 2D
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,3,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,4,6>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,1,5,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,3,3>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,3,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,4,6>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,P,2,5,8>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,3,4>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,4,6>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,1,5,8>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,3,4>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,4,6>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,P,3,5,8>();

   // 2D
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,3,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,3,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,4,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,4,4>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,5>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,6>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,6>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,5>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,6>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,7>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,5,6>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,2,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,6>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,4,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,4,8>();

   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,5>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,6>();

   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,5>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,6>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,6>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,7>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,8>();
}

void InitEvalKernels()
{
   auto &k = QuadratureInterpolator::EvalKernels::Get();
   // 2D
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,2,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,3,6>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,1,4,8>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,2,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,3,4>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,3,6>();
   k.AddSpecialization<2,QVectorLayout::byVDIM,2,4,8>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,2,4>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,3,6>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,1,4,8>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,2,4>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,3,6>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,4,8>();

   k.AddSpecialization<3,QVectorLayout::byVDIM,3,2,2>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,3,3>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,4,4>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,5,5>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,6,6>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,7,7>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,8,8>();
   k.AddSpecialization<3,QVectorLayout::byVDIM,3,9,9>();

   // 2D
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,2,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,3,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,4,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,1,4,4>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,5>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,2,6>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,3,6>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,3>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,5>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,6>();
   k.AddSpecialization<2,QVectorLayout::byNODES,2,4,7>();

   k.AddSpecialization<2,QVectorLayout::byNODES,2,5,6>();

   // 3D
   k.AddSpecialization<3,QVectorLayout::byNODES,1,2,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,3,6>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,1,4,8>();

   k.AddSpecialization<3,QVectorLayout::byNODES,2,2,2>();
   k.AddSpecialization<3,QVectorLayout::byNODES,2,2,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,2,3,4>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,5>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,2,6>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,5>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,3,6>();

   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,3>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,4>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,6>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,7>();
   k.AddSpecialization<3,QVectorLayout::byNODES,3,4,8>();
}

// Tensor-product evaluation of quadrature point determinants: dispatch
// function.
void TensorDeterminants(const int NE,
                        const int vdim,
                        const DofToQuad &maps,
                        const Vector &e_vec,
                        Vector &q_det,
                        Vector &d_buff);

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
