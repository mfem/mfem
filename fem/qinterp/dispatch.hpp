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

   // 2D
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,3,3,16>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,3,4,16>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,4,3,16>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,1,4,4,16>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,2,16>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,3,8>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,4,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,5,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,2,6,2>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,3,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,4,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,3,4>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,3,6,2>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,4,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,5,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,6,2>();
   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,4,7,2>();

   k.AddSpecialization<2,QVectorLayout::byNODES,P,2,5,6,2>();
   // 3D
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,3,6,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,4,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,1,4,8,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,5,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,2,6,1>();

   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,3,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,5,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,3,6,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,4,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,6,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,7,1>();
   k.AddSpecialization<3,QVectorLayout::byNODES,P,3,4,8,1>();
}

void InitEvalKernels()
{
   auto &k = QuadratureInterpolator::EvalKernels::Get();
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
