// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template<> void QuadratureInterpolator::Values<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_val) const
{
   MFEM_VERIFY(q_layout == QVectorLayout::byVDIM, "");
   if (fespace->GetNE() == 0) { return; }
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QValues(*fespace, &d2q, e_vec, q_val);
}

template<> void QuadratureInterpolator::Derivatives<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_der) const
{
   MFEM_VERIFY(q_layout == QVectorLayout::byVDIM, "");
   if (fespace->GetNE() == 0) { return; }
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QGrad(*fespace, &d2q, e_vec, q_der);
}

template<>
void QuadratureInterpolator::PhysDerivatives<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_der) const
{
   // q_layout == QVectorLayout::byVDIM
   Mesh *mesh = fespace->GetMesh();
   if (mesh->GetNE() == 0) { return; }
   // mesh->DeleteGeometricFactors(); // This should be done outside
   const IntegrationRule &ir = *IntRule;
   const GeometricFactors *geom =
      mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QPhysGrad(*fespace, geom, &d2q, e_vec, q_der);
}

template<> void QuadratureInterpolator::Mult<QVectorLayout::byVDIM>(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det) const
{
   constexpr QVectorLayout byVDIM = QVectorLayout::byVDIM;
   if (eval_flags & VALUES) { Values<byVDIM>(e_vec, q_val); }
   if (eval_flags & DERIVATIVES) { Derivatives<byVDIM>(e_vec, q_der); }
   if (eval_flags & DETERMINANTS)
   {
      MFEM_ABORT("evaluation of determinants with 'byVDIM' output layout"
                 " is not implemented yet!");
   }
}

} // namespace mfem
