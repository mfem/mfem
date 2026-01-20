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

#include "../bilininteg.hpp"
#include "../nonlininteg.hpp"

namespace mfem
{

template <typename Base>
void StressDivergenceIntegrator<Base>::SetUpQuadratureSpace(const FiniteElementSpace &fes)
{
   if (IntRule == nullptr)
   {
      // This is where it's assumed that all elements are the same.
      const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
      int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
      IntRule = &IntRules.Get(T.GetGeometryType(), quad_order);
   }

   Mesh &mesh = *fespace->GetMesh();

   q_space.reset(new QuadratureSpace(mesh, *IntRule));
   q_vec.reset(new QuadratureFunction(*q_space, vdim * vdim));
}

template <typename Base>
void StressDivergenceIntegrator<Base>::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Elasticity PA only implemented for byNODES ordering.");

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   MFEM_VERIFY(fespace->GetVDim() == mesh.Dimension(), "");
   vdim = fespace->GetVDim();
   ndofs = fespace->GetTypicalFE()->GetDof();

   SetUpQuadratureSpace(fes);

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL
                                                      : DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   geom = mesh.GetGeometricFactors(*IntRule, GeometricFactors::JACOBIANS);
}

template class StressDivergenceIntegrator<NonlinearFormIntegrator>;
template class StressDivergenceIntegrator<BilinearFormIntegrator>;
} // namespace mfem
