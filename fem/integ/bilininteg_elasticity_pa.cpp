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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_elasticity_kernels.hpp"

namespace mfem
{

void ElasticityIntegrator::SetUpQuadratureSpaceAndCoefficients(
   const FiniteElementSpace &fes)
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
   lambda_quad.reset(new CoefficientVector(lambda, *q_space,
                                           CoefficientStorage::FULL));
   mu_quad.reset(new CoefficientVector(mu, *q_space, CoefficientStorage::FULL));
   q_vec.reset(new QuadratureFunction(*q_space, vdim*vdim));
}

void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Elasticity PA only implemented for byNODES ordering.");

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   MFEM_VERIFY(fespace->GetVDim() == mesh.Dimension(), "");
   vdim = fespace->GetVDim();
   ndofs = fespace->GetTypicalFE()->GetDof();

   SetUpQuadratureSpaceAndCoefficients(fes);

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   geom = mesh.GetGeometricFactors(*IntRule, GeometricFactors::JACOBIANS);
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   q_vec->SetVDim(vdim*vdim*vdim*vdim);
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *lambda_quad, *mu_quad,
                                          *geom, *maps, *q_vec, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *lambda_quad, *mu_quad,
                                 *geom, *maps, x, *q_vec, y);
}

void ElasticityIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y); // Operator is symmetric
}

void ElasticityComponentIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   fespace = &fes;

   // Avoid projecting the coefficients more than once. If the coefficients
   // change, the parent ElasticityIntegrator must be reassembled.
   if (!parent.q_space)
   {
      parent.SetUpQuadratureSpaceAndCoefficients(fes);
   }
   else
   {
      IntRule = parent.IntRule;
   }

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   geom = fes.GetMesh()->GetGeometricFactors(*IntRule,
                                             GeometricFactors::JACOBIANS);
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
}

void ElasticityComponentIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, i_block, j_block);
}

void ElasticityComponentIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   // Each block in the operator is symmetric, so we can just switch the roles
   // of i_block and j_block
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, j_block, i_block);
}

} // namespace mfem
