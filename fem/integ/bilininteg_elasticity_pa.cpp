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
#include "../qfunction.hpp"
#include "bilininteg_elasticity_kernels.hpp"

namespace mfem
{
namespace
{
void VerifyElasticityPASpace(const FiniteElementSpace &fes,
                             const int expected_vdim)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();

   MFEM_VERIFY(dim == 2 || dim == 3,
               "Elasticity PA is implemented only in dimensions 2 and 3.");
   MFEM_VERIFY(dim == mesh.SpaceDimension(),
               "Elasticity PA does not support embedded meshes.");
   MFEM_VERIFY(mesh.GetNE() > 0,
               "Elasticity PA requires a nonempty mesh.");
   if (expected_vdim > 1)
   {
      MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
                  "Elasticity PA only supports Ordering::byNODES.");
   }
   MFEM_VERIFY(fes.GetVDim() == expected_vdim,
               "Unexpected vector dimension for elasticity PA.");
   MFEM_VERIFY(!fes.IsVariableOrder(),
               "Elasticity PA does not support variable-order spaces.");

   const FiniteElement &typical_fe = *fes.GetTypicalFE();
   MFEM_VERIFY(typical_fe.GetRangeType() == FiniteElement::SCALAR &&
               typical_fe.GetMapType() == FiniteElement::VALUE,
               "Elasticity PA requires scalar, VALUE-mapped finite elements.");

   const int typical_dofs = typical_fe.GetDof();
   const Geometry::Type typical_geometry =
      mesh.GetTypicalElementTransformation()->GetGeometryType();
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      const Geometry::Type geometry =
         mesh.GetElementTransformation(e)->GetGeometryType();
      MFEM_VERIFY(geometry == typical_geometry &&
                  fe.GetDof() == typical_dofs &&
                  fe.GetRangeType() == FiniteElement::SCALAR &&
                  fe.GetMapType() == FiniteElement::VALUE,
                  "Elasticity PA requires one uniform scalar VALUE-mapped "
                  "finite element type on all mesh elements.");
   }
}
} // namespace


void ElasticityIntegrator::SetUpQuadratureSpaceAndCoefficients(
   const FiniteElementSpace &fes)
{
   if (IntRule == nullptr)
   {
      const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
      const int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
      IntRule = &IntRules.Get(T.GetGeometryType(), quad_order);
   }

   Mesh &mesh = *fes.GetMesh();
   q_space.reset(new QuadratureSpace(mesh, *IntRule));

   MFEM_VERIFY(mu != nullptr, "The shear modulus coefficient is not set.");
   if (lambda != nullptr)
   {
      lambda_quad.reset(new CoefficientVector(*lambda, *q_space,
                                              CoefficientStorage::FULL));
      mu_quad.reset(new CoefficientVector(*mu, *q_space,
                                          CoefficientStorage::FULL));
   }
   else
   {
      ProductCoefficient lambda_coefficient(q_lambda, *mu);
      ProductCoefficient mu_coefficient(q_mu, *mu);
      lambda_quad.reset(new CoefficientVector(lambda_coefficient, *q_space,
                                              CoefficientStorage::FULL));
      mu_quad.reset(new CoefficientVector(mu_coefficient, *q_space,
                                          CoefficientStorage::FULL));
   }

   q_vec.reset(new QuadratureFunction(*q_space, vdim*vdim));
   pa_data.SetSize(0);
}

void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   VerifyElasticityPASpace(fes, mesh.Dimension());

   vdim = mesh.Dimension();
   ndofs = fespace->GetTypicalFE()->GetDof();

   SetUpQuadratureSpaceAndCoefficients(fes);
   const auto ordering = GetEVectorOrdering(*fespace);
   const auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
                     DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   geom = mesh.GetGeometricFactors(
             *IntRule,
             GeometricFactors::JACOBIANS |
             GeometricFactors::DETERMINANTS);
   internal::ElasticitySetupPAData(vdim, *IntRule, *lambda_quad, *mu_quad,
                                   *geom, pa_data);
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *maps, *IntRule,
                                          pa_data, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const FiniteElement &fe = *fespace->GetTypicalFE();
   const auto ordering = GetEVectorOrdering(*fespace);

   // Tensor kernels require lexicographic element vectors. Unsupported
   // D1D/Q1D pairs return false and use the generic fallback below. Define
   // MFEM_ELASTICITY_PA_DISABLE_TENSOR while building MFEM to benchmark or
   // debug the optimized generic fallback independently.
#ifndef MFEM_ELASTICITY_PA_DISABLE_TENSOR
   if (ordering == ElementDofOrdering::LEXICOGRAPHIC &&
       dynamic_cast<const TensorBasisElement *>(&fe) != nullptr)
   {
      const DofToQuad &tensor_maps =
         fe.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
      if (internal::ElasticityAddMultPATensor(vdim, fespace->GetNE(),
                                              tensor_maps, pa_data, x, y))
      {
         return;
      }
   }
#endif

   internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *maps, pa_data,
                                 x, *q_vec, y);
}

void ElasticityIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y); // Operator is symmetric
}

void ElasticityComponentIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   VerifyElasticityPASpace(fes, 1);
   MFEM_VERIFY(i_block >= 0 && i_block < dim &&
               j_block >= 0 && j_block < dim,
               "Elasticity component block index is out of range.");

   fespace = &fes;
   if (!parent.q_space)
   {
      if (parent.IntRule == nullptr)
      {
         parent.IntRule = IntRule;
      }
      else if (IntRule != nullptr)
      {
         MFEM_VERIFY(parent.IntRule == IntRule,
                     "Elasticity component and parent integration rules differ.");
      }
      parent.vdim = dim;
      parent.ndofs = fes.GetTypicalFE()->GetDof();
      parent.SetUpQuadratureSpaceAndCoefficients(fes);
   }
   else
   {
      MFEM_VERIFY(parent.q_space->GetMesh() == &mesh,
                  "Elasticity component and parent use different meshes.");
      MFEM_VERIFY(parent.vdim == dim,
                  "Elasticity component and parent dimensions differ.");
      MFEM_VERIFY(parent.ndofs == fes.GetTypicalFE()->GetDof(),
                  "Elasticity component and parent finite elements differ.");
      MFEM_VERIFY(IntRule == nullptr || IntRule == parent.IntRule,
                  "Elasticity component and parent integration rules differ.");
   }

   IntRule = parent.IntRule;
   MFEM_VERIFY(IntRule != nullptr,
               "Elasticity component integration rule was not initialized.");

   const auto ordering = GetEVectorOrdering(*fespace);
   const auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
                     DofToQuad::LEXICOGRAPHIC_FULL;
   geom = mesh.GetGeometricFactors(
             *IntRule,
             GeometricFactors::JACOBIANS |
             GeometricFactors::DETERMINANTS);
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);

   const int expected_size =
      (dim*dim + 2) * parent.lambda_quad->Size();
   if (parent.pa_data.Size() != expected_size)
   {
      internal::ElasticitySetupPAData(dim, *IntRule, *parent.lambda_quad,
                                      *parent.mu_quad, *geom, parent.pa_data);
   }
}

void ElasticityComponentIntegrator::AddMultPA(const Vector &x,
                                              Vector &y) const
{
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *maps, parent.pa_data,
      x, *parent.q_vec, y, i_block, j_block);
}

void ElasticityComponentIntegrator::AddMultTransposePA(
   const Vector &x, Vector &y) const
{
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *maps, parent.pa_data,
      x, *parent.q_vec, y, j_block, i_block);
}

} // namespace mfem
