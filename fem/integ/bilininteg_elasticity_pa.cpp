// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <memory>
#include <tuple>

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
   if (expected_vdim > 1)
   {
      MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
                  "Elasticity PA only supports Ordering::byNODES.");
   }
   MFEM_VERIFY(fes.GetVDim() == expected_vdim,
               "Unexpected vector dimension for elasticity PA.");
   MFEM_VERIFY(!fes.IsVariableOrder(),
               "Elasticity PA does not support variable-order spaces.");
   // Empty MPI partitions (GetNE()==0) must succeed: setup and apply kernels
   // are zero-length foralls. GetTypicalFE() already handles that case.
   MFEM_VERIFY(mesh.GetNumGeometries(dim) <= 1,
               "Elasticity PA requires a single element geometry.");
   const FiniteElement &typical_fe = *fes.GetTypicalFE();
   MFEM_VERIFY(typical_fe.GetRangeType() == FiniteElement::SCALAR &&
               typical_fe.GetMapType() == FiniteElement::VALUE,
               "Elasticity PA requires scalar, VALUE-mapped finite elements.");
}

bool SameQuadrature(const QuadratureSpace &qs, const Mesh &mesh,
                    const IntegrationRule &ir)
{
   if (qs.GetMesh() != &mesh) { return false; }
   // An empty partition has no element whose rule can be queried.
   if (mesh.GetNE() == 0) { return true; }
   return &qs.GetIntRule(0) == &ir;
}

bool SameValues(const Vector &a, const Vector &b)
{
   if (a.Size() != b.Size()) { return false; }
   Vector difference(a);
   difference -= b;
   return difference.Norml2() == 0.0;
}

} // namespace


void ElasticityIntegrator::ProjectCoefficients(
   CoefficientVector &lambda_out, CoefficientVector &mu_out) const
{
   MFEM_VERIFY(mu != nullptr, "The shear modulus coefficient is not set.");
   if (lambda != nullptr)
   {
      lambda_out.Project(*lambda);
      mu_out.Project(*mu);
   }
   else
   {
      ProductCoefficient lambda_coefficient(q_lambda, *mu);
      ProductCoefficient mu_coefficient(q_mu, *mu);
      lambda_out.Project(lambda_coefficient);
      mu_out.Project(mu_coefficient);
   }
}

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
   // Component QuadratureFunctions keep a raw pointer to this space, so a
   // refresh replaces it only when the mesh or the rule actually changes.
   q_vec.reset();
   const bool reuse = q_space && SameQuadrature(*q_space, mesh, *IntRule);
   if (!reuse)
   {
      lambda_quad.reset();
      mu_quad.reset();
      q_space = std::make_unique<QuadratureSpace>(mesh, *IntRule);
      lambda_quad = std::make_unique<CoefficientVector>(
                       *q_space, CoefficientStorage::FULL);
      mu_quad = std::make_unique<CoefficientVector>(
                   *q_space, CoefficientStorage::FULL);
   }

   ProjectCoefficients(*lambda_quad, *mu_quad);
   pa_data.SetSize(0);
}

void ElasticityIntegrator::EnsureSharedPAData(const FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   const long seq = mesh.GetNodesSequence();
   const bool same_space = q_space && IntRule != nullptr &&
                           SameQuadrature(*q_space, mesh, *IntRule);
   const int expected =
      same_space ? (vdim * vdim + 2) * lambda_quad->Size() : -1;
   const bool geom_ok =
      same_space && pa_geom_sequence == seq && pa_data.Size() == expected;

   if (geom_ok)
   {
      CoefficientVector lambda_next(*q_space, CoefficientStorage::FULL);
      CoefficientVector mu_next(*q_space, CoefficientStorage::FULL);
      ProjectCoefficients(lambda_next, mu_next);
      // Same nodes and the same material: the other blocks share pa_data.
      if (SameValues(*lambda_quad, lambda_next) &&
          SameValues(*mu_quad, mu_next))
      {
         return;
      }
      ProjectCoefficients(*lambda_quad, *mu_quad);
   }
   else
   {
      SetUpQuadratureSpaceAndCoefficients(fes);
   }

   const GeometricFactors *geom = mesh.GetGeometricFactors(
                                     *IntRule,
                                     GeometricFactors::JACOBIANS |
                                     GeometricFactors::DETERMINANTS);
   internal::ElasticitySetupPAData(vdim, *IntRule, *lambda_quad, *mu_quad,
                                   *geom, pa_data);
   pa_geom_sequence = seq;
}

void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   VerifyElasticityPASpace(fes, mesh.Dimension());

   vdim = mesh.Dimension();
   ndofs = fespace->GetTypicalFE()->GetDof();
   EnsureSharedPAData(fes);

   const auto ordering = GetEVectorOrdering(*fespace);
   const auto mode =
      ordering == ElementDofOrdering::NATIVE ?
      DofToQuad::FULL : DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);

   use_tensor_pa = false;
   tensor_maps = nullptr;
   tensor_apply = nullptr;
   d1d = 0;
   q1d = 0;
   const FiniteElement &fe = *fespace->GetTypicalFE();
   // Tensor kernels require lexicographic element vectors. Unspecialized
   // D1D/Q1D pairs keep the generic maps and skip the aborting Fallback.
   if (ordering == ElementDofOrdering::LEXICOGRAPHIC &&
       dynamic_cast<const TensorBasisElement *>(&fe) != nullptr)
   {
      tensor_maps = &fe.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
      d1d = tensor_maps->ndof;
      q1d = tensor_maps->nqpt;
      const auto &table = ApplyPAKernels::GetDispatchTable();
      const auto it = table.find(std::make_tuple(vdim, d1d, q1d));
      if (it != table.end())
      {
         use_tensor_pa = true;
         tensor_apply = it->second;
      }
   }
   if (!use_tensor_pa)
   {
      q_vec = std::make_unique<QuadratureFunction>(*q_space, 2*vdim*vdim);
   }
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *maps, *IntRule,
                                          pa_data, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (use_tensor_pa)
   {
      tensor_apply(fespace->GetNE(), *tensor_maps, pa_data, x, y);
      return;
   }

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
   if (parent.q_space)
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
   else if (parent.IntRule == nullptr)
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
   parent.EnsureSharedPAData(fes);

   IntRule = parent.IntRule;
   MFEM_VERIFY(IntRule != nullptr,
               "Elasticity component integration rule was not initialized.");

   const auto ordering = GetEVectorOrdering(*fespace);
   const auto mode =
      ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
      DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   q_vec = std::make_unique<QuadratureFunction>(*parent.q_space, 2*dim);
}

void ElasticityComponentIntegrator::AddMultPA(const Vector &x,
                                              Vector &y) const
{
   ApplyPAKernels::Run(parent.vdim, i_block, j_block,
                       parent.ndofs, *fespace, *maps, parent.pa_data,
                       x, *q_vec, y);
}

void ElasticityComponentIntegrator::AddMultTransposePA(
   const Vector &x, Vector &y) const
{
   ApplyPAKernels::Run(parent.vdim, j_block, i_block,
                       parent.ndofs, *fespace, *maps, parent.pa_data,
                       x, *q_vec, y);
}

} // namespace mfem
