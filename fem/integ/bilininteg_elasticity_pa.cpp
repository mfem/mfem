// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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


void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   if(parent){
      // This is a component integrator, so just make sure monolithic operator
      // is assembled.
      parent->AssemblePA(fes);
      return;
   }
   const bool alreadyAssembled = bool(lambda_quad);
   if(alreadyAssembled){
      // Don't reassemble vectors.
      return;
   }
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Elasticity PA only implemented for byNODES ordering.");

   fespace = &fes;
   const auto el = fespace->GetFE(0);
   ndofs = el->GetDof();
   const auto mesh = fespace->GetMesh();
   vdim = fespace->GetVDim();
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      //This is where it's assumed that all elements are the same.
      const auto Trans = fespace->GetElementTransformation(0);
      int order = 2 * Trans->OrderGrad(el);
      IntRule = &IntRules.Get(el->GetGeomType(), order);
   }
   geom = mesh->GetGeometricFactors(*IntRule, GeometricFactors::JACOBIANS);
   quad_space = std::make_shared<QuadratureSpace>(*mesh, *IntRule);
   lambda_quad = std::make_shared<CoefficientVector>(lambda, *quad_space,
                                                     CoefficientStorage::FULL);
   mu_quad = std::make_shared<CoefficientVector>(mu, *quad_space,
                                                 CoefficientStorage::FULL);
   q_vec = std::make_shared<QuadratureFunction>(*quad_space, vdim*vdim);
   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetFE(0)->GetDofToQuad(*IntRule, mode);
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   q_vec->SetVDim(vdim*vdim*vdim*vdim);
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *fespace, *lambda_quad,
                                          *mu_quad, *geom, *maps, *q_vec, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (!parent)
   {
      q_vec->SetVDim(vdim*vdim);
   }
   else
   {
      //If it has a parent, it is a component integrator.
      q_vec->SetVDim(vdim);
   }

   internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *lambda_quad, *mu_quad,
                                 *geom, *maps, x, *q_vec, y, IBlock, JBlock);
}

void ElasticityIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (!parent)
   {
      AddMultPA(x, y);
   }
   else
   {
      //This block operator is symmetric, so simply switch IBlock and JBlock.
      internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *lambda_quad, *mu_quad,
                                    *geom, *maps, x, *q_vec, y, JBlock, IBlock);
   }
}

} // namespace mfem
