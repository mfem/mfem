// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "fem.hpp"
#include <cmath>
#include <algorithm>
#include "bilininteg.hpp"
#include "geom_ext.hpp"
#include "kernels/mass.hpp"
#include "kernels/diffusion.hpp"

namespace mfem
{

// *****************************************************************************
static const IntegrationRule &DefaultGetRule(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}


// *****************************************************************************
// * PA DiffusionIntegrator Extension
// *****************************************************************************

// *****************************************************************************
void DiffusionIntegrator::Assemble(const FiniteElementSpace *fes)
{
   const Mesh *mesh = fes->GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *(fes->GetFE(0));
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int quadraturePoints = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes->GetMesh()->GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   const GeometryExtension *geo = GeometryExtension::Get(*fes,*ir);
   maps = DofToQuad::Get(*fes, *fes, *ir);
   vec.SetSize(symmDims * quadraturePoints * ne);
   const double coeff = static_cast<ConstantCoefficient*>(Q)->constant;
   kernels::fem::DiffusionAssemble(dim, quad1D, ne,
                                   maps->quadWeights,
                                   geo->J,
                                   coeff,
                                   vec);
   delete geo;
}

// *****************************************************************************
void DiffusionIntegrator::MultAssembled(Vector &x, Vector &y)
{
   kernels::fem::DiffusionMultAssembled(dim, dofs1D, quad1D, ne,
                                        maps->dofToQuad,
                                        maps->dofToQuadD,
                                        maps->quadToDof,
                                        maps->quadToDofD,
                                        vec, x, y);
}

// *****************************************************************************
// * PA Mass Integrator Extension
// *****************************************************************************
void MassIntegrator::Assemble(const FiniteElementSpace *fes)
{
   const Mesh *mesh = fes->GetMesh();
   const IntegrationRule *ir = IntRule;
   const FiniteElement &el = *(fes->GetFE(0));
   dim = mesh->Dimension();
   ne = fes->GetMesh()->GetNE();
   dofs1D = el.GetOrder() + 1;
   assert(ir);
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   maps = DofToQuad::Get(*fes, *fes, *IntRule);
}

// *****************************************************************************
void MassIntegrator::SetOperator(Vector &v)
{
   vec.SetSize(v.Size());
   vec = v;
}

// *****************************************************************************
void MassIntegrator::MultAssembled(Vector &x, Vector &y)
{
   kernels::fem::MassMultAssembled(dim, dofs1D, quad1D, ne,
                                   maps->dofToQuad,
                                   maps->dofToQuadD,
                                   maps->quadToDof,
                                   maps->quadToDofD,
                                   vec, x, y);
}

} // namespace mfem
