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

using namespace std;

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
// * PADiffusionIntegrator
// *****************************************************************************
void PADiffusionIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   const GeometryExtension *geo = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(symmDims * nq * ne);
   const double coeff = static_cast<ConstantCoefficient*>(Q)->constant;
   kernels::fem::DiffusionAssemble(dim, quad1D, ne,
                                   maps->W,
                                   geo->J,
                                   coeff,
                                   vec);
   delete geo;
}

// *****************************************************************************
void PADiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   kernels::fem::DiffusionMultAssembled(dim, dofs1D, quad1D, ne,
                                        maps->B,
                                        maps->G,
                                        maps->Bt,
                                        maps->Gt,
                                        vec, x, y);
}


// *****************************************************************************
// * PAMassIntegrator
// *****************************************************************************
void PAMassIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   const GeometryExtension *geo = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(ne*nq);
   ConstantCoefficient *const_coeff = dynamic_cast<ConstantCoefficient*>(Q);
   FunctionCoefficient *function_coeff = dynamic_cast<FunctionCoefficient*>(Q);
   // TODO: other types of coefficients ...
   if (dim==1) { mfem_error("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      double constant = 0.0;
      double (*function)(const DeviceVector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
      else if (function_coeff)
      {
         function = function_coeff->GetFunction();
      }
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      const int dims = el.GetDim();
      const double *w = maps->W.GetMmData();
      const double *x = geo->X.GetMmData();
      const double *J = geo->J.GetMmData();
      double *v = vec.GetMmData();
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J[ijklNM(0,0,q,e,2,NQ)];
            const double J12 = J[ijklNM(1,0,q,e,2,NQ)];
            const double J21 = J[ijklNM(0,1,q,e,2,NQ)];
            const double J22 = J[ijklNM(1,1,q,e,2,NQ)];
            const double detJ = (J11*J22)-(J21*J12);
            const int offset = dims*NQ*e+q;
            const double coeff =
            const_coeff ? constant:
               function_coeff ? function(DeviceVector3(x[offset],
                                                       x[offset+1],
                                                       x[offset+2])):
            0.0;
            v[ijN(q,e,NQ)] =  w[q] * coeff * detJ;
         }
      });
   }
   if (dim==3) { mfem_error("Not supported yet... stay tuned!"); }
   //delete geo;
}

// *****************************************************************************
void PAMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   kernels::fem::MassMultAssembled(dim, dofs1D, quad1D, ne,
                                   maps->B, maps->G,
                                   maps->Bt, maps->Gt,
                                   vec, x, y);
}

} // namespace mfem
