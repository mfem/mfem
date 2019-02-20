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
#include "../linalg/device.hpp"

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
void DiffusionIntegrator::Assemble(const FiniteElementSpace &fes)
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
void DiffusionIntegrator::MultAssembled(Vector &x, Vector &y)
{
   kernels::fem::DiffusionMultAssembled(dim, dofs1D, quad1D, ne,
                                        maps->B,
                                        maps->G,
                                        maps->Bt,
                                        maps->Gt,
                                        vec, x, y);
}

// *****************************************************************************
// * PA Mass Integrator Extension
// *****************************************************************************
void MassIntegrator::Assemble(const FiniteElementSpace &fes)
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
      double (*function)(const Vector3&) = NULL;
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
      const DeviceVector w(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geo->X.GetData(), 3,NQ,NE);
      const DeviceTensor<4> J(geo->J.GetData(), 2,2,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e);
            const double J12 = J(1,0,q,e);
            const double J21 = J(0,1,q,e);
            const double J22 = J(1,1,q,e);
            const double detJ = (J11*J22)-(J21*J12);
            const int offset = dims*NQ*e+q;
            const double coeff =
            const_coeff ? constant:
            function_coeff ?
               function(Vector3(x[offset], x[offset+1], x[offset+2])):
            0.0;
            v(q,e) =  w[q] * coeff * detJ;
         }
      });
   }
   if (dim==3)
   {
      double constant = 0.0;
      double (*function)(const Vector3&) = NULL;
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
      const DeviceVector W(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geo->X.GetData(), 3,NQ,NE);
      const DeviceTensor<4> J(geo->J.GetData(), 3,3,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ,NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e),J12 = J(1,0,q,e),J13 = J(2,0,q,e);
            const double J21 = J(0,1,q,e),J22 = J(1,1,q,e),J23 = J(2,1,q,e);
            const double J31 = J(0,2,q,e),J32 = J(1,2,q,e),J33 = J(2,2,q,e);
            const double detJ =
            ((J11 * J22 * J33) + (J12 * J23 * J31) + (J13 * J21 * J32) -
            (J13 * J22 * J31) - (J12 * J21 * J33) - (J11 * J23 * J32));
            const int offset = dims*NQ*e+q;
            const double coeff =
            const_coeff ? constant:
            function_coeff ?
               function(Vector3(x[offset], x[offset+1], x[offset+2])):
            0.0;
            v(q,e) = W(q) * coeff * detJ;
         }
      });
   }
   //delete geo;
}

// *****************************************************************************
void MassIntegrator::MultAssembled(Vector &x, Vector &y)
{
   kernels::fem::MassMultAssembled(dim, dofs1D, quad1D, ne,
                                   maps->B, maps->Bt,
                                   vec, x, y);
}

} // namespace mfem
