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
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes->GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   const GeometryExtension *geo = GeometryExtension::Get(*fes,*ir);
   maps = DofToQuad::Get(*fes, *fes, *ir);
   vec.SetSize(symmDims * nq * ne);
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
//MFEM_FUNCTION static double rho0(const Vector &x) { return 1.0; }

void MassIntegrator::Assemble(const FiniteElementSpace *fes)
{
   const Mesh *mesh = fes->GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *(fes->GetFE(0));
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   dim = mesh->Dimension();
   ne = fes->GetMesh()->GetNE();
   nq = ir->GetNPoints();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   const GeometryExtension *geo = GeometryExtension::Get(*fes,*ir);
   maps = DofToQuad::Get(*fes, *fes, *ir);
   vec.SetSize(ne*nq);
   // Values of Q->eval * DetJ at all quadrature points ************************
   /*for (int e = 0; e < ne; e++)
   {
      ElementTransformation *eltrans = fes->GetElementTransformation(e);
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         eltrans->SetIntPoint(&ip);            
         const double qeval = Q?Q->Eval(*eltrans, ip):1.0;
         const double weights = eltrans->Weight() * ip.weight;
         vec(e*nq + q) = weights * qeval;
      }
      }*/
   /*
   for (int e = 0; e < ne; e++){
      for (int q = 0; q < nq; q++){
         printf("\n\t%f",vec(e*nq + q));
      }
      }*/
   // **************************************************************************
   //const double coeff = 1.0;
   //Q?static_cast<ConstantCoefficient*>(Q)->constant:1.0;
   /*
   ConstantCoefficient *cstq = dynamic_cast<ConstantCoefficient*>(Q);
   if (!cstq) {
      mfem_error("ConstantCoefficient only supported yet...!");
      }
   dbg("Qeval=%f",cstq->constant);
   */
/*
#warning enabling Gpu
   config::useCuda();
   config::enableGpu(0);
*/
   if (dim==1) { mfem_error("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      //config::GPU mode;
      MFEM_ASSERT(Q, "Q not a Coefficient");      
      FunctionCoefficient *fctQ = dynamic_cast<FunctionCoefficient*>(Q);
      double (*coeffFunction)(const Vector &) = fctQ->Get();
      
      const Vector &nodes = *mesh->GetNodes();
      const int NE = ne;
      const int NQ = nq;
      const double *w = (const double*) mm::ptr((const double*)maps->quadWeights);
      const double *J = (const double*) mm::ptr((const double*)geo->J);
      double *v = (double*) mm::ptr((double*)vec.GetData());
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J[ijklNM(0,0,q,e,2,NQ)];
            const double J12 = J[ijklNM(1,0,q,e,2,NQ)];
            const double J21 = J[ijklNM(0,1,q,e,2,NQ)];
            const double J22 = J[ijklNM(1,1,q,e,2,NQ)];
            const double detJ = (J11*J22)-(J21*J12);
            //v[ijN(q,e,NQ)] =  w[q] * rho0(nodes) * detJ;
            v[ijN(q,e,NQ)] =  w[q] * coeffFunction(nodes) * detJ;
         }
      });
   }/*
   vec.Pull();
   
   dbg("Kernel:");
   for (int e = 0; e < ne; e++){
      for (int q = 0; q < nq; q++){
         printf("\n\t%f",vec(e*nq + q));
      }
      }
    */
   if (dim==3){ mfem_error("Not supported yet... stay tuned!"); }
   //delete geo;
   //exit(0);
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
