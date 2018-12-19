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

#include "diffusion.hpp"
#include "mfem4/general/mm.hpp"

// TODO: move
#define MFEM_RESTRICT __restrict
#define MFEM_FORALL(i, n, body) for (int i = 0; i < (n); i++) body

namespace mfem4
{

static IntegrationRule& DiffusionIntRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }

   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   else
   {
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }
}


//// Full Assembly /////////////////////////////////////////////////////////////




// TODO



//// Partial Assembly //////////////////////////////////////////////////////////

// raw pointer version
static void
DiffusionPartialAssemble2D(const int ne, const int nq_1d,
                           const double* MFEM_RESTRICT weights,
                           const double* MFEM_RESTRICT J,
                           const double coeff,
                           double* MFEM_RESTRICT oper)
{
   weights = MM::DevicePtr(weights);
   J       = MM::DevicePtr(J);
   oper    = MM::DevicePtr(oper);

   const int nq = nq_1d*nq_1d;

   MFEM_FORALL(e, ne,
   {
      for (int q = 0; q < nq; ++q)
      {
         double J11 = J[ijklNM(0,0,q,e, 2,nq)];
         double J12 = J[ijklNM(1,0,q,e, 2,nq)];
         double J21 = J[ijklNM(0,1,q,e, 2,nq)];
         double J22 = J[ijklNM(1,1,q,e, 2,nq)];

         double c_detJ = weights[q] * coeff / ((J11*J22)-(J21*J12));

         oper[ijkNM(0,q,e,3,NUM_QUAD)] =  c_detJ * (J21*J21 + J22*J22);
         oper[ijkNM(1,q,e,3,NUM_QUAD)] = -c_detJ * (J21*J11 + J22*J12);
         oper[ijkNM(2,q,e,3,NUM_QUAD)] =  c_detJ * (J11*J11 + J12*J12);
      }
   })
}

// DeviceTensor version
static void
DiffusionPartialAssemble2D(const int ne, const int nq_1d,
                           const DeviceTensor<1> weights,
                           const DeviceTensor<4> J,
                           const double coeff,
                           DeviceTensor<3> oper)
{
   const int nq = nq_1d*nq_1d;

   MFEM_FORALL(e, ne,
   {
      for (int q = 0; q < nq; ++q)
      {
         double J11 = J(0,0,q,e), J12 = J(1,0,q,e);
         double J21 = J(0,1,q,e), J22 = J(1,1,q,e);

         double c_detJ = weights[q] * coeff / ((J11*J22)-(J21*J12));

         oper(0,q,e) =  c_detJ * (J21*J21 + J22*J22);
         oper(1,q,e) = -c_detJ * (J21*J11 + J22*J12);
         oper(2,q,e) =  c_detJ * (J11*J11 + J12*J12);
      }
   })
}


AssemblyData*
DiffusionIntegrator::PartialAssemble(const BatchGeometry &geom,
                                     const FiniteElement &trial_fe,
                                     const FiniteElement &test_fe)
{
   Geometry::Type geom = trial_fe.GetGeomType();
   MFEM_VERIFY(geom == Geometry::SQUARE || geom == Geometry::CUBE, "");

   const IntegrationRule &ir = DiffusionIntRule(trial_fe, test_fe);
   const IntegrationRule &ir_1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());

   DiffusionAssemblyData *ad = new DiffusionAssemblyData;

   int dim = trial_fe.GetDim();
   if (dim == 2)
   {
      DiffusionPartialAssemble2D(geom.GetNE(),
                                 ir_1d.GetNPoints(),
                                 weights, // TODO
                                 geom.GetJacobians(ir),
                                 1.0, // TODO Coefficient
                                 ad->oper);
   }
   /*else if (dim == 3)
   {
   }*/
   else
   {
      MFEM_ABORT("Unsupported dim.");
   }

   return ad;
}



//// PA Action /////////////////////////////////////////////////////////////////

void DiffusionIntegrator::MultAdd(const AssemblyData &assembly,
                                  const Vector &x, Vector &y)
{

}




} // namespace mfem4
