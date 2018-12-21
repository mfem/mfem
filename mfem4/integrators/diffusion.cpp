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

/*// raw pointer version
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
}*/

// DeviceTensor version
static void
DiffusionPartialAssemble2D(const int num_elements, const int num_quad_1d,
                           const DeviceTensor<1> &weights,
                           const DeviceTensor<4> &J,
                           const double coeff,
                           DeviceTensor<3> &oper)
{
   const int nq = num_quad_1d*num_quad_1d;

   MFEM_FORALL(e, num_elements,
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
      MFEM_ABORT("Unsupported dimension.");
   }

   return ad;
}



//// PA Action /////////////////////////////////////////////////////////////////


template<int NUM_DOFS_1D,
         int NUM_QUAD_1D,
         int MAX_DOFS_1D = NUM_DOFS_1D,
         int MAX_QUAD_1D = NUM_QUAD_1D>
static void
DiffusionMultAdd2D(int num_elements, int num_dofs_1d, int num_quad_1d,
                   const DeviceTensor<2> &dofToQuad,
                   const DeviceTensor<2> &dofToQuadD,
                   const DeviceTensor<2> &quadToDof,
                   const DeviceTensor<2> &quadToDofD,
                   const DeviceTensor<3> &oper,
                   const DeviceTensor<3> &solIn,
                   DeviceTensor<3> &solOut)
{
   // attempt to support both compile-time and run-time sizes
   const int nd_1d = NUM_DOFS_1D ? NUM_DOFS_1D : num_dofs_1d;
   const int nq_1d = NUM_QUAD_1D ? NUM_QUAD_1D : num_quad_1d;

   MFEM_FORALL(e, num_elements,
   {
      double grad[MAX_QUAD_1D][MAX_QUAD_1D][2];
      for (int qy = 0; qy < nq_1d; ++qy)
      {
         for (int qx = 0; qx < nq_1d; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }

      for (int dy = 0; dy < nd_1d; ++dy)
      {
         double gradX[MAX_QUAD_1D][2];
         for (int qx = 0; qx < nq_1d; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }

         for (int dx = 0; dx < nd_1d; ++dx)
         {
            const double s = solIn(dx, dy, e);
            for (int qx = 0; qx < nq_1d; ++qx)
            {
               gradX[qx][0] += s * dofToQuad(qx, dx);
               gradX[qx][1] += s * dofToQuadD(qx, dx);
            }
         }

         for (int qy = 0; qy < nq_1d; ++qy)
         {
            const double wy  = dofToQuad(qy, dy);
            const double wDy = dofToQuadD(qy, dy);
            for (int qx = 0; qx < nq_1d; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }

      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < nq_1d; ++qy)
      {
         for (int qx = 0; qx < nq_1d; ++qx)
         {
            const int q = QUAD_2D_ID(qx, qy);

            const double O11 = oper(0, q, e);
            const double O12 = oper(1, q, e);
            const double O22 = oper(2, q, e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }

      for (int qy = 0; qy < nq_1d; ++qy)
      {
         double gradX[MAX_DOFS_1D][2];
         for (int dx = 0; dx < nd_1d; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }

         for (int qx = 0; qx < nq_1d; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < nd_1d; ++dx)
            {
               const double wx  = quadToDof(dx, qx);
               const double wDx = quadToDofD(dx, qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < nd_1d; ++dy)
         {
            const double wy  = quadToDof(dy, qy);
            const double wDy = quadToDofD(dy, qy);
            for (int dx = 0; dx < nd_1d; ++dx)
            {
               solOut(dx,dy,e) += ((gradX[dx][0] * wy) +
                                   (gradX[dx][1] * wDy));
            }
         }
      }
   })
}


typedef void (*fDiffusionMultAdd)(
      int num_elements, int num_dofs_1d, int num_quad_1d,
      const DeviceTensor<2> &dofToQuad,
      const DeviceTensor<2> &dofToQuadD,
      const DeviceTensor<2> &quadToDof,
      const DeviceTensor<2> &quadToDofD,
      const DeviceTensor<3> &oper,
      const DeviceTensor<3> &solIn,
      DeviceTensor<3> &solOut);


void DiffusionIntegrator::MultAdd(const AssemblyData &assembly,
                                  const Vector &x, Vector &y)
{
   DiffusionAssemblyData *ad = dynamic_cast<DiffusionAssemblyData*>(&assembly);
   MFEM_VERIFY(ad, "");

   int num_elements = ad->oper.Size(2);
   int num_quad_1d = ad->dof_quad.Size(0);
   int num_dofs_1d = ad->dof_quad.Size(1);

   int key = (num_dofs_1d << 8) | num_quad_1d;

   if (ad->dim == 2)
   {
      void* call;
      switch (key)
      {
         case 0x0101: call = DiffusionMultAdd2D<1, 1>; break;
         case 0x0201: call = DiffusionMultAdd2D<2, 1>; break;
         case 0x0202: call = DiffusionMultAdd2D<2, 2>; break;
         case 0x0303: call = DiffusionMultAdd2D<3, 3>; break;
         default:     call = DiffusionMultAdd2D<0, 0, 10, 10>; break;
      }

      ((fDiffusionMultAdd) call)(num_elements, num_dofs_1d, num_quad_1d,
                                 ad->dof_quad, ad->dof_quad_d,
                                 ad->quad_dof, ad->quad_dof_d,
                                 ad->oper,
                                 TODO);

   }
   /*else if (ad->dim == 3)
   {
   }*/
   else
   {
      MFEM_ABORT("Unsupported dimension.");
   }
}




} // namespace mfem4
