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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PABdrVectorMassApply2D(const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &Bt_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B  = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto D  = Reshape(d_.Read(), VDIM, VDIM, Q1D, Q1D, NE);
   auto x  = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y  = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      double sol_xy[max_Q1D][max_Q1D][VDIM];
      for (int c = 0; c < VDIM; ++c)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int cc = 0; cc < VDIM; ++cc)
               {
                  sol_xy[qy][qx][cc] = 0.0;
               }
            }
         }

         // dof -> quad.
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[max_Q1D][VDIM];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int cc = 0; cc < VDIM; ++cc)
               {
                  sol_x[qy][cc] = 0.0;
               }
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               for (int cc = 0; cc < VDIM; cc++)
               {
                  const double s = x(dx,dy,cc,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_x[qx][cc] += B(qx,dx)* s;
                  }
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double d2q = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int cc = 0; cc < VDIM; cc++)
                  {
                     sol_xy[qy][qx][cc] += d2q * sol_x[qx][cc];
                  }
               }
            }
         }

         // quad data.
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int cc = 0; cc < VDIM; cc++)
               {
                  sol_xy[qy][qx][cc] *= D(c,cc,qx,qy,e);
               }
            }
         }

         // quad -> dof.
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[max_D1D][VDIM];
            for (int dx = 0; dx < D1D; ++dx)
            {
               for (int cc = 0; cc < VDIM; cc++)
               {
                  sol_x[dx][cc] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int cc = 0; cc < VDIM; cc++)
               {
                  const double s = sol_xy[qy][qx][cc];
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     sol_x[dx][cc] += Bt(dx,qx) * s;
                  }
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double q2d = Bt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  for (int cc = 0; cc < VDIM; cc++)
                  {
                     y(dx,dy,c,e) += q2d * sol_x[dx][cc];
                  }
               }
            }
         }
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PABdrVectorMassApply3D(const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &Bt_,
                                   const Vector &q_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{

}

void BoundaryVectorMassIntegrator::
     AssemblePABoundaryFaces(const FiniteElementSpace &fes)
{
   nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf == 0) { return; }

   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0));

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * fes.FEColl()->GetOrder();
      ir = &IntRules.Get(mesh->GetFaceGeometry(0), order);
   }

   nq = ir->GetNPoints();
   dim = mesh->Dimension();

   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
}

void BoundaryVectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      MFEM_VERIFY(vdim == 2, "Not implemented for genereal vdim");
      return PABdrVectorMassApply2D(ne, maps->B, maps->Bt, pa_data,
                                    x, y, dofs1D, quad1D);
   }
   if (dim == 3)
   {
      MFEM_VERIFY(vdim == 3, "Not implemented for genereal vdim");
      return PABdrVectorMassApply3D(ne, maps->B, maps->Bt, pa_data,
                                    x, y, dofs1D, quad1D);
   }
}

} // namespace mfem
