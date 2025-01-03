// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../nonlininteg.hpp"
#include "../ceed/integrators/nlconvection/nlconvection.hpp"

namespace mfem
{

void VectorConvectionNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAVectorConvectionNLIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAVectorConvectionNLFIntegrator(fes, *ir, Q);
      }
      return;
   }
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   pa_data.SetSize(ne * nq * dim * dim, Device::GetMemoryType());
   real_t COEFF = 1.0;
   if (Q)
   {
      ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient *>(Q);
      MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
      COEFF = cQ->constant;
   }
   const int NE = ne;
   const int NQ = nq;
   auto W = ir->GetWeights().Read();
   if (dim == 1)
   {
      MFEM_ABORT("dim==1 not supported!");
   }
   if (dim == 2)
   {
      auto J = Reshape(geom->J.Read(), NQ, 2, 2, NE);
      auto G = Reshape(pa_data.Write(), NQ, 2, 2, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            const real_t J11 = J(q, 0, 0, e);
            const real_t J12 = J(q, 0, 1, e);
            const real_t J21 = J(q, 1, 0, e);
            const real_t J22 = J(q, 1, 1, e);
            // Store wq * Q * adj(J)
            G(q, 0, 0, e) = W[q] * COEFF * J22;  // 1,1
            G(q, 0, 1, e) = W[q] * COEFF * -J12; // 1,2
            G(q, 1, 0, e) = W[q] * COEFF * -J21; // 2,1
            G(q, 1, 1, e) = W[q] * COEFF * J11;  // 2,2
         }
      });
   }
   if (dim == 3)
   {
      auto J = Reshape(geom->J.Read(), NQ, 3, 3, NE);
      auto G = Reshape(pa_data.Write(), NQ, 3, 3, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            const real_t J11 = J(q, 0, 0, e);
            const real_t J21 = J(q, 1, 0, e);
            const real_t J31 = J(q, 2, 0, e);
            const real_t J12 = J(q, 0, 1, e);
            const real_t J22 = J(q, 1, 1, e);
            const real_t J32 = J(q, 2, 1, e);
            const real_t J13 = J(q, 0, 2, e);
            const real_t J23 = J(q, 1, 2, e);
            const real_t J33 = J(q, 2, 2, e);
            const real_t cw = W[q] * COEFF;
            // adj(J)
            const real_t A11 = (J22 * J33) - (J23 * J32);
            const real_t A12 = (J32 * J13) - (J12 * J33);
            const real_t A13 = (J12 * J23) - (J22 * J13);
            const real_t A21 = (J31 * J23) - (J21 * J33);
            const real_t A22 = (J11 * J33) - (J13 * J31);
            const real_t A23 = (J21 * J13) - (J11 * J23);
            const real_t A31 = (J21 * J32) - (J31 * J22);
            const real_t A32 = (J31 * J12) - (J11 * J32);
            const real_t A33 = (J11 * J22) - (J12 * J21);
            // Store wq * Q * adj(J)
            G(q, 0, 0, e) = cw * A11; // 1,1
            G(q, 0, 1, e) = cw * A12; // 1,2
            G(q, 0, 2, e) = cw * A13; // 1,3
            G(q, 1, 0, e) = cw * A21; // 2,1
            G(q, 1, 1, e) = cw * A22; // 2,2
            G(q, 1, 2, e) = cw * A23; // 2,3
            G(q, 2, 0, e) = cw * A31; // 3,1
            G(q, 2, 1, e) = cw * A32; // 3,2
            G(q, 2, 2, e) = cw * A33; // 3,3
         }
      });
   }
}

// PA Convection NL 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void PAConvectionNLApply2D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Array<real_t> &bt,
                                  const Vector &q_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Q = Reshape(q_.Read(), Q1D * Q1D, 2, 2, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, 2, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, 2, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t data[max_Q1D][max_Q1D][2];
      real_t grad0[max_Q1D][max_Q1D][2];
      real_t grad1[max_Q1D][max_Q1D][2];
      real_t Z[max_Q1D][max_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            data[qy][qx][0] = 0.0;
            data[qy][qx][1] = 0.0;
            grad0[qy][qx][0] = 0.0;
            grad0[qy][qx][1] = 0.0;
            grad1[qy][qx][0] = 0.0;
            grad1[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t dataX[max_Q1D][2];
         real_t gradX0[max_Q1D][2];
         real_t gradX1[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            dataX[qx][0] = 0.0;
            dataX[qx][1] = 0.0;
            gradX0[qx][0] = 0.0;
            gradX0[qx][1] = 0.0;
            gradX1[qx][0] = 0.0;
            gradX1[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s0 = x(dx, dy, 0, e);
            const real_t s1 = x(dx, dy, 1, e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t Bx = B(qx, dx);
               const real_t Gx = G(qx, dx);
               dataX[qx][0] += s0 * Bx;
               dataX[qx][1] += s1 * Bx;
               gradX0[qx][0] += s0 * Gx;
               gradX0[qx][1] += s0 * Bx;
               gradX1[qx][0] += s1 * Gx;
               gradX1[qx][1] += s1 * Bx;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t By = B(qy, dy);
            const real_t Gy = G(qy, dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               data[qy][qx][0] += dataX[qx][0] * By;
               data[qy][qx][1] += dataX[qx][1] * By;
               grad0[qy][qx][0] += gradX0[qx][0] * By;
               grad0[qy][qx][1] += gradX0[qx][1] * Gy;
               grad1[qy][qx][0] += gradX1[qx][0] * By;
               grad1[qy][qx][1] += gradX1[qx][1] * Gy;
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;
            const real_t u1 = data[qy][qx][0];
            const real_t u2 = data[qy][qx][1];
            const real_t grad00 = grad0[qy][qx][0];
            const real_t grad01 = grad0[qy][qx][1];
            const real_t grad10 = grad1[qy][qx][0];
            const real_t grad11 = grad1[qy][qx][1];
            const real_t Dxu1 = grad00 * Q(q, 0, 0, e) + grad01 * Q(q, 1, 0, e);
            const real_t Dyu1 = grad00 * Q(q, 0, 1, e) + grad01 * Q(q, 1, 1, e);
            const real_t Dxu2 = grad10 * Q(q, 0, 0, e) + grad11 * Q(q, 1, 0, e);
            const real_t Dyu2 = grad10 * Q(q, 0, 1, e) + grad11 * Q(q, 1, 1, e);
            Z[qy][qx][0] = u1 * Dxu1 + u2 * Dyu1;
            Z[qy][qx][1] = u1 * Dxu2 + u2 * Dyu2;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t Y[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y[dx][0] = 0.0;
            Y[dx][1] = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t Btx = Bt(dx, qx);
               Y[dx][0] += Btx * Z[qy][qx][0];
               Y[dx][1] += Btx * Z[qy][qx][1];
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t Bty = Bt(dy, qy);
               y(dx, dy, 0, e) += Bty * Y[dx][0];
               y(dx, dy, 1, e) += Bty * Y[dx][1];
            }
         }
      }
   });
}

// PA Convection NL 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void PAConvectionNLApply3D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Array<real_t> &bt,
                                  const Vector &q_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Q = Reshape(q_.Read(), Q1D * Q1D * Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t data[max_Q1D][max_Q1D][max_Q1D][VDIM];
      real_t grad0[max_Q1D][max_Q1D][max_Q1D][VDIM];
      real_t grad1[max_Q1D][max_Q1D][max_Q1D][VDIM];
      real_t grad2[max_Q1D][max_Q1D][max_Q1D][VDIM];
      real_t Z[max_Q1D][max_Q1D][max_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               data[qz][qy][qx][0] = 0.0;
               data[qz][qy][qx][1] = 0.0;
               data[qz][qy][qx][2] = 0.0;

               grad0[qz][qy][qx][0] = 0.0;
               grad0[qz][qy][qx][1] = 0.0;
               grad0[qz][qy][qx][2] = 0.0;

               grad1[qz][qy][qx][0] = 0.0;
               grad1[qz][qy][qx][1] = 0.0;
               grad1[qz][qy][qx][2] = 0.0;

               grad2[qz][qy][qx][0] = 0.0;
               grad2[qz][qy][qx][1] = 0.0;
               grad2[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t dataXY[max_Q1D][max_Q1D][VDIM];
         real_t gradXY0[max_Q1D][max_Q1D][VDIM];
         real_t gradXY1[max_Q1D][max_Q1D][VDIM];
         real_t gradXY2[max_Q1D][max_Q1D][VDIM];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               dataXY[qy][qx][0] = 0.0;
               dataXY[qy][qx][1] = 0.0;
               dataXY[qy][qx][2] = 0.0;

               gradXY0[qy][qx][0] = 0.0;
               gradXY0[qy][qx][1] = 0.0;
               gradXY0[qy][qx][2] = 0.0;

               gradXY1[qy][qx][0] = 0.0;
               gradXY1[qy][qx][1] = 0.0;
               gradXY1[qy][qx][2] = 0.0;

               gradXY2[qy][qx][0] = 0.0;
               gradXY2[qy][qx][1] = 0.0;
               gradXY2[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t dataX[max_Q1D][VDIM];
            real_t gradX0[max_Q1D][VDIM];
            real_t gradX1[max_Q1D][VDIM];
            real_t gradX2[max_Q1D][VDIM];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               dataX[qx][0] = 0.0;
               dataX[qx][1] = 0.0;
               dataX[qx][2] = 0.0;

               gradX0[qx][0] = 0.0;
               gradX0[qx][1] = 0.0;
               gradX0[qx][2] = 0.0;

               gradX1[qx][0] = 0.0;
               gradX1[qx][1] = 0.0;
               gradX1[qx][2] = 0.0;

               gradX2[qx][0] = 0.0;
               gradX2[qx][1] = 0.0;
               gradX2[qx][2] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s0 = x(dx, dy, dz, 0, e);
               const real_t s1 = x(dx, dy, dz, 1, e);
               const real_t s2 = x(dx, dy, dz, 2, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Bx = B(qx, dx);
                  const real_t Gx = G(qx, dx);

                  dataX[qx][0] += s0 * Bx;
                  dataX[qx][1] += s1 * Bx;
                  dataX[qx][2] += s2 * Bx;

                  gradX0[qx][0] += s0 * Gx;
                  gradX0[qx][1] += s0 * Bx;
                  gradX0[qx][2] += s0 * Bx;

                  gradX1[qx][0] += s1 * Gx;
                  gradX1[qx][1] += s1 * Bx;
                  gradX1[qx][2] += s1 * Bx;

                  gradX2[qx][0] += s2 * Gx;
                  gradX2[qx][1] += s2 * Bx;
                  gradX2[qx][2] += s2 * Bx;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t By = B(qy, dy);
               const real_t Gy = G(qy, dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  dataXY[qy][qx][0] += dataX[qx][0] * By;
                  dataXY[qy][qx][1] += dataX[qx][1] * By;
                  dataXY[qy][qx][2] += dataX[qx][2] * By;

                  gradXY0[qy][qx][0] += gradX0[qx][0] * By;
                  gradXY0[qy][qx][1] += gradX0[qx][1] * Gy;
                  gradXY0[qy][qx][2] += gradX0[qx][2] * By;

                  gradXY1[qy][qx][0] += gradX1[qx][0] * By;
                  gradXY1[qy][qx][1] += gradX1[qx][1] * Gy;
                  gradXY1[qy][qx][2] += gradX1[qx][2] * By;

                  gradXY2[qy][qx][0] += gradX2[qx][0] * By;
                  gradXY2[qy][qx][1] += gradX2[qx][1] * Gy;
                  gradXY2[qy][qx][2] += gradX2[qx][2] * By;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t Bz = B(qz, dz);
            const real_t Gz = G(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  data[qz][qy][qx][0] += dataXY[qy][qx][0] * Bz;
                  data[qz][qy][qx][1] += dataXY[qy][qx][1] * Bz;
                  data[qz][qy][qx][2] += dataXY[qy][qx][2] * Bz;

                  grad0[qz][qy][qx][0] += gradXY0[qy][qx][0] * Bz;
                  grad0[qz][qy][qx][1] += gradXY0[qy][qx][1] * Bz;
                  grad0[qz][qy][qx][2] += gradXY0[qy][qx][2] * Gz;

                  grad1[qz][qy][qx][0] += gradXY1[qy][qx][0] * Bz;
                  grad1[qz][qy][qx][1] += gradXY1[qy][qx][1] * Bz;
                  grad1[qz][qy][qx][2] += gradXY1[qy][qx][2] * Gz;

                  grad2[qz][qy][qx][0] += gradXY2[qy][qx][0] * Bz;
                  grad2[qz][qy][qx][1] += gradXY2[qy][qx][1] * Bz;
                  grad2[qz][qy][qx][2] += gradXY2[qy][qx][2] * Gz;
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + qz * Q1D);

               const real_t u1 = data[qz][qy][qx][0];
               const real_t u2 = data[qz][qy][qx][1];
               const real_t u3 = data[qz][qy][qx][2];

               const real_t grad00 = grad0[qz][qy][qx][0];
               const real_t grad01 = grad0[qz][qy][qx][1];
               const real_t grad02 = grad0[qz][qy][qx][2];

               const real_t grad10 = grad1[qz][qy][qx][0];
               const real_t grad11 = grad1[qz][qy][qx][1];
               const real_t grad12 = grad1[qz][qy][qx][2];

               const real_t grad20 = grad2[qz][qy][qx][0];
               const real_t grad21 = grad2[qz][qy][qx][1];
               const real_t grad22 = grad2[qz][qy][qx][2];

               const real_t Dxu1 = grad00 * Q(q, 0, 0, e)
                                   + grad01 * Q(q, 1, 0, e)
                                   + grad02 * Q(q, 2, 0, e);
               const real_t Dyu1 = grad00 * Q(q, 0, 1, e)
                                   + grad01 * Q(q, 1, 1, e)
                                   + grad02 * Q(q, 2, 1, e);
               const real_t Dzu1 = grad00 * Q(q, 0, 2, e)
                                   + grad01 * Q(q, 1, 2, e)
                                   + grad02 * Q(q, 2, 2, e);

               const real_t Dxu2 = grad10 * Q(q, 0, 0, e)
                                   + grad11 * Q(q, 1, 0, e)
                                   + grad12 * Q(q, 2, 0, e);
               const real_t Dyu2 = grad10 * Q(q, 0, 1, e)
                                   + grad11 * Q(q, 1, 1, e)
                                   + grad12 * Q(q, 2, 1, e);
               const real_t Dzu2 = grad10 * Q(q, 0, 2, e)
                                   + grad11 * Q(q, 1, 2, e)
                                   + grad12 * Q(q, 2, 2, e);

               const real_t Dxu3 = grad20 * Q(q, 0, 0, e)
                                   + grad21 * Q(q, 1, 0, e)
                                   + grad22 * Q(q, 2, 0, e);
               const real_t Dyu3 = grad20 * Q(q, 0, 1, e)
                                   + grad21 * Q(q, 1, 1, e)
                                   + grad22 * Q(q, 2, 1, e);
               const real_t Dzu3 = grad20 * Q(q, 0, 2, e)
                                   + grad21 * Q(q, 1, 2, e)
                                   + grad22 * Q(q, 2, 2, e);

               Z[qz][qy][qx][0] = u1 * Dxu1 + u2 * Dyu1 + u3 * Dzu1;
               Z[qz][qy][qx][1] = u1 * Dxu2 + u2 * Dyu2 + u3 * Dzu2;
               Z[qz][qy][qx][2] = u1 * Dxu3 + u2 * Dyu3 + u3 * Dzu3;
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t opXY[max_D1D][max_D1D][VDIM];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               opXY[dy][dx][0] = 0.0;
               opXY[dy][dx][1] = 0.0;
               opXY[dy][dx][2] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t opX[max_D1D][VDIM];
            for (int dx = 0; dx < D1D; ++dx)
            {
               opX[dx][0] = 0.0;
               opX[dx][1] = 0.0;
               opX[dx][2] = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Btx = Bt(dx, qx);
                  opX[dx][0] += Btx * Z[qz][qy][qx][0];
                  opX[dx][1] += Btx * Z[qz][qy][qx][1];
                  opX[dx][2] += Btx * Z[qz][qy][qx][2];
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t Bty = Bt(dy, qy);
                  opXY[dy][dx][0] += Bty * opX[dx][0];
                  opXY[dy][dx][1] += Bty * opX[dx][1];
                  opXY[dy][dx][2] += Bty * opX[dx][2];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t Btz = Bt(dz, qz);
                  y(dx, dy, dz, 0, e) += Btz * opXY[dy][dx][0];
                  y(dx, dy, dz, 1, e) += Btz * opXY[dy][dx][1];
                  y(dx, dy, dz, 2, e) += Btz * opXY[dy][dx][2];
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int T_MAX_D1D = 0, int T_MAX_Q1D = 0>
static void SmemPAConvectionNLApply3D(const int NE,
                                      const Array<real_t> &b_,
                                      const Array<real_t> &g_,
                                      const Vector &d_,
                                      const Vector &x_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : T_MAX_D1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX_Q1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX_Q1D;
      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      real_t(*B)[MD1] = (real_t(*)[MD1])(BG + 0);
      real_t(*G)[MD1] = (real_t(*)[MD1])(BG + 1);
      real_t(*Bt)[MQ1] = (real_t(*)[MQ1])(BG + 0);
      MFEM_SHARED real_t U[2][MQ1][MQ1][MQ1];
      MFEM_SHARED real_t sm0[3][MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[3][MQ1 * MQ1 * MQ1];
      real_t(*DDQ0)[MD1][MQ1] = (real_t(*)[MD1][MQ1])(sm0 + 0);
      real_t(*DDQ1)[MD1][MQ1] = (real_t(*)[MD1][MQ1])(sm0 + 1);
      real_t(*X)[MD1][MD1] = (real_t(*)[MD1][MD1])(sm0 + 2);
      real_t(*DQQ0)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm1 + 0);
      real_t(*DQQ1)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm1 + 1);
      real_t(*DQQ2)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm1 + 2);
      real_t(*QQQ0)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm0 + 0);
      real_t(*QQQ1)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm0 + 1);
      real_t(*QQQ2)[MQ1][MQ1] = (real_t(*)[MQ1][MQ1])(sm0 + 2);
      real_t(*QQD0)[MQ1][MD1] = (real_t(*)[MQ1][MD1])(sm1 + 0);
      real_t(*QDD0)[MD1][MD1] = (real_t(*)[MD1][MD1])(sm0 + 0);
      MFEM_SHARED real_t Z[MQ1][MQ1][MQ1];

      for (int cy = 0; cy < VDIM; ++cy)
      {
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(q, x, Q1D)
            {
               MFEM_FOREACH_THREAD(d, y, D1D)
               {
                  B[q][d] = b(q, d);
                  G[q][d] = g(q, d);
               }
            }
         }
         MFEM_FOREACH_THREAD(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D) { Z[qz][qy][qx] = 0.0; }
            }
         }
         MFEM_SYNC_THREAD;
         for (int c = 0; c < VDIM; ++c)
         {
            MFEM_FOREACH_THREAD(dz, z, D1D)
            {
               MFEM_FOREACH_THREAD(dy, y, D1D)
               {
                  MFEM_FOREACH_THREAD(dx, x, D1D)
                  {
                     X[dz][dy][dx] = x(dx, dy, dz, cy, e);
                     U[0][dz][dy][dx] = x(dx, dy, dz, c, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dz, z, D1D)
            {
               MFEM_FOREACH_THREAD(dy, y, D1D)
               {
                  MFEM_FOREACH_THREAD(qx, x, Q1D)
                  {
                     real_t u = 0.0;
                     real_t v = 0.0;
                     real_t z = 0.0;
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        const real_t coord = X[dz][dy][dx];
                        const real_t value = U[0][dz][dy][dx];
                        u += coord * B[qx][dx];
                        v += coord * G[qx][dx];
                        z += value * B[qx][dx];
                     }
                     DDQ0[dz][dy][qx] = u;
                     DDQ1[dz][dy][qx] = v;
                     U[1][dz][dy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dz, z, D1D)
            {
               MFEM_FOREACH_THREAD(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD(qx, x, Q1D)
                  {
                     real_t u = 0.0;
                     real_t v = 0.0;
                     real_t w = 0.0;
                     real_t z = 0.0;
                     for (int dy = 0; dy < D1D; ++dy)
                     {
                        u += DDQ1[dz][dy][qx] * B[qy][dy];
                        v += DDQ0[dz][dy][qx] * G[qy][dy];
                        w += DDQ0[dz][dy][qx] * B[qy][dy];
                        z += U[1][dz][dy][qx] * B[qy][dy];
                     }
                     DQQ0[dz][qy][qx] = u;
                     DQQ1[dz][qy][qx] = v;
                     DQQ2[dz][qy][qx] = w;
                     U[0][dz][qy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qz, z, Q1D)
            {
               MFEM_FOREACH_THREAD(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD(qx, x, Q1D)
                  {
                     real_t u = 0.0;
                     real_t v = 0.0;
                     real_t w = 0.0;
                     real_t z = 0.0;
                     for (int dz = 0; dz < D1D; ++dz)
                     {
                        u += DQQ0[dz][qy][qx] * B[qz][dz];
                        v += DQQ1[dz][qy][qx] * B[qz][dz];
                        w += DQQ2[dz][qy][qx] * G[qz][dz];
                        z += U[0][dz][qy][qx] * B[qz][dz];
                     }
                     QQQ0[qz][qy][qx] = u;
                     QQQ1[qz][qy][qx] = v;
                     QQQ2[qz][qy][qx] = w;
                     U[1][qz][qy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qz, z, Q1D)
            {
               MFEM_FOREACH_THREAD(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD(qx, x, Q1D)
                  {
                     const int q = qx + (qy + qz * Q1D) * Q1D;
                     const real_t z = U[1][qz][qy][qx];
                     const real_t gX = QQQ0[qz][qy][qx];
                     const real_t gY = QQQ1[qz][qy][qx];
                     const real_t gZ = QQQ2[qz][qy][qx];
                     const real_t d = gX * D(q, 0, c, e) + gY * D(q, 1, c, e)
                                      + gZ * D(q, 2, c, e);
                     Z[qz][qy][qx] += z * d;
                  }
               }
            }
            MFEM_SYNC_THREAD;
         } // for each conv component
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(d, y, D1D)
            {
               MFEM_FOREACH_THREAD(q, x, Q1D) { Bt[d][q] = b(q, d); }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += Z[qz][qy][qx] * Bt[dx][qx];
                  }
                  QQD0[qz][qy][dx] = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  }
                  QDD0[qz][dy][dx] = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  }
                  Y(dx, dy, dz, cy, e) += u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void VectorConvectionNLFIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      const int NE = ne;
      const int D1D = maps->ndof;
      const int Q1D = maps->nqpt;
      const Vector &QV = pa_data;
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Array<real_t> &Bt = maps->Bt;
      if (dim == 2)
      {
         return PAConvectionNLApply2D(NE, B, G, Bt, QV, x, y, D1D, Q1D);
      }
      if (dim == 3)
      {
         constexpr int T_MAX_D1D = 8;
         constexpr int T_MAX_Q1D = 8;
         MFEM_VERIFY(D1D <= T_MAX_D1D && Q1D <= T_MAX_Q1D, "Not yet implemented!");
         return SmemPAConvectionNLApply3D<0, 0, T_MAX_D1D, T_MAX_Q1D>
                (NE, B, G, QV, x, y, D1D, Q1D);
      }
      MFEM_ABORT("Not yet implemented!");
   }
}

} // namespace mfem
