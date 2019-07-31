// Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

// PA Gradient Integrator

// PA Gradient Assemble 2D kernel
static void PAGradientSetup2D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              const double COEFF,
                              Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto y = Reshape(op.Write(), NQ, 2, 2, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J12 = J(q,0,1,e);
         const double J21 = J(q,1,0,e);
         const double J22 = J(q,1,1,e);
         // Store wq * Q * adj(J)
         y(q,0,0,e) = W[q] * COEFF *  J22; // 1,1
         y(q,0,1,e) = W[q] * COEFF * -J12; // 1,2
         y(q,1,0,e) = W[q] * COEFF * -J21; // 2,1
         y(q,1,1,e) = W[q] * COEFF *  J11; // 2,2
      }
   });
}

// PA Gradient Assemble 3D kernel
static void PAGradientSetup3D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              const double COEFF,
                              Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto y = Reshape(op.Write(), NQ, 3, 3, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double cw  = W[q] * COEFF;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // Store wq * Q * adj(J)
         y(q,0,0,e) = cw * A11; // 1,1
         y(q,0,1,e) = cw * A12; // 1,2
         y(q,0,2,e) = cw * A13; // 1,3
         y(q,1,0,e) = cw * A21; // 2,1
         y(q,1,1,e) = cw * A22; // 2,2
         y(q,1,2,e) = cw * A23; // 2,3
         y(q,2,0,e) = cw * A31; // 3,1
         y(q,2,1,e) = cw * A32; // 3,2
         y(q,2,2,e) = cw * A33; // 3,3
      }
   });
}

static void PAGradientSetup(const int dim,
                            const int TR_D1D,
                            const int TE_D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &W,
                            const Vector &J,
                            const double COEFF,
                            Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAGradientSetup"); }
   if (dim == 2)
   {
      PAGradientSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
      PAGradientSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void GradientIntegrator::Setup(const FiniteElementSpace &trial_fes,
                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements ordered by nodes
   MFEM_ASSERT(trial_fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement &trial_fe = *trial_fes.GetFE(0);
   const FiniteElement &test_fe = *test_fes.GetFE(0);
   ElementTransformation *trans = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            *trans);
   const int dims = trial_fe.GetDim();
   const int dimsToStore = dims * dims;
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   trial_maps = &trial_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   trial_dofs1D = trial_maps->ndof;
   quad1D = trial_maps->nqpt;
   test_maps  = &test_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   test_dofs1D = test_maps->ndof;
   MFEM_ASSERT(quad1D == test_maps->nqpt,
               "PA requires test and trial space to have same number of quadrature points!");
   pa_data.SetSize(nq * dimsToStore * ne, Device::GetMemoryType());
   double coeff = 1.0;
   if (Q)
   {
      ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
      MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
      coeff = cQ->constant;
   }
   PAGradientSetup(dim, trial_dofs1D, test_dofs1D, quad1D,
                   ne, ir->GetWeights(), geom->J, coeff, pa_data);
}

// PA Gradient Apply 2D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void PAGradientApply2D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y,
                              const int tr_d1d = 0,
                              const int te_d1d = 0,
                              const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, TR_D1D);
   auto G = Reshape(g.Read(), Q1D, TR_D1D);
   auto Bt = Reshape(bt.Read(), TE_D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 2,2, NE);
   auto x = Reshape(_x.Read(), TR_D1D, TR_D1D, NE);
   auto y = Reshape(_y.ReadWrite(), TE_D1D, TE_D1D, 2, NE);
   MFEM_FORALL(e, NE,
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 2;
      // the following variables are evaluated at compile time
      constexpr int max_TE_D1D = T_TE_D1D ? T_TE_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[max_Q1D][max_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < TR_D1D; ++dy)
      {
         double gradX[max_Q1D][VDIM];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < TR_D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * G(qx,dx);
               gradX[qx][1] += s * B(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][0] * wy;
               grad[qy][qx][1] += gradX[qx][1] * wDy;
            }
         }
      }
      // We've now calculated grad(p) = [Dxy, xDy] in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;
            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = gradX*op(q,0,0,e) + gradY*op(q,1,0,e);
            grad[qy][qx][1] = gradX*op(q,0,1,e) + gradY*op(q,1,1,e);
         }
      }
      // We've now calculated grad = grad p * op
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double opX[max_TE_D1D][VDIM];
         for (int dx = 0; dx < TE_D1D; ++dx)
         {
            opX[dx][0] = 0.0;
            opX[dx][1] = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               opX[dx][0] += Bt(dx,qx)*grad[qy][qx][0];
               opX[dx][1] += Bt(dx,qx)*grad[qy][qx][1];
            }
         }
         for (int dy = 0; dy < TE_D1D; ++dy)
         {
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               y(dx,dy,0,e) += Bt(dy,qy)*opX[dx][0];
               y(dx,dy,1,e) += Bt(dy,qy)*opX[dx][1];
            }
         }
      }
      // We've now calculated y = u * grad
   });

}

// Shared memory PA Gradient Apply 2D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0,
         const int T_NBZ = 0>
static void SmemPAGradientApply2D(const int NE,
                                  const Array<double> &_b,
                                  const Array<double> &_g,
                                  const Array<double> &_bt,
                                  const Vector &_op,
                                  const Vector &_x,
                                  Vector &_y,
                                  const int tr_d1d = 0,
                                  const int te_d1d = 0,
                                  const int q1d = 0)
{
   // TODO
   MFEM_ASSERT(false, "SHARED MEM NOT PROGRAMMED YET");
}

// PA Gradient Apply 2D kernel transpose
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void PAGradientApplyTranspose2D(const int NE,
                                       const Array<double> &bt,
                                       const Array<double> &gt,
                                       const Array<double> &b,
                                       const Vector &_op,
                                       const Vector &_x,
                                       Vector &_y,
                                       const int tr_d1d = 0,
                                       const int te_d1d = 0,
                                       const int q1d = 0)
{
   // TODO
   MFEM_ASSERT(false, "Gradient PA Apply Transpose 2D NOT PROGRAMMED YET");
}

// PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void PAGradientApply3D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y,
                              int tr_d1d = 0,
                              int te_d1d = 0,
                              int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, TR_D1D);
   auto G = Reshape(g.Read(), Q1D, TR_D1D);
   auto Bt = Reshape(bt.Read(), TE_D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D*Q1D, 3,3, NE);
   auto x = Reshape(_x.Read(), TR_D1D, TR_D1D, TR_D1D, NE);
   auto y = Reshape(_y.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 3, NE);
   MFEM_FORALL(e, NE,
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 3;
      // the following variables are evaluated at compile time
      constexpr int max_TE_D1D = T_TE_D1D ? T_TE_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[max_Q1D][max_Q1D][max_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < TR_D1D; ++dz)
      {
         double gradXY[max_Q1D][max_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < TR_D1D; ++dy)
         {
            double gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < TR_D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      // We've now calculated grad(p) = [Dxyz, xDyz, xyDz] in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];

               grad[qz][qy][qx][0] = gradX*op(q,0,0,e) + gradY*op(q,1,0,e) + gradZ*op(q,2,0,e);
               grad[qz][qy][qx][1] = gradX*op(q,0,1,e) + gradY*op(q,1,1,e) + gradZ*op(q,2,1,e);
               grad[qz][qy][qx][2] = gradX*op(q,0,2,e) + gradY*op(q,1,2,e) + gradZ*op(q,2,2,e);
            }
         }
      }
      // We've now calculated grad = grad p * op
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double opXY[max_TE_D1D][max_TE_D1D][VDIM];
         for (int dy = 0; dy < TE_D1D; ++dy)
         {
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               opXY[dy][dx][0] = 0.0;
               opXY[dy][dx][1] = 0.0;
               opXY[dy][dx][2] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double opX[max_TE_D1D][VDIM];
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               opX[dx][0] = 0.0;
               opX[dx][1] = 0.0;
               opX[dx][2] = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  opX[dx][0] += Bt(dx,qx)*grad[qz][qy][qx][0];
                  opX[dx][1] += Bt(dx,qx)*grad[qz][qy][qx][1];
                  opX[dx][2] += Bt(dx,qx)*grad[qz][qy][qx][2];
               }
            }
            for (int dy = 0; dy < TE_D1D; ++dy)
            {
               for (int dx = 0; dx < TE_D1D; ++dx)
               {
                  opXY[dy][dx][0] += Bt(dy,qy)*opX[dx][0];
                  opXY[dy][dx][1] += Bt(dy,qy)*opX[dx][1];
                  opXY[dy][dx][2] += Bt(dy,qy)*opX[dx][2];
               }
            }
         }
         for (int dz = 0; dz < TE_D1D; ++dz)
         {
            for (int dy = 0; dy < TE_D1D; ++dy)
            {
               for (int dx = 0; dx < TE_D1D; ++dx)
               {
                  y(dx,dy,dz,0,e) += Bt(dz,qz)*opXY[dy][dx][0];
                  y(dx,dy,dz,1,e) += Bt(dz,qz)*opXY[dy][dx][1];
                  y(dx,dy,dz,2,e) += Bt(dz,qz)*opXY[dy][dx][2];
               }
            }
         }
      }
      // We've now calculated y = u * grad
   });
}

// PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void PAGradientApplyTranspose3D(const int NE,
                                       const Array<double> &bt,
                                       const Array<double> &gt,
                                       const Array<double> &b,
                                       const Vector &_op,
                                       const Vector &_x,
                                       Vector &_y,
                                       int tr_d1d = 0,
                                       int te_d1d = 0,
                                       int q1d = 0)
{
   // TODO
   MFEM_ASSERT(false, "Gradient PA Apply Transpose 3D NOT PROGRAMMED YET");
}

// Shared memory PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void SmemPAGradientApply3D(const int NE,
                                  const Array<double> &_b,
                                  const Array<double> &_g,
                                  const Array<double> &_bt,
                                  const Vector &_op,
                                  const Vector &_x,
                                  Vector &_y,
                                  const int tr_d1d = 0,
                                  const int te_d1d = 0,
                                  const int q1d = 0)
{
   // TODO
   MFEM_ASSERT(false, "SHARED MEM NOT PROGRAMMED YET");
}

static void PAGradientApply(const int dim,
                            const int TR_D1D,
                            const int TE_D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Vector &op,
                            const Vector &x,
                            Vector &y,
                            bool transpose=false)
{

   //if (Device::Allows(Backend::RAJA_CUDA))
   //{
   if (dim == 2)
   {
      switch ((TR_D1D << 4) | TE_D1D)
      {
         case 0x23: // Specialized for Taylor-Hood elements
            if (Q1D == 3)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<2,3,3>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<2,3,3>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x34:
            if (Q1D == 4)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<3,4,4>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<3,4,4>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x45:
            if (Q1D == 6)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<4,5,6>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<4,5,6>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x56:
            if (Q1D == 7)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<5,6,7>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<5,6,7>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x67:
            if (Q1D == 9)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<6,7,9>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<6,7,9>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x78:
            if (Q1D == 10)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<7,8,10>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<7,8,10>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x89:
            if (Q1D == 12)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose2D<8,9,12>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply2D<8,9,12>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         default:
            break;
      }
      return PAGradientApply2D(NE,B,G,Bt,op,x,y,TR_D1D,TE_D1D,Q1D);
   }
   if (dim == 3)
   {
      switch ((TR_D1D << 4) | TE_D1D)
      {
         case 0x23: // Specialized for Taylor-Hood elements
            if (Q1D == 4)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<2,3,4>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<2,3,4>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x34:
            if (Q1D == 6)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<3,4,6>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<3,4,6>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x45:
            if (Q1D == 8)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<4,5,8>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<4,5,8>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x56:
            if (Q1D == 10)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<5,6,10>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<5,6,10>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x67:
            if (Q1D == 12)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<6,7,12>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<6,7,12>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x78:
            if (Q1D == 14)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<7,8,14>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<7,8,14>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         case 0x89:
            if (Q1D == 16)
            {
               if (transpose)
               {
                  return PAGradientApplyTranspose3D<8,9,16>(NE,B,G,Bt,op,x,y);
               }
               else
               {
                  return PAGradientApply3D<8,9,16>(NE,B,G,Bt,op,x,y);
               }
            }
            break;
         default:
            break;
      }
      if (transpose)
      {
         return PAGradientApplyTranspose3D(NE,B,G,Bt,op,x,y,TR_D1D,TE_D1D,Q1D);
      }
      else
      {
         return PAGradientApply3D(NE,B,G,Bt,op,x,y,TR_D1D,TE_D1D,Q1D);
      }
   }
   //}
   MFEM_ABORT("Unknown kernel.");
}

// PA Gradient Apply kernel
void GradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PAGradientApply(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                   trial_maps->B, trial_maps->G, test_maps->Bt, pa_data, x, y,
                   false);
}

// PA Gradient Apply kernel
void GradientIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_ABORT("PA Gradient AddMultTransposePA not yet programmed!");
}

} // namespace mfem

