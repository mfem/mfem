// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleGradPA_C0_2D(const real_t lim_normal,
                               const ConstDeviceCube &LD,
                               const bool const_c0,
                               const DeviceTensor<3, const real_t> &C0,
                               const int NE,
                               const DeviceTensor<5, const real_t> &J,
                               const ConstDeviceMatrix &W,
                               const real_t *b,
                               const real_t *bld,
                               const DeviceTensor<4, const real_t> &X0,
                               const DeviceTensor<4, const real_t> &X1,
                               DeviceTensor<5> &H0,
                               const bool exp_lim,
                               const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, bld, sB);

      kernels::internal::s_regs2d_t<MQ1> rm0, rm1; // scalar LD
      kernels::internal::LoadDofs2d(e, D1D, LD, rm0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, rm0, rm1);

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::v_regs2d_t<2,MQ1> r00, r01; // vector X0
      kernels::internal::LoadDofs2d(e, D1D, X0, r00);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r00, r01);

      kernels::internal::v_regs2d_t<2,MQ1> r10, r11; // vector X1
      kernels::internal::LoadDofs2d(e, D1D, X1, r10);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r10, r11);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t weight = W(qx, qy) * detJtr;
            const real_t coeff0 = const_c0 ? C0(0, 0, 0) : C0(qx, qy, e);
            const real_t weight_m = weight * lim_normal * coeff0;

            const real_t D = rm1(qy, qx);
            const real_t p0[2] = { r01(0, qy, qx), r01(1, qy, qx) };
            const real_t p1[2] = { r11(0, qy, qx), r11(1, qy, qx) };

            const real_t dist = D; // GetValues, default comp set to 0

            // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
            real_t grad_grad[4];

            if (!exp_lim)
            {
               // d2.Diag(1.0 / (dist * dist), x.Size());
               const real_t c = 1.0 / (dist * dist);
               kernels::Diag<2>(c, grad_grad);
            }
            else
            {
               real_t tmp[2];
               kernels::Subtract<2>(1.0, p1, p0, tmp);
               real_t dsq = kernels::DistanceSquared<2>(p1, p0);
               real_t dist_squared = dist * dist;
               real_t dist_squared_squared = dist_squared * dist_squared;
               real_t f = exp(10.0 * ((dsq / dist_squared) - 1.0));
               grad_grad[0] =
                  ((400.0 * tmp[0] * tmp[0] * f) / dist_squared_squared) +
                  (20.0 * f / dist_squared);
               grad_grad[1] =
                  (400.0 * tmp[0] * tmp[1] * f) / dist_squared_squared;
               grad_grad[2] = grad_grad[1];
               grad_grad[3] =
                  ((400.0 * tmp[1] * tmp[1] * f) / dist_squared_squared) +
                  (20.0 * f / dist_squared);
            }
            ConstDeviceMatrix gg(grad_grad, 2, 2);

            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               {
                  H0(i, j, qx, qy, e) = weight_m * gg(i, j);
               }
            }
         }
      }
   });
}

// Assemble gradient and Hessian of ALF field at quadrature points for AdaptLim (2D)
template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleGradPA_AdaptLim_2D(const int NE,
                                      const real_t *b_nodes,
                                      const real_t *g_nodes,
                                      const real_t *B,
                                      const real_t *G,
                                      const DeviceTensor<4, const real_t> &X,
                                      const ConstDeviceCube &ALF,
                                      DeviceTensor<4> &ALF_grad,
                                      DeviceTensor<5> &ALF_hess,
                                      const int d1d,
                                      const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB_node[MD1][MD1];
      MFEM_SHARED real_t sG_node[MD1][MD1];
      MFEM_SHARED real_t sB_d[MD1][MD1];
      MFEM_SHARED real_t sG_d[MD1][MD1];
      MFEM_SHARED real_t smem_dof[MD1][MD1];

      // Position space basis for Jacobian computation
      kernels::internal::LoadMatrix(D1D, D1D, b_nodes, sB_node);
      kernels::internal::LoadMatrix(D1D, D1D, g_nodes, sG_node);

      // Limiting basis and gradient matrices (at limiting DOF nodes).
      kernels::internal::LoadMatrix(D1D, D1D, B, sB_d);
      kernels::internal::LoadMatrix(D1D, D1D, G, sG_d);

      kernels::internal::vd_regs2d_t<2, 2, MD1> r_X;
      kernels::internal::LoadDofs2d(e, D1D, X, r_X);

      kernels::internal::s_regs2d_t<MD1> ralf_dofs;
      kernels::internal::LoadDofs2d(e, D1D, ALF, ralf_dofs);

      kernels::internal::vd_regs2d_t<2, 2, MD1> r_X_grad;
      kernels::internal::Grad2d(D1D, D1D, smem_dof,
                                sB_node, sG_node, r_X, r_X_grad);

      // Project gradient of ALF at the DOFs.
      real_t grad_e[MD1][MD1][2];
      for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
      {
         for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
         {
            const real_t Jpr[4] =
            {
               r_X_grad[0][0][dy_dof][dx_dof], r_X_grad[1][0][dy_dof][dx_dof],
               r_X_grad[0][1][dy_dof][dx_dof], r_X_grad[1][1][dy_dof][dx_dof]
            };
            real_t Jpr_inv[4];
            kernels::CalcInverse<2>(Jpr, Jpr_inv);

            // Compute physical gradient at this DOF node
            // grad_phys = Jpr^{-T} * grad_ref
            grad_e[dy_dof][dx_dof][0] = 0.0;
            grad_e[dy_dof][dx_dof][1] = 0.0;
            for (int dy = 0; dy < D1D; dy++)
            {
               for (int dx = 0; dx < D1D; dx++)
               {
                  const real_t dsdx = sG_node[dx][dx_dof] * sB_node[dy][dy_dof];
                  const real_t dsdy = sB_node[dx][dx_dof] * sG_node[dy][dy_dof];
                  const real_t grad_phys_x = Jpr_inv[0] * dsdx + Jpr_inv[2] * dsdy;
                  const real_t grad_phys_y = Jpr_inv[1] * dsdx + Jpr_inv[3] * dsdy;
                  grad_e[dy_dof][dx_dof][0] += grad_phys_x * ralf_dofs(dy, dx);
                  grad_e[dy_dof][dx_dof][1] += grad_phys_y * ralf_dofs(dy, dx);
               }
            }
         }
      }

      // Project gradient of each component of grad_e (i.e., compute Hessian of ALF at DOFs).
      real_t hess_e[MD1][MD1][2][2];
      for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
      {
         for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
         {
            const real_t Jpr[4] = {
               r_X_grad[0][0][dy_dof][dx_dof], r_X_grad[1][0][dy_dof][dx_dof],
               r_X_grad[0][1][dy_dof][dx_dof], r_X_grad[1][1][dy_dof][dx_dof]
            };
            real_t Jpr_inv[4];
            kernels::CalcInverse<2>(Jpr, Jpr_inv);
            
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               {
                  hess_e[dy_dof][dx_dof][i][j] = 0.0;
               }
            }
            
            for (int dy = 0; dy < D1D; dy++)
            {
               for (int dx = 0; dx < D1D; dx++)
               {
                  const real_t dsdx = sG_node[dx][dx_dof] * sB_node[dy][dy_dof];
                  const real_t dsdy = sB_node[dx][dx_dof] * sG_node[dy][dy_dof];
                  const real_t grad_phys_x = Jpr_inv[0] * dsdx + Jpr_inv[2] * dsdy;
                  const real_t grad_phys_y = Jpr_inv[1] * dsdx + Jpr_inv[3] * dsdy;
                  
                  // d(grad_e[...][0]) / dx_i = d^2(ALF) / dx_0 dx_i
                  hess_e[dy_dof][dx_dof][0][0] += grad_phys_x * grad_e[dy][dx][0];
                  hess_e[dy_dof][dx_dof][0][1] += grad_phys_y * grad_e[dy][dx][0];
                  // d(grad_e[...][1]) / dx_i = d^2(ALF) / dx_1 dx_i
                  hess_e[dy_dof][dx_dof][1][0] += grad_phys_x * grad_e[dy][dx][1];
                  hess_e[dy_dof][dx_dof][1][1] += grad_phys_y * grad_e[dy][dx][1];
               }
            }
         }
      }

      // Evaluate gradient and Hessian at quadrature points using ALF shape functions
      MFEM_SHARED real_t sB_q[MD1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB_q);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            // Evaluate gradient at quad point.
            real_t grad_alf_q[2] = {0.0, 0.0};
            for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
            {
               for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
               {
                  const real_t shape = sB_q[dx_dof][qx] * sB_q[dy_dof][qy];
                  grad_alf_q[0] += grad_e[dy_dof][dx_dof][0] * shape;
                  grad_alf_q[1] += grad_e[dy_dof][dx_dof][1] * shape;
               }
            }
            // Store gradient at quadrature point.
            ALF_grad(0, qx, qy, e) = grad_alf_q[0];
            ALF_grad(1, qx, qy, e) = grad_alf_q[1];

            // Evaluate Hessian at quad point.
            real_t hess_alf_q[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
            {
               for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
               {
                  const real_t shape = sB_q[dx_dof][qx] * sB_q[dy_dof][qy];
                  for (int i = 0; i < 2; i++)
                  {
                     for (int j = 0; j < 2; j++)
                     {
                        hess_alf_q[i][j] += hess_e[dy_dof][dx_dof][i][j] * shape;
                     }
                  }
               }
            }
            // Store Hessian at quadrature point.
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               {
                  ALF_hess(i, j, qx, qy, e) = hess_alf_q[i][j];
               }
            }
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleGradCoef2D, TMOP_AssembleGradPA_C0_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleGradCoef2D);

void TMOP_Integrator::AssembleGradPA_C0_2D(const Vector &x) const
{
   const int NE = PA.ne, d = PA.maps_lim->ndof, q = PA.maps_lim->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const auto C0 = PA.C0.Size() == 1
                       ? Reshape(PA.C0.Read(), 1, 1, 1)
                       : Reshape(PA.C0.Read(), q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), 2, 2, q, q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto *b = PA.maps->B.Read(), *bld = PA.maps_lim->B.Read();
   const auto LD = Reshape(PA.LD.Read(), d, d, NE);
   const auto XL = Reshape(PA.XL.Read(), d, d, 2, NE);
   const auto X = Reshape(x.Read(), d, d, 2, NE);
   auto H0 = Reshape(PA.H0.Write(), 2, 2, q, q, NE);

   const auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = el ? true : false;

   TMOPAssembleGradCoef2D::Run(d, q, ln, LD, const_c0, C0, NE,
                               J, W, b, bld, XL, X, H0, exp_lim, d, q);
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleGradAdaptLim2D, TMOP_AssembleGradPA_AdaptLim_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleGradAdaptLim2D);

void TMOP_Integrator::AssembleGradPA_AdaptLim_2D(const Vector &x) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto *B_nodes = PA.maps_nodes->B.Read(), *G_nodes = PA.maps_nodes->G.Read();
   const auto *B = PA.maps->B.Read(), *G = PA.maps->G.Read();
   const auto X = Reshape(x.Read(), d, d, 2, NE);
   const auto ALF = Reshape(PA.ALF.Read(), d, d, NE);
   auto ALF_grad = Reshape(PA.ALFG.Write(), 2, q, q, NE);
   auto ALF_hess = Reshape(PA.ALFH.Write(), 2, 2, q, q, NE);

   TMOPAssembleGradAdaptLim2D::Run(d, q, NE, B_nodes, G_nodes, B, G, X, ALF,
                                   ALF_grad, ALF_hess, d, q);
}

} // namespace mfem
