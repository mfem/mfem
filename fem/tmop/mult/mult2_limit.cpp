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
void TMOP_AddMultPA_C0_2D(const real_t lim_normal,
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
                          DeviceTensor<4> &Y,
                          const bool exp_lim,
                          const int d1d,
                          const int q1d)
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

            const real_t ld = rm1(qy, qx);
            const real_t p0[2] = { r01(0, qy, qx), r01(1, qy, qx) };
            const real_t p1[2] = { r11(0, qy, qx), r11(1, qy, qx) };

            const real_t coeff0 = const_c0 ? C0(0, 0, 0) : C0(qx, qy, e);

            const real_t dist = ld; // GetValues, default comp set to 0

            real_t d1[2];
            // Eval_d1 (Quadratic Limiter)
            // subtract(1.0 / (dist * dist), x, x0, d1);
            // z = a * (x - y)
            // grad = a * (x - x0)

            // Eval_d1 (Exponential Limiter)
            // real_t dist_squared = dist*dist;
            // subtract(20.0*exp(10.0*((x.DistanceSquaredTo(x0) / dist_squared)
            // - 1.0)) / dist_squared, x, x0, d1); z = a * (x - y) grad = a * (x
            // - x0)

            real_t a = 0.0;
            const real_t w = weight * lim_normal * coeff0;
            const real_t dist_squared = dist * dist;

            if (!exp_lim) { a = 1.0 / dist_squared; }
            else
            {
               real_t dsq = kernels::DistanceSquared<2>(p1, p0) / dist_squared;
               a = 20.0 * exp(10.0 * (dsq - 1.0)) / dist_squared;
            }
            kernels::Subtract<2>(w * a, p1, p0, d1);
            r00(0,qy,qx) = d1[0];
            r00(1,qy,qx) = d1[1];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r00, r01);
      kernels::internal::WriteDofs2d(e, D1D, r01, Y);
   });
}

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultPA_AdaptLim_2D(const real_t lim_normal,
                                const real_t adapt_lim_delta_max,
                                const bool const_coeff,
                                const DeviceTensor<3, const real_t> &ALC,
                                const int NE,
                                const DeviceTensor<5, const real_t> &J,
                                const ConstDeviceMatrix &W,
                                const real_t *b,
                                const real_t *g,
                                const real_t *b_nodes,
                                const real_t *g_nodes,
                                const DeviceTensor<4, const real_t> &X,
                                const ConstDeviceCube &ALF,
                                const ConstDeviceCube &ALF0,
                                DeviceTensor<4> &Y,
                                const int d1d,
                                const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB_node[MD1][MD1];
      MFEM_SHARED real_t sG_node[MD1][MD1];
      MFEM_SHARED real_t smem_dof[MD1][MD1];

      // Position space basis matrices for computing Jacobian at nodes.
      kernels::internal::LoadMatrix(D1D, D1D, b_nodes, sB_node);
      kernels::internal::LoadMatrix(D1D, D1D, g_nodes, sG_node);

      // Load positions.
      kernels::internal::vd_regs2d_t<2, 2, MD1> r_X;
      kernels::internal::LoadDofs2d(e, D1D, X, r_X);

      // Load ALF DOFs into MD1 array for gradient computation.
      kernels::internal::s_regs2d_t<MD1> ralf_dofs;
      kernels::internal::LoadDofs2d(e, D1D, ALF, ralf_dofs);

      // Compute gradient of X at DOF nodes using position space basis.
      kernels::internal::vd_regs2d_t<2, 2, MD1> r_X_grad;
      kernels::internal::Grad2d(D1D, D1D, smem_dof,
                                sB_node, sG_node, r_X, r_X_grad);

      // Project gradient of ALF at the DOFs.
      // grad_e(k,d) = sum_j grad_phys(k*dim+d, j) * alf_dof(j).
      real_t grad_e[MD1][MD1][2];
      for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
      {
         for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
         {
            // Jpr = X^t.DSh at this DOF node.
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
                  // Reference gradient at DOF location (dx_dof, dy_dof).
                  const real_t dsdx = sG_node[dx][dx_dof] * sB_node[dy][dy_dof];
                  const real_t dsdy = sB_node[dx][dx_dof] * sG_node[dy][dy_dof];

                  // Physical gradient: grad_phys = Jpr^{-T} * grad_ref.
                  const real_t grad_phys_x = Jpr_inv[0] * dsdx + Jpr_inv[2] * dsdy;
                  const real_t grad_phys_y = Jpr_inv[1] * dsdx + Jpr_inv[3] * dsdy;

                  // Grad of ALF at the DOFs.
                  grad_e[dy_dof][dx_dof][0] += grad_phys_x * ralf_dofs(dy, dx);
                  grad_e[dy_dof][dx_dof][1] += grad_phys_y * ralf_dofs(dy, dx);
               }
            }
         }
      }

      // Load quad basis matrices for evaluation at quadrature points
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB_quad[MD1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB_quad);

      // Load ALF DOFs into MQ1 array for Eval2d.
      kernels::internal::s_regs2d_t<MQ1> ralf_val_dof;
      kernels::internal::LoadDofs2d(e, D1D, ALF, ralf_val_dof);

      // Evaluate ALF and ALF0 at quad points
      kernels::internal::s_regs2d_t<MQ1> ralf_val_quad;
      kernels::internal::Eval2d(D1D, Q1D, smem, sB_quad,
                                ralf_val_dof, ralf_val_quad);

      kernels::internal::s_regs2d_t<MQ1> ralf0_val_dof, ralf0_val_quad;
      kernels::internal::LoadDofs2d(e, D1D, ALF0, ralf0_val_dof);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB_quad,
                                ralf0_val_dof, ralf0_val_quad);

      // Storage for output gradient.
      kernels::internal::v_regs2d_t<2,MQ1> r00, r01;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t weight = W(qx, qy) * detJtr;

            const real_t gf_val = ralf_val_quad(qy, qx);
            const real_t gf0_val = ralf0_val_quad(qy, qx);

            // Evaluate gradient field at quad point:
            // grad_q = sum_k grad_e(k,:) * shape(k)
            real_t grad_alf[2] = {0.0, 0.0};

            for (int dy_dof = 0; dy_dof < D1D; dy_dof++)
            {
               for (int dx_dof = 0; dx_dof < D1D; dx_dof++)
               {
                  const real_t shp = sB_quad[dx_dof][qx] * sB_quad[dy_dof][qy];
                  grad_alf[0] += grad_e[dy_dof][dx_dof][0] * shp;
                  grad_alf[1] += grad_e[dy_dof][dx_dof][1] * shp;
               }
            }

            // Apply scaling:
            // 2.0 * (gf_q - gf0_q) / delta_max^2 * weight * lim_normal * coeff
            const real_t coeff = const_coeff ? ALC(0, 0, 0) : ALC(qx, qy, e);
            const real_t factor = weight * lim_normal * coeff *
                                  2.0 * (gf_val - gf0_val) /
                                  (adapt_lim_delta_max * adapt_lim_delta_max);

            r00(0, qy, qx) = factor * grad_alf[0];
            r00(1, qy, qx) = factor * grad_alf[1];
         }
      }
      MFEM_SYNC_THREAD;

      // Apply transpose: shape functions times gradient (AddMultVWt in full assembly)
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB_quad, r00, r01);
      kernels::internal::WriteDofs2d(e, D1D, r01, Y);
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultCoefKernels, TMOP_AddMultPA_C0_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultCoefKernels);

void TMOP_Integrator::AddMultPA_C0_2D(const Vector &x, Vector &y) const
{
   const real_t ln = lim_normal;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   MFEM_VERIFY(PA.maps_lim->ndof == d, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == q, "");

   const bool const_c0 = PA.C0.Size() == 1;
   const auto C0 = const_c0
                   ? Reshape(PA.C0.Read(), 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, NE);
   const auto LD = Reshape(PA.LD.Read(), d, d, NE);
   const auto J = Reshape(PA.Jtr.Read(), 2, 2, q, q, NE);
   const auto *b = PA.maps->B.Read(), *bld = PA.maps_lim->B.Read();
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto XL = Reshape(PA.XL.Read(), d, d, 2, NE);
   const auto X = Reshape(x.Read(), d, d, 2, NE);
   auto Y = Reshape(y.ReadWrite(), d, d, 2, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPMultCoefKernels::Run(d, q, ln, LD, const_c0, C0, NE, J, W, b, bld, XL, X,
                            Y, exp_lim, d, q);
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultAdaptLim, TMOP_AddMultPA_AdaptLim_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultAdaptLim);

void TMOP_Integrator::AddMultPA_AdaptLim_2D(const Vector &x, Vector &y) const
{
   const real_t ln = lim_normal;
   const real_t delta_max = PA.al_delta;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const bool const_coeff = PA.ALC.Size() == 1;
   const auto ALC = const_coeff
                    ? Reshape(PA.ALC.Read(), 1, 1, 1)
                    : Reshape(PA.ALC.Read(), q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), 2, 2, q, q, NE);
   const auto *B = PA.maps->B.Read(), *G = PA.maps->G.Read();
   const auto *B_nodes = PA.maps_nodes->B.Read(), *G_nodes = PA.maps_nodes->G.Read();
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto X = Reshape(x.Read(), d, d, 2, NE);
   const auto ALF = Reshape(PA.ALF.Read(), d, d, NE);
   const auto ALF0 = Reshape(PA.ALF0.Read(), d, d, NE);
   auto Y = Reshape(y.ReadWrite(), d, d, 2, NE);

   TMOPMultAdaptLim::Run(d, q, ln, delta_max, const_coeff, ALC, NE, J, W,
                         B, G, B_nodes, G_nodes, X, ALF, ALF0, Y, d, q);
}

} // namespace mfem
