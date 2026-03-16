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
void TMOP_AssembleGradPA_C0_3D(const real_t lim_normal,
                               const DeviceTensor<4, const real_t> &LD,
                               const bool const_c0,
                               const DeviceTensor<4, const real_t> &C0,
                               const int NE,
                               const DeviceTensor<6, const real_t> &J,
                               const ConstDeviceCube &W,
                               const real_t *b,
                               const real_t *bld,
                               const DeviceTensor<5, const real_t> &X0,
                               const DeviceTensor<5, const real_t> &X1,
                               DeviceTensor<6> &H0,
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

      kernels::internal::s_regs3d_t<MQ1> rm0, rm1; // scalar LD
      kernels::internal::LoadDofs3d(e, D1D, LD, rm0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, rm0, rm1);

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      kernels::internal::v_regs3d_t<3,MQ1> r00, r01; // vector X0
      kernels::internal::LoadDofs3d(e, D1D, X0, r00);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r00, r01);

      kernels::internal::v_regs3d_t<3,MQ1> r10, r11; // vector X1
      kernels::internal::LoadDofs3d(e, D1D, X1, r10);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r10, r11);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx, qy, qz) * detJtr;
               const real_t coeff0 = const_c0
                                     ? C0(0, 0, 0, 0)
                                     : C0(qx, qy, qz, e);
               const real_t weight_m = weight * lim_normal * coeff0;
               const real_t D = rm1(qz, qy, qx);
               const real_t p0[3] = { r01(0, qz, qy, qx),
                                      r01(1, qz, qy, qx),
                                      r01(2, qz, qy, qx)
                                    };
               const real_t p1[3] = { r11(0, qz, qy, qx),
                                      r11(1, qz, qy, qx),
                                      r11(2, qz, qy, qx)
                                    };

               const real_t dist = D; // GetValues, default comp set to 0

               // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);

               real_t grad_grad[9];

               if (!exp_lim)
               {
                  // d2.Diag(1.0 / (dist * dist), x.Size());
                  const real_t c = 1.0 / (dist * dist);
                  kernels::Diag<3>(c, grad_grad);
               }
               else
               {
                  real_t tmp[3];
                  kernels::Subtract<3>(1.0, p1, p0, tmp);
                  real_t dsq = kernels::DistanceSquared<3>(p1, p0);
                  real_t dist_squared = dist * dist;
                  real_t dist_squared_squared = dist_squared * dist_squared;
                  real_t f = exp(10.0 * ((dsq / dist_squared) - 1.0));
                  grad_grad[0] =
                     ((400.0 * tmp[0] * tmp[0] * f) / dist_squared_squared) +
                     (20.0 * f / dist_squared);
                  grad_grad[1] =
                     (400.0 * tmp[0] * tmp[1] * f) / dist_squared_squared;
                  grad_grad[2] =
                     (400.0 * tmp[0] * tmp[2] * f) / dist_squared_squared;
                  grad_grad[3] = grad_grad[1];
                  grad_grad[4] =
                     ((400.0 * tmp[1] * tmp[1] * f) / dist_squared_squared) +
                     (20.0 * f / dist_squared);
                  grad_grad[5] =
                     (400.0 * tmp[1] * tmp[2] * f) / dist_squared_squared;
                  grad_grad[6] = grad_grad[2];
                  grad_grad[7] = grad_grad[5];
                  grad_grad[8] =
                     ((400.0 * tmp[2] * tmp[2] * f) / dist_squared_squared) +
                     (20.0 * f / dist_squared);
               }
               ConstDeviceMatrix gg(grad_grad, 3, 3);

               for (int i = 0; i < 3; i++)
               {
                  for (int j = 0; j < 3; j++)
                  {
                     H0(i, j, qx, qy, qz, e) = weight_m * gg(i, j);
                  }
               }
            }
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleGradCoef3D, TMOP_AssembleGradPA_C0_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleGradCoef3D);

void TMOP_Integrator::AssembleGradPA_C0_3D(const Vector &x) const
{
   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const int NE = PA.ne, d = PA.maps_lim->ndof, q = PA.maps_lim->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto C0 = const_c0
                   ? Reshape(PA.C0.Read(), 1, 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), 3, 3, q, q, q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto *b = PA.maps->B.Read(), *bld = PA.maps_lim->B.Read();
   const auto LD = Reshape(PA.LD.Read(), d, d, d, NE);
   const auto XL = Reshape(PA.XL.Read(), d, d, d, 3, NE);
   const auto X = Reshape(x.Read(), d, d, d, 3, NE);
   auto H0 = Reshape(PA.H0.Write(), 3, 3, q, q, q, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPAssembleGradCoef3D::Run(d, q, ln, LD, const_c0, C0, NE,
                               J, W, b, bld, XL, X, H0, exp_lim, d, q);
}

// Assemble gradient and Hessian of ALF field at quadrature points for AdaptLim (3D)
template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleGradPA_AdaptLim_3D(const int NE,
                                     const real_t *b_nodes,
                                     const real_t *g_nodes,
                                     const real_t *B,
                                     const DeviceTensor<5, const real_t> &X,
                                     const DeviceTensor<4, const real_t> &ALF,
                                     DeviceTensor<5> &ALF_grad,
                                     DeviceTensor<6> &ALF_hess,
                                     const int d1d,
                                     const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem_d[MD1][MD1];
      MFEM_SHARED real_t smem_q[MQ1][MQ1];
      MFEM_SHARED real_t sB_nodes[MD1][MD1], sG_nodes[MD1][MD1];
      MFEM_SHARED real_t sB_q[MD1][MQ1];

      // Maps nodes - nodes.
      kernels::internal::LoadMatrix(D1D, D1D, b_nodes, sB_nodes);
      kernels::internal::LoadMatrix(D1D, D1D, g_nodes, sG_nodes);
      // Map nodes - quads.
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB_q);

      // Compute the physical Jacobian at DOF nodes.
      kernels::internal::vd_regs3d_t<3, 3, MD1> r_X, r_X_grad_nodes;
      kernels::internal::LoadDofs3d(e, D1D, X, r_X);
      kernels::internal::Grad3d(D1D, D1D, smem_d,
                                sB_nodes, sG_nodes, r_X, r_X_grad_nodes);

      // Compute the reference derivatives of ALF at DOF nodes.
      kernels::internal::s_regs3d_t<MD1> alf_n, dalf_dxi_n, dalf_deta_n, dalf_dzeta_n;
      kernels::internal::LoadDofs3d(e, D1D, ALF, alf_n);
      kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                sG_nodes, sB_nodes, sB_nodes,
                                                alf_n, dalf_dxi_n);
      kernels::internal::LoadDofs3d(e, D1D, ALF, alf_n);
      kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                sB_nodes, sG_nodes, sB_nodes,
                                                alf_n, dalf_deta_n);
      kernels::internal::LoadDofs3d(e, D1D, ALF, alf_n);
      kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                sB_nodes, sB_nodes, sG_nodes,
                                                alf_n, dalf_dzeta_n);

      // Physical gradient coefficients at DOF nodes into grad_e.
      // Takes derivatives of alf.
      real_t grad_e[MD1][MD1][MD1][3];
      for (int dz = 0; dz < D1D; dz++)
      {
         for (int dy = 0; dy < D1D; dy++)
         {
            for (int dx = 0; dx < D1D; dx++)
            {
               const real_t Jpr[9] =
               {
                  r_X_grad_nodes[0][0][dz][dy][dx], r_X_grad_nodes[1][0][dz][dy][dx], r_X_grad_nodes[2][0][dz][dy][dx],
                  r_X_grad_nodes[0][1][dz][dy][dx], r_X_grad_nodes[1][1][dz][dy][dx], r_X_grad_nodes[2][1][dz][dy][dx],
                  r_X_grad_nodes[0][2][dz][dy][dx], r_X_grad_nodes[1][2][dz][dy][dx], r_X_grad_nodes[2][2][dz][dy][dx]
               };
               real_t Jpr_inv[9];
               kernels::CalcInverse<3>(Jpr, Jpr_inv);

               const real_t dalf_dxi   = dalf_dxi_n[dz][dy][dx];
               const real_t dalf_deta  = dalf_deta_n[dz][dy][dx];
               const real_t dalf_dzeta = dalf_dzeta_n[dz][dy][dx];

               grad_e[dz][dy][dx][0] =
                  Jpr_inv[0] * dalf_dxi + Jpr_inv[1] * dalf_deta + Jpr_inv[2] * dalf_dzeta;
               grad_e[dz][dy][dx][1] =
                  Jpr_inv[3] * dalf_dxi + Jpr_inv[4] * dalf_deta + Jpr_inv[5] * dalf_dzeta;
               grad_e[dz][dy][dx][2] =
                  Jpr_inv[6] * dalf_dxi + Jpr_inv[7] * dalf_deta + Jpr_inv[8] * dalf_dzeta;
            }
         }
      }

      // Compute the Hessian coefficients at DOF nodes into hess_e.
      // Takes derivatives of grad_e.
      real_t hess_e[MD1][MD1][MD1][3][3];
      for (int dz = 0; dz < D1D; dz++)
      {
         for (int dy = 0; dy < D1D; dy++)
         {
            for (int dx = 0; dx < D1D; dx++)
            {
               for (int i = 0; i < 3; i++)
               {
                  for (int j = 0; j < 3; j++) { hess_e[dz][dy][dx][i][j] = 0.0; }
               }

               const real_t Jpr[9] =
               {
                  r_X_grad_nodes[0][0][dz][dy][dx], r_X_grad_nodes[1][0][dz][dy][dx], r_X_grad_nodes[2][0][dz][dy][dx],
                  r_X_grad_nodes[0][1][dz][dy][dx], r_X_grad_nodes[1][1][dz][dy][dx], r_X_grad_nodes[2][1][dz][dy][dx],
                  r_X_grad_nodes[0][2][dz][dy][dx], r_X_grad_nodes[1][2][dz][dy][dx], r_X_grad_nodes[2][2][dz][dy][dx]
               };
               real_t Jpr_inv[9];
               kernels::CalcInverse<3>(Jpr, Jpr_inv);

               for (int c = 0; c < 3; c++)
               {
                  kernels::internal::s_regs3d_t<MD1> rgrad_nodes, dd_dxi_n, dd_deta_n, dd_dzeta_n;
                  for (int zz = 0; zz < D1D; zz++)
                  {
                     for (int yy = 0; yy < D1D; yy++)
                     {
                        for (int xx = 0; xx < D1D; xx++)
                        {
                           rgrad_nodes[zz][yy][xx] = grad_e[zz][yy][xx][c];
                        }
                     }
                  }
                  kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                            sG_nodes, sB_nodes, sB_nodes,
                                                            rgrad_nodes, dd_dxi_n);
                  for (int zz = 0; zz < D1D; zz++)
                  {
                     for (int yy = 0; yy < D1D; yy++)
                     {
                        for (int xx = 0; xx < D1D; xx++)
                        {
                           rgrad_nodes[zz][yy][xx] = grad_e[zz][yy][xx][c];
                        }
                     }
                  }
                  kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                            sB_nodes, sG_nodes, sB_nodes,
                                                            rgrad_nodes, dd_deta_n);
                  for (int zz = 0; zz < D1D; zz++)
                  {
                     for (int yy = 0; yy < D1D; yy++)
                     {
                        for (int xx = 0; xx < D1D; xx++)
                        {
                           rgrad_nodes[zz][yy][xx] = grad_e[zz][yy][xx][c];
                        }
                     }
                  }
                  kernels::internal::Contract3d<false, MD1>(D1D, D1D, smem_d,
                                                            sB_nodes, sB_nodes, sG_nodes,
                                                            rgrad_nodes, dd_dzeta_n);

                  const real_t dd_dxi   = dd_dxi_n[dz][dy][dx];
                  const real_t dd_deta  = dd_deta_n[dz][dy][dx];
                  const real_t dd_dzeta = dd_dzeta_n[dz][dy][dx];

                  const real_t ddx =
                     Jpr_inv[0] * dd_dxi + Jpr_inv[1] * dd_deta + Jpr_inv[2] * dd_dzeta;
                  const real_t ddy =
                     Jpr_inv[3] * dd_dxi + Jpr_inv[4] * dd_deta + Jpr_inv[5] * dd_dzeta;
                  const real_t ddz =
                     Jpr_inv[6] * dd_dxi + Jpr_inv[7] * dd_deta + Jpr_inv[8] * dd_dzeta;

                  hess_e[dz][dy][dx][c][0] = ddx;
                  hess_e[dz][dy][dx][c][1] = ddy;
                  hess_e[dz][dy][dx][c][2] = ddz;
               }
            }
         }
      }

      // Interpolate gradient and Hessian at quad points.
      kernels::internal::s_regs3d_t<MQ1> r_node, r_quad;

      // Gradient at quad points: 3 scalar evals.
      for (int i = 0; i < 3; i++)
      {
         for (int dz = 0; dz < D1D; dz++)
         {
            for (int dy = 0; dy < D1D; dy++)
            {
               for (int dx = 0; dx < D1D; dx++) { r_node[dz][dy][dx] = grad_e[dz][dy][dx][i]; }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Eval3d<MQ1>(D1D, Q1D, smem_q, sB_q, r_node, r_quad);
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  ALF_grad(i, qx, qy, qz, e) = r_quad[qz][qy][qx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }

      // Hessian at quad points: 9 scalar evals.
      for (int i = 0; i < 3; i++)
      {
         for (int j = 0; j < 3; j++)
         {
            for (int dz = 0; dz < D1D; dz++)
            {
               for (int dy = 0; dy < D1D; dy++)
               {
                  for (int dx = 0; dx < D1D; dx++) { r_node[dz][dy][dx] = hess_e[dz][dy][dx][i][j]; }
               }
            }
            MFEM_SYNC_THREAD;
            kernels::internal::Eval3d<MQ1>(D1D, Q1D, smem_q, sB_q, r_node, r_quad);
            for (int qz = 0; qz < Q1D; ++qz)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
                  {
                     ALF_hess(i, j, qx, qy, qz, e) = r_quad[qz][qy][qx];
                  }
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleGradAdaptLim3D,
                       TMOP_AssembleGradPA_AdaptLim_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleGradAdaptLim3D);

void TMOP_Integrator::AssembleGradPA_AdaptLim_3D(const Vector &x) const
{
   if (PA.AL_grads_assembled) { return; }

   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto *B_nodes = PA.maps_nodes->B.Read(),
               *G_nodes = PA.maps_nodes->G.Read();
   const auto *B = PA.maps->B.Read();
   const auto X = Reshape(x.Read(), d, d, d, 3, NE);
   const auto ALF = Reshape(PA.ALF.Read(), d, d, d, NE);
   auto ALF_grad = Reshape(PA.ALFG.Write(), 3, q, q, q, NE);
   auto ALF_hess = Reshape(PA.ALFH.Write(), 3, 3, q, q, q, NE);

   TMOPAssembleGradAdaptLim3D::Run(d, q, NE, B_nodes, G_nodes, B, X, ALF,
                                   ALF_grad, ALF_hess, d, q);
   PA.AL_grads_assembled = true;
}

} // namespace mfem
