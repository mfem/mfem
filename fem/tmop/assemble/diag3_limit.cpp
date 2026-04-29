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
void TMOP_AssembleDiagPA_C0_3D(const int NE,
                               const ConstDeviceMatrix &B,
                               const DeviceTensor<6, const real_t> &H0,
                               DeviceTensor<5> &D,
                               const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::s_regs3d_t<MQ1> r0, r1;

      for (int v = 0; v < 3; ++v)
      {
         // first tensor contraction, along z direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = B(qz, dz);
                     u += Bz * H0(v, v, qx, qy, qz, e) * Bz;
                  }
                  r0[dz][qy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // second tensor contraction, along y direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[qy][qx] = r0[dz][qy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy);
                     u += By * smem[qy][qx] * By;
                  }
                  r1[dz][dy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // third tensor contraction, along x direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[dy][qx] = r1[dz][dy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx);
                     u += Bx * smem[dy][qx] * Bx;
                  }
                  D(dx, dy, dz, v, e) += u;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiagCoef3D, TMOP_AssembleDiagPA_C0_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiagCoef3D);

void TMOP_Integrator::AssembleDiagonalPA_C0_3D(Vector &diagonal) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto H0 = Reshape(PA.H0.Read(), 3, 3, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, 3, NE);

   TMOPAssembleDiagCoef3D::Run(d, q, NE, B, H0, D, d, q);
}

// Diagonal assembly for AdaptLim limiting (3D)
template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleDiagPA_AdaptLim_3D(const real_t lim_normal,
                                     const real_t adapt_lim_delta_max,
                                     const bool const_coeff,
                                     const DeviceTensor<4, const real_t> &ALC,
                                     const int NE,
                                     const DeviceTensor<6, const real_t> &J,
                                     const ConstDeviceCube &W,
                                     const real_t *b,
                                     const DeviceTensor<5, const real_t> &ALF_grad,
                                     const DeviceTensor<6, const real_t> &ALF_hess,
                                     const DeviceTensor<4, const real_t> &ALF,
                                     const DeviceTensor<4, const real_t> &ALF0,
                                     DeviceTensor<5> &D,
                                     const int d1d,
                                     const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const real_t inv_delta_sq =
      1.0 / (adapt_lim_delta_max * adapt_lim_delta_max);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      // ALF and ALF0 values at quad points.
      kernels::internal::s_regs3d_t<MQ1> alf_dof, alf_quad;
      kernels::internal::LoadDofs3d(e, D1D, ALF, alf_dof);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, alf_dof, alf_quad);
      kernels::internal::s_regs3d_t<MQ1> alf0_dof, alf0_quad;
      kernels::internal::LoadDofs3d(e, D1D, ALF0, alf0_dof);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, alf0_dof, alf0_quad);

      kernels::internal::s_regs3d_t<MQ1> r0, r1;

      for (int v = 0; v < 3; ++v)
      {
         // Contract in z.
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = sB[dz][qz];
                     const real_t bb = Bz * Bz;

                     const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                     const real_t detJtr = kernels::Det<3>(Jtr);
                     const real_t weight = W(qx, qy, qz) * detJtr;
                     const real_t coeff = const_coeff ? ALC(0, 0, 0, 0) : ALC(qx, qy, qz, e);
                     const real_t factor = weight * lim_normal * coeff * 2.0 * inv_delta_sq;

                     const real_t diff = alf_quad(qz, qy, qx) - alf0_quad(qz, qy, qx);
                     const real_t grad_v = ALF_grad(v, qx, qy, qz, e);
                     const real_t hess_vv = ALF_hess(v, v, qx, qy, qz, e);
                     const real_t hdiag = factor * (grad_v * grad_v + diff * hess_vv);

                     u += bb * hdiag;
                  }
                  r0[dz][qy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // Contract in y.
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[qy][qx] = r0[dz][qy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = sB[dy][qy];
                     u += (By * By) * smem[qy][qx];
                  }
                  r1[dz][dy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // Contract in x.
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[dy][qx] = r1[dz][dy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = sB[dx][qx];
                     u += (Bx * Bx) * smem[dy][qx];
                  }
                  D(dx, dy, dz, v, e) += u;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiagAdaptLim3D,
                       TMOP_AssembleDiagPA_AdaptLim_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiagAdaptLim3D);

void TMOP_Integrator::AssembleDiagonalPA_AdaptLim_3D(Vector &diagonal) const
{
   const real_t ln = lim_normal;
   const real_t delta_max = PA.al_delta;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const bool const_coeff = PA.ALC.Size() == 1;
   const auto ALC = const_coeff
                    ? Reshape(PA.ALC.Read(), 1, 1, 1, 1)
                    : Reshape(PA.ALC.Read(), q, q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), 3, 3, q, q, q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto *B = PA.maps->B.Read();
   const auto ALF = Reshape(PA.ALF.Read(), d, d, d, NE);
   const auto ALF0 = Reshape(PA.ALF0.Read(), d, d, d, NE);
   const auto ALF_grad = Reshape(PA.ALFG.Read(), 3, q, q, q, NE);
   const auto ALF_hess = Reshape(PA.ALFH.Read(), 3, 3, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, 3, NE);

   TMOPAssembleDiagAdaptLim3D::Run(d, q, ln, delta_max, const_coeff, ALC, NE,
                                   J, W, B, ALF_grad, ALF_hess, ALF, ALF0, D,
                                   d, q);
}

} // namespace mfem
