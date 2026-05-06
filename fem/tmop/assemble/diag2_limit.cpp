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
void TMOP_AssembleDiagPA_C0_2D(const int NE,
                               const ConstDeviceMatrix &B,
                               const DeviceTensor<5, const real_t> &H0,
                               DeviceTensor<4> &D,
                               const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t qd[MQ1 * MD1];
      DeviceTensor<2, real_t> QD(qd, MQ1, MD1);

      for (int v = 0; v < 2; v++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               QD(qx, dy) = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t bb = B(qy, dy) * B(qy, dy);
                  QD(qx, dy) += bb * H0(v, v, qx, qy, e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t d = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t bb = B(qx, dx) * B(qx, dx);
                  d += bb * QD(qx, dy);
               }
               D(dx, dy, v, e) += d;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Diagonal assembly for AdaptLim limiting (2D)
template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleDiagPA_AdaptLim_2D(const real_t lim_normal,
                                     const real_t adapt_lim_delta_max,
                                     const bool const_coeff,
                                     const DeviceTensor<3, const real_t> &ALC,
                                     const int NE,
                                     const DeviceTensor<5, const real_t> &J,
                                     const ConstDeviceMatrix &W,
                                     const real_t *b,
                                     const DeviceTensor<4, const real_t> &ALF_grad,
                                     const DeviceTensor<5, const real_t> &ALF_hess,
                                     const ConstDeviceCube &ALFmF0,
                                     DeviceTensor<4> &D,
                                     const int d1d,
                                     const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const real_t normal_inv_delta_sq =
      2.0 * lim_normal / (adapt_lim_delta_max * adapt_lim_delta_max);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      // ALF and ALF0 values at quad points.
      kernels::internal::s_regs2d_t<MQ1> alf_dof, alf_quad;
      kernels::internal::LoadDofs2d(e, D1D, ALFmF0, alf_dof);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, alf_dof, alf_quad);

      MFEM_SHARED real_t qd[MQ1 * MD1];
      DeviceTensor<2, real_t> QD(qd, MQ1, MD1);

      for (int v = 0; v < 2; v++)
      {
         // Contract in y.
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               QD(qx, dy) = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t By = sB[dy][qy];
                  const real_t bb = By * By;

                  const real_t *Jtr = &J(0, 0, qx, qy, e);
                  const real_t detJtr = kernels::Det<2>(Jtr);
                  const real_t weight = W(qx, qy) * detJtr;
                  const real_t coeff = const_coeff ? ALC(0, 0, 0) : ALC(qx, qy, e);
                  const real_t factor = weight * coeff * normal_inv_delta_sq;

                  const real_t diff = alf_quad(qy, qx);
                  const real_t grad_v = ALF_grad(v, qx, qy, e);
                  const real_t hess_vv = ALF_hess(v, v, qx, qy, e);
                  const real_t hdiag = factor * (grad_v * grad_v + diff * hess_vv);

                  QD(qx, dy) += bb * hdiag;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Contract in x.
         MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
            {
               real_t d = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t Bx = sB[dx][qx];
                  const real_t bb = Bx * Bx;
                  d += bb * QD(qx, dy);
               }
               D(dx, dy, v, e) += d;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiagCoef2D, TMOP_AssembleDiagPA_C0_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiagCoef2D);

void TMOP_Integrator::AssembleDiagonalPA_C0_2D(Vector &diagonal) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto H0 = Reshape(PA.H0.Read(), 2, 2, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, 2, NE);

   TMOPAssembleDiagCoef2D::Run(d, q, NE, B, H0, D, d, q);
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiagAdaptLim2D,
                       TMOP_AssembleDiagPA_AdaptLim_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiagAdaptLim2D);

void TMOP_Integrator::AssembleDiagonalPA_AdaptLim_2D(Vector &diagonal) const
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
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto *B = PA.maps->B.Read();
   const auto ALFmF0 = Reshape(PA.ALFmF0.Read(), d, d, NE);
   const auto ALF_grad = Reshape(PA.ALFG.Read(), 2, q, q, NE);
   const auto ALF_hess = Reshape(PA.ALFH.Read(), 2, 2, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, 2, NE);

   TMOPAssembleDiagAdaptLim2D::Run(d, q, ln, delta_max, const_coeff, ALC, NE,
                                   J, W, B, ALF_grad, ALF_hess, ALFmF0, D, d, q);
}

} // namespace mfem
