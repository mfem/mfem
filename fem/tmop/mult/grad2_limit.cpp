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
void TMOP_AddMultGradPA_C0_2D(const int NE,
                              const real_t *b,
                              const DeviceTensor<5, const real_t> &H0,
                              const DeviceTensor<4, const real_t> &X,
                              DeviceTensor<4> &Y,
                              const int d1d,
                              const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      kernels::internal::v_regs2d_t<2,MQ1> r0, r1;
      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            // Xh = X^T . Sh
            const real_t Xh[2] = { r1(0, qy, qx), r1(1, qy, qx) };

            real_t H_data[4];
            DeviceMatrix H(H_data, 2, 2);
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++) { H(i, j) = H0(i, j, qx, qy, e); }
            }

            // p2 = H . Xh
            real_t p2[2];
            kernels::Mult(2, 2, H_data, Xh, p2);
            r0(0,qy,qx) = p2[0];
            r0(1,qy,qx) = p2[1];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, D1D, r1, Y);
   });
}

// Gradient action for AdaptLim limiting (2D)
template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultGradPA_AdaptLim_2D(const real_t lim_normal,
                                   const real_t adapt_lim_delta_max,
                                   const bool const_coeff,
                                   const DeviceTensor<3, const real_t> &ALC,
                                   const int NE,
                                   const DeviceTensor<5, const real_t> &J,
                                   const ConstDeviceMatrix &W,
                                   const real_t *b,
                                   const DeviceTensor<4, const real_t> &R,
                                   const DeviceTensor<4, const real_t> &ALF_grad,
                                   const DeviceTensor<5, const real_t> &ALF_hess,
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
      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);

      // Load ALF and ALF0 values at quadrature points
      kernels::internal::s_regs2d_t<MQ1> ralf_val_dof, ralf_val_quad;
      kernels::internal::LoadDofs2d(e, D1D, ALF, ralf_val_dof);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, ralf_val_dof, ralf_val_quad);

      kernels::internal::s_regs2d_t<MQ1> ralf0_val_dof, ralf0_val_quad;
      kernels::internal::LoadDofs2d(e, D1D, ALF0, ralf0_val_dof);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, ralf0_val_dof, ralf0_val_quad);

      // Load input vector R (the direction for Hessian action)
      kernels::internal::v_regs2d_t<2,MQ1> r_R_dof, r_R_quad;
      kernels::internal::LoadDofs2d(e, D1D, R, r_R_dof);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r_R_dof, r_R_quad);

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

            // Load precomputed gradient at this quadrature point
            real_t grad_alf[2] = { ALF_grad(0, qx, qy, e),
                                   ALF_grad(1, qx, qy, e) };

            // Load precomputed Hessian at this quadrature point
            real_t hess_alf[2][2];
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               {
                  hess_alf[i][j] = ALF_hess(i, j, qx, qy, e);
               }
            }

            // Get input vector at this quad point
            const real_t R_q[2] = { r_R_quad(0, qy, qx), r_R_quad(1, qy, qx) };

            // Compute Hessian action:
            // H*R = 2.0 / delta_max^2 * (grad_alf . R_q * grad_alf + (gf_val - gf0_val) * Hess_alf * R_q)
            const real_t coeff = const_coeff ? ALC(0, 0, 0) : ALC(qx, qy, e);
            const real_t factor = weight * lim_normal * coeff *
                                  2.0 / (adapt_lim_delta_max * adapt_lim_delta_max);
            
            // First term: grad_alf . R_q (outer product contribution)
            const real_t grad_dot_R = grad_alf[0] * R_q[0] + grad_alf[1] * R_q[1];
            
            // Second term: Hess_alf * R_q
            real_t hess_R[2];
            hess_R[0] = hess_alf[0][0] * R_q[0] + hess_alf[0][1] * R_q[1];
            hess_R[1] = hess_alf[1][0] * R_q[0] + hess_alf[1][1] * R_q[1];

            r00(0, qy, qx) = factor * (grad_dot_R * grad_alf[0] + (gf_val - gf0_val) * hess_R[0]);
            r00(1, qy, qx) = factor * (grad_dot_R * grad_alf[1] + (gf_val - gf0_val) * hess_R[1]);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r00, r01);
      kernels::internal::WriteDofs2d(e, D1D, r01, Y);
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultGradCoefKernels, TMOP_AddMultGradPA_C0_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultGradCoefKernels);

void TMOP_Integrator::AddMultGradPA_C0_2D(const Vector &R, Vector &C) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto H0 = Reshape(PA.H0.Read(), 2, 2, q, q, NE);
   const auto *b = PA.maps->B.Read();
   const auto X = Reshape(R.Read(), d, d, 2, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, 2, NE);

   TMOPMultGradCoefKernels::Run(d, q, NE, b, H0, X, Y, d, q);
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultGradAdaptLim, TMOP_AddMultGradPA_AdaptLim_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultGradAdaptLim);

void TMOP_Integrator::AddMultGradPA_AdaptLim_2D(const Vector &R, Vector &C) const
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
   const auto *B = PA.maps->B.Read();
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto RR = Reshape(R.Read(), d, d, 2, NE);
   const auto ALF = Reshape(PA.ALF.Read(), d, d, NE);
   const auto ALF0 = Reshape(PA.ALF0.Read(), d, d, NE);
   const auto ALF_grad = Reshape(PA.ALFG.Read(), 2, q, q, NE);
   const auto ALF_hess = Reshape(PA.ALFH.Read(), 2, 2, q, q, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, 2, NE);

   TMOPMultGradAdaptLim::Run(d, q, ln, delta_max, const_coeff, ALC, NE, J, W, B,
                             RR, ALF_grad, ALF_hess, ALF, ALF0, Y, d, q);
}

} // namespace mfem
