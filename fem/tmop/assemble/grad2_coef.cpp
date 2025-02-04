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

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_SetupGradPA_C0_2D(const real_t lim_normal,
                            const ConstDeviceCube &LD,
                            const bool const_c0,
                            const DeviceTensor<3, const real_t> &C0,
                            const int NE,
                            const DeviceTensor<5, const real_t> &J,
                            const ConstDeviceMatrix &W,
                            const ConstDeviceMatrix &b,
                            const ConstDeviceMatrix &bld,
                            const DeviceTensor<4, const real_t> &X0,
                            const DeviceTensor<4, const real_t> &X1,
                            DeviceTensor<5> &H0,
                            const bool exp_lim,
                            const int d1d,
                            const int q1d,
                            const int max)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int DIM = 2;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t B[MQ1 * MD1];
      MFEM_SHARED real_t BLD[MQ1 * MD1];

      MFEM_SHARED real_t XY[NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ[NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ[NBZ][MQ1 * MQ1];

      MFEM_SHARED real_t XY0[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ0[2][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ0[2][NBZ][MQ1 * MQ1];

      MFEM_SHARED real_t XY1[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ1[2][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ1[2][NBZ][MQ1 * MQ1];

      kernels::internal::LoadX<MD1, NBZ>(e, D1D, LD, XY);
      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X0, XY0);
      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X1, XY1);

      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, b, B);
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, bld, BLD);

      kernels::internal::EvalX<MD1, MQ1, NBZ>(D1D, Q1D, BLD, XY, DQ);
      kernels::internal::EvalY<MD1, MQ1, NBZ>(D1D, Q1D, BLD, DQ, QQ);

      kernels::internal::EvalX<MD1, MQ1, NBZ>(D1D, Q1D, B, XY0, DQ0);
      kernels::internal::EvalY<MD1, MQ1, NBZ>(D1D, Q1D, B, DQ0, QQ0);

      kernels::internal::EvalX<MD1, MQ1, NBZ>(D1D, Q1D, B, XY1, DQ1);
      kernels::internal::EvalY<MD1, MQ1, NBZ>(D1D, Q1D, B, DQ1, QQ1);

      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t weight = W(qx, qy) * detJtr;
            const real_t coeff0 = const_c0 ? C0(0, 0, 0) : C0(qx, qy, e);
            const real_t weight_m = weight * lim_normal * coeff0;

            real_t D, p0[2], p1[2];
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ, D);
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ0, p0);
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ1, p1);

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
            ConstDeviceMatrix gg(grad_grad, DIM, DIM);

            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  H0(i, j, qx, qy, e) = weight_m * gg(i, j);
               }
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPAssembleGradCoef2D, TMOP_SetupGradPA_C0_2D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPAssembleGradCoef2D);

void TMOP_Integrator::AssembleGradPA_C0_2D(const Vector &x) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, d = PA.maps_lim->ndof, q = PA.maps_lim->nqpt;
   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;

   const auto C0 = PA.C0.Size() == 1 ? Reshape(PA.C0.Read(), 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), q, d);
   const auto LD = Reshape(PA.LD.Read(), d, d, NE);
   const auto X0 = Reshape(PA.X0.Read(), d, d, DIM, NE);
   const auto X = Reshape(x.Read(), d, d, DIM, NE);
   auto H0 = Reshape(PA.H0.Write(), DIM, DIM, q, q, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPAssembleGradCoef2D::Run(d, q, ln, LD, const_c0, C0, NE, J, W, B, BLD, X0,
                               X, H0, exp_lim, d, q, 4);
}

} // namespace mfem
