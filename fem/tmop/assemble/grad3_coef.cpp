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

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_SetupGradPA_C0_3D(const real_t lim_normal,
                            const DeviceTensor<4, const real_t> &LD,
                            const bool const_c0,
                            const DeviceTensor<4, const real_t> &C0,
                            const int NE,
                            const DeviceTensor<6, const real_t> &J,
                            const ConstDeviceCube &W,
                            const ConstDeviceMatrix &b,
                            const ConstDeviceMatrix &bld,
                            const DeviceTensor<5, const real_t> &X0,
                            const DeviceTensor<5, const real_t> &X1,
                            DeviceTensor<6> &H0,
                            const bool exp_lim,
                            const int d1d,
                            const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t B[MQ1 * MD1];
      MFEM_SHARED real_t sBLD[MQ1 * MD1];
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, bld, sBLD);

      MFEM_SHARED real_t sm0[MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ * MDQ];
      DeviceCube QQQ(sm1, MD1,MD1,MD1);

      MFEM_SHARED real_t s00[3][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t s01[3][MDQ * MDQ * MDQ];

      MFEM_SHARED real_t s10[3][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t s11[3][MDQ * MDQ * MDQ];

      kernels::internal::LoadX<MDQ>(e, D1D, LD, sm0);
      kernels::internal::LoadX_v<MDQ>(e, D1D, X0, s00);
      kernels::internal::LoadX_v<MDQ>(e, D1D, X1, s10);

      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, b, B);

      kernels::internal::EvalX<MD1, MQ1>(D1D, Q1D, B, sm0, sm1);
      kernels::internal::EvalY<MD1, MQ1>(D1D, Q1D, B, sm1, sm0);
      kernels::internal::EvalZ<MD1, MQ1>(D1D, Q1D, B, sm0, sm1);

      kernels::internal::EvalX<MD1, MQ1>(D1D, Q1D, B, s00, s01);
      kernels::internal::EvalY<MD1, MQ1>(D1D, Q1D, B, s01, s00);
      kernels::internal::EvalZ<MD1, MQ1>(D1D, Q1D, B, s00, s01);

      kernels::internal::EvalX<MD1, MQ1>(D1D, Q1D, B, s10, s11);
      kernels::internal::EvalY<MD1, MQ1>(D1D, Q1D, B, s11, s10);
      kernels::internal::EvalZ<MD1, MQ1>(D1D, Q1D, B, s10, s11);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx, qy, qz) * detJtr;
               const real_t coeff0 =
                  const_c0 ? C0(0, 0, 0, 0) : C0(qx, qy, qz, e);
               const real_t weight_m = weight * lim_normal * coeff0;

               real_t D, p0[3], p1[3];
               kernels::internal::PullEval(qx, qy, qz, QQQ, D);
               kernels::internal::PullEval<MDQ>(Q1D, qx, qy, qz, s01, p0);
               kernels::internal::PullEval<MDQ>(Q1D, qx, qy, qz, s11, p1);

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
               ConstDeviceMatrix gg(grad_grad, DIM, DIM);

               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     H0(i, j, qx, qy, qz, e) = weight_m * gg(i, j);
                  }
               }
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPAssembleGradCoef3D, TMOP_SetupGradPA_C0_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPAssembleGradCoef3D);

void TMOP_Integrator::AssembleGradPA_C0_3D(const Vector &x) const
{
   constexpr int DIM = 3;
   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const int NE = PA.ne, d = PA.maps_lim->ndof, q = PA.maps_lim->nqpt;

   const auto C0 = const_c0 ? Reshape(PA.C0.Read(), 1, 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, q, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), q, d);
   const auto LD = Reshape(PA.LD.Read(), d, d, d, NE);
   const auto X0 = Reshape(PA.X0.Read(), d, d, d, DIM, NE);
   const auto X = Reshape(x.Read(), d, d, d, DIM, NE);
   auto H0 = Reshape(PA.H0.Write(), DIM, DIM, q, q, q, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPAssembleGradCoef3D::Run(d, q, ln, LD, const_c0, C0, NE, J, W, B, BLD, X0,
                               X, H0, exp_lim, d, q);
}

} // namespace mfem
