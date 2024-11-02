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

#include "../../../tmop.hpp"
#include "../../../kernels.hpp"
#include "../../../../general/forall.hpp"
#include "../../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultPA_C0_2D(const real_t lim_normal,
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
                          DeviceTensor<4> &Y,
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

            real_t ld, p0[2], p1[2];
            const real_t coeff0 = const_c0 ? C0(0, 0, 0) : C0(qx, qy, e);
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ, ld);
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ0, p0);
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ1, p1);

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
            kernels::internal::PushEval<MQ1, NBZ>(Q1D, qx, qy, d1, QQ0);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1, MQ1>(D1D, Q1D, b, B);
      kernels::internal::EvalXt<MD1, MQ1, NBZ>(D1D, Q1D, B, QQ0, DQ0);
      kernels::internal::EvalYt<MD1, MQ1, NBZ>(D1D, Q1D, B, DQ0, Y, e);
   });
}

void TMOP_Integrator::AddMultPA_C0_2D(const Vector &x, Vector &y) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const real_t ln = lim_normal;
   MFEM_VERIFY(PA.maps_lim->ndof == d, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == q, "");

   const bool const_c0 = PA.C0.Size() == 1;
   const auto C0 = const_c0 ? Reshape(PA.C0.Read(), 1, 1, 1)
                            : Reshape(PA.C0.Read(), q, q, NE);
   const auto LD = Reshape(PA.LD.Read(), d, d, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), q, d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto X0 = Reshape(PA.X0.Read(), d, d, DIM, NE);
   const auto X = Reshape(x.Read(), d, d, DIM, NE);
   auto Y = Reshape(y.ReadWrite(), d, d, DIM, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   decltype(&TMOP_AddMultPA_C0_2D<>) ker = TMOP_AddMultPA_C0_2D;

   if (d == 2 && q == 2) { ker = TMOP_AddMultPA_C0_2D<2, 2>; }
   if (d == 2 && q == 3) { ker = TMOP_AddMultPA_C0_2D<2, 3>; }
   if (d == 2 && q == 4) { ker = TMOP_AddMultPA_C0_2D<2, 4>; }
   if (d == 2 && q == 5) { ker = TMOP_AddMultPA_C0_2D<2, 5>; }
   if (d == 2 && q == 6) { ker = TMOP_AddMultPA_C0_2D<2, 6>; }

   if (d == 3 && q == 3) { ker = TMOP_AddMultPA_C0_2D<3, 3>; }
   if (d == 3 && q == 4) { ker = TMOP_AddMultPA_C0_2D<3, 4>; }
   if (d == 3 && q == 5) { ker = TMOP_AddMultPA_C0_2D<3, 5>; }
   if (d == 3 && q == 6) { ker = TMOP_AddMultPA_C0_2D<3, 6>; }

   if (d == 4 && q == 4) { ker = TMOP_AddMultPA_C0_2D<4, 4>; }
   if (d == 4 && q == 5) { ker = TMOP_AddMultPA_C0_2D<4, 5>; }
   if (d == 4 && q == 6) { ker = TMOP_AddMultPA_C0_2D<4, 6>; }

   if (d == 5 && q == 5) { ker = TMOP_AddMultPA_C0_2D<5, 5>; }
   if (d == 5 && q == 6) { ker = TMOP_AddMultPA_C0_2D<5, 6>; }

   ker(ln, LD, const_c0, C0, NE, J, W, B, BLD, X0, X, Y, exp_lim, d, q, 4);
}

} // namespace mfem
