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

#include "../../tmop.hpp"
#include "../../../fem/kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"
#include "../../../linalg/dinvariants.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_EnergyPA_2D(const real_t metric_normal,
                      const real_t *w,
                      const bool const_m0,
                      const real_t *mc,
                      const real_t *metric_param,
                      const int mid,
                      const int NE,
                      const DeviceTensor<5, const real_t> &J,
                      const ConstDeviceMatrix &W,
                      const ConstDeviceMatrix &B,
                      const ConstDeviceMatrix &G,
                      const DeviceTensor<4, const real_t> &X,
                      DeviceTensor<3> &E,
                      const int d1d,
                      const int q1d,
                      const int max)
{
   using Args = kernels::InvariantsEvaluator2D::Buffers;
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 56 || mid == 77 ||
               mid == 80 || mid == 94,
               "2D metric not yet implemented!");

   constexpr int NBZ = 1;

   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto MC = const_m0 ? Reshape(mc, 1, 1, 1) : Reshape(mc, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t XY[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ[4][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ[4][NBZ][MQ1 * MQ1];

      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X, XY);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1, NBZ>(D1D, Q1D, BG, XY, DQ);
      kernels::internal::GradY<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, QQ);

      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t m_coef = const_m0 ? MC(0, 0, 0) : MC(qx, qy, e);
            const real_t weight = metric_normal * m_coef * W(qx, qy) * detJtr;

            // Jrt = Jtr^{-1}
            real_t Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^t.DSh
            real_t Jpr[4];
            kernels::internal::PullGrad<MQ1, NBZ>(Q1D, qx, qy, QQ, Jpr);

            // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
            real_t Jpt[4];
            kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

            // metric->EvalW(Jpt);
            kernels::InvariantsEvaluator2D ie(Args().J(Jpt));

            auto EvalW_01 = [&]() { return ie.Get_I1(); };

            auto EvalW_02 = [&]() { return 0.5 * ie.Get_I1b() - 1.0; };

            auto EvalW_07 = [&]()
            { return ie.Get_I1() * (1.0 + 1.0 / ie.Get_I2()) - 4.0; };

            auto EvalW_56 = [&]()
            {
               const real_t I2b = ie.Get_I2b();
               return 0.5 * (I2b + 1.0 / I2b) - 1.0;
            };

            auto EvalW_77 = [&]()
            {
               const real_t I2b = ie.Get_I2b();
               return 0.5 * (I2b * I2b + 1. / (I2b * I2b) - 2.);
            };

            auto EvalW_80 = [&]()
            { return w[0] * EvalW_02() + w[1] * EvalW_77(); };

            auto EvalW_94 = [&]()
            { return w[0] * EvalW_02() + w[1] * EvalW_56(); };

            const real_t EvalW = mid == 1  ? EvalW_01()
                                 : mid == 2  ? EvalW_02()
                                 : mid == 7  ? EvalW_07()
                                 : mid == 56 ? EvalW_56()
                                 : mid == 77 ? EvalW_77()
                                 : mid == 80 ? EvalW_80()
                                 : mid == 94 ? EvalW_94()
                                 : 0.0;

            E(qx, qy, e) = weight * EvalW;
         }
      }
   });
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_2D(const Vector &x) const
{
   constexpr int DIM = 2;
   const real_t mn = metric_normal;
   const int NE = PA.ne, MId = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;

   Array<real_t> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }

   const Vector &mc = PA.MC;

   const bool const_m0 = mc.Size() == 1;

   const auto MC =
      const_m0 ? Reshape(mc.Read(), 1, 1, 1) : Reshape(mc.Read(), q, q, NE);

   const real_t *w = mp.Read();

   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto G = Reshape(PA.maps->G.Read(), q, d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q);
   const auto X = Reshape(x.Read(), d, d, DIM, NE);

   auto E = Reshape(PA.E.Write(), q, q, NE);

   decltype(&TMOP_EnergyPA_2D<>) ker = TMOP_EnergyPA_2D;

   if (d == 2 && q == 2) { ker = TMOP_EnergyPA_2D<2, 2>; }
   if (d == 2 && q == 3) { ker = TMOP_EnergyPA_2D<2, 3>; }
   if (d == 2 && q == 4) { ker = TMOP_EnergyPA_2D<2, 4>; }
   if (d == 2 && q == 5) { ker = TMOP_EnergyPA_2D<2, 5>; }
   if (d == 2 && q == 6) { ker = TMOP_EnergyPA_2D<2, 6>; }

   if (d == 3 && q == 3) { ker = TMOP_EnergyPA_2D<3, 3>; }
   if (d == 3 && q == 4) { ker = TMOP_EnergyPA_2D<3, 4>; }
   if (d == 3 && q == 5) { ker = TMOP_EnergyPA_2D<3, 5>; }
   if (d == 3 && q == 6) { ker = TMOP_EnergyPA_2D<3, 6>; }

   if (d == 4 && q == 4) { ker = TMOP_EnergyPA_2D<4, 4>; }
   if (d == 4 && q == 5) { ker = TMOP_EnergyPA_2D<4, 5>; }
   if (d == 4 && q == 6) { ker = TMOP_EnergyPA_2D<4, 6>; }

   if (d == 5 && q == 5) { ker = TMOP_EnergyPA_2D<5, 5>; }
   if (d == 5 && q == 6) { ker = TMOP_EnergyPA_2D<5, 6>; }

   ker(mn, w, const_m0, MC, mp, MId, NE, J, W, B, G, X, E, d, q, 4);
   return PA.E * PA.O;
}

} // namespace mfem
