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

#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{
class TMOPEnergyPA3D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;
   real_t energy;

public:
   TMOPEnergyPA3D(const TMOP_Integrator *ti, const Vector &x): ti(ti), x(x) {}

   int Ndof() const { return ti->PA.maps->ndof; }
   int Nqpt() const { return ti->PA.maps->nqpt; }

   real_t Energy() const { return energy; }

   template <typename METRIC, int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
   static void Mult(TMOPEnergyPA3D &ker)
   {
      const mfem::TMOP_Integrator *ti = ker.ti;

      constexpr int DIM = 3;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d = ker.Ndof(), q = ker.Nqpt();

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const auto *w = mp.Read();

      const Vector &mc = ti->PA.MC;
      const bool const_m0 = mc.Size() == 1;
      const auto MC =
         const_m0 ? Reshape(mc.Read(), 1, 1, 1, 1) : Reshape(mc.Read(), q, q, q, NE);

      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
      const auto B = Reshape(ti->PA.maps->B.Read(), q, d);
      const auto G = Reshape(ti->PA.maps->G.Read(), q, d);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), q, q, q);
      const auto X = Reshape(ker.x.Read(), d, d, d, DIM, NE);
      auto E = Reshape(ti->PA.E.Write(), q, q, q, NE);

      const int Q1D = T_Q1D ? T_Q1D : q;

      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         const int D1D = T_D1D ? T_D1D : q;
         const int Q1D = T_Q1D ? T_Q1D : q;
         constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
         constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

         MFEM_SHARED real_t BG[2][MQ1 * MD1];
         MFEM_SHARED real_t DDD[3][MD1 * MD1 * MD1];
         MFEM_SHARED real_t DDQ[6][MD1 * MD1 * MQ1];
         MFEM_SHARED real_t DQQ[9][MD1 * MQ1 * MQ1];
         MFEM_SHARED real_t QQQ[9][MQ1 * MQ1 * MQ1];

         kernels::internal::LoadX<MD1>(e, D1D, X, DDD);
         kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

         kernels::internal::GradX<MD1, MQ1>(D1D, Q1D, BG, DDD, DDQ);
         kernels::internal::GradY<MD1, MQ1>(D1D, Q1D, BG, DDQ, DQQ);
         kernels::internal::GradZ<MD1, MQ1>(D1D, Q1D, BG, DQQ, QQQ);

         MFEM_FOREACH_THREAD(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJtr = kernels::Det<3>(Jtr);
                  const real_t m_coef =
                     const_m0 ? MC(0, 0, 0, 0) : MC(qx, qy, qz, e);
                  const real_t weight =
                     metric_normal * m_coef * W(qx, qy, qz) * detJtr;

                  // Jrt = Jtr^{-1}
                  real_t Jrt[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);

                  // Jpr = X^t.DSh
                  real_t Jpr[9];
                  kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, QQQ, Jpr);

                  // Jpt = X^t.DS = (X^t.DSh).Jrt = Jpr.Jrt
                  real_t Jpt[9];
                  kernels::Mult(3, 3, 3, Jpr, Jrt, Jpt);

                  // metric->EvalW(Jpt);
                  // real_t B[9];
                  // kernels::InvariantsEvaluator3D ie(Args().J(Jpt).B(B));

                  // auto EvalW_302 = [&]() // I1b * I2b / 9 - 1
                  // { return ie.Get_I1b() * ie.Get_I2b() / 9. - 1.; };

                  // auto EvalW_303 = [&]() // mu_303 = I1b/3 - 1
                  // { return ie.Get_I1b() / 3. - 1.; };

                  // auto EvalW_315 = [&]() // (I3b - 1)^2
                  // {
                  //    const real_t a = ie.Get_I3b() - 1.0;
                  //    return a * a;
                  // };

                  // auto EvalW_318 = [&]() //  0.5 * (I3 + 1/I3) - 1.
                  // {
                  //    const real_t I3 = ie.Get_I3();
                  //    return 0.5 * (I3 + 1.0 / I3) - 1.0;
                  // };

                  // auto EvalW_321 = [&]() // I1 + I2/I3 - 6
                  // { return ie.Get_I1() + ie.Get_I2() / ie.Get_I3() - 6.0; };

                  // auto EvalW_332 = [&]()
                  // { return w[0] * EvalW_302() + w[1] * EvalW_315(); };

                  // auto EvalW_338 = [&]()
                  // { return w[0] * EvalW_302() + w[1] * EvalW_318(); };

                  // const real_t EvalW = mid == 302 ? EvalW_302()
                  //                      : mid == 303 ? EvalW_303()
                  //                      : mid == 315 ? EvalW_315()
                  //                      : mid == 318 ? EvalW_318()
                  //                      : mid == 321 ? EvalW_321()
                  //                      : mid == 332 ? EvalW_332()
                  //                      : mid == 338 ? EvalW_338()
                  //                      : 0.0;

                  const real_t EvalW = METRIC{}.EvalW(Jpt, w);

                  E(qx, qy, qz, e) = weight * EvalW;
               }
            }
         }
      });
      ker.energy = ti->PA.E * ti->PA.O;
   }

   // MFEM_TMOP_REGISTER_KERNELS(TMOPEnergyPA3D, TMOP_EnergyPA_3D);
   // MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPEnergyPA3D);

   /*real_t TMOP_Integrator::GetLocalStateEnergyPA_3D(const Vector &x) const
   {
      constexpr int DIM = 3;
      const real_t mn = metric_normal;
      const int NE = PA.ne, M = metric->Id();
      const int d = PA.maps->ndof, q = PA.maps->nqpt;

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
      {
         m->GetWeights(mp);
      }
      const auto *w = mp.Read();

      const Vector &mc = PA.MC;
      const bool const_m0 = mc.Size() == 1;
      const auto MC =
         const_m0 ? Reshape(mc.Read(), 1, 1, 1, 1) : Reshape(mc.Read(), q, q, q, NE);

      const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
      const auto B = Reshape(PA.maps->B.Read(), q, d);
      const auto G = Reshape(PA.maps->G.Read(), q, d);
      const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
      const auto X = Reshape(x.Read(), d, d, d, DIM, NE);
      auto E = Reshape(PA.E.Write(), q, q, q, NE);

      TMOPEnergyPA3D::Run(d, q, //
                          mn, w, const_m0, MC, M, NE, J, W, B, G, X, E, d, q, 4);

      return PA.E * PA.O;
   }*/
};

} // namespace mfem
