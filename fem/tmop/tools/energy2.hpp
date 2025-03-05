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

class TMOPEnergyPA2D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;
   real_t energy;

public:
   TMOPEnergyPA2D(const TMOP_Integrator *ti, const Vector &x): ti(ti), x(x),
      energy(std::nan("1")) {}

   int Ndof() const { return ti->PA.maps->ndof; }
   int Nqpt() const { return ti->PA.maps->nqpt; }

   real_t Energy() const
   {
      assert(!std::isnan(energy));
      return energy;
   }

   template <typename METRIC, int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
   static void Mult(TMOPEnergyPA2D &ker)
   {
      const mfem::TMOP_Integrator *ti = ker.ti;
      constexpr int DIM = 2, NBZ = 1;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d = ker.Ndof(), q = ti->PA.maps->nqpt;

      const auto B = Reshape(ti->PA.maps->B.Read(), q, d);
      const auto G = Reshape(ti->PA.maps->G.Read(), q, d);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), q, q);
      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, q, q, NE);
      const auto X = Reshape(ker.x.Read(), d, d, DIM, NE);
      auto E = Reshape(ti->PA.E.Write(), q, q, NE);

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const auto *metric_data = mp.Read();

      const int Q1D = T_Q1D ? T_Q1D : q;

      const Vector &mc_ = ti->PA.MC;
      const bool const_m0 = mc_.Size() == 1;
      const auto MC = const_m0 ? Reshape(mc_.Read(), 1, 1, 1)
                      : Reshape(mc_.Read(), Q1D, Q1D, NE);

      mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int NBZ = 1;
         constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
         constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
         const int D1D = T_D1D ? T_D1D : q;
         const int Q1D = T_Q1D ? T_Q1D : q;

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
               // kernels::InvariantsEvaluator2D ie(Args().J(Jpt));

               // auto EvalW_01 = [&]() { return ie.Get_I1(); };

               // auto EvalW_02 = [&]() { return 0.5 * ie.Get_I1b() - 1.0; };

               // auto EvalW_07 = [&]()
               // { return ie.Get_I1() * (1.0 + 1.0 / ie.Get_I2()) - 4.0; };

               // auto EvalW_56 = [&]()
               // {
               //    const real_t I2b = ie.Get_I2b();
               //    return 0.5 * (I2b + 1.0 / I2b) - 1.0;
               // };

               // auto EvalW_77 = [&]()
               // {
               //    const real_t I2b = ie.Get_I2b();
               //    return 0.5 * (I2b * I2b + 1. / (I2b * I2b) - 2.);
               // };

               // auto EvalW_80 = [&](const real_t *w)
               // { return w[0] * EvalW_02() + w[1] * EvalW_77(); };

               // auto EvalW_94 = [&](const real_t *w)
               // { return w[0] * EvalW_02() + w[1] * EvalW_56(); };

               // const real_t EvalW = mid == 1  ? EvalW_01()
               //                      : mid == 2  ? EvalW_02()
               //                      : mid == 7  ? EvalW_07()
               //                      : mid == 56 ? EvalW_56()
               //                      : mid == 77 ? EvalW_77()
               //                      : mid == 80 ? EvalW_80(metric_data)
               //                      : mid == 94 ? EvalW_94(metric_data)
               //                      : 0.0;

               const real_t EvalW = METRIC{}.EvalW(Jpt, metric_data);

               E(qx, qy, e) = weight * EvalW;
            }
         }
      });
      ker.energy = ti->PA.E * ti->PA.O;
   }
};

} // namespace mfem
