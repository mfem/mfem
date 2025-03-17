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
#pragma once

#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

class TMOPAddMultPA2D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;
   Vector &y;

public:
   TMOPAddMultPA2D(const TMOP_Integrator *ti, const Vector &x, Vector &y):
      ti(ti),
      x(x),
      y(y)
   {
   }

   int Ndof() const { return ti->PA.maps->ndof; }
   int Nqpt() const { return ti->PA.maps->nqpt; }

   template <typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPAddMultPA2D &ker)
   {
      const mfem::TMOP_Integrator *ti = ker.ti;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d1d = ti->PA.maps->ndof, q1d = ti->PA.maps->nqpt;

      constexpr int DIM = 2, NBZ = 1;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");
      MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const real_t *w = mp.Read();

      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, NE);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), Q1D, Q1D);
      const auto B = Reshape(ti->PA.maps->B.Read(), Q1D, D1D);
      const auto G = Reshape(ti->PA.maps->G.Read(), Q1D, D1D);
      const auto X = Reshape(ker.x.Read(), D1D, D1D, DIM, NE);
      auto Y = Reshape(ker.y.ReadWrite(), D1D, D1D, DIM, NE);

      const Vector &mc_ = ti->PA.MC;
      const bool const_m0 = mc_.Size() == 1;
      const auto MC = const_m0 ? Reshape(mc_.Read(), 1, 1, 1)
                      : Reshape(mc_.Read(), Q1D, Q1D, NE);

      mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int NBZ = 1;
         constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_TMOP_1D;
         constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_TMOP_1D;

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
               const real_t weight =
                  metric_normal * m_coef * W(qx, qy) * detJtr;

               // Jrt = Jtr^{-1}
               real_t Jrt[4];
               kernels::CalcInverse<2>(Jtr, Jrt);

               // Jpr = X{^T}.DSh
               real_t Jpr[4];
               kernels::internal::PullGrad<MQ1, NBZ>(Q1D, qx, qy, QQ, Jpr);

               // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = Jpr.Jrt
               real_t Jpt[4];
               kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

               real_t P[4];
               METRIC{}.EvalP(Jpt, w, P);

               for (int i = 0; i < 4; i++) { P[i] *= weight; }

               // PMatO += DS . P^t += DSh . (Jrt . P^t)
               real_t A[4];
               kernels::MultABt(2, 2, 2, Jrt, P, A);
               kernels::internal::PushGrad<MQ1, NBZ>(Q1D, qx, qy, A, QQ);
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::LoadBGt<MD1, MQ1>(D1D, Q1D, B, G, BG);
         kernels::internal::GradYt<MD1, MQ1, NBZ>(D1D, Q1D, BG, QQ, DQ);
         kernels::internal::GradXt<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, Y, e);
      });
   }
};

} // namespace mfem
