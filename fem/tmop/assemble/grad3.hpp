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
#include "../../kernels_sm.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

class TMOPSetupGradPA3D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;

public:
   TMOPSetupGradPA3D(const TMOP_Integrator *ti, const Vector &x): ti(ti), x(x)
   {
   }

   int Ndof() const { return ti->PA.maps->ndof; }

   int Nqpt() const { return ti->PA.maps->nqpt; }

   template <typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPSetupGradPA3D &ker)
   {
      constexpr int DIM = 3;
      const mfem::TMOP_Integrator *ti = ker.ti;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d1d = ker.Ndof(), q1d = ti->PA.maps->nqpt;

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const real_t *w = mp.Read();

      const auto B = Reshape(ti->PA.maps->B.Read(), Q1D, D1D);
      const auto G = Reshape(ti->PA.maps->G.Read(), Q1D, D1D);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), Q1D, Q1D, Q1D);
      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
      const auto X = Reshape(ker.x.Read(), D1D, D1D, D1D, DIM, NE);
      auto H = Reshape(ti->PA.H.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);


      const bool const_m0 = ti->PA.MC.Size() == 1;
      const auto MC = const_m0 ? Reshape(ti->PA.MC.Read(), 1, 1, 1, 1)
                      : Reshape(ti->PA.MC.Read(), Q1D, Q1D, Q1D, NE);

      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         const int D1D = T_D1D ? T_D1D : d1d;
         const int Q1D = T_Q1D ? T_Q1D : q1d;
         constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
         constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
         constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;

         MFEM_SHARED real_t s_BG[2][MQ1 * MD1];
         MFEM_SHARED real_t sm0[9][MDQ * MDQ * MDQ];
         MFEM_SHARED real_t sm1[9][MDQ * MDQ * MDQ];

         kernels::internal::sm::LoadX<MDQ>(e, D1D, X, sm0);
         kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, s_BG);

         kernels::internal::sm::GradX<MD1, MQ1>(D1D, Q1D, s_BG, sm0, sm1);
         kernels::internal::sm::GradY<MD1, MQ1>(D1D, Q1D, s_BG, sm1, sm0);
         kernels::internal::sm::GradZ<MD1, MQ1>(D1D, Q1D, s_BG, sm0, sm1);

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

                  // Jpr = X^T.DSh
                  real_t Jpr[9];
                  kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, sm1, Jpr);

                  // Jpt = X^T . DS = (X^T.DSh) . Jrt = Jpr . Jrt
                  real_t Jpt[9];
                  kernels::Mult(3, 3, 3, Jpr, Jrt, Jpt);

                  METRIC{}.AssembleH(qx, qy, qz, e, weight, Jrt, Jpr, Jpt, w,
                                     H);
               } // qx
            } // qy
         } // qz
      });
   }
};

} // namespace mfem
