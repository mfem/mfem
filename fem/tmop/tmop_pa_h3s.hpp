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

#include "tmop_pa.hpp"

namespace mfem
{

class TMOPSetupGradPA3D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;

public:
   TMOPSetupGradPA3D(const TMOP_Integrator *ti, const Vector &x) : ti(ti), x(x)
   {
   }

   int Ndof() const { return ti->PA.maps->ndof; }

   int Nqpt() const { return ti->PA.maps->nqpt; }

   template <typename METRIC, int T_D1D, int T_Q1D, int T_MAX = 4>
   void operator()()
   {
      constexpr int DIM = 3;
      const double metric_normal = ti->metric_normal;

      Array<double> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const double *w = mp.Read();

      const int d = ti->PA.maps->ndof, q = ti->PA.maps->nqpt;

      const int NE = ti->PA.ne;

      const auto B = Reshape(ti->PA.maps->B.Read(), q, d);
      const auto G = Reshape(ti->PA.maps->G.Read(), q, d);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), q, q, q);
      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
      const auto X = Reshape(x.Read(), d, d, d, DIM, NE);
      auto H = Reshape(ti->PA.H.Write(), DIM, DIM, DIM, DIM, q, q, q, NE);

      const int Q1D = T_Q1D ? T_Q1D : q;

      const bool const_m0 = ti->PA.MC.Size() == 1;
      const auto MC = const_m0 ? Reshape(ti->PA.MC.Read(), 1, 1, 1, 1)
                      : Reshape(ti->PA.MC.Read(), Q1D, Q1D, Q1D, NE);

      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         const int D1D = T_D1D ? T_D1D : d;
         const int Q1D = T_Q1D ? T_Q1D : q;
         constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
         constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

         MFEM_SHARED real_t s_BG[2][MQ1 * MD1];
         MFEM_SHARED real_t s_DDD[3][MD1 * MD1 * MD1];
         MFEM_SHARED real_t s_DDQ[9][MD1 * MD1 * MQ1];
         MFEM_SHARED real_t s_DQQ[9][MD1 * MQ1 * MQ1];
         MFEM_SHARED real_t s_QQQ[9][MQ1 * MQ1 * MQ1];

         kernels::internal::LoadX<MD1>(e, D1D, X, s_DDD);
         kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, s_BG);

         kernels::internal::GradX<MD1, MQ1>(D1D, Q1D, s_BG, s_DDD, s_DDQ);
         kernels::internal::GradY<MD1, MQ1>(D1D, Q1D, s_BG, s_DDQ, s_DQQ);
         kernels::internal::GradZ<MD1, MQ1>(D1D, Q1D, s_BG, s_DQQ, s_QQQ);

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
                  kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, s_QQQ, Jpr);

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
