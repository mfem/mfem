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

#include "../pa.hpp"
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

   template <int MD1, int MQ1, typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPAddMultPA2D &ker)
   {
      const mfem::TMOP_Integrator *ti = ker.ti;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d1d = ti->PA.maps->ndof, q1d = ti->PA.maps->nqpt;

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
      const auto *b = ti->PA.maps->B.Read(), *g = ti->PA.maps->G.Read();

      const auto X = Reshape(ker.x.Read(), D1D, D1D, 2, NE);
      const auto J = Reshape(ti->PA.Jtr.Read(), 2, 2, Q1D, Q1D, NE);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), Q1D, Q1D);
      auto Y = Reshape(ker.y.ReadWrite(), D1D, D1D, 2, NE);

      const Vector &mc = ti->PA.MC;
      const bool const_m0 = mc.Size() == 1;
      const auto MC = const_m0
                      ? Reshape(mc.Read(), 1, 1, 1)
                      : Reshape(mc.Read(), Q1D, Q1D, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         kernels::internal::vd_regs2d_t<2, 2, MQ1> r0, r1;

         kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
         kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

         kernels::internal::LoadDofs2d(e, D1D, X, r0);
         kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);

         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, e);
               const real_t detJtr = kernels::Det<2>(Jtr);
               const real_t m_coef = const_m0 ? MC(0, 0, 0) : MC(qx, qy, e);
               const real_t weight = metric_normal * m_coef * W(qx, qy) * detJtr;

               // Jrt = Jtr^{-1}
               real_t Jrt[4];
               kernels::CalcInverse<2>(Jtr, Jrt);

               // Jpr = X{^T}.DSh
               const real_t Jpr[4] =
               {
                  r1[0][0][qy][qx], r1[1][0][qy][qx],
                  r1[0][1][qy][qx], r1[1][1][qy][qx]
               };

               // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = Jpr.Jrt
               real_t Jpt[4];
               kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

               real_t P[4];
               METRIC{}.EvalP(Jpt, w, P);

               for (int i = 0; i < 4; i++) { P[i] *= weight; }

               // PMatO += DS . P^t += DSh . (Jrt . P^t)
               real_t A[4];
               kernels::MultABt(2, 2, 2, Jrt, P, A);
               r0[0][0][qy][qx] = A[0], r0[0][1][qy][qx] = A[1];
               r0[1][0][qy][qx] = A[2], r0[1][1][qy][qx] = A[3];
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1);
         kernels::internal::WriteDofs2d(e, D1D, r1, Y);
      });
   }
};

} // namespace mfem
