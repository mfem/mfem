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

   template <typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPEnergyPA3D &ker)
   {
      constexpr int DIM = 3, VDIM = 3;

      const mfem::TMOP_Integrator *ti = ker.ti;
      const real_t metric_normal = ti->metric_normal;
      const int NE = ti->PA.ne, d1d = ker.Ndof(), q1d = ker.Nqpt();
      const int D1D = T_D1D ? T_D1D : d1d, Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

      Array<real_t> mp;
      if (auto metric = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         metric->GetWeights(mp);
      }

      const auto *w = mp.Read();
      const auto *b = ti->PA.maps->B.Read(), *g = ti->PA.maps->G.Read();

      const auto X = Reshape(ker.x.Read(), D1D, D1D, D1D, DIM, NE);
      const auto W = Reshape(ti->PA.ir->GetWeights().Read(), Q1D, Q1D, Q1D);
      const auto J = Reshape(ti->PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
      auto E = Reshape(ti->PA.E.Write(), Q1D, Q1D, Q1D, NE);

      const Vector &mc = ti->PA.MC;
      const bool const_m0 = mc.Size() == 1;
      const auto MC = const_m0
                      ? Reshape(mc.Read(), 1, 1, 1, 1)
                      : Reshape(mc.Read(), Q1D, Q1D, Q1D, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
         constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         regs5d_t<VDIM, DIM, MQ1> r0, r1;

         LoadMatrix(D1D, Q1D, b, sB);
         LoadMatrix(D1D, Q1D, g, sG);

         LoadDofs3d(e, D1D, X, r0);
         Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

         for (int qz = 0; qz < Q1D; ++qz)
         {
            foreach_y_thread(Q1D, [&](int qy)
            {
               foreach_x_thread(Q1D, [&](int qx)
               {
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJtr = kernels::Det<3>(Jtr);
                  const real_t m_coef = const_m0
                                        ? MC(0, 0, 0, 0)
                                        : MC(qx, qy, qz, e);
                  const real_t weight = metric_normal * m_coef * W(qx, qy, qz) * detJtr;

                  // Jrt = Jtr^{-1}
                  real_t Jrt[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);

                  // Jpr = X^t.DSh
                  const real_t Jpr[9] =
                  {
                     r1(0, 0, qz, qy, qx), r1(1, 0, qz, qy, qx), r1(2, 0, qz, qy, qx),
                     r1(0, 1, qz, qy, qx), r1(1, 1, qz, qy, qx), r1(2, 1, qz, qy, qx),
                     r1(0, 2, qz, qy, qx), r1(1, 2, qz, qy, qx), r1(2, 2, qz, qy, qx)
                  };

                  // Jpt = X^t.DS = (X^t.DSh).Jrt = Jpr.Jrt
                  real_t Jpt[9];
                  kernels::Mult(3, 3, 3, Jpr, Jrt, Jpt);

                  const real_t EvalW = METRIC{}.EvalW(Jpt, w);

                  E(qx, qy, qz, e) = weight * EvalW;
               });
            });
         }
      });
      ker.energy = ti->PA.E * ti->PA.O;
   }
};

} // namespace mfem
