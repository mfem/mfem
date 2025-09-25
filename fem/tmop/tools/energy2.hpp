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

class TMOPEnergyPA2D
{
   const Vector &X;
   Vector &E, &L;
   const Vector &O;
   const bool use_detA;

   real_t metric_energy, limiting_energy;

   int ndof, nqpt;
   real_t metric_normal;
   int ne;
   TMOP_QualityMetric *metric;
   const Array<real_t> &B, &G;
   const DenseTensor &J;
   const IntegrationRule &ir;
   const Vector &metric_coeff;

public:
   TMOPEnergyPA2D(const TMOP_Integrator *ti,
                  const Vector &x,
                  Vector &l,
                  const bool detA):
      X(x),
      E(ti->PA.E),
      L(l),
      O(ti->PA.O),
      use_detA(detA),
      metric_energy{},
      limiting_energy{},
      ndof(ti->PA.maps->ndof),
      nqpt(ti->PA.maps->nqpt),
      metric_normal(ti->metric_normal),
      ne(ti->PA.ne),
      metric(ti->metric),
      B(ti->PA.maps->B),
      G(ti->PA.maps->G),
      J(ti->PA.Jtr),
      ir(*ti->PA.ir),
      metric_coeff(ti->PA.MC)
   { }

   TMOPEnergyPA2D(const TMOP_Integrator *ti,
                  const Vector &x,
                  Vector &l,
                  const real_t normal,
                  const Vector &coeff,
                  const bool detA):
      X(x),
      E(ti->PA.E),
      L(l),
      O(ti->PA.O),
      use_detA(detA),
      metric_energy{},
      limiting_energy{},
      ndof(ti->PA.maps->ndof),
      nqpt(ti->PA.maps->nqpt),
      metric_normal(normal),
      ne(ti->PA.ne),
      metric(ti->metric),
      B(ti->PA.maps->B),
      G(ti->PA.maps->G),
      J(ti->PA.Jtr),
      ir(*ti->PA.ir),
      metric_coeff(coeff)
   { }

   TMOPEnergyPA2D(const Vector &x,
                  Vector &e,
                  Vector &l,
                  const Vector &o,
                  const bool detA,
                  const int d1d,
                  const int q1d,
                  const real_t normal,
                  const int NE,
                  TMOP_QualityMetric *quality_metric,
                  const Array<real_t> &b,
                  const Array<real_t> &g,
                  const DenseTensor &J,
                  const IntegrationRule &integration_rule,
                  const Vector &mc):
      X(x),
      E(e),
      L(l),
      O(o),
      use_detA(detA),
      metric_energy{},
      limiting_energy{},
      ndof(d1d),
      nqpt(q1d),
      metric_normal(normal),
      ne(NE),
      metric(quality_metric),
      B(b),
      G(g),
      J(J),
      ir(integration_rule),
      metric_coeff(mc)
   { }

   int Ndof() const { return ndof; }
   int Nqpt() const { return nqpt; }

   template <int MD1, int MQ1, typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPEnergyPA2D &ker)
   {
      const real_t metric_normal = ker.metric_normal;
      const int NE = ker.ne, d1d = ker.ndof, q1d = ker.nqpt;
      const int D1D = T_D1D ? T_D1D : d1d, Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_VERIFY(T_D1D > 0 || D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(T_Q1D > 0 || Q1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");

      Array<real_t> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ker.metric))
      {
         m->GetWeights(mp);
      }

      const auto *w = mp.Read();
      const auto *b = ker.B.Read();
      const auto *g = ker.G.Read();

      const auto X = Reshape(ker.X.Read(), D1D, D1D, 2, NE);
      const auto J = Reshape(ker.J.Read(), 2, 2, Q1D, Q1D, NE);
      const auto W = Reshape(ker.ir.GetWeights().Read(), Q1D, Q1D);

      const Vector &mc = ker.metric_coeff;
      const bool const_m0 = mc.Size() == 1;
      const auto MC = const_m0
                      ? Reshape(mc.Read(), 1, 1, 1)
                      : Reshape(mc.Read(), Q1D, Q1D, NE);

      const bool use_detA = ker.use_detA;
      auto E = Reshape(ker.E.Write(), Q1D, Q1D, NE);
      auto L = Reshape(ker.L.Write(), Q1D, Q1D, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         kernels::internal::vd_regs2d_t<2, 2, MQ1> r0, r1;

         kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
         kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

         kernels::internal::LoadDofs2d(e, D1D, X, r0);
         kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);

         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, e);
               const real_t coeff = const_m0 ? MC(0, 0, 0) : MC(qx, qy, e);

               // Jrt = Jtr^{-1}
               real_t Jrt[4];
               kernels::CalcInverse<2>(Jtr, Jrt);

               // Jpr = X^t.DSh
               const real_t Jpr[4] =
               {
                  r1[0][0][qy][qx], r1[1][0][qy][qx],
                  r1[0][1][qy][qx], r1[1][1][qy][qx]
               };

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               real_t Jpt[4];
               kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

               const real_t det = kernels::Det<2>(use_detA ? Jpr : Jtr);
               const real_t weight = metric_normal * coeff * W(qx,qy) * det;

               const real_t EvalW = METRIC{}.EvalW(Jpt, w);

               E(qx, qy, e) = weight * EvalW;
               L(qx, qy, e) = weight;
            }
         }
      });
      ker.metric_energy = ker.E * ker.O;
      ker.limiting_energy = ker.L * ker.O;
   }

   void GetEnergy(real_t &met_energy, real_t &lim_energy) const
   {
      met_energy = this->metric_energy;
      lim_energy = this->limiting_energy;
   }
};

} // namespace mfem
