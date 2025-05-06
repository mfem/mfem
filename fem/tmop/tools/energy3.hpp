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
   const Vector &x;
   Vector &E, &L;
   const Vector &O;
   const bool use_detA;

   real_t metric_energy, limiting_energy;

   int ndof, nqpt;
   real_t metric_normal;
   int ne;
   TMOP_QualityMetric *metric;
   const Array<real_t> &B, &G;
   const DenseTensor &Jtr;
   const IntegrationRule &ir;
   const Vector &metric_coeff;

public:
   TMOPEnergyPA3D(const TMOP_Integrator *ti,
                  const Vector &X,
                  Vector &L,
                  const bool use_detA):
      x(X),
      E(ti->PA.E),
      L(L),
      O(ti->PA.O),
      use_detA(use_detA),
      metric_energy{},
      limiting_energy{},
      ndof(ti->PA.maps->ndof),
      nqpt(ti->PA.maps->nqpt),
      metric_normal(ti->metric_normal),
      ne(ti->PA.ne),
      metric(ti->metric),
      B(ti->PA.maps->B),
      G(ti->PA.maps->G),
      Jtr(ti->PA.Jtr),
      ir(*ti->PA.ir),
      metric_coeff(ti->PA.MC)
   { }

   TMOPEnergyPA3D(const TMOP_Integrator *ti,
                  const Vector &X,
                  Vector &L,
                  const real_t metric_normal,
                  const Vector &metric_coeff,
                  const bool use_detA):
      x(X),
      E(ti->PA.E),
      L(L),
      O(ti->PA.O),
      use_detA(use_detA),
      metric_energy{},
      limiting_energy{},
      ndof(ti->PA.maps->ndof),
      nqpt(ti->PA.maps->nqpt),
      metric_normal(metric_normal),
      ne(ti->PA.ne),
      metric(ti->metric),
      B(ti->PA.maps->B),
      G(ti->PA.maps->G),
      Jtr(ti->PA.Jtr),
      ir(*ti->PA.ir),
      metric_coeff(metric_coeff)
   { }

   TMOPEnergyPA3D(const Vector &X,
                  Vector &E,
                  Vector &L,
                  const Vector &O,
                  const bool use_detA,
                  const int ndof,
                  const int nqpt,
                  const real_t metric_normal,
                  const int ne,
                  TMOP_QualityMetric *metric,
                  const Array<real_t> &B,
                  const Array<real_t> &G,
                  const DenseTensor &Jtr,
                  const IntegrationRule &ir,
                  const Vector &mc):
      x(X),
      E(E),
      L(L),
      O(O),
      use_detA(use_detA),
      metric_energy{},
      limiting_energy{},
      ndof(ndof),
      nqpt(nqpt),
      metric_normal(metric_normal),
      ne(ne),
      metric(metric),
      B(B),
      G(G),
      Jtr(Jtr),
      ir(ir),
      metric_coeff(mc)
   { }

   int Ndof() const { return ndof; }
   int Nqpt() const { return nqpt; }

   template <int MD1, int MQ1, typename METRIC, int T_D1D = 0, int T_Q1D = 0>
   static void Mult(TMOPEnergyPA3D &ker)
   {
      const real_t metric_normal = ker.metric_normal;
      const int NE = ker.ne, d1d = ker.ndof, q1d = ker.nqpt;
      const int D1D = T_D1D ? T_D1D : d1d, Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_VERIFY(T_D1D > 0 || D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
      MFEM_VERIFY(T_Q1D > 0 || Q1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");

      Array<real_t> mp;
      if (auto metric = dynamic_cast<TMOP_Combo_QualityMetric *>(ker.metric))
      {
         metric->GetWeights(mp);
      }

      const auto *w = mp.Read();
      const auto *b = ker.B.Read();
      const auto *g = ker.G.Read();

      const auto X = Reshape(ker.x.Read(), D1D, D1D, D1D, 3, NE);
      const auto J = Reshape(ker.Jtr.Read(), 3, 3, Q1D, Q1D, Q1D, NE);
      const auto W = Reshape(ker.ir.GetWeights().Read(), Q1D, Q1D, Q1D);

      const Vector &mc = ker.metric_coeff;
      const bool const_m0 = mc.Size() == 1;
      const auto MC = const_m0
                      ? Reshape(mc.Read(), 1, 1, 1, 1)
                      : Reshape(mc.Read(), Q1D, Q1D, Q1D, NE);

      const bool use_detA = ker.use_detA;
      auto E = Reshape(ker.E.Write(), Q1D, Q1D, Q1D, NE);
      auto L = Reshape(ker.L.Write(), Q1D, Q1D, Q1D, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         regs5d_t<3, 3, MQ1> r0, r1;

         LoadMatrix(D1D, Q1D, b, sB);
         LoadMatrix(D1D, Q1D, g, sG);

         LoadDofs3d(e, D1D, X, r0);
         Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

         for (int qz = 0; qz < Q1D; ++qz)
         {
            mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
            {
               mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
               {
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t metric_coeff = const_m0
                                              ? MC(0, 0, 0, 0)
                                              : MC(qx, qy, qz, e);

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

                  const real_t det = kernels::Det<3>(use_detA ? Jpr : Jtr);
                  const real_t weight = metric_normal * metric_coeff * W(qx, qy, qz) * det;

                  const real_t EvalW = METRIC{}.EvalW(Jpt, w);

                  E(qx, qy, qz, e) = weight * EvalW;
                  L(qx, qy, qz, e) = weight;
               });
            });
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
