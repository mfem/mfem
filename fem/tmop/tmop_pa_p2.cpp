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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

using Args = kernels::InvariantsEvaluator2D::Buffers;

static MFEM_HOST_DEVICE inline
void EvalP_001(const real_t *Jpt, real_t *P)
{
   real_t dI1[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
   kernels::Set(2,2, 1.0, ie.Get_dI1(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_002(const real_t *Jpt, real_t *P)
{
   real_t dI1b[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
   kernels::Set(2,2, 0.5, ie.Get_dI1b(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_007(const real_t *Jpt, real_t *P)
{
   real_t dI1[4], dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1)
                                     .dI2(dI2).dI2b(dI2b));
   const real_t I2 = ie.Get_I2();
   kernels::Add(2,2, 1.0 + 1.0 / I2, ie.Get_dI1(),
                -ie.Get_I1() / (I2*I2), ie.Get_dI2(), P);
}

// P_56 = 0.5*(1 - 1/I2b^2)*dI2b.
static MFEM_HOST_DEVICE inline
void EvalP_056(const real_t *Jpt, real_t *P)
{
   real_t dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
   const real_t I2b = ie.Get_I2b();
   kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_077(const real_t *Jpt, real_t *P)
{
   real_t dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().
                                     J(Jpt).
                                     dI2(dI2).dI2b(dI2b));
   const real_t I2 = ie.Get_I2();
   kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
}

// P_80 = w0 P_2 + w1 P_77.
static MFEM_HOST_DEVICE inline
void EvalP_080(const real_t *Jpt, const real_t *w, real_t *P)
{
   real_t dI1b[4], dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).
                                     dI1b(dI1b).dI2(dI2).dI2b(dI2b));

   kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);

   const real_t I2 = ie.Get_I2();
   kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
}

// P_94 = w0 P_2 + w1 P_56.
static MFEM_HOST_DEVICE inline
void EvalP_094(const real_t *Jpt, const real_t *w, real_t *P)
{
   real_t dI1b[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).
                                     dI1b(dI1b).dI2b(dI2b));

   kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);

   const real_t I2b = ie.Get_I2b();
   kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
}

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_2D,
                           const real_t metric_normal,
                           const Vector &mc_,
                           const Array<real_t> &metric_param,
                           const int mid,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<real_t> &w_,
                           const Array<real_t> &b_,
                           const Array<real_t> &g_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 77
               || mid == 80 || mid == 94,
               "2D metric not yet implemented!");

   const bool const_m0 = mc_.Size() == 1;

   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto MC = const_m0 ?
                   Reshape(mc_.Read(), 1, 1, 1) :
                   Reshape(mc_.Read(), Q1D, Q1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   const real_t *metric_data = metric_param.Read();

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t XY[2][NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED real_t QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const real_t *Jtr = &J(0,0,qx,qy,e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t m_coef = const_m0 ? MC(0,0,0) : MC(qx,qy,e);
            const real_t weight = metric_normal * m_coef *
                                  W(qx,qy) * detJtr;

            // Jrt = Jtr^{-1}
            real_t Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X{^T}.DSh
            real_t Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jpr);

            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = Jpr.Jrt
            real_t Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->EvalP(Jpt, P);
            real_t P[4];
            if (mid ==  1) { EvalP_001(Jpt, P); }
            if (mid ==  2) { EvalP_002(Jpt, P); }
            if (mid ==  7) { EvalP_007(Jpt, P); }
            if (mid == 56) { EvalP_056(Jpt, P); }
            if (mid == 77) { EvalP_077(Jpt, P); }
            if (mid == 80) { EvalP_080(Jpt, metric_data, P); }
            if (mid == 94) { EvalP_094(Jpt, metric_data, P); }
            for (int i = 0; i < 4; i++) { P[i] *= weight; }

            // PMatO += DS . P^t += DSh . (Jrt . P^t)
            real_t A[4];
            kernels::MultABt(2,2,2, Jrt, P, A);
            kernels::internal::PushGrad<MQ1,NBZ>(Q1D,qx,qy,A,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::internal::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::internal::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_2D(const Vector &X, Vector &Y) const
{
   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W = PA.ir->GetWeights();
   const Array<real_t> &B = PA.maps->B;
   const Array<real_t> &G = PA.maps->G;
   const real_t mn = metric_normal;
   const Vector &MC = PA.MC;

   Array<real_t> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_2D,id,mn,MC,mp,M,N,J,W,B,G,X,Y);
}

} // namespace mfem
