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
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

using Args = kernels::InvariantsEvaluator3D::Buffers;

// P_302 = (I1b/9)*dI2b + (I2b/9)*dI1b
static MFEM_HOST_DEVICE inline
void EvalP_302(const real_t *J, real_t *P)
{
   real_t B[9];
   real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b)
                                     .dI2(dI2).dI2b(dI2b)
                                     .dI3b(dI3b));
   const real_t alpha = ie.Get_I1b()/9.;
   const real_t beta = ie.Get_I2b()/9.;
   kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
}

// P_303 = dI1b/3
static MFEM_HOST_DEVICE inline
void EvalP_303(const real_t *J, real_t *P)
{
   real_t B[9];
   real_t dI1b[9], dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B).dI1b(dI1b).dI3b(dI3b));
   kernels::Set(3,3, 1./3., ie.Get_dI1b(), P);
}

// P_315 = 2*(I3b - 1)*dI3b
static MFEM_HOST_DEVICE inline
void EvalP_315(const real_t *J, real_t *P)
{
   real_t dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).dI3b(dI3b));

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
   kernels::Set(3,3, 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
}

// P_318 = (I3b - 1/I3b^3)*dI3b.
// Uses the I3b form, as dI3 and ddI3 were not implemented at the time.
static MFEM_HOST_DEVICE inline
void EvalP_318(const real_t *J, real_t *P)
{
   real_t dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).dI3b(dI3b));

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
   kernels::Set(3,3, I3b - 1.0/(I3b * I3b * I3b), ie.Get_dI3b(sign_detJ), P);
}

// P_321 = dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
static MFEM_HOST_DEVICE inline
void EvalP_321(const real_t *J, real_t *P)
{
   real_t B[9];
   real_t dI1[9], dI2[9], dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B)
                                     .dI1(dI1).dI2(dI2).dI3b(dI3b));
   real_t sign_detJ;
   const real_t I3 = ie.Get_I3();
   const real_t alpha = 1.0/I3;
   const real_t beta = -2.*ie.Get_I2()/(I3*ie.Get_I3b(sign_detJ));
   kernels::Add(3,3, alpha, ie.Get_dI2(), beta, ie.Get_dI3b(sign_detJ), P);
   kernels::Add(3,3, ie.Get_dI1(), P);
}

// P_332 = w0 P_302 + w1 P_315.
static MFEM_HOST_DEVICE inline
void EvalP_332(const real_t *J, const real_t *w, real_t *P)
{
   real_t B[9];
   real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b)
                                     .dI2(dI2).dI2b(dI2b)
                                     .dI3b(dI3b));
   const real_t alpha = w[0] * ie.Get_I1b()/9.;
   const real_t beta = w[0]* ie.Get_I2b()/9.;
   kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
   kernels::Add(3,3, w[1] * 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
}

// P_338 = w0 P_302 + w1 P_318.
static MFEM_HOST_DEVICE inline
void EvalP_338(const real_t *J, const real_t *w, real_t *P)
{
   real_t B[9];
   real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b)
                                     .dI2(dI2).dI2b(dI2b)
                                     .dI3b(dI3b));
   const real_t alpha = w[0] * ie.Get_I1b()/9.;
   const real_t beta = w[0]* ie.Get_I2b()/9.;
   kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
   kernels::Add(3,3, w[1] * (I3b - 1.0/(I3b * I3b * I3b)),
                ie.Get_dI3b(sign_detJ), P);
}

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_3D,
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
   MFEM_VERIFY(mid == 302 || mid == 303 || mid == 315 || mid == 318 ||
               mid == 321 || mid == 332 || mid == 338,
               "3D metric not yet implemented!");

   const bool const_m0 = mc_.Size() == 1;

   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto MC = const_m0 ?
                   Reshape(mc_.Read(), 1, 1, 1, 1) :
                   Reshape(mc_.Read(), Q1D, Q1D, Q1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   const real_t *metric_data = metric_param.Read();

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t s_BG[2][MQ1*MD1];
      MFEM_SHARED real_t s_DDD[3][MD1*MD1*MD1];
      MFEM_SHARED real_t s_DDQ[9][MD1*MD1*MQ1];
      MFEM_SHARED real_t s_DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED real_t s_QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,s_DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,s_BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,s_BG,s_DDD,s_DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,s_DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const real_t *Jtr = &J(0,0,qx,qy,qz,e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t m_coef = const_m0 ? MC(0,0,0,0) : MC(qx,qy,qz,e);
               const real_t weight = metric_normal * m_coef *
                                     W(qx,qy,qz) * detJtr;

               // Jrt = Jtr^{-1}
               real_t Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               real_t Jpr[9];
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz,s_QQQ,Jpr);

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               real_t Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // metric->EvalP(Jpt, P);
               real_t P[9];
               if (mid == 302) { EvalP_302(Jpt, P); }
               if (mid == 303) { EvalP_303(Jpt, P); }
               if (mid == 315) { EvalP_315(Jpt, P); }
               if (mid == 318) { EvalP_318(Jpt, P); }
               if (mid == 321) { EvalP_321(Jpt, P); }
               if (mid == 332) { EvalP_332(Jpt, metric_data, P); }
               if (mid == 338) { EvalP_338(Jpt, metric_data, P); }
               for (int i = 0; i < 9; i++) { P[i] *= weight; }

               // Y += DS . P^t += DSh . (Jrt . P^t)
               real_t A[9];
               kernels::MultABt(3,3,3, Jrt, P, A);
               kernels::internal::PushGrad<MQ1>(Q1D,qx,qy,qz,A,s_QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,s_BG);
      kernels::internal::GradZt<MD1,MQ1>(D1D,Q1D,s_BG,s_QQQ,s_DQQ);
      kernels::internal::GradYt<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_DDQ);
      kernels::internal::GradXt<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_3D(const Vector &X, Vector &Y) const
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

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_3D,id,mn,MC,mp,M,N,J,W,B,G,X,Y);
}

} // namespace mfem
