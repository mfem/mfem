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

// mu_302 = I1b * I2b / 9 - 1
static MFEM_HOST_DEVICE inline
real_t EvalW_302(const real_t *J)
{
   real_t B[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B));
   return ie.Get_I1b()*ie.Get_I2b()/9. - 1.;
}

// mu_303 = I1b/3 - 1
static MFEM_HOST_DEVICE inline
real_t EvalW_303(const real_t *J)
{
   real_t B[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B));
   return ie.Get_I1b()/3. - 1.;
}

// mu_315 = (I3b - 1)^2
static MFEM_HOST_DEVICE inline
real_t EvalW_315(const real_t *J)
{
   real_t B[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B));
   const real_t a = ie.Get_I3b() - 1.0;
   return a*a;
}

// mu_318 = 0.5 * (I3 + 1/I3) - 1.
static MFEM_HOST_DEVICE inline
real_t EvalW_318(const real_t *J)
{
   real_t B[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B));
   const real_t I3 = ie.Get_I3();
   return 0.5*(I3 + 1.0/I3) - 1.0;
}

// mu_321 = I1 + I2/I3 - 6
static MFEM_HOST_DEVICE inline
real_t EvalW_321(const real_t *J)
{
   real_t B[9];
   kernels::InvariantsEvaluator3D ie(Args().J(J).B(B));
   return ie.Get_I1() + ie.Get_I2()/ie.Get_I3() - 6.0;
}

static MFEM_HOST_DEVICE inline
real_t EvalW_332(const real_t *J, const real_t *w)
{
   return w[0] * EvalW_302(J) + w[1] * EvalW_315(J);
}

static MFEM_HOST_DEVICE inline
real_t EvalW_338(const real_t *J, const real_t *w)
{
   return w[0] * EvalW_302(J) + w[1] * EvalW_318(J);
}

MFEM_REGISTER_TMOP_KERNELS(real_t, EnergyPA_3D,
                           const real_t metric_normal,
                           const Vector &mc_,
                           const Array<real_t> &metric_param,
                           const int mid,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<real_t> &w_,
                           const Array<real_t> &b_,
                           const Array<real_t> &g_,
                           const Vector &ones,
                           const Vector &x_,
                           Vector &energy,
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
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);

   auto E = Reshape(energy.Write(), Q1D, Q1D, Q1D, NE);

   const real_t *metric_data = metric_param.Read();

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t DDD[3][MD1*MD1*MD1];
      MFEM_SHARED real_t DDQ[6][MD1*MD1*MQ1];
      MFEM_SHARED real_t DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

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

               // Jpr = X^t.DSh
               real_t Jpr[9];
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz, QQQ, Jpr);

               // Jpt = X^t.DS = (X^t.DSh).Jrt = Jpr.Jrt
               real_t Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // metric->EvalW(Jpt);
               const real_t EvalW =
                  mid == 302 ? EvalW_302(Jpt) :
                  mid == 303 ? EvalW_303(Jpt) :
                  mid == 315 ? EvalW_315(Jpt) :
                  mid == 318 ? EvalW_318(Jpt) :
                  mid == 321 ? EvalW_321(Jpt) :
                  mid == 332 ? EvalW_332(Jpt, metric_data) :
                  mid == 338 ? EvalW_338(Jpt, metric_data) : 0.0;

               E(qx,qy,qz,e) = weight * EvalW;
            }
         }
      }
   });
   return energy * ones;
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_3D(const Vector &X) const
{
   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t mn = metric_normal;
   const Vector &MC = PA.MC;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W = PA.ir->GetWeights();
   const Array<real_t> &B = PA.maps->B;
   const Array<real_t> &G = PA.maps->G;
   const Vector &O = PA.O;
   Vector &E = PA.E;

   Array<real_t> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_3D,id,mn,MC,mp,M,N,J,W,B,G,O,X,E);
}

} // namespace mfem
