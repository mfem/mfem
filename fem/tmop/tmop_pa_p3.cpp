// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultPA_3D(const double metric_normal,
                       const double gamma,
                       const int mid,
                       const int NE,
                       const DeviceTensor<6, const double> &J,
                       const ConstDeviceCube &W,
                       const ConstDeviceMatrix &B,
                       const ConstDeviceMatrix &G,
                       const DeviceTensor<5,const double> &X,
                       DeviceTensor<5> &Y,
                       const int d1d,
                       const int q1d,
                       const int max)
{
   using Args = kernels::InvariantsEvaluator3D::Buffers;
   MFEM_VERIFY(mid == 302 || mid == 303 || mid == 315 ||
               mid == 321 || mid == 332, "3D metric not yet implemented!");

   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
      MFEM_SHARED double s_DDQ[9][MD1*MD1*MQ1];
      MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED double s_QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,s_DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,s_BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,s_BG,s_DDD,s_DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,s_DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double *Jtr = &J(0,0,qx,qy,qz,e);
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight = metric_normal * W(qx,qy,qz) * detJtr;

               // Jrt = Jtr^{-1}
               double Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               double Jpr[9];
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz,s_QQQ,Jpr);

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               double Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // metric->EvalP(Jpt, P);
               double P[9];
               if (mid == 302)
               {
                  // (I1b/9)*dI2b + (I2b/9)*dI1b
                  double B[9];
                  double dI1b[9], dI2[9], dI2b[9], dI3b[9];
                  kernels::InvariantsEvaluator3D ie
                  (Args().J(Jpt).B(B).dI1b(dI1b)
                   .dI2(dI2).dI2b(dI2b) .dI3b(dI3b));
                  const double alpha = ie.Get_I1b()/9.;
                  const double beta = ie.Get_I2b()/9.;
                  kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
               }

               if (mid == 303)
               {
                  // dI1b/3
                  double B[9];
                  double dI1b[9], dI3b[9];
                  kernels::InvariantsEvaluator3D ie
                  (Args().J(Jpt).B(B).dI1b(dI1b).dI3b(dI3b));
                  kernels::Set(3,3, 1./3., ie.Get_dI1b(), P);
               }

               if (mid == 315)
               {
                  // 2*(I3b - 1)*dI3b
                  double dI3b[9];
                  kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));
                  double sign_detJ;
                  const double I3b = ie.Get_I3b(sign_detJ);
                  kernels::Set(3,3, 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
               }

               if (mid == 321)
               {
                  // dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
                  double B[9];
                  double dI1[9], dI2[9], dI3b[9];
                  kernels::InvariantsEvaluator3D ie
                  (Args().J(Jpt).B(B).dI1(dI1).dI2(dI2).dI3b(dI3b));
                  double sign_detJ;
                  const double I3 = ie.Get_I3();
                  const double alpha = 1.0/I3;
                  const double beta = -2.*ie.Get_I2()/(I3*ie.Get_I3b(sign_detJ));
                  kernels::Add(3,3, alpha, ie.Get_dI2(), beta, ie.Get_dI3b(sign_detJ), P);
                  kernels::Add(3,3, ie.Get_dI1(), P);
               }

               if (mid == 332)
               {
                  // (1-gamma) P_302 + gamma P_315.
                  double B[9];
                  double dI1b[9], dI2[9], dI2b[9], dI3b[9];
                  kernels::InvariantsEvaluator3D ie
                  (Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
                  const double alpha = (1.0 - gamma) * ie.Get_I1b()/9.;
                  const double beta = (1.0 - gamma) * ie.Get_I2b()/9.;
                  kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
                  double sign_detJ;
                  const double I3b = ie.Get_I3b(sign_detJ);
                  kernels::Add(3,3, gamma*2.0*(I3b-1.0), ie.Get_dI3b(sign_detJ), P);
               }

               for (int i = 0; i < 9; i++) { P[i] *= weight; }

               // Y += DS . P^t += DSh . (Jrt . P^t)
               double A[9];
               kernels::MultABt(3,3,3, Jrt, P, A);
               kernels::internal::PushGrad<MQ1>(Q1D,qx,qy,qz,A,s_QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,B,G,s_BG);
      kernels::internal::GradZt<MD1,MQ1>(D1D,Q1D,s_BG,s_QQQ,s_DQQ);
      kernels::internal::GradYt<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_DDQ);
      kernels::internal::GradXt<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_3D(const Vector &x, Vector &y) const
{
   const int NE = PA.ne;
   constexpr int DIM = 3;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;

   const double mn = metric_normal;

   double mp = 0.0;
   if (auto m = dynamic_cast<TMOP_Metric_332 *>(metric)) { mp = m->GetGamma(); }

   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), Q1D, Q1D, Q1D);
   const auto B = Reshape(PA.maps->B.Read(), Q1D, D1D);
   const auto G = Reshape(PA.maps->G.Read(), Q1D, D1D);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, DIM, NE);

#ifndef MFEM_USE_JIT
   decltype(&TMOP_AddMultPA_3D<>) ker = TMOP_AddMultPA_3D<>;

   const int d=D1D, q=Q1D;
   if (d == 2 && q==2) { ker = TMOP_AddMultPA_3D<2,2>; }
   if (d == 2 && q==3) { ker = TMOP_AddMultPA_3D<2,3>; }
   if (d == 2 && q==4) { ker = TMOP_AddMultPA_3D<2,4>; }
   if (d == 2 && q==5) { ker = TMOP_AddMultPA_3D<2,5>; }
   if (d == 2 && q==6) { ker = TMOP_AddMultPA_3D<2,6>; }

   if (d == 3 && q==3) { ker = TMOP_AddMultPA_3D<3,3>; }
   if (d == 3 && q==4) { ker = TMOP_AddMultPA_3D<4,4>; }
   if (d == 3 && q==5) { ker = TMOP_AddMultPA_3D<5,5>; }
   if (d == 3 && q==6) { ker = TMOP_AddMultPA_3D<6,6>; }

   if (d == 4 && q==4) { ker = TMOP_AddMultPA_3D<4,4>; }
   if (d == 4 && q==5) { ker = TMOP_AddMultPA_3D<4,5>; }
   if (d == 4 && q==6) { ker = TMOP_AddMultPA_3D<4,6>; }

   if (d == 5 && q==5) { ker = TMOP_AddMultPA_3D<5,5>; }
   if (d == 5 && q==6) { ker = TMOP_AddMultPA_3D<5,6>; }

   ker(mn,mp,M,NE,J,W,B,G,X,Y,D1D,Q1D,4);
#else
   TMOP_AddMultPA_3D(mn,mp,M,NE,J,W,B,G,X,Y,D1D,Q1D,4);
#endif
}

} // namespace mfem
