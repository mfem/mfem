// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultPA_2D(const double metric_normal,
                       const double *w,
                       const int mid,
                       const int NE,
                       const DeviceTensor<5, const double> &J,
                       const ConstDeviceMatrix &W,
                       const ConstDeviceMatrix &B,
                       const ConstDeviceMatrix &G,
                       const DeviceTensor<4,const double> &X,
                       DeviceTensor<4> &Y,
                       const int d1d,
                       const int q1d,
                       const int max)
{
   using Args = kernels::InvariantsEvaluator2D::Buffers;
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 ||
               mid == 56 || mid == 77 || mid == 80 || mid == 94,
               "2D metric not yet implemented!");

   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = metric_normal * W(qx,qy) * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X{^T}.DSh
            double Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jpr);

            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->EvalP(Jpt, P);
            double P[4];

            if (mid == 1)
            {
               double dI1[4];
               kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
               kernels::Set(2,2, 1.0, ie.Get_dI1(), P);
            }

            if (mid == 2)
            {
               double dI1b[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
               kernels::Set(2,2, 1./2., ie.Get_dI1b(), P);
            }

            if (mid == 7)
            {
               double dI1[4], dI2[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI1(dI1).dI2(dI2).dI2b(dI2b));
               const double I2 = ie.Get_I2();
               kernels::Add(2,2, 1.0 + 1.0 / I2, ie.Get_dI1(),
                            -ie.Get_I1() / (I2*I2), ie.Get_dI2(), P);
            }

            if (mid == 56)
            {
               // 0.5*(1 - 1/I2b^2)*dI2b
               double dI2b[4];
               kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
               const double I2b = ie.Get_I2b();
               kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
            }

            if (mid == 77)
            {
               double dI2[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI2(dI2).dI2b(dI2b));
               const double I2 = ie.Get_I2();
               kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
            }

            if (mid == 80)
            {
               // w0 P_2 + w1 P_77
               double dI1b[4], dI2[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI1b(dI1b).dI2(dI2).dI2b(dI2b));
               kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);
               const double I2 = ie.Get_I2();
               kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
            }

            if (mid == 94)
            {
               // w0 P_2 + w1 P_56
               double dI1b[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
               kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);
               const double I2b = ie.Get_I2b();
               kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
            }

            for (int i = 0; i < 4; i++) { P[i] *= weight; }

            // PMatO += DS . P^t += DSh . (Jrt . P^t)
            double A[4];
            kernels::MultABt(2,2,2, Jrt, P, A);
            kernels::internal::PushGrad<MQ1,NBZ>(Q1D,qx,qy,A,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,B,G,BG);
      kernels::internal::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::internal::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_2D(const Vector &x, Vector &y) const
{
   constexpr int DIM = 2;
   const double mn = metric_normal;
   const int NE = PA.ne, M = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
   const double *w = mp.Read();

   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q);
   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   auto X = Reshape(x.Read(), d,d, DIM, NE);
   auto Y = Reshape(y.ReadWrite(), d,d, DIM, NE);

   decltype(&TMOP_AddMultPA_2D<>) ker = TMOP_AddMultPA_2D;

   if (d==2 && q==2) { ker = TMOP_AddMultPA_2D<2,2>; }
   if (d==2 && q==3) { ker = TMOP_AddMultPA_2D<2,3>; }
   if (d==2 && q==4) { ker = TMOP_AddMultPA_2D<2,4>; }
   if (d==2 && q==5) { ker = TMOP_AddMultPA_2D<2,5>; }
   if (d==2 && q==6) { ker = TMOP_AddMultPA_2D<2,6>; }

   if (d==3 && q==3) { ker = TMOP_AddMultPA_2D<3,3>; }
   if (d==3 && q==4) { ker = TMOP_AddMultPA_2D<3,4>; }
   if (d==3 && q==5) { ker = TMOP_AddMultPA_2D<3,5>; }
   if (d==3 && q==6) { ker = TMOP_AddMultPA_2D<3,6>; }

   if (d==4 && q==4) { ker = TMOP_AddMultPA_2D<4,4>; }
   if (d==4 && q==5) { ker = TMOP_AddMultPA_2D<4,5>; }
   if (d==4 && q==6) { ker = TMOP_AddMultPA_2D<4,6>; }

   if (d==5 && q==5) { ker = TMOP_AddMultPA_2D<5,5>; }
   if (d==5 && q==6) { ker = TMOP_AddMultPA_2D<5,6>; }

   ker(mn,w,M,NE,J,W,B,G,X,Y,d,q,4);
}

} // namespace mfem
