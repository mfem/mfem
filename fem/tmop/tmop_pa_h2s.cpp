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
void TMOP_SetupGradPA_2D(const DeviceTensor<4,const double> &X,
                         const double metric_normal,
                         const double *w,
                         const int mid,
                         const int NE,
                         const ConstDeviceMatrix &W,
                         const ConstDeviceMatrix &B,
                         const ConstDeviceMatrix &G,
                         const DeviceTensor<5, const double> &J,
                         DeviceTensor<7> &H,
                         const int d1d,
                         const int q1d,
                         const int max)
{
   using Args = kernels::InvariantsEvaluator2D::Buffers;
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 ||
               mid == 56 || mid == 77 || mid==80 || mid==94,
               "Metric not yet implemented!");

   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int DIM = 2;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double s_QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,s_X);
      kernels::internal::LoadBG<MD1,MQ1>(D1D, Q1D, B, G, s_BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_X, s_DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_DQ, s_QQ);

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

            // Jpr = X^t.DSh
            double Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,s_QQ,Jpr);

            // Jpt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->AssembleH
            if (mid == 1)
            {
               // weight * ddI1
               double ddI1[4];
               kernels::InvariantsEvaluator2D ie(Args().J(Jpt).ddI1(ddI1));
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           const double h = ddi1(r,c);
                           H(r,c,i,j,qx,qy,e) = weight * h;
                        }
                     }
                  }
               }
            }

            if (mid == 2)
            {
               // 0.5 * weight * dI1b
               double ddI1[4], ddI1b[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b));
               const double w = 0.5 * weight;
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           const double h = ddi1b(r,c);
                           H(r,c,i,j,qx,qy,e) = w * h;
                        }
                     }
                  }
               }
            }

            if (mid == 7)
            {
               double ddI1[4], ddI2[4], dI1[4], dI2[4], dI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).ddI1(ddI1).ddI2(ddI2)
                .dI1(dI1).dI2(dI2).dI2b(dI2b));
               const double c1 = 1./ie.Get_I2();
               const double c2 = weight*c1*c1;
               const double c3 = ie.Get_I1()*c2;
               ConstDeviceMatrix di1(ie.Get_dI1(),DIM,DIM);
               ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);

               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
                     ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           H(r,c,i,j,qx,qy,e) =
                              weight * (1.0 + c1) * ddi1(r,c)
                              - c3 * ddi2(r,c)
                              - c2 * (di1(i,j) * di2(r,c) + di2(i,j) * di1(r,c))
                              + 2.0 * c1 * c3 * di2(r,c) * di2(i,j);
                        }
                     }
                  }
               }
            }

            if (mid == 56)
            {
               // (0.5 - 0.5/I2b^2)*ddI2b + (1/I2b^3)*(dI2b x dI2b)
               double dI2b[4], ddI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI2b(dI2b).ddI2b(ddI2b));
               const double I2b = ie.Get_I2b();
               ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           H(r,c,i,j,qx,qy,e) =
                              weight * (0.5 - 0.5/(I2b*I2b)) * ddi2b(r,c) +
                              weight / (I2b*I2b*I2b) * di2b(r,c) * di2b(i,j);
                        }
                     }
                  }
               }
            }

            if (mid == 77)
            {
               double dI2[4], dI2b[4], ddI2[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI2(dI2).dI2b(dI2b).ddI2(ddI2));
               const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
               ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           H(r,c,i,j,qx,qy,e) =
                              weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r,c) +
                              weight * (I2inv_sq / I2) * di2(r,c) * di2(i,j);
                        }
                     }
                  }
               }
            }

            if (mid == 80)
            {
               // w0 H_2 + w1 H_77
               double ddI1[4], ddI1b[4], dI2[4], dI2b[4], ddI2[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).dI2(dI2).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2(ddI2));

               const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
               ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
                     ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           H(r,c,i,j,qx,qy,e) =
                              w[0] * 0.5 * weight * ddi1b(r,c) +
                              w[1] * (weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r,c) +
                                      weight * (I2inv_sq / I2) * di2(r,c) * di2(i,j));
                        }
                     }
                  }
               }
            }

            if (mid == 94)
            {
               // w0 H_2 + w1 H_56
               double ddI1[4], ddI1b[4], dI2b[4], ddI2b[4];
               kernels::InvariantsEvaluator2D ie
               (Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2b(ddI2b));
               const double I2b = ie.Get_I2b();
               ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
                     ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           H(r,c,i,j,qx,qy,e) =
                              w[0] * 0.5 * weight * ddi1b(r,c) +
                              w[1] * (weight * (0.5 - 0.5/(I2b*I2b)) * ddi2b(r,c) +
                                      weight / (I2b*I2b*I2b) * di2b(r,c) * di2b(i,j) );
                        }
                     }
                  }
               }
            }
         } // qx
      } // qy
   });
}

void TMOP_Integrator::AssembleGradPA_2D(const Vector &x) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, M = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;
   const double mn = metric_normal;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
   const double *w = mp.Read();

   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q);
   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q, NE);
   const auto X = Reshape(x.Read(), d,d, DIM, NE);
   auto H = Reshape(PA.H.Write(), DIM,DIM, DIM,DIM, q,q, NE);

   decltype(&TMOP_SetupGradPA_2D<>) ker = TMOP_SetupGradPA_2D;

   if (d==2 && q==2) { ker = TMOP_SetupGradPA_2D<2,2>; }
   if (d==2 && q==3) { ker = TMOP_SetupGradPA_2D<2,3>; }
   if (d==2 && q==4) { ker = TMOP_SetupGradPA_2D<2,4>; }
   if (d==2 && q==5) { ker = TMOP_SetupGradPA_2D<2,5>; }
   if (d==2 && q==6) { ker = TMOP_SetupGradPA_2D<2,6>; }

   if (d==3 && q==3) { ker = TMOP_SetupGradPA_2D<3,3>; }
   if (d==3 && q==4) { ker = TMOP_SetupGradPA_2D<3,4>; }
   if (d==3 && q==5) { ker = TMOP_SetupGradPA_2D<3,5>; }
   if (d==3 && q==6) { ker = TMOP_SetupGradPA_2D<3,6>; }

   if (d==4 && q==4) { ker = TMOP_SetupGradPA_2D<4,4>; }
   if (d==4 && q==5) { ker = TMOP_SetupGradPA_2D<4,5>; }
   if (d==4 && q==6) { ker = TMOP_SetupGradPA_2D<4,6>; }

   if (d==5 && q==5) { ker = TMOP_SetupGradPA_2D<5,5>; }
   if (d==5 && q==6) { ker = TMOP_SetupGradPA_2D<5,6>; }

   ker(X,mn,w,M,NE,W,B,G,J,H,d,q,4);
}

} // namespace mfem
