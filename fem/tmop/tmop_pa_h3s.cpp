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
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_SetupGradPA_3D(const double metric_normal,
                         const double metric_param,
                         const int mid,
                         const int NE,
                         const ConstDeviceCube &W,
                         const ConstDeviceMatrix &B,
                         const ConstDeviceMatrix &G,
                         const DeviceTensor<6, const double> &J,
                         const DeviceTensor<5,const double> &X,
                         DeviceTensor<8> &H,
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
      constexpr int DIM = 3;
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
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz, s_QQQ, Jpr);

               // Jpt = X^T . DS = (X^T.DSh) . Jrt = Jpr . Jrt
               double Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // InvariantsEvaluator3D buffers used for the metrics
               double B[9];
               double         dI1b[9], ddI1[9], ddI1b[9];
               double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
               // reuse local arrays, to help register allocation
               double        *dI3b=Jrt,        *ddI3b=Jpr;

               // metric->AssembleH
               if (mid == 302)
               {
                  // (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
                  kernels::InvariantsEvaluator3D ie
                  (Args()
                   .J(Jpt).B(B)
                   .dI1b(dI1b).ddI1b(ddI1b)
                   .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                   .dI3b(dI3b));
                  const double c1 = weight/9.;
                  const double I1b = ie.Get_I1b();
                  const double I2b = ie.Get_I2b();
                  ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
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
                              const double dp =
                              (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                              + ddi2b(r,c)*I1b
                              + ddi1b(r,c)*I2b;
                              H(r,c,i,j,qx,qy,qz,e) = c1 * dp;
                           }
                        }
                     }
                  }
               }

               if (mid == 303)
               {
                  // ddI1b/3
                  kernels::InvariantsEvaluator3D ie
                  (Args()
                   .J(Jpt).B(B)
                   .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
                   .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                   .dI3b(dI3b).ddI3b(ddI3b));
                  const double c1 = weight/3.;
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
                        for (int r = 0; r < DIM; r++)
                        {
                           for (int c = 0; c < DIM; c++)
                           {
                              const double dp = ddi1b(r,c);
                              H(r,c,i,j,qx,qy,qz,e) = c1 * dp;
                           }
                        }
                     }
                  }
               }

               if (mid == 315)
               {
                  // 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
                  kernels::InvariantsEvaluator3D ie(Args().
                                                    J(Jpt).
                                                    dI3b(dI3b).ddI3b(ddI3b));

                  double sign_detJ;
                  const double I3b = ie.Get_I3b(sign_detJ);
                  ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
                        for (int r = 0; r < DIM; r++)
                        {
                           for (int c = 0; c < DIM; c++)
                           {
                              const double dp =
                                 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                 2.0 * weight * di3b(r,c) * di3b(i,j);
                              H(r,c,i,j,qx,qy,qz,e) = dp;
                           }
                        }
                     }
                  }
               }

               if (mid == 321)
               {
                  // ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
                  //      + (1/I3)*ddI2
                  //      + (6*I2/I3b^4)*(dI3b x dI3b)
                  //      + (-2*I2/I3b^3)*ddI3b
                  kernels::InvariantsEvaluator3D ie
                  (Args()
                   .J(Jpt).B(B)
                   .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
                   .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                   .dI3b(dI3b).ddI3b(ddI3b));
                  double sign_detJ;
                  const double I2 = ie.Get_I2();
                  const double I3b = ie.Get_I3b(sign_detJ);
                  ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
                  ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
                  const double c0 = 1.0/I3b;
                  const double c1 = weight*c0*c0;
                  const double c2 = -2*c0*c1;
                  const double c3 = c2*I2;
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
                        ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
                        ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
                        for (int r = 0; r < DIM; r++)
                        {
                           for (int c = 0; c < DIM; c++)
                           {
                              const double dp =
                                 weight * ddi1(r,c)
                                 + c1 * ddi2(r,c)
                                 + c3 * ddi3b(r,c)
                                 + c2 * ((di2(r,c)*di3b(i,j) + di3b(r,c)*di2(i,j)))
                                 -3*c0*c3 * di3b(r,c)*di3b(i,j);
                              H(r,c,i,j,qx,qy,qz,e) = dp;
                           }
                        }
                     }
                  }
               }

               if (mid == 332)
               {
                  // (1-gamma) H_302 + gamma H_315
                  kernels::InvariantsEvaluator3D ie
                  (Args()
                   .J(Jpt).B(B)
                   .dI1b(dI1b).ddI1b(ddI1b)
                   .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                   .dI3b(dI3b).ddI3b(ddI3b));
                  double sign_detJ;
                  const double c1 = weight / 9.0;
                  const double I1b = ie.Get_I1b();
                  const double I2b = ie.Get_I2b();
                  const double I3b = ie.Get_I3b(sign_detJ);
                  ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
                  ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
                  ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
                        ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
                        ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
                        for (int r = 0; r < DIM; r++)
                        {
                           for (int c = 0; c < DIM; c++)
                           {
                              const double dp_302 =
                                 (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                                 + ddi2b(r,c)*I1b
                                 + ddi1b(r,c)*I2b;
                              const double dp_315 =
                                 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                 2.0 * weight * di3b(r,c) * di3b(i,j);
                              H(r,c,i,j,qx,qy,qz,e) =
                                 (1.0 - metric_param) * c1 * dp_302 +
                                 metric_param * dp_315;
                           }
                        }
                     }
                  }
               }
            } // qx
         } // qy
      } // qz
   });
}

void TMOP_Integrator::AssembleGradPA_3D(const Vector &x) const
{
   const int NE = PA.ne;
   constexpr int DIM = 3;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int M = metric->Id();
   const double mn = metric_normal;

   double mp = 0.0;
   if (auto m = dynamic_cast<TMOP_Metric_332 *>(metric)) { mp = m->GetGamma(); }

   const auto B = Reshape(PA.maps->B.Read(), Q1D, D1D);
   const auto G = Reshape(PA.maps->G.Read(), Q1D, D1D);
   const auto W = Reshape(PA.ir->GetWeights().Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, DIM, NE);
   auto H = Reshape(PA.H.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

#ifndef MFEM_USE_JIT
   decltype(&TMOP_SetupGradPA_3D<>) ker = TMOP_SetupGradPA_3D<>;

   const int d=D1D, q=Q1D;
   if (d == 2 && q==2) { ker = TMOP_SetupGradPA_3D<2,2>; }
   if (d == 2 && q==3) { ker = TMOP_SetupGradPA_3D<2,3>; }
   if (d == 2 && q==4) { ker = TMOP_SetupGradPA_3D<2,4>; }
   if (d == 2 && q==5) { ker = TMOP_SetupGradPA_3D<2,5>; }
   if (d == 2 && q==6) { ker = TMOP_SetupGradPA_3D<2,6>; }

   if (d == 3 && q==3) { ker = TMOP_SetupGradPA_3D<3,3>; }
   if (d == 3 && q==4) { ker = TMOP_SetupGradPA_3D<4,4>; }
   if (d == 3 && q==5) { ker = TMOP_SetupGradPA_3D<5,5>; }
   if (d == 3 && q==6) { ker = TMOP_SetupGradPA_3D<6,6>; }

   if (d == 4 && q==4) { ker = TMOP_SetupGradPA_3D<4,4>; }
   if (d == 4 && q==5) { ker = TMOP_SetupGradPA_3D<4,5>; }
   if (d == 4 && q==6) { ker = TMOP_SetupGradPA_3D<4,6>; }

   if (d == 5 && q==5) { ker = TMOP_SetupGradPA_3D<5,5>; }
   if (d == 5 && q==6) { ker = TMOP_SetupGradPA_3D<5,6>; }

   ker(mn,mp,M,NE,W,B,G,J,X,H,D1D,Q1D,4);
#else
   TMOP_SetupGradPA_3D(mn,mp,M,NE,W,B,G,J,X,H,D1D,Q1D,4);
#endif
}

} // namespace mfem
