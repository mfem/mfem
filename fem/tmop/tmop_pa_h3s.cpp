// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

using Args = kernels::InvariantsEvaluator3D::Buffers;

// dP_302 = (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
static MFEM_HOST_DEVICE inline
void EvalH_302(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9],          ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double dI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
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
               dP(r,c,i,j,qx,qy,qz,e) = c1 * dp;
            }
         }
      }
   }
}

// dP_303 = ddI1b/3
static MFEM_HOST_DEVICE inline
void EvalH_303(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9], ddI1[9], ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double dI3b[9], ddI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
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
               dP(r,c,i,j,qx,qy,qz,e) = c1 * dp;
            }
         }
      }
   }
}

// dP_315 = 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
static MFEM_HOST_DEVICE inline
void EvalH_315(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double dI3b[9], ddI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args().
                                     J(J).
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
               const double dp = 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                 2.0 * weight * di3b(r,c) * di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = dp;
            }
         }
      }
   }
}

// dP_321 = ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
//               + (1/I3)*ddI2
//               + (6*I2/I3b^4)*(dI3b x dI3b)
//               + (-2*I2/I3b^3)*ddI3b
static MFEM_HOST_DEVICE inline
void EvalH_321(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9], ddI1[9], ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double dI3b[9], ddI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
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
               dP(r,c,i,j,qx,qy,qz,e) = dp;
            }
         }
      }
   }
}

// H_332 = (1-gamma) H_302 + gamma H_315
static MFEM_HOST_DEVICE inline
void EvalH_332(const int e, const int qx, const int qy, const int qz,
               const double weight, const double gamma,
               const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9],          ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double         dI3b[9],          ddI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b).ddI3b(ddI3b));
   double sign_detJ;
   const double c1 = weight/9.;
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
               const double dp_315 = 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                     2.0 * weight * di3b(r,c) * di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = (1.0 - gamma) * c1 * dp_302 +
                                        gamma * dp_315;
            }
         }
      }
   }
}

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_3D,
                           const double metric_normal,
                           const double metric_param,
                           const int mid,
                           const Vector &x_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           Vector &h_,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(mid == 302 || mid == 303 || mid == 315 ||
               mid == 321 || mid == 332, "3D metric not yet implemented!");

   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);
   auto H = Reshape(h_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

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

               // metric->AssembleH
               if (mid == 302) { EvalH_302(e,qx,qy,qz,weight,Jpt,H); }
               if (mid == 303) { EvalH_303(e,qx,qy,qz,weight,Jpt,H); }
               if (mid == 315) { EvalH_315(e,qx,qy,qz,weight,Jpt,H); }
               if (mid == 321) { EvalH_321(e,qx,qy,qz,weight,Jpt,H); }
               if (mid == 332) { EvalH_332(e,qx,qy,qz,weight,metric_param,Jpt,H); }
            } // qx
         } // qy
      } // qz
   });
}

void TMOP_Integrator::AssembleGradPA_3D(const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int M = metric->Id();
   const int id = (D1D << 4 ) | Q1D;
   const double mn = metric_normal;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &H = PA.H;

   double mp = 0.0;
   if (auto m = dynamic_cast<TMOP_Metric_332 *>(metric)) { mp = m->GetGamma(); }

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_3D,id,mn,mp,M,X,N,W,B,G,J,H);
}

} // namespace mfem
