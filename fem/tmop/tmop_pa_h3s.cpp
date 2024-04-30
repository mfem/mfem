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

using Args = kernels::InvariantsEvaluator3D::Buffers;

// dP_302 = (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
static MFEM_HOST_DEVICE inline
void EvalH_302(const int e, const int qx, const int qy, const int qz,
               const real_t weight, const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *B, real_t *dI1b, real_t *ddI1b,
               real_t *dI2, real_t *dI2b, real_t *ddI2, real_t *ddI2b,
               real_t *dI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b));
   const real_t c1 = weight/9.;
   const real_t I1b = ie.Get_I1b();
   const real_t I2b = ie.Get_I2b();
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
               const real_t dp =
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
               const real_t weight, const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *B, real_t *dI1b, real_t *ddI1, real_t *ddI1b,
               real_t *dI2, real_t *dI2b, real_t *ddI2, real_t *ddI2b,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b).ddI3b(ddI3b));
   const real_t c1 = weight/3.;
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
         for (int r = 0; r < DIM; r++)
         {
            for (int c = 0; c < DIM; c++)
            {
               const real_t dp = ddi1b(r,c);
               dP(r,c,i,j,qx,qy,qz,e) = c1 * dp;
            }
         }
      }
   }
}

// dP_315 = 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
static MFEM_HOST_DEVICE inline
void EvalH_315(const int e, const int qx, const int qy, const int qz,
               const real_t weight, const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args().
                                     J(J).
                                     dI3b(dI3b).ddI3b(ddI3b));

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
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
               const real_t dp = 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                 2.0 * weight * di3b(r,c) * di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = dp;
            }
         }
      }
   }
}

// dP_318 = (I3b - 1/I3b^3)*ddI3b + (1 + 3/I3b^4)*(dI3b x dI3b)
// Uses the I3b form, as dI3 and ddI3 were not implemented at the time.
static MFEM_HOST_DEVICE inline
void EvalH_318(const int e, const int qx, const int qy, const int qz,
               const real_t weight, const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args().
                                     J(J).
                                     dI3b(dI3b).ddI3b(ddI3b));

   real_t sign_detJ;
   const real_t I3b = ie.Get_I3b(sign_detJ);
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
               const real_t dp =
                  weight * (I3b - 1.0/(I3b*I3b*I3b)) * ddi3b(r,c) +
                  weight * (1.0 + 3.0/(I3b*I3b*I3b*I3b)) * di3b(r,c)*di3b(i,j);
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
               const real_t weight, const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *B, real_t *dI1b, real_t *ddI1, real_t *ddI1b,
               real_t *dI2, real_t *dI2b, real_t *ddI2, real_t *ddI2b,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b).ddI3b(ddI3b));
   real_t sign_detJ;
   const real_t I2 = ie.Get_I2();
   const real_t I3b = ie.Get_I3b(sign_detJ);

   ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
   ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);

   const real_t c0 = 1.0/I3b;
   const real_t c1 = weight*c0*c0;
   const real_t c2 = -2*c0*c1;
   const real_t c3 = c2*I2;

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
               const real_t dp =
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

// H_332 = w0 H_302 + w1 H_315
static MFEM_HOST_DEVICE inline
void EvalH_332(const int e, const int qx, const int qy, const int qz,
               const real_t weight, const real_t *w,
               const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *B, real_t *dI1b, real_t *ddI1b,
               real_t *dI2, real_t *dI2b, real_t *ddI2, real_t *ddI2b,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b).ddI3b(ddI3b));
   real_t sign_detJ;
   const real_t c1 = weight/9.;
   const real_t I1b = ie.Get_I1b();
   const real_t I2b = ie.Get_I2b();
   const real_t I3b = ie.Get_I3b(sign_detJ);
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
               const real_t dp_302 =
                  (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                  + ddi2b(r,c)*I1b
                  + ddi1b(r,c)*I2b;
               const real_t dp_315 = 2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                                     2.0 * weight * di3b(r,c) * di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = w[0] * c1 * dp_302 +
                                        w[1] * dp_315;
            }
         }
      }
   }
}

// H_338 = w0 H_302 + w1 H_318
static MFEM_HOST_DEVICE inline
void EvalH_338(const int e, const int qx, const int qy, const int qz,
               const real_t weight, const real_t *w,
               const real_t *J, DeviceTensor<8,real_t> dP,
               real_t *B, real_t *dI1b, real_t *ddI1b,
               real_t *dI2, real_t *dI2b, real_t *ddI2, real_t *ddI2b,
               real_t *dI3b, real_t *ddI3b)
{
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(Args()
                                     .J(J).B(B)
                                     .dI1b(dI1b).ddI1b(ddI1b)
                                     .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
                                     .dI3b(dI3b).ddI3b(ddI3b));
   real_t sign_detJ;
   const real_t c1 = weight/9.;
   const real_t I1b = ie.Get_I1b();
   const real_t I2b = ie.Get_I2b();
   const real_t I3b = ie.Get_I3b(sign_detJ);
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
               const real_t dp_302 =
                  (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                  + ddi2b(r,c)*I1b
                  + ddi1b(r,c)*I2b;
               const real_t dp_318 =
                  weight * (I3b - 1.0/(I3b*I3b*I3b)) * ddi3b(r,c) +
                  weight * (1.0 + 3.0/(I3b*I3b*I3b*I3b)) * di3b(r,c)*di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = w[0] * c1 * dp_302 +
                                        w[1] * dp_318;
            }
         }
      }
   }
}

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_3D,
                           const real_t metric_normal,
                           const Vector &mc_,
                           const Array<real_t> &metric_param,
                           const int mid,
                           const Vector &x_,
                           const int NE,
                           const Array<real_t> &w_,
                           const Array<real_t> &b_,
                           const Array<real_t> &g_,
                           const DenseTensor &j_,
                           Vector &h_,
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
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);
   auto H = Reshape(h_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

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
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz, s_QQQ, Jpr);

               // Jpt = X^T . DS = (X^T.DSh) . Jrt = Jpr . Jrt
               real_t Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // InvariantsEvaluator3D buffers used for the metrics
               real_t B[9];
               real_t         dI1b[9], ddI1[9], ddI1b[9];
               real_t dI2[9], dI2b[9], ddI2[9], ddI2b[9];
               // reuse local arrays, to help register allocation
               real_t        *dI3b=Jrt,        *ddI3b=Jpr;

               // metric->AssembleH
               if (mid == 302)
               {
                  EvalH_302(e,qx,qy,qz,weight,Jpt,H,
                            B,dI1b,ddI1b,dI2,dI2b,ddI2,ddI2b,dI3b);
               }
               if (mid == 303)
               {
                  EvalH_303(e,qx,qy,qz,weight,Jpt,H,
                            B,dI1b,ddI1,ddI1b,dI2,dI2b,ddI2,ddI2b,dI3b,ddI3b);
               }
               if (mid == 315)
               {
                  EvalH_315(e,qx,qy,qz,weight,Jpt,H, dI3b,ddI3b);
               }
               if (mid == 318)
               {
                  EvalH_318(e,qx,qy,qz,weight,Jpt,H, dI3b,ddI3b);
               }
               if (mid == 321)
               {
                  EvalH_321(e,qx,qy,qz,weight,Jpt,H,
                            B,dI1b,ddI1,ddI1b,dI2,dI2b,ddI2,ddI2b,dI3b,ddI3b);
               }
               if (mid == 332)
               {
                  EvalH_332(e,qx,qy,qz,weight,metric_data,Jpt,H,
                            B,dI1b,ddI1b,dI2,dI2b,ddI2,ddI2b,dI3b,ddI3b);
               }
               if (mid == 338)
               {
                  EvalH_338(e,qx,qy,qz,weight,metric_data,Jpt,H,
                            B,dI1b,ddI1b,dI2,dI2b,ddI2,ddI2b,dI3b,ddI3b);
               }
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
   const real_t mn = metric_normal;
   const Vector &MC = PA.MC;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W = PA.ir->GetWeights();
   const Array<real_t> &B = PA.maps->B;
   const Array<real_t> &G = PA.maps->G;
   Vector &H = PA.H;

   Array<real_t> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_3D,id,mn,MC,mp,M,X,N,W,B,G,J,H);
}

} // namespace mfem
