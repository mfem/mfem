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

using Args = kernels::InvariantsEvaluator2D::Buffers;

// weight * ddI1
static MFEM_HOST_DEVICE inline
void EvalH_001(const int e, const int qx, const int qy,
               const double weight, const double *Jpt,
               DeviceTensor<7,double> H)
{
   constexpr int DIM = 2;
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

// 0.5 * weight * dI1b
static MFEM_HOST_DEVICE inline
void EvalH_002(const int e, const int qx, const int qy,
               const double weight, const double *Jpt,
               DeviceTensor<7,double> H)
{
   constexpr int DIM = 2;
   double ddI1[4], ddI1b[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args()
                                     .J(Jpt)
                                     .ddI1(ddI1)
                                     .ddI1b(ddI1b)
                                     .dI2b(dI2b));
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

static MFEM_HOST_DEVICE inline
void EvalH_007(const int e, const int qx, const int qy,
               const double weight, const double *Jpt,
               DeviceTensor<7,double> H)
{
   constexpr int DIM = 2;
   double ddI1[4], ddI2[4], dI1[4], dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args()
                                     .J(Jpt)
                                     .ddI1(ddI1)
                                     .ddI2(ddI2)
                                     .dI1(dI1)
                                     .dI2(dI2)
                                     .dI2b(dI2b));
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
                  - c2 * ( di1(i,j) * di2(r,c) + di2(i,j) * di1(r,c) )
                  + 2.0 * c1 * c3 * di2(r,c) * di2(i,j);
            }
         }
      }
   }
}

static MFEM_HOST_DEVICE inline
void EvalH_077(const int e, const int qx, const int qy,
               const double weight, const double *Jpt,
               DeviceTensor<7,double> H)
{
   constexpr int DIM = 2;
   double dI2[4], dI2b[4], ddI2[4];
   kernels::InvariantsEvaluator2D ie(Args()
                                     .J(Jpt)
                                     .dI2(dI2)
                                     .dI2b(dI2b)
                                     .ddI2(ddI2));
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
                  weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r,c)
                  + weight * (I2inv_sq / I2) * di2(r,c) * di2(i,j);
            }
         }
      }
   }
}

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_2D,
                           const Vector &x_,
                           const double metric_normal,
                           const int mid,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           Vector &h_,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 77,
               "Metric not yet implemented!");

   constexpr int DIM = 2;
   constexpr int NBZ = 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   auto H = Reshape(h_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double s_QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,s_X);
      kernels::internal::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, s_BG);

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
            kernels::internal::PullGrad<MQ1,NBZ>(qx,qy,s_QQ,Jpr);

            // Jpt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->AssembleH
            if (mid ==  1) { EvalH_001(e,qx,qy,weight,Jpt,H); }
            if (mid ==  2) { EvalH_002(e,qx,qy,weight,Jpt,H); }
            if (mid ==  7) { EvalH_007(e,qx,qy,weight,Jpt,H); }
            if (mid == 77) { EvalH_077(e,qx,qy,weight,Jpt,H); }
         } // qx
      } // qy
   });
}

void TMOP_Integrator::AssembleGradPA_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double mn = metric_normal;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_2D,id,X,mn,M,N,W,B,G,J,H);
}

} // namespace mfem
