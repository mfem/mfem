// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop.hpp"
#include "tmop_pa.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dinvariants.hpp"

namespace mfem
{

// weight * ddI1
static MFEM_HOST_DEVICE inline
void EvalH_001(const int e, const int qx, const int qy,
               const double weight, const double *Jpt,
               DeviceTensor<7,double> H)
{
   constexpr int DIM = 2;
   double ddI1[4];
   kernels::InvariantsEvaluator2D ie(Jpt,
                                     nullptr, nullptr, ddI1, nullptr,
                                     nullptr, nullptr, nullptr, nullptr);
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
   kernels::InvariantsEvaluator2D ie(Jpt,
                                     nullptr, nullptr, ddI1, ddI1b,
                                     nullptr, dI2b, nullptr, nullptr);
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

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0, int T_MAX = 0>
static void SetupGradPA_2D(const Vector &x_,
                           const double metric_normal,
                           const int mid,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           Vector &h_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

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
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double s_QQ[4][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X,s_X);
      kernels::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, s_BG);

      kernels::GradX<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_X, s_DQ);
      kernels::GradY<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_DQ, s_QQ);

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

            // Jpr = X^T.DSh
            double Jpr[4];
            kernels::PullGradXY<MQ1,NBZ>(qx,qy,s_QQ,Jpr);

            // Jpt = GX^T.DS = (GX^T.DSh).Jrt = GX.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->AssembleH
            if (mid == 1) { EvalH_001(e,qx,qy,weight,Jpt,H); }
            if (mid == 2) { EvalH_002(e,qx,qy,weight,Jpt,H); }
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
   const DenseTensor &J = PA.Jtr;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &H = PA.A;
   const double mn = metric_normal;

   switch (id)
   {
      case 0x21: return SetupGradPA_2D<2,1,1>(X,mn,M,N,W,B,G,J,H);
      case 0x22: return SetupGradPA_2D<2,2,1>(X,mn,M,N,W,B,G,J,H);
      case 0x23: return SetupGradPA_2D<2,3,1>(X,mn,M,N,W,B,G,J,H);
      case 0x24: return SetupGradPA_2D<2,4,1>(X,mn,M,N,W,B,G,J,H);
      case 0x25: return SetupGradPA_2D<2,5,1>(X,mn,M,N,W,B,G,J,H);
      case 0x26: return SetupGradPA_2D<2,6,1>(X,mn,M,N,W,B,G,J,H);

      case 0x31: return SetupGradPA_2D<3,1,1>(X,mn,M,N,W,B,G,J,H);
      case 0x32: return SetupGradPA_2D<3,2,1>(X,mn,M,N,W,B,G,J,H);
      case 0x33: return SetupGradPA_2D<3,3,1>(X,mn,M,N,W,B,G,J,H);
      case 0x34: return SetupGradPA_2D<3,4,1>(X,mn,M,N,W,B,G,J,H);
      case 0x35: return SetupGradPA_2D<3,5,1>(X,mn,M,N,W,B,G,J,H);
      case 0x36: return SetupGradPA_2D<3,6,1>(X,mn,M,N,W,B,G,J,H);

      case 0x41: return SetupGradPA_2D<4,1,1>(X,mn,M,N,W,B,G,J,H);
      case 0x42: return SetupGradPA_2D<4,2,1>(X,mn,M,N,W,B,G,J,H);
      case 0x43: return SetupGradPA_2D<4,3,1>(X,mn,M,N,W,B,G,J,H);
      case 0x44: return SetupGradPA_2D<4,4,1>(X,mn,M,N,W,B,G,J,H);
      case 0x45: return SetupGradPA_2D<4,5,1>(X,mn,M,N,W,B,G,J,H);
      case 0x46: return SetupGradPA_2D<4,6,1>(X,mn,M,N,W,B,G,J,H);

      case 0x51: return SetupGradPA_2D<5,1,1>(X,mn,M,N,W,B,G,J,H);
      case 0x52: return SetupGradPA_2D<5,2,1>(X,mn,M,N,W,B,G,J,H);
      case 0x53: return SetupGradPA_2D<5,3,1>(X,mn,M,N,W,B,G,J,H);
      case 0x54: return SetupGradPA_2D<5,4,1>(X,mn,M,N,W,B,G,J,H);
      case 0x55: return SetupGradPA_2D<5,5,1>(X,mn,M,N,W,B,G,J,H);
      case 0x56: return SetupGradPA_2D<5,6,1>(X,mn,M,N,W,B,G,J,H);

      default:
      {
         constexpr int T_MAX = 8;
         MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");
         return SetupGradPA_2D<0,0,0,T_MAX>(X,mn,M,N,W,B,G,J,H,D1D,Q1D);
      }
   }
}

} // namespace mfem
