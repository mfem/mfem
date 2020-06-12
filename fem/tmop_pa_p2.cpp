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

static MFEM_HOST_DEVICE inline
void EvalP_001(const double *Jpt, double *P)
{
   double dI1[4];
   kernels::InvariantsEvaluator2D ie(Jpt,
                                     dI1, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr);
   kernels::Set(2,2, 1.0, ie.Get_dI1(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_002(const double *Jpt, double *P)
{
   double dI1b[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Jpt,
                                     nullptr, dI1b, nullptr, nullptr,
                                     nullptr, dI2b, nullptr, nullptr);
   kernels::Set(2,2, 1./2., ie.Get_dI1b(), P);
}

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0, int T_MAX = 0>
static void AddMultPA_Kernel_2D(const double metric_normal,
                                const int mid,
                                const int NE,
                                const DenseTensor &j_,
                                const Array<double> &w_,
                                const Array<double> &b_,
                                const Array<double> &g_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   constexpr int VDIM = 2;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
   constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   const auto J = Reshape(j_.Read(), VDIM, VDIM, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   auto X = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

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

            // G = X{^T}.DSh
            double Jpr[4];
            kernels::PullGradXY<MQ1,NBZ>(qx,qy,QQ,Jpr);

            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = G.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->EvalP(Jpt, P);
            double P[4];
            if (mid == 1) { EvalP_001(Jpt, P); }
            if (mid == 2) { EvalP_002(Jpt, P); }
            for (int i = 0; i < 4; i++) { P[i] *= weight; }

            // PMatO +=  DS . P^t += DSh . (Jrt . P^t)
            double A[4];
            kernels::MultABt(2,2,2, Jrt, P, A);
            kernels::PushGradXY<MQ1,NBZ>(qx,qy,A,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBGt<MD1,MQ1>(D1D, Q1D, b, g, BG);
      kernels::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_2D(const Vector &X, Vector &Y) const
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
   const double mn = metric_normal;

   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_2D<2,1,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x22: return AddMultPA_Kernel_2D<2,2,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x23: return AddMultPA_Kernel_2D<2,3,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x24: return AddMultPA_Kernel_2D<2,4,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x25: return AddMultPA_Kernel_2D<2,5,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x26: return AddMultPA_Kernel_2D<2,6,1>(mn,M,N,J,W,B,G,X,Y);

      case 0x31: return AddMultPA_Kernel_2D<3,1,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x32: return AddMultPA_Kernel_2D<3,2,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x33: return AddMultPA_Kernel_2D<3,3,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x34: return AddMultPA_Kernel_2D<3,4,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x35: return AddMultPA_Kernel_2D<3,5,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x36: return AddMultPA_Kernel_2D<3,6,1>(mn,M,N,J,W,B,G,X,Y);

      case 0x41: return AddMultPA_Kernel_2D<4,1,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x42: return AddMultPA_Kernel_2D<4,2,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x43: return AddMultPA_Kernel_2D<4,3,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x44: return AddMultPA_Kernel_2D<4,4,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x45: return AddMultPA_Kernel_2D<4,5,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x46: return AddMultPA_Kernel_2D<4,6,1>(mn,M,N,J,W,B,G,X,Y);

      case 0x51: return AddMultPA_Kernel_2D<5,1,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x52: return AddMultPA_Kernel_2D<5,2,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x53: return AddMultPA_Kernel_2D<5,3,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x54: return AddMultPA_Kernel_2D<5,4,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x55: return AddMultPA_Kernel_2D<5,5,1>(mn,M,N,J,W,B,G,X,Y);
      case 0x56: return AddMultPA_Kernel_2D<5,6,1>(mn,M,N,J,W,B,G,X,Y);

      default:
      {
         constexpr int T_MAX = 8;
         MFEM_VERIFY(D1D <= T_MAX && Q1D <= T_MAX, "Max size error!");
         return AddMultPA_Kernel_2D<0,0,0,T_MAX>(mn,M,N,J,W,B,G,X,Y,D1D,Q1D);
      }
   }
}

} // namespace mfem
