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

static MFEM_HOST_DEVICE inline
void EvalP_001(const double *Jpt, double *P)
{
   double dI1[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
   kernels::Set(2,2, 1.0, ie.Get_dI1(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_002(const double *Jpt, double *P)
{
   double dI1b[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
   kernels::Set(2,2, 1./2., ie.Get_dI1b(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_007(const double *Jpt, double *P)
{
   double dI1[4], dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1)
                                     .dI2(dI2).dI2b(dI2b));
   const double I2 = ie.Get_I2();
   kernels::Add(2,2, 1.0 + 1.0 / I2, ie.Get_dI1(),
                -ie.Get_I1() / (I2*I2), ie.Get_dI2(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_077(const double *Jpt, double *P)
{
   double dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().
                                     J(Jpt).
                                     dI2(dI2).dI2b(dI2b));
   const double I2 = ie.Get_I2();
   kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
}

static MFEM_HOST_DEVICE inline
void EvalP_080(const double *Jpt, double gamma, double *P)
{
   // p_80 = (1-gamma) p_2 + gamma p_77.

   double dI1b[4], dI2[4], dI2b[4];
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt).
                                     dI1b(dI1b).dI2(dI2).dI2b(dI2b));

   kernels::Set(2,2, (1.0 - gamma) * 1./2., ie.Get_dI1b(), P);

   const double I2 = ie.Get_I2();
   kernels::Add(2,2, gamma * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
}

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_2D,
                           const double metric_normal,
                           const double metric_param,
                           const int mid,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 77 || mid == 80,
               "Metric not yet implemented!");

   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
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
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

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
            if (mid ==  1) { EvalP_001(Jpt, P); }
            if (mid ==  2) { EvalP_002(Jpt, P); }
            if (mid ==  7) { EvalP_007(Jpt, P); }
            if (mid == 77) { EvalP_077(Jpt, P); }
            if (mid == 80) { EvalP_080(Jpt, metric_param, P); }
            for (int i = 0; i < 4; i++) { P[i] *= weight; }

            // PMatO += DS . P^t += DSh . (Jrt . P^t)
            double A[4];
            kernels::MultABt(2,2,2, Jrt, P, A);
            kernels::internal::PushGrad<MQ1,NBZ>(Q1D,qx,qy,A,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::internal::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::internal::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
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
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const double mn = metric_normal;

   double mp = 0.0;
   if (auto m = dynamic_cast<TMOP_Metric_080 *>(metric)) { mp = m->GetGamma(); }

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_2D,id,mn,mp,M,N,J,W,B,G,X,Y);
}

} // namespace mfem
