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
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

using Args = kernels::InvariantsEvaluator2D::Buffers;

static MFEM_HOST_DEVICE inline
double EvalW_001(const double *Jpt)
{
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
   return ie.Get_I1();
}

static MFEM_HOST_DEVICE inline
double EvalW_002(const double *Jpt)
{
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
   return 0.5 * ie.Get_I1b() - 1.0;
}

static MFEM_HOST_DEVICE inline
double EvalW_007(const double *Jpt)
{
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
   return ie.Get_I1() * (1.0 + 1.0/ie.Get_I2()) - 4.0;
}

static MFEM_HOST_DEVICE inline
double EvalW_077(const double *Jpt)
{
   kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
   const double I2b = ie.Get_I2b();
   return 0.5*(I2b*I2b + 1./(I2b*I2b) - 2.);
}

MFEM_REGISTER_TMOP_KERNELS(double, EnergyPA_2D,
                           const double metric_normal,
                           const int mid,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const Vector &x_,
                           const Vector &ones,
                           Vector &energy,
                           const int d1d,
                           const int q1d)
{
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 77,
               "2D metric not yet implemented!");

   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);

   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);

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

            // Jpr = X^t.DSh
            double Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jpr);

            // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2,Jpr,Jrt,Jpt);

            // metric->EvalW(Jpt);
            const double EvalW =
            mid ==  1 ? EvalW_001(Jpt) :
            mid ==  2 ? EvalW_002(Jpt) :
            mid ==  7 ? EvalW_007(Jpt) :
            mid == 77 ? EvalW_077(Jpt) : 0.0;

            E(qx,qy,e) = weight * EvalW;
         }
      }
   });
   return energy * ones;
}

double TMOP_Integrator::GetLocalStateEnergyPA_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double m = metric_normal;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &O = PA.O;
   Vector &E = PA.E;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_2D,id,m,M,N,J,W,B,G,X,O,E);
}

} // namespace mfem
