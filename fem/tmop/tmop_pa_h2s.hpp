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

template<typename METRIC_KERNEL, int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_SetupGradPA_2D(const DeviceTensor<4,const double> &X,
                         const double metric_normal,
                         const double *w,
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
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
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
            METRIC_KERNEL{}.AssembleH(qx,qy,e, weight, Jpt, w, H);
         } // qx
      } // qy
   });
}

template<typename M, typename... Args>
static void Launch(const int d, const int q, Args&&... args)
{
   decltype(&TMOP_SetupGradPA_2D<M>) ker = TMOP_SetupGradPA_2D<M>;

   if (d==2 && q==2) { ker = TMOP_SetupGradPA_2D<M,2,2>; }
   if (d==2 && q==3) { ker = TMOP_SetupGradPA_2D<M,2,3>; }
   if (d==2 && q==4) { ker = TMOP_SetupGradPA_2D<M,2,4>; }
   if (d==2 && q==5) { ker = TMOP_SetupGradPA_2D<M,2,5>; }
   if (d==2 && q==6) { ker = TMOP_SetupGradPA_2D<M,2,6>; }

   if (d==3 && q==3) { ker = TMOP_SetupGradPA_2D<M,3,3>; }
   if (d==3 && q==4) { ker = TMOP_SetupGradPA_2D<M,3,4>; }
   if (d==3 && q==5) { ker = TMOP_SetupGradPA_2D<M,3,5>; }
   if (d==3 && q==6) { ker = TMOP_SetupGradPA_2D<M,3,6>; }

   if (d==4 && q==4) { ker = TMOP_SetupGradPA_2D<M,4,4>; }
   if (d==4 && q==5) { ker = TMOP_SetupGradPA_2D<M,4,5>; }
   if (d==4 && q==6) { ker = TMOP_SetupGradPA_2D<M,4,6>; }

   if (d==5 && q==5) { ker = TMOP_SetupGradPA_2D<M,5,5>; }
   if (d==5 && q==6) { ker = TMOP_SetupGradPA_2D<M,5,6>; }

   ker(std::forward<Args>(args)...,d,q,4);
}

} // namespace mfem
