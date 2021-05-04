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

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_2D,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           const Vector &h_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   const auto H = Reshape(h_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

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

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            double Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(qx,qy,QQ,Jpr);

            // Jpt = Jpr . Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // B = Jpt : H
            double B[4];
            DeviceMatrix M(B,2,2);
            ConstDeviceMatrix J(Jpt,2,2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  M(i,j) = 0.0;
                  for (int r = 0; r < DIM; r++)
                  {
                     for (int c = 0; c < DIM; c++)
                     {
                        M(i,j) += H(r,c,i,j,qx,qy,e) * J(r,c);
                     }
                  }
               }
            }
            // C = Jrt . B
            double C[4];
            kernels::MultABt(2,2,2, Jrt, B, C);

            // Overwrite QQ = Jrt . (Jpt : H)^t
            kernels::internal::PushGrad<MQ1,NBZ>(qx,qy, C, QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::internal::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::internal::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultGradPA_2D(const Vector &R, Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_2D,id,N,B,G,J,H,R,C);
}

} // namespace mfem
