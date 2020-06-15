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
#include "../linalg/dtensor.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0>
static void AddMultGradPA_Kernel_3D(const int NE,
                                    const Array<double> &b_,
                                    const Array<double> &g_,
                                    const DenseTensor &j_,
                                    const Vector &dp_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);
   const auto dP = Reshape(dp_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

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

      kernels::LoadX<MD1>(e,D1D,X,s_DDD);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,s_BG);

      kernels::GradX<MD1,MQ1>(D1D,Q1D,s_BG,s_DDD,s_DDQ);
      kernels::GradY<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,s_DQQ);
      kernels::GradZ<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double *Jtr = &J(0,0,qx,qy,qz,e);

               // Jrt = Jtr^{-1}
               double Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               double Jpr[9];
               kernels::PullGradXYZ<MQ1>(qx,qy,qz, s_QQQ, Jpr);

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               double Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // M = Jpt : dP
               double M[9];
               for (int r = 0; r < DIM; r++)
               {
                  for (int c = 0; c < DIM; c++)
                  {
                     M[r+DIM*c] = 0.0;
                     for (int i = 0; i < DIM; i++)
                     {
                        for (int j = 0; j < DIM; j++)
                        {
                           M[r+DIM*c] += dP(i,j,r,c,qx,qy,qz,e) * Jpt[i+DIM*j];
                        }
                     }
                  }
               }

               // Y =  DS . M^t += DSh . (Jrt . M^t)
               double A[9];
               kernels::MultABt(3,3,3, Jrt, M, A);
               kernels::PushGradXYZ<MQ1>(qx,qy,qz, A, s_QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::LoadBGt<MD1,MQ1>(D1D, Q1D, b, g, s_BG);

      kernels::GradZt<MD1,MQ1>(D1D,Q1D,s_BG,s_QQQ,s_DQQ);
      kernels::GradYt<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_DDQ);
      kernels::GradXt<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,Y,e);
   });
}

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_3D,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           const Vector &dp_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d);

void TMOP_Integrator::AddMultGradPA_3D(const Vector &X, const Vector &R,
                                       Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &A = PA.A;

   if (!PA.setup)
   {
      PA.setup = true;
      AssembleGradPA_3D(X);
   }

   if (KAddMultGradPA_Kernel_3D.Find(id))
   {
      return KAddMultGradPA_Kernel_3D.At(id)(N,B,G,J,A,R,C,0,0);
   }
   else
   {
      constexpr int T_MAX = 4;
      MFEM_VERIFY(D1D <= T_MAX && Q1D <= T_MAX, "Max size error!");
      return AddMultGradPA_Kernel_3D<0,0,T_MAX>(N,B,G,J,A,R,C,D1D,Q1D);
   }
}

} // namespace mfem
