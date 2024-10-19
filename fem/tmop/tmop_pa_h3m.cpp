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

#include "tmop_pa.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultGradPA_3D(const int NE, const ConstDeviceMatrix &B,
                           const ConstDeviceMatrix &G,
                           const DeviceTensor<6, const double> &J,
                           const DeviceTensor<8, const double> &H,
                           const DeviceTensor<5, const double> &X,
                           DeviceTensor<5> &Y, const int d1d, const int q1d,
                           const int max)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t DDD[3][MD1 * MD1 * MD1];
      MFEM_SHARED real_t DDQ[9][MD1 * MD1 * MQ1];
      MFEM_SHARED real_t DQQ[9][MD1 * MQ1 * MQ1];
      MFEM_SHARED real_t QQQ[9][MQ1 * MQ1 * MQ1];

      kernels::internal::LoadX<MD1>(e, D1D, X, DDD);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1>(D1D, Q1D, BG, DDD, DDQ);
      kernels::internal::GradY<MD1, MQ1>(D1D, Q1D, BG, DDQ, DQQ);
      kernels::internal::GradZ<MD1, MQ1>(D1D, Q1D, BG, DQQ, QQQ);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);

               // Jrt = Jtr^{-1}
               real_t Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               real_t Jpr[9];
               kernels::internal::PullGrad<MQ1>(Q1D, qx, qy, qz, QQQ, Jpr);

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               real_t Jpt[9];
               kernels::Mult(3, 3, 3, Jpr, Jrt, Jpt);

               // B = Jpt : H
               real_t B[9];
               DeviceMatrix M(B, 3, 3);
               ConstDeviceMatrix J(Jpt, 3, 3);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     M(i, j) = 0.0;
                     for (int r = 0; r < DIM; r++)
                     {
                        for (int c = 0; c < DIM; c++)
                        {
                           M(i, j) += H(r, c, i, j, qx, qy, qz, e) * J(r, c);
                        }
                     }
                  }
               }

               // Y +=  DS . M^t += DSh . (Jrt . M^t)
               real_t A[9];
               kernels::MultABt(3, 3, 3, Jrt, B, A);
               kernels::internal::PushGrad<MQ1>(Q1D, qx, qy, qz, A, QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1, MQ1>(D1D, Q1D, B, G, BG);
      kernels::internal::GradZt<MD1, MQ1>(D1D, Q1D, BG, QQQ, DQQ);
      kernels::internal::GradYt<MD1, MQ1>(D1D, Q1D, BG, DQQ, DDQ);
      kernels::internal::GradXt<MD1, MQ1>(D1D, Q1D, BG, DDQ, Y, e);
   });
}

void TMOP_Integrator::AddMultGradPA_3D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto G = Reshape(PA.maps->G.Read(), q, d);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
   const auto X = Reshape(R.Read(), d, d, d, DIM, NE);
   const auto H = Reshape(PA.H.Read(), DIM, DIM, DIM, DIM, q, q, q, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, d, DIM, NE);

   decltype(&TMOP_AddMultGradPA_3D<>) ker = TMOP_AddMultGradPA_3D;

   if (d == 2 && q == 2)
   {
      ker = TMOP_AddMultGradPA_3D<2, 2>;
   }
   if (d == 2 && q == 3)
   {
      ker = TMOP_AddMultGradPA_3D<2, 3>;
   }
   if (d == 2 && q == 4)
   {
      ker = TMOP_AddMultGradPA_3D<2, 4>;
   }
   if (d == 2 && q == 5)
   {
      ker = TMOP_AddMultGradPA_3D<2, 5>;
   }
   if (d == 2 && q == 6)
   {
      ker = TMOP_AddMultGradPA_3D<2, 6>;
   }

   if (d == 3 && q == 3)
   {
      ker = TMOP_AddMultGradPA_3D<3, 3>;
   }
   if (d == 3 && q == 4)
   {
      ker = TMOP_AddMultGradPA_3D<3, 4>;
   }
   if (d == 3 && q == 5)
   {
      ker = TMOP_AddMultGradPA_3D<3, 5>;
   }
   if (d == 3 && q == 6)
   {
      ker = TMOP_AddMultGradPA_3D<3, 6>;
   }

   if (d == 4 && q == 4)
   {
      ker = TMOP_AddMultGradPA_3D<4, 4>;
   }
   if (d == 4 && q == 5)
   {
      ker = TMOP_AddMultGradPA_3D<4, 5>;
   }
   if (d == 4 && q == 6)
   {
      ker = TMOP_AddMultGradPA_3D<4, 6>;
   }

   if (d == 5 && q == 5)
   {
      ker = TMOP_AddMultGradPA_3D<5, 5>;
   }
   if (d == 5 && q == 6)
   {
      ker = TMOP_AddMultGradPA_3D<5, 6>;
   }

   ker(NE, B, G, J, H, X, Y, d, q, 4);
}

}  // namespace mfem
