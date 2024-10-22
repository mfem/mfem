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
#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultGradPA_2D(const int NE,
                           const ConstDeviceMatrix &B,
                           const ConstDeviceMatrix &G,
                           const DeviceTensor<5, const real_t> &J,
                           const DeviceTensor<7, const real_t> &H,
                           const DeviceTensor<4, const real_t> &X,
                           DeviceTensor<4> &Y,
                           const int d1d,
                           const int q1d,
                           const int max)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 2;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t XY[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ[4][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ[4][NBZ][MQ1 * MQ1];

      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X, XY);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::GradX<MD1, MQ1, NBZ>(D1D, Q1D, BG, XY, DQ);
      kernels::internal::GradY<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, QQ);

      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);

            // Jrt = Jtr^{-1}
            real_t Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            real_t Jpr[4];
            kernels::internal::PullGrad<MQ1, NBZ>(Q1D, qx, qy, QQ, Jpr);

            // Jpt = Jpr . Jrt
            real_t Jpt[4];
            kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

            // B = Jpt : H
            real_t B[4];
            DeviceMatrix M(B, 2, 2);
            ConstDeviceMatrix J(Jpt, 2, 2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  M(i, j) = 0.0;
                  for (int r = 0; r < DIM; r++)
                  {
                     for (int c = 0; c < DIM; c++)
                     {
                        M(i, j) += H(r, c, i, j, qx, qy, e) * J(r, c);
                     }
                  }
               }
            }
            // C = Jrt . B
            real_t C[4];
            kernels::MultABt(2, 2, 2, Jrt, B, C);

            // Overwrite QQ = Jrt . (Jpt : H)^t
            kernels::internal::PushGrad<MQ1, NBZ>(Q1D, qx, qy, C, QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1, MQ1>(D1D, Q1D, B, G, BG);
      kernels::internal::GradYt<MD1, MQ1, NBZ>(D1D, Q1D, BG, QQ, DQ);
      kernels::internal::GradXt<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, Y, e);
   });
}

void TMOP_Integrator::AddMultGradPA_2D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto G = Reshape(PA.maps->G.Read(), q, d);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, NE);
   const auto H = Reshape(PA.H.Read(), DIM, DIM, DIM, DIM, q, q, NE);
   const auto X = Reshape(R.Read(), d, d, DIM, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, DIM, NE);

   decltype(&TMOP_AddMultGradPA_2D<>) ker = TMOP_AddMultGradPA_2D;

   if (d == 2 && q == 2) { ker = TMOP_AddMultGradPA_2D<2, 2>; }
   if (d == 2 && q == 3) { ker = TMOP_AddMultGradPA_2D<2, 3>; }
   if (d == 2 && q == 4) { ker = TMOP_AddMultGradPA_2D<2, 4>; }
   if (d == 2 && q == 5) { ker = TMOP_AddMultGradPA_2D<2, 5>; }
   if (d == 2 && q == 6) { ker = TMOP_AddMultGradPA_2D<2, 6>; }

   if (d == 3 && q == 3) { ker = TMOP_AddMultGradPA_2D<3, 3>; }
   if (d == 3 && q == 4) { ker = TMOP_AddMultGradPA_2D<3, 4>; }
   if (d == 3 && q == 5) { ker = TMOP_AddMultGradPA_2D<3, 5>; }
   if (d == 3 && q == 6) { ker = TMOP_AddMultGradPA_2D<3, 6>; }

   if (d == 4 && q == 4) { ker = TMOP_AddMultGradPA_2D<4, 4>; }
   if (d == 4 && q == 5) { ker = TMOP_AddMultGradPA_2D<4, 5>; }
   if (d == 4 && q == 6) { ker = TMOP_AddMultGradPA_2D<4, 6>; }

   if (d == 5 && q == 5) { ker = TMOP_AddMultGradPA_2D<5, 5>; }
   if (d == 5 && q == 6) { ker = TMOP_AddMultGradPA_2D<5, 6>; }

   ker(NE, B, G, J, H, X, Y, d, q, 4);
}

} // namespace mfem
