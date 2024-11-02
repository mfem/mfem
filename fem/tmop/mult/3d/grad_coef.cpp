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

#include "../../../tmop.hpp"
#include "../../../kernels.hpp"
#include "../../../../general/forall.hpp"
#include "../../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultGradPA_C0_3D(const int NE,
                              const ConstDeviceMatrix &b,
                              const DeviceTensor<6, const real_t> &H0,
                              const DeviceTensor<5, const real_t> &X,
                              DeviceTensor<5> &Y,
                              const int d1d,
                              const int q1d,
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

      MFEM_SHARED real_t B[MQ1 * MD1];

      MFEM_SHARED real_t DDD[3][MD1 * MD1 * MD1];
      MFEM_SHARED real_t DDQ[3][MD1 * MD1 * MQ1];
      MFEM_SHARED real_t DQQ[3][MD1 * MQ1 * MQ1];
      MFEM_SHARED real_t QQQ[3][MQ1 * MQ1 * MQ1];

      kernels::internal::LoadX<MD1>(e, D1D, X, DDD);
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, b, B);

      kernels::internal::EvalX<MD1, MQ1>(D1D, Q1D, B, DDD, DDQ);
      kernels::internal::EvalY<MD1, MQ1>(D1D, Q1D, B, DDQ, DQQ);
      kernels::internal::EvalZ<MD1, MQ1>(D1D, Q1D, B, DQQ, QQQ);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               // Xh = X^T . Sh
               real_t Xh[3];
               kernels::internal::PullEval<MQ1>(Q1D, qx, qy, qz, QQQ, Xh);

               real_t H_data[9];
               DeviceMatrix H(H_data, 3, 3);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     H(i, j) = H0(i, j, qx, qy, qz, e);
                  }
               }

               // p2 = H . Xh
               real_t p2[3];
               kernels::Mult(3, 3, H_data, Xh, p2);
               kernels::internal::PushEval<MQ1>(Q1D, qx, qy, qz, p2, QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1, MQ1>(D1D, Q1D, b, B);
      kernels::internal::EvalXt<MD1, MQ1>(D1D, Q1D, B, QQQ, DQQ);
      kernels::internal::EvalYt<MD1, MQ1>(D1D, Q1D, B, DQQ, DDQ);
      kernels::internal::EvalZt<MD1, MQ1>(D1D, Q1D, B, DDQ, Y, e);
   });
}

void TMOP_Integrator::AddMultGradPA_C0_3D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto H0 = Reshape(PA.H0.Read(), DIM, DIM, q, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto X = Reshape(R.Read(), d, d, d, DIM, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, d, DIM, NE);

   decltype(&TMOP_AddMultGradPA_C0_3D<>) ker = TMOP_AddMultGradPA_C0_3D;

   if (d == 2 && q == 2) { ker = TMOP_AddMultGradPA_C0_3D<2, 2>; }
   if (d == 2 && q == 3) { ker = TMOP_AddMultGradPA_C0_3D<2, 3>; }
   if (d == 2 && q == 4) { ker = TMOP_AddMultGradPA_C0_3D<2, 4>; }
   if (d == 2 && q == 5) { ker = TMOP_AddMultGradPA_C0_3D<2, 5>; }
   if (d == 2 && q == 6) { ker = TMOP_AddMultGradPA_C0_3D<2, 6>; }

   if (d == 3 && q == 3) { ker = TMOP_AddMultGradPA_C0_3D<3, 3>; }
   if (d == 3 && q == 4) { ker = TMOP_AddMultGradPA_C0_3D<3, 4>; }
   if (d == 3 && q == 5) { ker = TMOP_AddMultGradPA_C0_3D<3, 5>; }
   if (d == 3 && q == 6) { ker = TMOP_AddMultGradPA_C0_3D<3, 6>; }

   if (d == 4 && q == 4) { ker = TMOP_AddMultGradPA_C0_3D<4, 4>; }
   if (d == 4 && q == 5) { ker = TMOP_AddMultGradPA_C0_3D<4, 5>; }
   if (d == 4 && q == 6) { ker = TMOP_AddMultGradPA_C0_3D<4, 6>; }

   if (d == 5 && q == 5) { ker = TMOP_AddMultGradPA_C0_3D<5, 5>; }
   if (d == 5 && q == 6) { ker = TMOP_AddMultGradPA_C0_3D<5, 6>; }

   ker(NE, B, H0, X, Y, d, q, 4);
}

} // namespace mfem
