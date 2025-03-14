// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../kernels_smem.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultGradPA_3D(const int NE,
                           const ConstDeviceMatrix &B,
                           const ConstDeviceMatrix &G,
                           const DeviceTensor<6, const real_t> &J,
                           const DeviceTensor<8, const real_t> &H,
                           const DeviceTensor<5, const real_t> &X,
                           DeviceTensor<5> &Y,
                           const int d1d,
                           const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_TMOP_1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_TMOP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_TMOP_1D;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;

      MFEM_SHARED real_t BG[2][MQ1 * MD1];
      MFEM_SHARED real_t sm0[9][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[9][MDQ * MDQ * MDQ];

      kernels::internal::sm::LoadX<MDQ>(e, D1D, X, sm0);
      kernels::internal::LoadBG<MD1, MQ1>(D1D, Q1D, B, G, BG);

      kernels::internal::sm::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
      kernels::internal::sm::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
      kernels::internal::sm::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);

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
               kernels::internal::sm::PullGrad<MDQ>(Q1D, qx, qy, qz, sm1, Jpr);

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
               kernels::internal::sm::PushGrad<MDQ>(Q1D, qx, qy, qz, A, sm0);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBGt<MD1, MQ1>(D1D, Q1D, B, G, BG);
      kernels::internal::sm::GradZt<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
      kernels::internal::sm::GradYt<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
      kernels::internal::sm::GradXt<MD1, MQ1>(D1D, Q1D, BG, sm0, Y, e);
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPMultGradKernels3D, TMOP_AddMultGradPA_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPMultGradKernels3D);

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

   TMOPMultGradKernels3D::Run(d, q, NE, B, G, J, H, X, Y, d, q);
}

} // namespace mfem
