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
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultGradPA_2D(const int NE,
                           const real_t *b,
                           const real_t *g,
                           const DeviceTensor<5, const real_t> &J,
                           const DeviceTensor<7, const real_t> &H,
                           const DeviceTensor<4, const real_t> &X,
                           DeviceTensor<4> &Y,
                           const int d1d,
                           const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      kernels::internal::vd_regs2d_t<2, 2, MQ1> r0, r1;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t *Jtr = &J(0, 0, qx, qy, e);

            // Jrt = Jtr^{-1}
            real_t Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            const real_t Jpr[4] =
            {
               r1[0][0][qy][qx], r1[1][0][qy][qx],
               r1[0][1][qy][qx], r1[1][1][qy][qx]
            };

            // Jpt = Jpr . Jrt
            real_t Jpt[4];
            kernels::Mult(2, 2, 2, Jpr, Jrt, Jpt);

            // B = Jpt : H
            real_t B[4];
            DeviceMatrix M(B, 2, 2);
            ConstDeviceMatrix J(Jpt, 2, 2);
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               {
                  M(i, j) = 0.0;
                  for (int r = 0; r < 2; r++)
                  {
                     for (int c = 0; c < 2; c++)
                     {
                        M(i, j) += H(r, c, i, j, qx, qy, e) * J(r, c);
                     }
                  }
               }
            }
            // C = Jrt . B
            real_t C[4];
            kernels::MultABt(2, 2, 2, Jrt, B, C);
            r0[0][0][qy][qx] = C[0], r0[0][1][qy][qx] = C[1];
            r0[1][0][qy][qx] = C[2], r0[1][1][qy][qx] = C[3];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1);
      kernels::internal::WriteDofs2d(e, D1D, r1, Y);
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultGradKernels, TMOP_AddMultGradPA_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultGradKernels);

void TMOP_Integrator::AddMultGradPA_2D(const Vector &R, Vector &C) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto *b = PA.maps->B.Read(), *g = PA.maps->G.Read();
   const auto J = Reshape(PA.Jtr.Read(), 2, 2, q, q, NE);
   const auto H = Reshape(PA.H.Read(), 2, 2, 2, 2, q, q, NE);
   const auto X = Reshape(R.Read(), d, d, 2, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, 2, NE);

   TMOPMultGradKernels::Run(d, q, NE, b, g, J, H, X, Y, d, q);
}

} // namespace mfem
