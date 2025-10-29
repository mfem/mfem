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
void TMOP_AddMultGradPA_3D(const int NE,
                           const real_t *b,
                           const real_t *g,
                           const DeviceTensor<6, const real_t> &J,
                           const DeviceTensor<8, const real_t> &H,
                           const DeviceTensor<5, const real_t> &X,
                           DeviceTensor<5> &Y, const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
      kernels::internal::vd_regs3d_t<3, 3, MQ1> r0, r1;

      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);

      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);

               // Jrt = Jtr^{-1}
               real_t Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               const real_t Jpr[9] =
               {
                  r1(0, 0, qz, qy, qx), r1(1, 0, qz, qy, qx), r1(2, 0, qz, qy, qx),
                  r1(0, 1, qz, qy, qx), r1(1, 1, qz, qy, qx), r1(2, 1, qz, qy, qx),
                  r1(0, 2, qz, qy, qx), r1(1, 2, qz, qy, qx), r1(2, 2, qz, qy, qx)
               };

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               real_t Jpt[9];
               kernels::Mult(3, 3, 3, Jpr, Jrt, Jpt);

               // B = Jpt : H
               real_t B[9];
               DeviceMatrix M(B, 3, 3);
               ConstDeviceMatrix J(Jpt, 3, 3);
               for (int i = 0; i < 3; i++)
               {
                  for (int j = 0; j < 3; j++)
                  {
                     M(i, j) = 0.0;
                     for (int r = 0; r < 3; r++)
                     {
                        for (int c = 0; c < 3; c++)
                        {
                           M(i, j) += H(r, c, i, j, qx, qy, qz, e) * J(r, c);
                        }
                     }
                  }
               }

               // Y +=  DS . M^t += DSh . (Jrt . M^t)
               real_t A[9];
               kernels::MultABt(3, 3, 3, Jrt, B, A);
               r0(0,0, qz,qy,qx) = A[0], r0(0,1, qz,qy,qx) = A[1], r0(0,2, qz,qy,qx) = A[2];
               r0(1,0, qz,qy,qx) = A[3], r0(1,1, qz,qy,qx) = A[4], r0(1,2, qz,qy,qx) = A[5];
               r0(2,0, qz,qy,qx) = A[6], r0(2,1, qz,qy,qx) = A[7], r0(2,2, qz,qy,qx) = A[8];
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1);
      kernels::internal::WriteDofs3d(e, D1D, r1, Y);
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPMultGradKernels3D, TMOP_AddMultGradPA_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPMultGradKernels3D);

void TMOP_Integrator::AddMultGradPA_3D(const Vector &R, Vector &C) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto *b = PA.maps->B.Read(), *g = PA.maps->G.Read();
   const auto J = Reshape(PA.Jtr.Read(), 3, 3, q, q, q, NE);
   const auto X = Reshape(R.Read(), d, d, d, 3, NE);
   const auto H = Reshape(PA.H.Read(), 3, 3, 3, 3, q, q, q, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, d, 3, NE);

   TMOPMultGradKernels3D::Run(d, q, NE, b, g, J, H, X, Y, d, q);
}

} // namespace mfem
