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
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_AddMultGradPA_C0_3D(const int NE,
                              const real_t *b,
                              const DeviceTensor<6, const real_t> &H0,
                              const DeviceTensor<5, const real_t> &X,
                              DeviceTensor<5> &Y,
                              const int d1d,
                              const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      static constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      static constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];

      LoadMatrix(D1D, Q1D, b, sB);

      regs5d_t<3,1,MQ1> r0, r1; // vector X
      LoadDofs3d(e, D1D, X, r0);
      Eval3d(D1D, Q1D, smem, sB, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
         {
            mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
            {
               // Xh = X^T . Sh
               const real_t Xh[3] =
               {
                  r1(0, 0, qz, qy, qx),
                  r1(1, 0, qz, qy, qx),
                  r1(2, 0, qz, qy, qx)
               };

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
               r0(0,0, qz,qy,qx) = p2[0];
               r0(0,1, qz,qy,qx) = p2[1];
               r0(0,2, qz,qy,qx) = p2[2];
            });
         });
      }
      MFEM_SYNC_THREAD;
      EvalTranspose3d(D1D, Q1D, smem, sB, r0, r1);
      WriteDofs3d(e, D1D, r1, Y);
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPMultGradCoefKernels3D, TMOP_AddMultGradPA_C0_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPMultGradCoefKernels3D);

void TMOP_Integrator::AddMultGradPA_C0_3D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto H0 = Reshape(PA.H0.Read(), DIM, DIM, q, q, q, NE);
   const auto *b = PA.maps->B.Read();
   const auto X = Reshape(R.Read(), d, d, d, DIM, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, d, DIM, NE);

   TMOPMultGradCoefKernels3D::Run(d, q, NE, b, H0, X, Y, d, q);
}

} // namespace mfem
