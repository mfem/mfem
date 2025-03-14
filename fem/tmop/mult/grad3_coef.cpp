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
void TMOP_AddMultGradPA_C0_3D(const int NE,
                              const ConstDeviceMatrix &b,
                              const DeviceTensor<6, const real_t> &H0,
                              const DeviceTensor<5, const real_t> &X,
                              DeviceTensor<5> &Y,
                              const int d1d,
                              const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DIM = 3;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;

      MFEM_SHARED real_t B[MQ1 * MD1];
      MFEM_SHARED real_t sm0[3][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[3][MDQ * MDQ * MDQ];

      kernels::internal::sm::LoadX<MDQ>(e, D1D, X, sm0);
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, b, B);

      kernels::internal::sm::EvalX<MD1, MQ1>(D1D, Q1D, B, sm0, sm1);
      kernels::internal::sm::EvalY<MD1, MQ1>(D1D, Q1D, B, sm1, sm0);
      kernels::internal::sm::EvalZ<MD1, MQ1>(D1D, Q1D, B, sm0, sm1);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               // Xh = X^T . Sh
               real_t Xh[3];
               kernels::internal::PullEval<MQ1>(Q1D, qx, qy, qz, sm1, Xh);

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
               kernels::internal::sm::PushEval<MQ1>(Q1D, qx, qy, qz, p2, sm0);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1, MQ1>(D1D, Q1D, b, B);
      kernels::internal::sm::EvalXt<MD1, MQ1>(D1D, Q1D, B, sm0, sm1);
      kernels::internal::sm::EvalYt<MD1, MQ1>(D1D, Q1D, B, sm1, sm0);
      kernels::internal::sm::EvalZt<MD1, MQ1>(D1D, Q1D, B, sm0, Y, e);
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPMultGradCoefKernels3D, TMOP_AddMultGradPA_C0_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPMultGradCoefKernels3D);

void TMOP_Integrator::AddMultGradPA_C0_3D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto H0 = Reshape(PA.H0.Read(), DIM, DIM, q, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto X = Reshape(R.Read(), d, d, d, DIM, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, d, DIM, NE);

   TMOPMultGradCoefKernels3D::Run(d, q, NE, B, H0, X, Y, d, q);
}

} // namespace mfem
