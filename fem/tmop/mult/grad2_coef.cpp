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

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AddMultGradPA_C0_2D(const int NE,
                              const ConstDeviceMatrix &B,
                              const DeviceTensor<5, const real_t> &H0,
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

      MFEM_SHARED real_t sB[MQ1 * MD1];

      MFEM_SHARED real_t XY[2][NBZ][MD1 * MD1];
      MFEM_SHARED real_t DQ[2][NBZ][MD1 * MQ1];
      MFEM_SHARED real_t QQ[2][NBZ][MQ1 * MQ1];

      kernels::internal::LoadX<MD1, NBZ>(e, D1D, X, XY);
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, B, sB);

      kernels::internal::EvalX<MD1, MQ1, NBZ>(D1D, Q1D, sB, XY, DQ);
      kernels::internal::EvalY<MD1, MQ1, NBZ>(D1D, Q1D, sB, DQ, QQ);

      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            // Xh = X^T . Sh
            real_t Xh[2];
            kernels::internal::PullEval<MQ1, NBZ>(Q1D, qx, qy, QQ, Xh);

            real_t H_data[4];
            DeviceMatrix H(H_data, 2, 2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++) { H(i, j) = H0(i, j, qx, qy, e); }
            }

            // p2 = H . Xh
            real_t p2[2];
            kernels::Mult(2, 2, H_data, Xh, p2);
            kernels::internal::PushEval<MQ1, NBZ>(Q1D, qx, qy, p2, QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1, MQ1>(D1D, Q1D, B, sB);
      kernels::internal::EvalXt<MD1, MQ1, NBZ>(D1D, Q1D, sB, QQ, DQ);
      kernels::internal::EvalYt<MD1, MQ1, NBZ>(D1D, Q1D, sB, DQ, Y, e);
   });
}

TMOP_REGISTER_KERNELS(TMOPMultGradCoefKernels, TMOP_AddMultGradPA_C0_2D);

void TMOP_Integrator::AddMultGradPA_C0_2D(const Vector &R, Vector &C) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto H0 = Reshape(PA.H0.Read(), DIM, DIM, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto X = Reshape(R.Read(), d, d, DIM, NE);
   auto Y = Reshape(C.ReadWrite(), d, d, DIM, NE);

   const static auto specialized_kernels = []
   { return tmop::KernelSpecializations<TMOPMultGradCoefKernels>(); }();

   TMOPMultGradCoefKernels::Run(d, q, NE, B, H0, X, Y, d, q, 4);
}

} // namespace mfem
