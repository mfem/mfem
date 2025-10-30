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

namespace mfem
{

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_AssembleDiagPA_C0_3D(const int NE,
                               const ConstDeviceMatrix &B,
                               const DeviceTensor<6, const real_t> &H0,
                               DeviceTensor<5> &D,
                               const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[MQ1][MQ1];
      kernels::internal::s_regs3d_t<MQ1> r0, r1;

      for (int v = 0; v < 3; ++v)
      {
         // first tensor contraction, along z direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = B(qz, dz);
                     u += Bz * H0(v, v, qx, qy, qz, e) * Bz;
                  }
                  r0[dz][qy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // second tensor contraction, along y direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[qy][qx] = r0[dz][qy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy);
                     u += By * smem[qy][qx] * By;
                  }
                  r1[dz][dy][qx] = u;
               }
            }
            MFEM_SYNC_THREAD;
         }

         // third tensor contraction, along x direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  smem[dy][qx] = r1[dz][dy][qx];
               }
            }
            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx);
                     u += Bx * smem[dy][qx] * Bx;
                  }
                  D(dx, dy, dz, v, e) += u;
               }
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPAssembleDiagCoef3D, TMOP_AssembleDiagPA_C0_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPAssembleDiagCoef3D);

void TMOP_Integrator::AssembleDiagonalPA_C0_3D(Vector &diagonal) const
{
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto H0 = Reshape(PA.H0.Read(), 3, 3, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, 3, NE);

   TMOPAssembleDiagCoef3D::Run(d, q, NE, B, H0, D, d, q);
}

} // namespace mfem
