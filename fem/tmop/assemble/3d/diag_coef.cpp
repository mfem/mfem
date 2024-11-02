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

#include "../../pa.hpp"
#include "../../../tmop.hpp"
#include "../../../kernel_dispatch.hpp"
#include "../../../../general/forall.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AssembleDiagonalPA_C0_3D(const int NE,
                                   const ConstDeviceMatrix &B,
                                   const DeviceTensor<6, const real_t> &H0,
                                   DeviceTensor<5> &D,
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
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t qqd[MQ1 * MQ1 * MD1];
      MFEM_SHARED real_t qdd[MQ1 * MD1 * MD1];
      DeviceTensor<3, real_t> QQD(qqd, MQ1, MQ1, MD1);
      DeviceTensor<3, real_t> QDD(qdd, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         // first tensor contraction, along z direction
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(dz, z, D1D)
               {
                  QQD(qx, qy, dz) = 0.0;
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t Bz = B(qz, dz);
                     QQD(qx, qy, dz) += Bz * H0(v, v, qx, qy, qz, e) * Bz;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
         // second tensor contraction, along y direction
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(dz, z, D1D)
            {
               MFEM_FOREACH_THREAD(dy, y, D1D)
               {
                  QDD(qx, dy, dz) = 0.0;
                  MFEM_UNROLL(MQ1)
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t By = B(qy, dy);
                     QDD(qx, dy, dz) += By * QQD(qx, qy, dz) * By;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
         // third tensor contraction, along x direction
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t d = 0.0;
                  MFEM_UNROLL(MQ1)
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t Bx = B(qx, dx);
                     d += Bx * QDD(qx, dy, dz) * Bx;
                  }
                  D(dx, dy, dz, v, e) += d;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

using kernel_t = decltype(&TMOP_AssembleDiagonalPA_C0_3D<>);

MFEM_REGISTER_KERNELS(TMOPAssembleDiagCoef3D, kernel_t, (int, int));

template <int D, int Q>
kernel_t TMOPAssembleDiagCoef3D::Kernel()
{
   return TMOP_AssembleDiagonalPA_C0_3D<D, Q>;
}

kernel_t TMOPAssembleDiagCoef3D::Fallback(int, int)
{
   return TMOP_AssembleDiagonalPA_C0_3D<>;
}

void TMOP_Integrator::AssembleDiagonalPA_C0_3D(Vector &diagonal) const
{
   constexpr int DIM = 3;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto H0 = Reshape(PA.H0.Read(), DIM, DIM, q, q, q, NE);
   auto D = Reshape(diagonal.ReadWrite(), d, d, d, DIM, NE);

   const static auto specialized_kernels = []
   { return KernelSpecializations<TMOPAssembleDiagCoef3D>(); }();

   TMOPAssembleDiagCoef3D::Run(d, q, NE, B, H0, D, d, q, 4);
}

} // namespace mfem
