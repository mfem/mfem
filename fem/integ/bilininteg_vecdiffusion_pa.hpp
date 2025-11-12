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
#pragma once

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"
#include "../kernels.hpp"

using mfem::kernels::internal::SetMaxOf;

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template<int T_SDIM = 0, int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorDiffusionApply2D(const int NE,
                                  const int coeff_vdim,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int sdim = 0,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   static constexpr int DIM = 2;
   const int SDIM = T_SDIM ? T_SDIM : sdim;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int PA_SIZE = DIM*DIM;
   const bool matrix_coeff = coeff_vdim == DIM*DIM;

   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, PA_SIZE,
                           SDIM * (matrix_coeff ? SDIM : 1), NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, SDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, SDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<3, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      for (int i = 0; i < SDIM; i++)
      {
         for (int j = 0; j < (matrix_coeff ? SDIM : 1); j++)
         {
            kernels::internal::LoadDofs2d(e, D1D, i, XE, r0);
            kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t gradX = r1[i][0][qy][qx];
                  const real_t gradY = r1[i][1][qy][qx];
                  const int k = matrix_coeff ? (j + i * SDIM) : i;
                  const real_t O11 = DE(qx,qy,0,k,e), O12 = DE(qx,qy,1,k,e);
                  const real_t O21 = DE(qx,qy,2,k,e), O22 = DE(qx,qy,3,k,e);
                  r0[i][0][qy][qx] = (O11 * gradX) + (O12 * gradY);
                  r0[i][1][qy][qx] = (O21 * gradX) + (O22 * gradY);
               } // qx
            } // qy
            MFEM_SYNC_THREAD;
            kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            const int ij =  matrix_coeff ? j : i;
            kernels::internal::WriteDofs2d(e, D1D, i, ij, r1, YE);
         } // j
      } // i
   });
}

template<int T_SDIM = 0, int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorDiffusionApply3D(const int NE,
                                  const int coeff_vdim,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int sdim = 0,
                                  const int d1d = 0,
                                  const int q1d = 0)
{

   static constexpr int DIM = 3;
   const int SDIM = T_SDIM ? T_SDIM : sdim;
   MFEM_VERIFY(SDIM == 3, "SDIM must be 3");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int PA_SIZE = DIM*DIM;
   const bool matrix_coeff = coeff_vdim == DIM*DIM;

   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, PA_SIZE,
                           SDIM * (matrix_coeff ? SDIM : 1), NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, SDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, SDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<3, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      for (int i = 0; i < SDIM; i++)
      {
         for (int j = 0; j < (matrix_coeff ? SDIM : 1); j++)
         {
            kernels::internal::LoadDofs3d(e, D1D, i, XE, r0);
            kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            for (int qz = 0; qz < Q1D; qz++)
            {
               MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
               {
                  MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
                  {
                     const real_t gradX = r1[i][0][qz][qy][qx];
                     const real_t gradY = r1[i][1][qz][qy][qx];
                     const real_t gradZ = r1[i][2][qz][qy][qx];
                     const int k = matrix_coeff ? (j + i * SDIM) : i;
                     const real_t O11 = DE(qx,qy,qz,0,k,e), O12 = DE(qx,qy,qz,1,k,e),
                                  O13 = DE(qx,qy,qz,2,k,e);
                     const real_t O22 = DE(qx,qy,qz,3,k,e), O23 = DE(qx,qy,qz,4,k,e);
                     const real_t O33 = DE(qx,qy,qz,5,k,e);
                     r0[i][0][qz][qy][qx] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                     r0[i][1][qz][qy][qx] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
                     r0[i][2][qz][qy][qx] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
                  } // qx
               } // qy
            } // qz
            MFEM_SYNC_THREAD;
            kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, i);
            const int ij =  matrix_coeff ? j : i;
            kernels::internal::WriteDofs3d(e, D1D, i, ij, r1, YE);
         } // j
      } // i
   });
}

} // namespace internal

template<int DIM, int T_SDIM, int T_D1D, int T_Q1D>
VectorDiffusionIntegrator::ApplyKernelType
VectorDiffusionIntegrator::ApplyPAKernels::Kernel()
{
   if (DIM == 2)
   {
      return internal::SmemPAVectorDiffusionApply2D<T_SDIM, T_D1D, T_Q1D>;
   }
   else if (DIM == 3)
   {
      return internal::SmemPAVectorDiffusionApply3D<T_SDIM, T_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

inline VectorDiffusionIntegrator::ApplyKernelType
VectorDiffusionIntegrator::ApplyPAKernels::Fallback(int dim, int sdim,
                                                    int d1d, int q1d)
{
   if (dim == 2)
   {
      return internal::SmemPAVectorDiffusionApply2D;
   }
   else if (dim == 3)
   {
      return internal::SmemPAVectorDiffusionApply3D;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
