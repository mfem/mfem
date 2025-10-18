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

template <int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorMassApply2D(const int NE,
                             const int coeff_vdim,
                             const Array<real_t> &b,
                             const Vector &d,
                             const Vector &x,
                             Vector &y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   static constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == DIM;
   const bool matrix_coeff = coeff_vdim == DIM*DIM;

   const auto B = b.Read();
   const auto D = Reshape(d.Read(), Q1D, Q1D, coeff_vdim, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t Qx = r1[0][qy][qx];
            const real_t Qy = r1[1][qy][qx];
            const real_t D0 = D(qx, qy, 0, e);

            if (const_coeff)
            {
               r0[0][qy][qx] = D0 * Qx;
               r0[1][qy][qx] = D0 * Qy;
            }
            if (vector_coeff)
            {
               const real_t D1 = D(qx, qy, 1, e);
               r0[0][qy][qx] = D0 * Qx;
               r0[1][qy][qx] = D1 * Qy;
            }
            if (matrix_coeff)
            {
               const real_t D1 = D(qx, qy, 1, e);
               const real_t D2 = D(qx, qy, 2, e);
               const real_t D3 = D(qx, qy, 3, e);
               r0[0][qy][qx] = D0 * Qx + D1 * Qy;
               r0[1][qy][qx] = D2 * Qx + D3 * Qy;
            }
         }
      }
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, D1D, r1, Y);
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorMassApply3D(const int NE,
                             const int coeff_vdim,
                             const Array<real_t> &b,
                             const Vector &d,
                             const Vector &x,
                             Vector &y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   static constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == VDIM;
   const bool matrix_coeff = coeff_vdim == VDIM*VDIM;

   const auto B = b.Read();
   const auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, coeff_vdim, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r1);

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t Qx = r1[0][qz][qy][qx];
               const real_t Qy = r1[1][qz][qy][qx];
               const real_t Qz = r1[2][qz][qy][qx];
               const real_t D0 = D(qx, qy, qz, 0, e);
               if (const_coeff)
               {
                  r0[0][qz][qy][qx] = D0 * Qx;
                  r0[1][qz][qy][qx] = D0 * Qy;
                  r0[2][qz][qy][qx] = D0 * Qz;
               }
               if (vector_coeff)
               {
                  const real_t D1 = D(qx, qy, qz, 1, e);
                  const real_t D2 = D(qx, qy, qz, 2, e);
                  r0[0][qz][qy][qx] = D0 * Qx;
                  r0[1][qz][qy][qx] = D1 * Qy;
                  r0[2][qz][qy][qx] = D2 * Qz;
               }
               if (matrix_coeff)
               {
                  const real_t D1 = D(qx, qy, qz, 1, e);
                  const real_t D2 = D(qx, qy, qz, 2, e);
                  const real_t D3 = D(qx, qy, qz, 3, e);
                  const real_t D4 = D(qx, qy, qz, 4, e);
                  const real_t D5 = D(qx, qy, qz, 5, e);
                  const real_t D6 = D(qx, qy, qz, 6, e);
                  const real_t D7 = D(qx, qy, qz, 7, e);
                  const real_t D8 = D(qx, qy, qz, 8, e);
                  r0[0][qz][qy][qx] = D0 * Qx + D1 * Qy + D2 * Qz;
                  r0[1][qz][qy][qx] = D3 * Qx + D4 * Qy + D5 * Qz;
                  r0[2][qz][qy][qx] = D6 * Qx + D7 * Qy + D8 * Qz;
               }
            }
         }
      }
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs3d(e, D1D, r1, Y);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
VectorMassIntegrator::VectorMassAddMultPAType
VectorMassIntegrator::VectorMassAddMultPA::Kernel()
{
   if (DIM == 2)
   {
      return internal::SmemPAVectorMassApply2D<T_D1D,T_Q1D>;
   }
   else if (DIM == 3)
   {
      return internal::SmemPAVectorMassApply3D<T_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

inline VectorMassIntegrator::VectorMassAddMultPAType
VectorMassIntegrator::VectorMassAddMultPA::Fallback(int dim, int d1d, int q1d)
{
   if (dim == 2)
   {
      return internal::SmemPAVectorMassApply2D;
   }
   else if (dim == 3)
   {
      return internal::SmemPAVectorMassApply3D;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
