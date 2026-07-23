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

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

// Shared memory PA Divergence Apply 2D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
inline void SmemPADivergenceApply2D(const int NE,
                                    const Array<real_t> &b_,
                                    const Array<real_t> &g_,
                                    const Array<real_t> &bt_,
                                    const Vector &q_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int tr_d1d = 0,
                                    const int te_d1d = 0,
                                    const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, 2, 2, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, 2, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::vd_regs2d_t<2, 2, MQ1> g0, g1;
      kernels::internal::v_regs2d_t<1, MQ1> r0, r1;

      kernels::internal::LoadMatrix(TR_D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(TR_D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, TR_D1D, X, g0);
      kernels::internal::Grad2d(TR_D1D, Q1D, smem, sB, sG, g0, g1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            r0[0][qy][qx] =
               g1[0][0][qy][qx] * Q(qx, qy, 0, 0, e) +
               g1[0][1][qy][qx] * Q(qx, qy, 1, 0, e) +
               g1[1][0][qy][qx] * Q(qx, qy, 0, 1, e) +
               g1[1][1][qy][qx] * Q(qx, qy, 1, 1, e);
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TE_D1D, Q1D, Bt, sB);
      kernels::internal::EvalTranspose2d(TE_D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, TE_D1D, r1, Y);
   });
}

// Shared memory PA Divergence Apply 2D kernel transpose
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
inline void SmemPADivergenceApplyTranspose2D(const int NE,
                                             const Array<real_t> &bt,
                                             const Array<real_t> &gt,
                                             const Array<real_t> &b,
                                             const Vector &q_,
                                             const Vector &x_,
                                             Vector &y_,
                                             const int tr_d1d = 0,
                                             const int te_d1d = 0,
                                             const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, 2, 2, NE);
   const auto X  = Reshape(x_.Read(), TE_D1D, TE_D1D, 1, NE);
   auto Y  = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, 2, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::v_regs2d_t<1, MQ1> r0, r1;
      kernels::internal::vd_regs2d_t<2, 2, MQ1> g0, g1;

      kernels::internal::LoadMatrix(TE_D1D, Q1D, B, sB);
      kernels::internal::LoadDofs2d(e, TE_D1D, X, r0);
      kernels::internal::Eval2d(TE_D1D, Q1D, smem, sB, r0, r1);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            g0[0][0][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 0, 0, e);
            g0[0][1][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 1, 0, e);
            g0[1][0][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 0, 1, e);
            g0[1][1][qy][qx] = r1[0][qy][qx] * Q(qx, qy, 1, 1, e);
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Bt, sB);
      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Gt, sG);
      kernels::internal::GradTranspose2d(TR_D1D, Q1D, smem, sB, sG, g0, g1);
      kernels::internal::WriteDofs2d(e, TR_D1D, g1, Y);
   });
}

// Shared memory PA Divergence Apply 3D kernel transpose
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
inline void SmemPADivergenceApplyTranspose3D(const int NE,
                                             const Array<real_t> &bt,
                                             const Array<real_t> &gt,
                                             const Array<real_t> &b,
                                             const Vector &q_,
                                             const Vector &x_,
                                             Vector &y_,
                                             int tr_d1d = 0,
                                             int te_d1d = 0,
                                             int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::v_regs3d_t<1, MQ1> r0, r1;
      kernels::internal::vd_regs3d_t<3, 3, MQ1> g0, g1;

      kernels::internal::LoadMatrix(TE_D1D, Q1D, B, sB);
      kernels::internal::LoadDofs3d(e, TE_D1D, X, r0);
      kernels::internal::Eval3d(TE_D1D, Q1D, smem, sB, r0, r1);

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const auto r = r1[0][qz][qy][qx];
               g0[0][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 0, e);
               g0[0][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 0, e);
               g0[0][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 0, e);

               g0[1][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 1, e);
               g0[1][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 1, e);
               g0[1][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 1, e);

               g0[2][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 2, e);
               g0[2][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 2, e);
               g0[2][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Bt, sB);
      kernels::internal::LoadMatrix<MQ1,true>(TR_D1D, Q1D, Gt, sG);
      kernels::internal::GradTranspose3d(TR_D1D, Q1D, smem, sB, sG, g0, g1);
      kernels::internal::WriteDofs3d(e, TR_D1D, g1, Y);
   });
}

// Shared memory PA Divergence Apply 3D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
inline void SmemPADivergenceApply3D(const int NE,
                                    const Array<real_t> &b_,
                                    const Array<real_t> &g_,
                                    const Array<real_t> &bt_,
                                    const Vector &q_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int tr_d1d = 0,
                                    const int te_d1d = 0,
                                    const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = Reshape(q_.Read(), Q1D, Q1D, Q1D, 3,3, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

      kernels::internal::vd_regs3d_t<3, 3, MQ1> g0, g1;
      kernels::internal::v_regs3d_t<1, MQ1> r0, r1;

      kernels::internal::LoadMatrix(TR_D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(TR_D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, TR_D1D, X, g0);
      kernels::internal::Grad3d(TR_D1D, Q1D, smem, sB, sG, g0, g1);

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               r0[0][qz][qy][qx] =
                  // c = 0
                  g1[0][0][qz][qy][qx] * Q(qx, qy, qz, 0, 0, e) +
                  g1[0][1][qz][qy][qx] * Q(qx, qy, qz, 1, 0, e) +
                  g1[0][2][qz][qy][qx] * Q(qx, qy, qz, 2, 0, e) +
                  // c = 1
                  g1[1][0][qz][qy][qx] * Q(qx, qy, qz, 0, 1, e) +
                  g1[1][1][qz][qy][qx] * Q(qx, qy, qz, 1, 1, e) +
                  g1[1][2][qz][qy][qx] * Q(qx, qy, qz, 2, 1, e) +
                  // c = 2
                  g1[2][0][qz][qy][qx] * Q(qx, qy, qz, 0, 2, e) +
                  g1[2][1][qz][qy][qx] * Q(qx, qy, qz, 1, 2, e) +
                  g1[2][2][qz][qy][qx] * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::LoadMatrix<MQ1, true>(TE_D1D, Q1D, Bt, sB);
      kernels::internal::EvalTranspose3d(TE_D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs3d(e, TE_D1D, r1, Y);
   });
}

} // namespace internal

template<int DIM, int T_TR_D1D, int T_TE_D1D, int T_Q1D>
VectorDivergenceIntegrator::VectorDivergenceAddMultPAType
VectorDivergenceIntegrator::VectorDivergenceAddMultPA::Kernel()
{
   static_assert(T_TR_D1D <= T_Q1D && T_TE_D1D <= T_Q1D);
   if constexpr (DIM == 2)
   {
      return internal::SmemPADivergenceApply2D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPADivergenceApply3D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

inline VectorDivergenceIntegrator::VectorDivergenceAddMultPAType
VectorDivergenceIntegrator::VectorDivergenceAddMultPA::Fallback
(int dim, int tr_d1d, int te_d1d, int q1d)
{
   MFEM_VERIFY(tr_d1d <= q1d && te_d1d <= q1d, "");
   MFEM_VERIFY(tr_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(te_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   if (dim == 2)
   {
      return internal::SmemPADivergenceApply2D;
   }
   else if (dim == 3)
   {
      return internal::SmemPADivergenceApply3D;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

template<int DIM, int T_TR_D1D, int T_TE_D1D, int T_Q1D>
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePAType
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::Kernel()
{
   static_assert(T_TR_D1D <= T_Q1D && T_TE_D1D <= T_Q1D);
   if constexpr (DIM == 2)
   {
      return internal::SmemPADivergenceApplyTranspose2D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPADivergenceApplyTranspose3D<T_TR_D1D, T_TE_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

inline VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePAType
VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::Fallback
(int dim, int tr_d1d, int te_d1d, int q1d)
{
   MFEM_VERIFY(tr_d1d <= q1d && te_d1d <= q1d, "");
   MFEM_VERIFY(tr_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(te_d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   if (dim == 2)
   {
      return internal::SmemPADivergenceApplyTranspose2D;
   }
   else if (dim == 3)
   {
      return internal::SmemPADivergenceApplyTranspose3D;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
