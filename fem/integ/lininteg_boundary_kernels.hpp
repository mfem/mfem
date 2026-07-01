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

#ifndef MFEM_LININTEG_BOUNDARY_KERNELS_HPP
#define MFEM_LININTEG_BOUNDARY_KERNELS_HPP

#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../fem.hpp"

using mfem::kernels::internal::SetMaxOf;

/// \cond DO_NOT_DOCUMENT

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
static void VBFEvalAssemble2D(const int vdim, const int nbe, const int d,
                              const int q, const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const real_t *n, const Vector &coeff, const real_t sign, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, nbe);
   const auto N = Reshape(n, q, 2, nbe);
   const auto W = Reshape(weights, q);
   const bool cst = coeff.Size() == 1;
   const auto C = cst ? Reshape(F,1,1,1) : Reshape(F,1,q,nbe);
   auto Y = Reshape(y, d, vdim, nbe);

   mfem::forall_2D(nbe, d, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t WCdetJ[Q];
      for (int qx = 0; qx < q; ++qx)
      {
         const real_t cval = cst ? C(0,0,0) : C(0,qx,e);
         WCdetJ[qx] = W(qx) * cval * detJ(qx,e) * sign;
      }

      for (int c = 0; c < vdim; ++c)
      {
         MFEM_FOREACH_THREAD(dx, x, d)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < q; ++qx)
            {
               u += WCdetJ[qx] * N(qx,c,e) * B(qx, dx);
            }
            Y(dx,c,e) += u;
         }
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void VBFEvalAssemble3D(const int vdim, const int nbe, const int d,
                              const int q, const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const real_t *n, const Vector &coeff, const real_t sign, real_t *y)
{
   static constexpr int VDIM = 3;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto detJ = Reshape(detj, q, q, nbe);
   const auto N = Reshape(n, q, q, 3, nbe);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == 1;
   const auto C = cst ? Reshape(F,1,1,1,1) : Reshape(F,1,q,q,nbe);
   auto Y = Reshape(y, d, d, VDIM, nbe);

   mfem::forall_2D<T_Q1D*T_Q1D>(nbe, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; }

      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sB[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(d, q, b, sB);

      MFEM_FOREACH_THREAD_DIRECT(qy, y, q)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q)
         {
            const real_t cval = cst ? C(0,0,0,0) : C(0,qx,qy,e);
            const real_t wcd = W(qx,qy) * sign * cval * detJ(qx,qy,e);
            for (int c = 0; c < VDIM; ++c)
            {
               r0[c][qy][qx] = wcd * N(qx,qy,c,e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      kernels::internal::EvalTranspose2d(d, q, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, d, r1, Y);
   });
}

template <int DIM, int T_D1D, int T_Q1D>
VectorBoundaryFluxLFIntegrator::AssembleKernelType
VectorBoundaryFluxLFIntegrator::AssembleKernels::Kernel()
{
   if constexpr (DIM == 2) { return VBFEvalAssemble2D<T_D1D, T_Q1D>; }
   if constexpr (DIM == 3) { return VBFEvalAssemble3D<T_D1D, T_Q1D>; }
   MFEM_ABORT("");
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

#endif // MFEM_LININTEG_BOUNDARY_KERNELS_HPP
