// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

template <int T_D1D = 0, int T_Q1D = 0>
static void VBFEvalAssemble2D(const int vdim, const int nbe, const int d,
                              const int q, const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const real_t *n, const Vector &coeff,
                              const real_t sign, real_t *y)
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
                              const real_t *n, const Vector &coeff,
                              const real_t sign, real_t *y)
{
   static constexpr int VDIM = 3;
   MFEM_VERIFY(vdim == VDIM, "vdim must be " << VDIM);

   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto detJ = Reshape(detj, q, q, nbe);
   const auto N = Reshape(n, q, q, 3, nbe);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == 1;
   const auto C = cst ? Reshape(F,1,1,1,1) : Reshape(F,1,q,q,nbe);
   auto Y = Reshape(y, d, d, VDIM, nbe);

   // The default LEGACY rule has q < d for p > 1: use max(d, q) threads and
   // max(D1D, Q1D) storage, as the 2D tile holds both dofs and points.
   constexpr int T_DQ = T_D1D > T_Q1D ? T_D1D : T_Q1D;
   const int dq = d > q ? d : q;
   mfem::forall_2D<T_DQ*T_DQ>(nbe, dq, dq, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; }

      constexpr int MDQ = T_DQ > 0 ? kernels::internal::SetMaxOf(T_DQ) :
                          DofQuadLimits::MAX_D1D > DofQuadLimits::MAX_Q1D ?
                          DofQuadLimits::MAX_D1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED real_t sB[MDQ][MDQ], smem[MDQ][MDQ];
      kernels::internal::v_regs2d_t<VDIM, MDQ> r0, r1;
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
   static_assert(DIM == 2 || DIM == 3, "only 2D and 3D are supported");
   if constexpr (DIM == 2) { return VBFEvalAssemble2D<T_D1D, T_Q1D>; }
   else { return VBFEvalAssemble3D<T_D1D, T_Q1D>; }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

#endif // MFEM_LININTEG_BOUNDARY_KERNELS_HPP
