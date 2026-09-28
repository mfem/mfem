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
#pragma once

/** @file nddual.hpp
    Nedelec 2x2 face dual on E-vectors and on simplex MMA smem tiles.
    Shared by the standalone Fo kernel and fused GEMM apply.
*/

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

enum class NdDofTransOp : int { InvPrimal, Dual, Primal, InvDual };

/** Face-orientation dual for one MMA batch. Empty / p<2 → Active() is false. */
struct NdDualCtx
{
   const int *fo = nullptr; ///< Fo(f,e) packed nfaces*NE, first index fastest
   int nfaces = 0;
   int face_base = 0;       ///< nedges * p
   int ntdofs = 0;          ///< p*(p-1) interior face dofs

   MFEM_HOST_DEVICE bool Active() const
   {
      return fo != nullptr && ntdofs >= 2;
   }
};

// Same layout as ND_DofTransformation::T_data / TInv_data (column-major 2x2).
MFEM_HOST_DEVICE inline void NdApplyFacePair(int mode, int ori,
                                             real_t &x0, real_t &x1)
{
   constexpr real_t T[24] =
   {
      1.0,  0.0,  0.0,  1.0,
      -1.0, -1.0,  0.0,  1.0,
      0.0,  1.0, -1.0, -1.0,
      1.0,  0.0, -1.0, -1.0,
      -1.0, -1.0,  1.0,  0.0,
      0.0,  1.0,  1.0,  0.0
   };
   constexpr real_t TInv[24] =
   {
      1.0,  0.0,  0.0,  1.0,
      -1.0, -1.0,  0.0,  1.0,
      -1.0, -1.0,  1.0,  0.0,
      1.0,  0.0, -1.0, -1.0,
      0.0,  1.0, -1.0, -1.0,
      0.0,  1.0,  1.0,  0.0
   };
   const int o = 4 * ori;
   const real_t a00 = (mode <= 1) ? TInv[o] : T[o];
   const real_t a10 = (mode <= 1) ? TInv[o + 1] : T[o + 1];
   const real_t a01 = (mode <= 1) ? TInv[o + 2] : T[o + 2];
   const real_t a11 = (mode <= 1) ? TInv[o + 3] : T[o + 3];
   real_t y0, y1;
   if (mode == 0 || mode == 2) // Mult
   {
      y0 = a00 * x0 + a01 * x1;
      y1 = a10 * x0 + a11 * x1;
   }
   else // MultTranspose
   {
      y0 = a00 * x0 + a10 * x1;
      y1 = a01 * x0 + a11 * x1;
   }
   x0 = y0;
   x1 = y1;
}

/** Apply 2x2 face maps on smem XY[r + x_ld * b] for a batch of elements. */
MFEM_HOST_DEVICE inline void ApplyNdDofTransSmem(
   int mode, NdDualCtx ctx,
   real_t *XY, int x_ld, int ndof,
   int e0, int NE, int nb,
   int tid, int nthreads)
{
   if (!ctx.Active()) { return; }
   (void)ndof;
   const int npairs = ctx.ntdofs / 2;
   const int nwork = ctx.nfaces * npairs;
   for (int idx = tid; idx < nwork * nb; idx += nthreads)
   {
      const int b = idx / nwork;
      const int w = idx - b * nwork;
      const int e = e0 + b;
      if (e >= NE) { continue; }
      const int f = w / npairs;
      const int i = w - f * npairs;
      const int ori = ctx.fo[f + ctx.nfaces * e];
      const int r = ctx.face_base + f * ctx.ntdofs + 2 * i;
      real_t x0 = XY[r + x_ld * b];
      real_t x1 = XY[r + 1 + x_ld * b];
      NdApplyFacePair(mode, ori, x0, x1);
      XY[r + x_ld * b] = x0;
      XY[r + 1 + x_ld * b] = x1;
   }
}

MFEM_HOST_DEVICE inline void ZeroSmemTile(real_t *XY, int x_ld, int nb,
                                          int tid, int nthreads)
{
   for (int i = tid; i < x_ld * nb; i += nthreads) { XY[i] = real_t(0); }
}

MFEM_HOST_DEVICE inline void AddSmemTileToY(const real_t *XY, real_t *y,
                                            int ndof, int x_ld,
                                            int e0, int NE, int nb,
                                            int tid, int nthreads)
{
   for (int idx = tid; idx < ndof * nb; idx += nthreads)
   {
      const int b = idx / ndof;
      const int r = idx - b * ndof;
      const int e = e0 + b;
      if (e < NE) { y[r + ndof * e] += XY[r + x_ld * b]; }
   }
}

} // namespace internal

/// \endcond

} // namespace mfem
