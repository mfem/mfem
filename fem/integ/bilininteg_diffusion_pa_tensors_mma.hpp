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

#include "../bilininteg.hpp"
#include "bilininteg_pa_mma.hpp"

#ifdef MFEM_USE_LAPACK
#include <vector>
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Prefer fat 1D Lapack for tensor diffusion host apply (quad/hex). */
inline bool PreferTensorDiffLapack(int D1D, int Q1D, int NE)
{
#ifdef MFEM_USE_LAPACK
   (void)Q1D;
   return NE >= 4 && D1D >= 4;
#else
   (void)D1D; (void)Q1D; (void)NE;
   return false;
#endif
}

#ifdef MFEM_USE_LAPACK
/** One 2D diffusion tile: B/G forward, metric, Bt/Gt backward. */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensorsSumLapack2DTile(
   const int e0, const int nbe, const int NB,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y,
   real_t *xloc, real_t *BX, real_t *GX, real_t *BXt, real_t *GXt,
   real_t *gX, real_t *gY, real_t *gXp, real_t *gYp,
   real_t *A0, real_t *A1, real_t *A0t, real_t *A1t,
   real_t *Y0, real_t *Y1)
{
   constexpr int PA_SIZE = SYM ? 3 : 4;
   const int ndof = D1D * D1D;
   const int n_xy = D1D * NB;
   const int n_qy = Q1D * NB;

   const real_t *Xsrc;
   if (nbe == NB)
   {
      Xsrc = X + static_cast<size_t>(ndof) * e0;
   }
   else
   {
      std::fill(xloc, xloc + static_cast<size_t>(D1D) * n_xy, real_t(0));
      for (int b = 0; b < nbe; ++b)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               xloc[dx + D1D * (dy + D1D * b)] =
                  X[dx + D1D * (dy + D1D * (e0 + b))];
            }
         }
      }
      Xsrc = xloc;
   }

   mma::LapackGemm('N', 'N', Q1D, n_xy, D1D, real_t(1), B, Q1D,
                   Xsrc, D1D, real_t(0), BX, Q1D);
   mma::LapackGemm('N', 'N', Q1D, n_xy, D1D, real_t(1), G, Q1D,
                   Xsrc, D1D, real_t(0), GX, Q1D);

   for (int b = 0; b < NB; ++b)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            BXt[dy + D1D * (qx + Q1D * b)] = BX[qx + Q1D * (dy + D1D * b)];
            GXt[dy + D1D * (qx + Q1D * b)] = GX[qx + Q1D * (dy + D1D * b)];
         }
      }
   }
   // gX = B * GX^T-pack; gY = G * BX^T-pack
   mma::LapackGemm('N', 'N', Q1D, n_qy, D1D, real_t(1), B, Q1D,
                   GXt, D1D, real_t(0), gX, Q1D);
   mma::LapackGemm('N', 'N', Q1D, n_qy, D1D, real_t(1), G, Q1D,
                   BXt, D1D, real_t(0), gY, Q1D);

   for (int b = 0; b < nbe; ++b)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const int idx = qx + Q1D * qy;
            const real_t gx = gX[qy + Q1D * (qx + Q1D * b)];
            const real_t gy = gY[qy + Q1D * (qx + Q1D * b)];
            const real_t O11 = Dv[idx + Q1D * Q1D * (0 + PA_SIZE * (e0 + b))];
            const real_t O21 = Dv[idx + Q1D * Q1D * (1 + PA_SIZE * (e0 + b))];
            const real_t O12 = SYM ? O21
               : Dv[idx + Q1D * Q1D * (2 + PA_SIZE * (e0 + b))];
            const real_t O22 = SYM
               ? Dv[idx + Q1D * Q1D * (2 + PA_SIZE * (e0 + b))]
               : Dv[idx + Q1D * Q1D * (3 + PA_SIZE * (e0 + b))];
            gX[qy + Q1D * (qx + Q1D * b)] = O11 * gx + O12 * gy;
            gY[qy + Q1D * (qx + Q1D * b)] = O21 * gx + O22 * gy;
         }
      }
   }
   for (int b = nbe; b < NB; ++b)
   {
      for (int i = 0; i < Q1D * Q1D; ++i)
      {
         gX[i + Q1D * Q1D * b] = real_t(0);
         gY[i + Q1D * Q1D * b] = real_t(0);
      }
   }

   // Pack gX/gY to [qx, qy + Q1D*b] for Gt/Bt along x
   for (int b = 0; b < NB; ++b)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gXp[qx + Q1D * (qy + Q1D * b)] = gX[qy + Q1D * (qx + Q1D * b)];
            gYp[qx + Q1D * (qy + Q1D * b)] = gY[qy + Q1D * (qx + Q1D * b)];
         }
      }
   }
   mma::LapackGemm('N', 'N', D1D, n_qy, Q1D, real_t(1), Gt, D1D,
                   gXp, Q1D, real_t(0), A0, D1D);
   mma::LapackGemm('N', 'N', D1D, n_qy, Q1D, real_t(1), Bt, D1D,
                   gYp, Q1D, real_t(0), A1, D1D);

   // Pack A0/A1 to [qy, dx + D1D*b] then apply Bt/Gt along y
   for (int b = 0; b < NB; ++b)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            A0t[qy + Q1D * (dx + D1D * b)] = A0[dx + D1D * (qy + Q1D * b)];
            A1t[qy + Q1D * (dx + D1D * b)] = A1[dx + D1D * (qy + Q1D * b)];
         }
      }
   }
   mma::LapackGemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Bt, D1D,
                   A0t, Q1D, real_t(0), Y0, D1D);
   mma::LapackGemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Gt, D1D,
                   A1t, Q1D, real_t(0), Y1, D1D);

   // Y0/Y1 are [dy, dx + D1D*b]; scatter as Y[dx,dy] += Y0 + Y1
   for (int b = 0; b < nbe; ++b)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            Y[dx + D1D * (dy + D1D * (e0 + b))] +=
               Y0[dy + D1D * (dx + D1D * b)] + Y1[dy + D1D * (dx + D1D * b)];
         }
      }
   }
}

template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensorsSumLapack2D(
   const int NE,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   const int NB = mma::TensorLapackNB(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   const int n_xy = D1D * NB;
   const int n_qy = Q1D * NB;

   auto run_tile = [&](int tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      std::vector<real_t> xloc(static_cast<size_t>(D1D) * n_xy);
      std::vector<real_t> BX(static_cast<size_t>(Q1D) * n_xy);
      std::vector<real_t> GX(static_cast<size_t>(Q1D) * n_xy);
      std::vector<real_t> BXt(static_cast<size_t>(D1D) * n_qy);
      std::vector<real_t> GXt(static_cast<size_t>(D1D) * n_qy);
      std::vector<real_t> gX(static_cast<size_t>(Q1D) * n_qy);
      std::vector<real_t> gY(static_cast<size_t>(Q1D) * n_qy);
      std::vector<real_t> gXp(static_cast<size_t>(Q1D) * n_qy);
      std::vector<real_t> gYp(static_cast<size_t>(Q1D) * n_qy);
      std::vector<real_t> A0(static_cast<size_t>(D1D) * n_qy);
      std::vector<real_t> A1(static_cast<size_t>(D1D) * n_qy);
      std::vector<real_t> A0t(static_cast<size_t>(Q1D) * n_xy);
      std::vector<real_t> A1t(static_cast<size_t>(Q1D) * n_xy);
      std::vector<real_t> Y0(static_cast<size_t>(D1D) * n_xy);
      std::vector<real_t> Y1(static_cast<size_t>(D1D) * n_xy);
      DiffusionApplyTensorsSumLapack2DTile<D1D, Q1D, SYM>(
         e0, nbe, NB, B, G, Bt, Gt, Dv, X, Y,
         xloc.data(), BX.data(), GX.data(), BXt.data(), GXt.data(),
         gX.data(), gY.data(), gXp.data(), gYp.data(),
         A0.data(), A1.data(), A0t.data(), A1t.data(),
         Y0.data(), Y1.data());
   };

#if defined(__APPLE__)
   dispatch_apply(static_cast<size_t>(ntiles), DISPATCH_APPLY_AUTO,
                  ^(size_t tile) { run_tile(static_cast<int>(tile)); });
#else
   for (int tile = 0; tile < ntiles; ++tile) { run_tile(tile); }
#endif
}

/** 3D diffusion one element: fat RHS = D1D² (same contractions as PADiffusion). */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensorsSumLapack3DElement(
   const int e,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y,
   real_t *BX, real_t *GX, real_t *BXt, real_t *GXt,
   real_t *XY0, real_t *XY1, real_t *XY2,
   real_t *g0, real_t *g1, real_t *g2,
   real_t *QQ0, real_t *QQ1, real_t *QQ2,
   real_t *gvec0, real_t *gvec1, real_t *gvec2,
   real_t *A0, real_t *A1, real_t *A2,
   real_t *Yacc)
{
   constexpr int PA_SIZE = SYM ? 6 : 9;
   const int nd2 = D1D * D1D;
   const int nq3 = Q1D * Q1D * Q1D;
   const int ndof = D1D * nd2;
   const real_t *Xe = X + static_cast<size_t>(ndof) * e;

   mma::LapackGemm('N', 'N', Q1D, nd2, D1D, real_t(1), B, Q1D,
                   Xe, D1D, real_t(0), BX, Q1D);
   mma::LapackGemm('N', 'N', Q1D, nd2, D1D, real_t(1), G, Q1D,
                   Xe, D1D, real_t(0), GX, Q1D);

   std::fill(g0, g0 + nq3, real_t(0));
   std::fill(g1, g1 + nq3, real_t(0));
   std::fill(g2, g2 + nq3, real_t(0));

   for (int dz = 0; dz < D1D; ++dz)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            BXt[dy + D1D * qx] = BX[qx + Q1D * (dy + D1D * dz)];
            GXt[dy + D1D * qx] = GX[qx + Q1D * (dy + D1D * dz)];
         }
      }
      mma::LapackGemm('N', 'N', Q1D, Q1D, D1D, real_t(1), B, Q1D,
                      GXt, D1D, real_t(0), XY0, Q1D);
      mma::LapackGemm('N', 'N', Q1D, Q1D, D1D, real_t(1), G, Q1D,
                      BXt, D1D, real_t(0), XY1, Q1D);
      mma::LapackGemm('N', 'N', Q1D, Q1D, D1D, real_t(1), B, Q1D,
                      BXt, D1D, real_t(0), XY2, Q1D);
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t v0 = XY0[qy + Q1D * qx];
            const real_t v1 = XY1[qy + Q1D * qx];
            const real_t v2 = XY2[qy + Q1D * qx];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const int idx = qx + Q1D * (qy + Q1D * qz);
               g0[idx] += v0 * B[qz + Q1D * dz];
               g1[idx] += v1 * B[qz + Q1D * dz];
               g2[idx] += v2 * G[qz + Q1D * dz];
            }
         }
      }
   }

   for (int q = 0; q < nq3; ++q)
   {
      const real_t gx = g0[q], gy = g1[q], gz = g2[q];
      const real_t O11 = Dv[q + nq3 * (0 + PA_SIZE * e)];
      const real_t O12 = Dv[q + nq3 * (1 + PA_SIZE * e)];
      const real_t O13 = Dv[q + nq3 * (2 + PA_SIZE * e)];
      real_t O21, O22, O23, O31, O32, O33;
      if constexpr (SYM)
      {
         O21 = O12;
         O22 = Dv[q + nq3 * (3 + PA_SIZE * e)];
         O23 = Dv[q + nq3 * (4 + PA_SIZE * e)];
         O31 = O13; O32 = O23;
         O33 = Dv[q + nq3 * (5 + PA_SIZE * e)];
      }
      else
      {
         O21 = Dv[q + nq3 * (3 + PA_SIZE * e)];
         O22 = Dv[q + nq3 * (4 + PA_SIZE * e)];
         O23 = Dv[q + nq3 * (5 + PA_SIZE * e)];
         O31 = Dv[q + nq3 * (6 + PA_SIZE * e)];
         O32 = Dv[q + nq3 * (7 + PA_SIZE * e)];
         O33 = Dv[q + nq3 * (8 + PA_SIZE * e)];
      }
      g0[q] = O11 * gx + O12 * gy + O13 * gz;
      g1[q] = O21 * gx + O22 * gy + O23 * gz;
      g2[q] = O31 * gx + O32 * gy + O33 * gz;
   }

   std::fill(Yacc, Yacc + ndof, real_t(0));
   for (int qz = 0; qz < Q1D; ++qz)
   {
      std::fill(A0, A0 + nd2, real_t(0));
      std::fill(A1, A1 + nd2, real_t(0));
      std::fill(A2, A2 + nd2, real_t(0));
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + Q1D * (qy + Q1D * qz);
            gvec0[qx] = g0[q];
            gvec1[qx] = g1[q];
            gvec2[qx] = g2[q];
         }
         mma::LapackGemm('N', 'N', D1D, 1, Q1D, real_t(1), Gt, D1D,
                         gvec0, Q1D, real_t(0), QQ0, D1D);
         mma::LapackGemm('N', 'N', D1D, 1, Q1D, real_t(1), Bt, D1D,
                         gvec1, Q1D, real_t(0), QQ1, D1D);
         mma::LapackGemm('N', 'N', D1D, 1, Q1D, real_t(1), Bt, D1D,
                         gvec2, Q1D, real_t(0), QQ2, D1D);
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t wy = Bt[dy + D1D * qy];
            const real_t wDy = Gt[dy + D1D * qy];
            for (int dx = 0; dx < D1D; ++dx)
            {
               A0[dx + D1D * dy] += QQ0[dx] * wy;
               A1[dx + D1D * dy] += QQ1[dx] * wDy;
               A2[dx + D1D * dy] += QQ2[dx] * wy;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         const real_t wz = Bt[dz + D1D * qz];
         const real_t wDz = Gt[dz + D1D * qz];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               const int id = dx + D1D * (dy + D1D * dz);
               Yacc[id] += A0[dx + D1D * dy] * wz
                         + A1[dx + D1D * dy] * wz
                         + A2[dx + D1D * dy] * wDz;
            }
         }
      }
   }

   real_t *Ye = Y + static_cast<size_t>(ndof) * e;
   for (int i = 0; i < ndof; ++i) { Ye[i] += Yacc[i]; }
}

/** 3D diffusion SUM via per-element Lapack + Apple GCD over elements. */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensorsSumLapack3D(
   const int NE,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   const int nd2 = D1D * D1D;
   const int nq3 = Q1D * Q1D * Q1D;

   auto run_e = [&](int e)
   {
      std::vector<real_t> BX(static_cast<size_t>(Q1D) * nd2);
      std::vector<real_t> GX(static_cast<size_t>(Q1D) * nd2);
      std::vector<real_t> BXt(static_cast<size_t>(D1D) * Q1D);
      std::vector<real_t> GXt(static_cast<size_t>(D1D) * Q1D);
      std::vector<real_t> XY0(static_cast<size_t>(Q1D) * Q1D);
      std::vector<real_t> XY1(static_cast<size_t>(Q1D) * Q1D);
      std::vector<real_t> XY2(static_cast<size_t>(Q1D) * Q1D);
      std::vector<real_t> g0(static_cast<size_t>(nq3));
      std::vector<real_t> g1(static_cast<size_t>(nq3));
      std::vector<real_t> g2(static_cast<size_t>(nq3));
      std::vector<real_t> QQ0(static_cast<size_t>(D1D));
      std::vector<real_t> QQ1(static_cast<size_t>(D1D));
      std::vector<real_t> QQ2(static_cast<size_t>(D1D));
      std::vector<real_t> gvec0(static_cast<size_t>(Q1D));
      std::vector<real_t> gvec1(static_cast<size_t>(Q1D));
      std::vector<real_t> gvec2(static_cast<size_t>(Q1D));
      std::vector<real_t> A0(static_cast<size_t>(nd2));
      std::vector<real_t> A1(static_cast<size_t>(nd2));
      std::vector<real_t> A2(static_cast<size_t>(nd2));
      std::vector<real_t> Yacc(static_cast<size_t>(D1D) * nd2);
      DiffusionApplyTensorsSumLapack3DElement<D1D, Q1D, SYM>(
         e, B, G, Bt, Gt, Dv, X, Y,
         BX.data(), GX.data(), BXt.data(), GXt.data(),
         XY0.data(), XY1.data(), XY2.data(),
         g0.data(), g1.data(), g2.data(),
         QQ0.data(), QQ1.data(), QQ2.data(),
         gvec0.data(), gvec1.data(), gvec2.data(),
         A0.data(), A1.data(), A2.data(),
         Yacc.data());
   };

#if defined(__APPLE__)
   dispatch_apply(static_cast<size_t>(NE), DISPATCH_APPLY_AUTO,
                  ^(size_t e) { run_e(static_cast<int>(e)); });
#else
   for (int e = 0; e < NE; ++e) { run_e(e); }
#endif
}

template <int D1D, int Q1D>
inline bool TryDiffusionApplyTensorsSumLapack2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!PreferTensorDiffLapack(D1D, Q1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(), *Gt = gt.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (symmetric)
   {
      DiffusionApplyTensorsSumLapack2D<D1D, Q1D, true>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   else
   {
      DiffusionApplyTensorsSumLapack2D<D1D, Q1D, false>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   return true;
}

/** Hand diffusion 3D (PADiffusion contractions) + GCD over element tiles. */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensorsHandGcd3D(
   const int NE,
   const real_t *B, const real_t *G,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? 6 : 9;
   const int nq3 = Q1D * Q1D * Q1D;
   auto apply_e = [&](int e)
   {
      real_t grad[Q1D][Q1D][Q1D][3];
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = real_t(0);
               grad[qz][qy][qx][1] = real_t(0);
               grad[qz][qy][qx][2] = real_t(0);
            }
      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t gradXY[Q1D][Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = real_t(0);
               gradXY[qy][qx][1] = real_t(0);
               gradXY[qy][qx][2] = real_t(0);
            }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t gradX[Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = real_t(0);
               gradX[qx][1] = real_t(0);
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X[dx + D1D * (dy + D1D * (dz + D1D * e))];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B[qx + Q1D * dx];
                  gradX[qx][1] += s * G[qx + Q1D * dx];
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = B[qy + Q1D * dy];
               const real_t wDy = G[qy + Q1D * dy];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradXY[qy][qx][0] += gradX[qx][1] * wy;
                  gradXY[qy][qx][1] += gradX[qx][0] * wDy;
                  gradXY[qy][qx][2] += gradX[qx][0] * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz = B[qz + Q1D * dz];
            const real_t wDz = G[qz + Q1D * dz];
            for (int qy = 0; qy < Q1D; ++qy)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + Q1D * qz);
               const real_t gX = grad[qz][qy][qx][0];
               const real_t gY = grad[qz][qy][qx][1];
               const real_t gZ = grad[qz][qy][qx][2];
               const real_t O11 = Dv[q + nq3 * (0 + PA_SIZE * e)];
               const real_t O12 = Dv[q + nq3 * (1 + PA_SIZE * e)];
               const real_t O13 = Dv[q + nq3 * (2 + PA_SIZE * e)];
               real_t O21, O22, O23, O31, O32, O33;
               if constexpr (SYM)
               {
                  O21 = O12; O22 = Dv[q + nq3 * (3 + PA_SIZE * e)];
                  O23 = Dv[q + nq3 * (4 + PA_SIZE * e)];
                  O31 = O13; O32 = O23;
                  O33 = Dv[q + nq3 * (5 + PA_SIZE * e)];
               }
               else
               {
                  O21 = Dv[q + nq3 * (3 + PA_SIZE * e)];
                  O22 = Dv[q + nq3 * (4 + PA_SIZE * e)];
                  O23 = Dv[q + nq3 * (5 + PA_SIZE * e)];
                  O31 = Dv[q + nq3 * (6 + PA_SIZE * e)];
                  O32 = Dv[q + nq3 * (7 + PA_SIZE * e)];
                  O33 = Dv[q + nq3 * (8 + PA_SIZE * e)];
               }
               grad[qz][qy][qx][0] = O11*gX + O12*gY + O13*gZ;
               grad[qz][qy][qx][1] = O21*gX + O22*gY + O23*gZ;
               grad[qz][qy][qx][2] = O31*gX + O32*gY + O33*gZ;
            }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t gradXY[D1D][D1D][3];
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = real_t(0);
               gradXY[dy][dx][1] = real_t(0);
               gradXY[dy][dx][2] = real_t(0);
            }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t gradX[D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = real_t(0);
               gradX[dx][1] = real_t(0);
               gradX[dx][2] = real_t(0);
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t gX = grad[qz][qy][qx][0];
               const real_t gY = grad[qz][qy][qx][1];
               const real_t gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradX[dx][0] += gX * G[qx + Q1D * dx]; // Gt
                  gradX[dx][1] += gY * B[qx + Q1D * dx];
                  gradX[dx][2] += gZ * B[qx + Q1D * dx];
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy = B[qy + Q1D * dy];
               const real_t wDy = G[qy + Q1D * dy];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz = B[qz + Q1D * dz];
            const real_t wDz = G[qz + Q1D * dz];
            for (int dy = 0; dy < D1D; ++dy)
               for (int dx = 0; dx < D1D; ++dx)
                  Y[dx + D1D * (dy + D1D * (dz + D1D * e))] +=
                     gradXY[dy][dx][0] * wz +
                     gradXY[dy][dx][1] * wz +
                     gradXY[dy][dx][2] * wDz;
         }
      }
   };
   const int NB = mma::TensorLapackNB3D(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
#if defined(__APPLE__)
   dispatch_apply(static_cast<size_t>(ntiles), DISPATCH_APPLY_AUTO,
                  ^(size_t tile)
   {
      const int e0 = static_cast<int>(tile) * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   });
#else
   for (int e = 0; e < NE; ++e) { apply_e(e); }
#endif
}

template <int D1D, int Q1D>
inline bool TryDiffusionApplyTensorsSumLapack3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!PreferTensorDiffLapack(D1D, Q1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(), *Gt = gt.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (D1D <= 5)
   {
      if (symmetric)
      {
         DiffusionApplyTensorsHandGcd3D<D1D, Q1D, true>(NE, B, G, Dv, X, Y);
      }
      else
      {
         DiffusionApplyTensorsHandGcd3D<D1D, Q1D, false>(NE, B, G, Dv, X, Y);
      }
      return true;
   }
   if (symmetric)
   {
      DiffusionApplyTensorsSumLapack3D<D1D, Q1D, true>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   else
   {
      DiffusionApplyTensorsSumLapack3D<D1D, Q1D, false>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   return true;
}
#endif // MFEM_USE_LAPACK

template <int T_D1D = 0, int T_Q1D = 0, bool SYM = true>
inline void MmaDiffusionApplyTensors3D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> &g,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : mma::TensorsMmaMaxQ1D;
   constexpr int PA_SIZE = SYM ? 6 : 9;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1,
               "Tensors MMA diffusion 3D D1D/Q1D exceeds shell cap");
   MFEM_VERIFY(d.Size() == PA_SIZE * Q1D * Q1D * Q1D * NE, "");

   const int NB = T_D1D ? mma::DiffNB3D<T_D1D, T_Q1D>()
                        : mma::DiffNB3DRuntime(D1D);
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::DiffThreads3D<T_D1D, T_Q1D>()
                                 : mma::DiffThreads3DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D * Q1D, PA_SIZE, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[3][MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[3][MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];

      mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, B, G, BG, BGt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         mma::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;

         {
            const int tid = mma::getThreadIdxX();
            const int nq = Q1D * Q1D * Q1D;
            const int stride = mma::getBlockNthreadsX();
            for (int thread = tid; thread < nq; thread += stride)
            {
               const int qx = thread % Q1D;
               const int div = thread / Q1D;
               const int qy = div % Q1D;
               const int qz = div / Q1D;
               const int idx = qx + Q1D * qy + Q1D * Q1D * qz;
               const int q = qx + Q1D * (qy + Q1D * qz);
               const real_t gX = sm1[0][idx], gY = sm1[1][idx], gZ = sm1[2][idx];
               const real_t O11 = D(q, 0, e);
               const real_t O12 = D(q, 1, e);
               const real_t O13 = D(q, 2, e);
               real_t O21, O22, O23, O31, O32, O33;
               if constexpr (SYM)
               {
                  O21 = O12; O22 = D(q, 3, e); O23 = D(q, 4, e);
                  O31 = O13; O32 = O23; O33 = D(q, 5, e);
               }
               else
               {
                  O21 = D(q, 3, e); O22 = D(q, 4, e); O23 = D(q, 5, e);
                  O31 = D(q, 6, e); O32 = D(q, 7, e); O33 = D(q, 8, e);
               }
               sm0[0][idx] = O11 * gX + O12 * gY + O13 * gZ;
               sm0[1][idx] = O21 * gX + O22 * gY + O23 * gZ;
               sm0[2][idx] = O31 * gX + O32 * gY + O33 * gZ;
            }
         }
         MFEM_SYNC_THREAD;

         mma::GradZt<MD1, MQ1>(D1D, Q1D, BGt, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::GradYt<MD1, MQ1>(D1D, Q1D, BGt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::GradXt<MD1, MQ1>(D1D, Q1D, BGt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaDiffusionApplyTensors3D_Dispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MFEM_CONTRACT_VAR(gt);
   using Fn = decltype(&MmaDiffusionApplyTensors3D<T_D1D, T_Q1D>);
   const Fn apply = symmetric ? &MmaDiffusionApplyTensors3D<T_D1D, T_Q1D>
                              : &MmaDiffusionApplyTensors3D<T_D1D, T_Q1D, false>;
   apply(NE, b, g, d, x, y, d1d, q1d);
}

template <int T_D1D = 0, int T_Q1D = 0, bool SYM = true>
inline void MmaDiffusionApplyTensors2D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> &g,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : mma::TensorsMmaMaxQ1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int PA_SIZE = SYM ? 3 : 4;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1,
               "Tensors MMA diffusion 2D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::DiffNB2D<T_D1D, T_Q1D>()
                        : mma::NB2DRuntime(D1D);
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::DiffThreads2D<T_D1D, T_Q1D>()
                                 : mma::Threads2DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D, PA_SIZE, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[2][MDQ * MDQ];
      MFEM_SHARED real_t sm1[2][MDQ * MDQ];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];

      mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, B, G, BG, BGt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX2D<MQ1>(e, D1D, X, sm0[0]);
         MFEM_SYNC_THREAD;

         mma::GradX2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::GradY2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;

         {
            const int tid = mma::getThreadIdxX();
            const int nq = Q1D * Q1D;
            const int stride = mma::getBlockNthreadsX();
            for (int t = tid; t < nq; t += stride)
            {
               const int qx = t % Q1D;
               const int qy = t / Q1D;
               const int idx = qx + Q1D * qy;
               const real_t gX = sm0[0][idx];
               const real_t gY = sm0[1][idx];
               const real_t O11 = D(idx, 0, e);
               const real_t O21 = D(idx, 1, e);
               const real_t O12 = SYM ? O21 : D(idx, 2, e);
               const real_t O22 = SYM ? D(idx, 2, e) : D(idx, 3, e);
               sm1[0][idx] = O11 * gX + O12 * gY;
               sm1[1][idx] = O21 * gX + O22 * gY;
            }
         }
         MFEM_SYNC_THREAD;

         mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaDiffusionApplyTensors2D_Dispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MFEM_CONTRACT_VAR(gt);
   using Fn = decltype(&MmaDiffusionApplyTensors2D<T_D1D, T_Q1D>);
   const Fn apply = symmetric ? &MmaDiffusionApplyTensors2D<T_D1D, T_Q1D>
                              : &MmaDiffusionApplyTensors2D<T_D1D, T_Q1D, false>;
   apply(NE, b, g, d, x, y, d1d, q1d);
}

/** Runtime overload for Fallback / unregistered (D1D,Q1D). */
inline void MmaDiffusionApplyTensors2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors2D_Dispatch<0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

inline void MmaDiffusionApplyTensors3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors3D_Dispatch<0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaDiffusionApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   // Host: fat 1D Lapack when profitable; else MMA Emulate shell.
   // Device: unchanged MMA/Emulate shell.
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
#ifdef MFEM_USE_LAPACK
      if constexpr (DIM == 3)
      {
         if (TryDiffusionApplyTensorsSumLapack3D<T_D1D, T_Q1D>(
                NE, symmetric, b, g, bt, gt, d, x, y))
         { return; }
      }
      else
      {
         if (TryDiffusionApplyTensorsSumLapack2D<T_D1D, T_Q1D>(
                NE, symmetric, b, g, bt, gt, d, x, y))
         { return; }
      }
#endif
   }
   if constexpr (DIM == 3)
   {
      MmaDiffusionApplyTensors3D_Dispatch<T_D1D, T_Q1D>(
         NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
   }
   else
   {
      MmaDiffusionApplyTensors2D_Dispatch<T_D1D, T_Q1D>(
         NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaDiffusionApplyTensors<DIM, T_D1D, T_Q1D>;
}

// Fallback defined in bilininteg_diffusion_pa_tensors_mma.cpp.

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
