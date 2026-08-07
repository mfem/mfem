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

/** @file tensor_grad.hpp
    Generic tensor-product Grad×Grad apply (host multi-RHS tiles + device shells).
    QFn is a template parameter; define QFns under fem/integ/, not here.
*/

#include "../mma.hpp"
#include "fields.hpp"
#include "tensor_metric.hpp"
#include "../../../../general/array.hpp"
#include "../../../../linalg/vector.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal
{


// ---------------------------------------------------------------------------
// Host: multi-RHS sum-fact GEMM tiles (lapack)
// ---------------------------------------------------------------------------
namespace mma::lapack
{
#ifdef MFEM_USE_LAPACK
/** Named slices into one host_Arena allocation for 2D Grad tiles. */
template <int D1D, int Q1D>
struct Diff2DWs
{
   real_t *xloc, *BX, *GX, *BXt, *GXt;
   real_t *gX, *gY, *gXp, *gYp;
   real_t *A0, *A1, *A0t, *A1t, *Y0, *Y1;

   static size_t Words(int NB)
   {
      // 3*D1D² + 8*D1D*Q1D + 4*Q1D² words per element column of the tile.
      return size_t(NB) * (3 * D1D * D1D + 8 * D1D * Q1D + 4 * Q1D * Q1D);
   }

   void Bind(mma::host_Arena &a, int NB)
   {
      const int n_xy = D1D * NB;
      const int n_qy = Q1D * NB;
      a.reset(Words(NB));
      xloc = a.take(size_t(D1D) * n_xy);
      BX = a.take(size_t(Q1D) * n_xy);
      GX = a.take(size_t(Q1D) * n_xy);
      BXt = a.take(size_t(D1D) * n_qy);
      GXt = a.take(size_t(D1D) * n_qy);
      gX = a.take(size_t(Q1D) * n_qy);
      gY = a.take(size_t(Q1D) * n_qy);
      gXp = a.take(size_t(Q1D) * n_qy);
      gYp = a.take(size_t(Q1D) * n_qy);
      A0 = a.take(size_t(D1D) * n_qy);
      A1 = a.take(size_t(D1D) * n_qy);
      A0t = a.take(size_t(Q1D) * n_xy);
      A1t = a.take(size_t(Q1D) * n_xy);
      Y0 = a.take(size_t(D1D) * n_xy);
      Y1 = a.take(size_t(D1D) * n_xy);
   }
};

/** One 2D Grad tile: B/G forward, metric, Bt/Gt backward. */
template <typename QFn, int D1D, int Q1D, bool SYM>
inline void TensorGradHost2DTile(
   const int e0, const int nbe, const int NB,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y,
   const Diff2DWs<D1D, Q1D> &ws)
{
   constexpr int PA_SIZE = SYM ? 3 : 4;
   const int n_xy = D1D * NB;
   const int n_qy = Q1D * NB;
   const real_t *Xsrc = mma::lapack::PackX2D<D1D>(e0, nbe, NB, X, ws.xloc);

   mma::lapack::Gemm('N', 'N', Q1D, n_xy, D1D, real_t(1), B, Q1D,
                     Xsrc, D1D, real_t(0), ws.BX, Q1D);
   mma::lapack::Gemm('N', 'N', Q1D, n_xy, D1D, real_t(1), G, Q1D,
                     Xsrc, D1D, real_t(0), ws.GX, Q1D);

   mma::lapack::TransposeAB<Q1D, D1D>(ws.BX, ws.BXt, NB);
   mma::lapack::TransposeAB<Q1D, D1D>(ws.GX, ws.GXt, NB);
   mma::lapack::Gemm('N', 'N', Q1D, n_qy, D1D, real_t(1), B, Q1D,
                     ws.GXt, D1D, real_t(0), ws.gX, Q1D);
   mma::lapack::Gemm('N', 'N', Q1D, n_qy, D1D, real_t(1), G, Q1D,
                     ws.BXt, D1D, real_t(0), ws.gY, Q1D);

   for (int b = 0; b < nbe; ++b)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const int idx = qx + Q1D * qy;
            const int e = e0 + b;
            real_t gv[2] = {ws.gX[qy + Q1D * (qx + Q1D * b)],
                            ws.gY[qy + Q1D * (qx + Q1D * b)]
                           };
            real_t O[PA_SIZE];
            for (int c = 0; c < PA_SIZE; ++c)
            {
               O[c] = Dv[idx + Q1D * Q1D * (c + PA_SIZE * e)];
            }
            mma::form::ApplyGradQFnVec<2, SYM>(QFn{}, gv, O);
            ws.gX[qy + Q1D * (qx + Q1D * b)] = gv[0];
            ws.gY[qy + Q1D * (qx + Q1D * b)] = gv[1];
         }
      }
   }
   for (int b = nbe; b < NB; ++b)
   {
      for (int i = 0; i < Q1D * Q1D; ++i)
      {
         ws.gX[i + Q1D * Q1D * b] = real_t(0);
         ws.gY[i + Q1D * Q1D * b] = real_t(0);
      }
   }

   mma::lapack::TransposeAB<Q1D, Q1D>(ws.gX, ws.gXp, NB);
   mma::lapack::TransposeAB<Q1D, Q1D>(ws.gY, ws.gYp, NB);
   mma::lapack::Gemm('N', 'N', D1D, n_qy, Q1D, real_t(1), Gt, D1D,
                     ws.gXp, Q1D, real_t(0), ws.A0, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_qy, Q1D, real_t(1), Bt, D1D,
                     ws.gYp, Q1D, real_t(0), ws.A1, D1D);

   mma::lapack::TransposeAB<D1D, Q1D>(ws.A0, ws.A0t, NB);
   mma::lapack::TransposeAB<D1D, Q1D>(ws.A1, ws.A1t, NB);
   mma::lapack::Gemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Bt, D1D,
                     ws.A0t, Q1D, real_t(0), ws.Y0, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Gt, D1D,
                     ws.A1t, Q1D, real_t(0), ws.Y1, D1D);

   for (int b = 0; b < nbe; ++b)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            Y[dx + D1D * (dy + D1D * (e0 + b))] +=
               ws.Y0[dy + D1D * (dx + D1D * b)] +
               ws.Y1[dy + D1D * (dx + D1D * b)];
         }
      }
   }
}

template <typename QFn, int D1D, int Q1D, bool SYM>
inline void TensorGradHost2D(
   const int NE,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   const int NB = mma::TensorTileNB(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   mma::host_Arena arena;
   Diff2DWs<D1D, Q1D> ws;
   ws.Bind(arena, NB);
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      TensorGradHost2DTile<QFn, D1D, Q1D, SYM>(
         e0, nbe, NB, B, G, Bt, Gt, Dv, X, Y, ws);
   }
}

template <typename QFn, int D1D, int Q1D>
inline bool TryTensorGradHost2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!mma::PreferTensorDense(D1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(), *Gt = gt.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (symmetric)
   {
      TensorGradHost2D<QFn, D1D, Q1D, true>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   else
   {
      TensorGradHost2D<QFn, D1D, Q1D, false>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   return true;
}

/** Named slices for 3D multi-RHS tiles (same story as Diff2DWs). */
template <int D1D, int Q1D>
struct Diff3DWs
{
   real_t *xloc;                 // D³·NB (also ytmp after X is consumed)
   real_t *BX, *GX, *BXt, *GXt;  // after x / transpose
   real_t *BB, *GB, *BG;         // after y: Q·(Q·D·NB)
   real_t *BBt, *GBt, *BGt;      // after y→z transpose: D·(Q²·NB)
   real_t *gX, *gY, *gZ;         // after z: Q·(Q²·NB)
   real_t *uX, *uY, *uZ;         // z-back
   real_t *tX, *tY, *tZ;         // z-back transpose → y-back
   real_t *vX, *vY, *vZ;         // y-back
   real_t *wX, *wY, *wZ;         // y-back transpose → x-back

   static size_t Words(int NB)
   {
      constexpr size_t D = D1D, Q = Q1D;
      // xloc + 10·Q·D² + 12·Q²·D + 3·Q³  (see Bind takes)
      return size_t(NB) * (D * D * D + 10 * Q * D * D + 12 * Q * Q * D +
                           3 * Q * Q * Q);
   }

   void Bind(mma::host_Arena &a, int NB)
   {
      constexpr size_t D = D1D, Q = Q1D;
      const size_t n_yz = D * D * NB;
      const size_t n_q_dz = Q * D * NB;
      const size_t n_qq = Q * Q * NB;
      a.reset(Words(NB));
      xloc = a.take(D * D * D * NB);
      BX = a.take(Q * n_yz);
      GX = a.take(Q * n_yz);
      BXt = a.take(D * n_q_dz); // D × (Q·D·NB)
      GXt = a.take(D * n_q_dz);
      BB = a.take(Q * n_q_dz);
      GB = a.take(Q * n_q_dz);
      BG = a.take(Q * n_q_dz);
      BBt = a.take(D * n_qq);
      GBt = a.take(D * n_qq);
      BGt = a.take(D * n_qq);
      gX = a.take(Q * n_qq);
      gY = a.take(Q * n_qq);
      gZ = a.take(Q * n_qq);
      uX = a.take(D * n_qq);
      uY = a.take(D * n_qq);
      uZ = a.take(D * n_qq);
      tX = a.take(Q * n_q_dz);
      tY = a.take(Q * n_q_dz);
      tZ = a.take(Q * n_q_dz);
      vX = a.take(D * n_q_dz);
      vY = a.take(D * n_q_dz);
      vZ = a.take(D * n_q_dz);
      wX = a.take(Q * n_yz);
      wY = a.take(Q * n_yz);
      wZ = a.take(Q * n_yz);
   }
};

/** One 3D Grad tile: sum-fact via multi-RHS GEMM (B/G × tile of elements).
    gX=(B⊗B⊗G)X, gY=(B⊗G⊗B)X, gZ=(G⊗B⊗B)X, then O·g, then adjoints. */
template <typename QFn, int D1D, int Q1D, bool SYM>
inline void TensorGradHost3DTile(
   const int e0, const int nbe, const int NB,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y,
   const Diff3DWs<D1D, Q1D> &ws)
{
   constexpr int PA_SIZE = SYM ? 6 : 9;
   constexpr int QQ = Q1D * Q1D;
   constexpr int QQQ = Q1D * Q1D * Q1D;
   const int n_yz = D1D * D1D * NB;
   const int n_q_dz_b = Q1D * D1D * NB;
   const int n_qq_b = QQ * NB;

   const real_t *Xsrc = mma::lapack::PackX3D<D1D>(e0, nbe, NB, X, ws.xloc);

   // ---- forward x: BX/GX = (B|G) X ---------------------------------------
   mma::lapack::Gemm('N', 'N', Q1D, n_yz, D1D, real_t(1), B, Q1D,
                     Xsrc, D1D, real_t(0), ws.BX, Q1D);
   mma::lapack::Gemm('N', 'N', Q1D, n_yz, D1D, real_t(1), G, Q1D,
                     Xsrc, D1D, real_t(0), ws.GX, Q1D);
   mma::lapack::TransposeAB<Q1D, D1D>(ws.BX, ws.BXt, D1D * NB);
   mma::lapack::TransposeAB<Q1D, D1D>(ws.GX, ws.GXt, D1D * NB);

   // ---- forward y ---------------------------------------------------------
   mma::lapack::Gemm('N', 'N', Q1D, n_q_dz_b, D1D, real_t(1), B, Q1D,
                     ws.BXt, D1D, real_t(0), ws.BB, Q1D); // By Bx
   mma::lapack::Gemm('N', 'N', Q1D, n_q_dz_b, D1D, real_t(1), G, Q1D,
                     ws.BXt, D1D, real_t(0), ws.GB, Q1D); // Gy Bx
   mma::lapack::Gemm('N', 'N', Q1D, n_q_dz_b, D1D, real_t(1), B, Q1D,
                     ws.GXt, D1D, real_t(0), ws.BG, Q1D); // By Gx
   mma::lapack::TransposeAB<QQ, D1D>(ws.BB, ws.BBt, NB);
   mma::lapack::TransposeAB<QQ, D1D>(ws.GB, ws.GBt, NB);
   mma::lapack::TransposeAB<QQ, D1D>(ws.BG, ws.BGt, NB);

   // ---- forward z → gX,gY,gZ ----------------------------------------------
   mma::lapack::Gemm('N', 'N', Q1D, n_qq_b, D1D, real_t(1), B, Q1D,
                     ws.BGt, D1D, real_t(0), ws.gX, Q1D); // Bz By Gx
   mma::lapack::Gemm('N', 'N', Q1D, n_qq_b, D1D, real_t(1), B, Q1D,
                     ws.GBt, D1D, real_t(0), ws.gY, Q1D); // Bz Gy Bx
   mma::lapack::Gemm('N', 'N', Q1D, n_qq_b, D1D, real_t(1), G, Q1D,
                     ws.BBt, D1D, real_t(0), ws.gZ, Q1D); // Gz By Bx

   // g[qz + Q*(qy + Q*(qx + Q*b))]; PA q = qx + Q*(qy + Q*qz)
   for (int b = 0; b < nbe; ++b)
   {
      for (int qx = 0; qx < Q1D; ++qx)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const int q = qx + Q1D * (qy + Q1D * qz);
               const int idx = qz + Q1D * (qy + Q1D * (qx + Q1D * b));
               real_t gv[3] = {ws.gX[idx], ws.gY[idx], ws.gZ[idx]};
               real_t O[PA_SIZE];
               for (int c = 0; c < PA_SIZE; ++c)
               {
                  O[c] = Dv[q + QQQ * (c + PA_SIZE * (e0 + b))];
               }
               mma::form::ApplyGradQFnVec<3, SYM>(QFn{}, gv, O);
               ws.gX[idx] = gv[0];
               ws.gY[idx] = gv[1];
               ws.gZ[idx] = gv[2];
            }
   }
   for (int b = nbe; b < NB; ++b)
   {
      std::fill(ws.gX + QQQ * b, ws.gX + QQQ * (b + 1), real_t(0));
      std::fill(ws.gY + QQQ * b, ws.gY + QQQ * (b + 1), real_t(0));
      std::fill(ws.gZ + QQQ * b, ws.gZ + QQQ * (b + 1), real_t(0));
   }

   // ---- backward z --------------------------------------------------------
   mma::lapack::Gemm('N', 'N', D1D, n_qq_b, Q1D, real_t(1), Bt, D1D,
                     ws.gX, Q1D, real_t(0), ws.uX, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_qq_b, Q1D, real_t(1), Bt, D1D,
                     ws.gY, Q1D, real_t(0), ws.uY, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_qq_b, Q1D, real_t(1), Gt, D1D,
                     ws.gZ, Q1D, real_t(0), ws.uZ, D1D);
   mma::lapack::TransposeAB<D1D, QQ>(ws.uX, ws.tX, NB);
   mma::lapack::TransposeAB<D1D, QQ>(ws.uY, ws.tY, NB);
   mma::lapack::TransposeAB<D1D, QQ>(ws.uZ, ws.tZ, NB);

   // ---- backward y --------------------------------------------------------
   mma::lapack::Gemm('N', 'N', D1D, n_q_dz_b, Q1D, real_t(1), Bt, D1D,
                     ws.tX, Q1D, real_t(0), ws.vX, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_q_dz_b, Q1D, real_t(1), Gt, D1D,
                     ws.tY, Q1D, real_t(0), ws.vY, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_q_dz_b, Q1D, real_t(1), Bt, D1D,
                     ws.tZ, Q1D, real_t(0), ws.vZ, D1D);
   mma::lapack::TransposeAB<D1D, Q1D>(ws.vX, ws.wX, D1D * NB);
   mma::lapack::TransposeAB<D1D, Q1D>(ws.vY, ws.wY, D1D * NB);
   mma::lapack::TransposeAB<D1D, Q1D>(ws.vZ, ws.wZ, D1D * NB);

   // ---- backward x → Y (reuse xloc as ytmp) -------------------------------
   real_t *ytmp = ws.xloc;
   mma::lapack::Gemm('N', 'N', D1D, n_yz, Q1D, real_t(1), Gt, D1D,
                     ws.wX, Q1D, real_t(0), ytmp, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_yz, Q1D, real_t(1), Bt, D1D,
                     ws.wY, Q1D, real_t(1), ytmp, D1D);
   mma::lapack::Gemm('N', 'N', D1D, n_yz, Q1D, real_t(1), Bt, D1D,
                     ws.wZ, Q1D, real_t(1), ytmp, D1D);
   mma::lapack::ScatterAddY3D<D1D>(e0, nbe, ytmp, Y);
}

template <typename QFn, int D1D, int Q1D, bool SYM>
inline void TensorGradHost3D(
   const int NE,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   const int NB = mma::TensorTileNB3D(D1D);
   const int ntiles = (NE + NB - 1) / NB;
   mma::host_Arena arena;
   Diff3DWs<D1D, Q1D> ws;
   ws.Bind(arena, NB);
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      TensorGradHost3DTile<QFn, D1D, Q1D, SYM>(
         e0, nbe, NB, B, G, Bt, Gt, Dv, X, Y, ws);
   }
}

template <typename QFn, int D1D, int Q1D>
inline bool TryTensorGradHost3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!mma::PreferTensorDense(D1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(), *Gt = gt.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (symmetric)
   {
      TensorGradHost3D<QFn, D1D, Q1D, true>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   else
   {
      TensorGradHost3D<QFn, D1D, Q1D, false>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   return true;
}

#endif // MFEM_USE_LAPACK

} // namespace mma::lapack

// ---------------------------------------------------------------------------
// Device (or host Emulate) smem shells — Grad → QFn → Gradt
// Host entry uses Try* first (see TensorGradApply).
// ---------------------------------------------------------------------------

/** One 3D element: LoadX → GradXYZ → O·g → Gradt → Y. */
template <typename QFn, int MD1, int MQ1, bool SYM, typename TD, typename TX, typename TY>
MFEM_HOST_DEVICE inline
void TensorGradElement3D(const int D1D, const int Q1D, const int e,
                              real_t (&BG)[2][MQ1 * MD1],
                              real_t (&BGt)[2][MQ1 * MD1],
                              real_t (&sm0)[3][MQ1 * MQ1 * MQ1],
                              real_t (&sm1)[3][MQ1 * MQ1 * MQ1],
                              TD D, TX X, TY Y)
{
   constexpr int plane_ld = MQ1 * MQ1 * MQ1;
   mma::LoadX<MQ1>(e, D1D, X, sm0);
   MFEM_SYNC_THREAD;

   mma::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
   MFEM_SYNC_THREAD;
   mma::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
   MFEM_SYNC_THREAD;
   mma::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
   MFEM_SYNC_THREAD;

   mma::form::ApplyGradQFnSmem<3, SYM>(QFn{}, 
      sm1[0], sm0[0], plane_ld, D, e, Q1D,
      mma::getThreadIdxX(), mma::getBlockNthreadsX());
   MFEM_SYNC_THREAD;

   mma::GradZt<MD1, MQ1>(D1D, Q1D, BGt, sm0, sm1);
   MFEM_SYNC_THREAD;
   mma::GradYt<MD1, MQ1>(D1D, Q1D, BGt, sm1, sm0);
   MFEM_SYNC_THREAD;
   mma::GradXt<MD1, MQ1>(D1D, Q1D, BGt, sm0, Y, e);
   MFEM_SYNC_THREAD;
}

/** One 2D element: LoadX → GradXY → O·g → Gradt → Y. */
template <typename QFn, int MD1, int MQ1, int MDQ, bool SYM, typename TD, typename TX,
          typename TY>
MFEM_HOST_DEVICE inline
void TensorGradElement2D(const int D1D, const int Q1D, const int e,
                              real_t (&BG)[2][MQ1 * MD1],
                              real_t (&BGt)[2][MQ1 * MD1],
                              real_t (&sm0)[2][MDQ * MDQ],
                              real_t (&sm1)[2][MDQ * MDQ],
                              TD D, TX X, TY Y)
{
   constexpr int plane_ld = MDQ * MDQ;
   mma::LoadX2D<MQ1>(e, D1D, X, sm0[0]);
   MFEM_SYNC_THREAD;

   mma::GradX2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm0, sm1);
   MFEM_SYNC_THREAD;
   mma::GradY2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm1, sm0);
   MFEM_SYNC_THREAD;

   mma::form::ApplyGradQFnSmem<2, SYM>(QFn{}, 
      sm0[0], sm1[0], plane_ld, D, e, Q1D,
      mma::getThreadIdxX(), mma::getBlockNthreadsX());
   MFEM_SYNC_THREAD;

   mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
   MFEM_SYNC_THREAD;
   mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y, e);
   MFEM_SYNC_THREAD;
}

template <typename QFn, int T_D1D = 0, int T_Q1D = 0, bool SYM = true>
inline void TensorGradApply3D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> &g,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   constexpr int PA_SIZE = SYM ? 6 : 9;
   dq.Verify(NE, "Tensor Grad MMA 3D D1D/Q1D exceeds shell cap");
   MFEM_VERIFY(d.Size() == PA_SIZE * Q1D * Q1D * Q1D * NE, "");

   const int NB = T_D1D
                  ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                  : mma::TensorNB3DRuntime(D1D, mma::kTensorCostHeavy);
   const int nthreads = mma::TensorShellNthreads(
                           T_D1D
                           ? mma::TensorThreads3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                           : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                         mma::kTensorCostHeavy));

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
         TensorGradElement3D<QFn, MD1, MQ1, SYM>(
            D1D, Q1D, e, BG, BGt, sm0, sm1, D, X, Y);
      }
   });
}

template <typename QFn, int T_D1D = 0, int T_Q1D = 0>
inline void TensorGradApply3D_Dispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MFEM_CONTRACT_VAR(gt);
   if (symmetric)
   {
      TensorGradApply3D<QFn, T_D1D, T_Q1D, true>(
         NE, b, g, d, x, y, d1d, q1d);
   }
   else
   {
      TensorGradApply3D<QFn, T_D1D, T_Q1D, false>(
         NE, b, g, d, x, y, d1d, q1d);
   }
}

template <typename QFn, int T_D1D = 0, int T_Q1D = 0, bool SYM = true>
inline void TensorGradApply2D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> &g,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int PA_SIZE = SYM ? 3 : 4;
   dq.Verify(NE, "Tensor Grad MMA 2D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                  : mma::NB2DRuntime(D1D);
   const int nthreads = mma::TensorShellNthreads(
                           T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                           : mma::Threads2DRuntime(D1D, Q1D));

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
         TensorGradElement2D<QFn, MD1, MQ1, MDQ, SYM>(
            D1D, Q1D, e, BG, BGt, sm0, sm1, D, X, Y);
      }
   });
}

template <typename QFn, int T_D1D = 0, int T_Q1D = 0>
inline void TensorGradApply2D_Dispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MFEM_CONTRACT_VAR(gt);
   if (symmetric)
   {
      TensorGradApply2D<QFn, T_D1D, T_Q1D, true>(
         NE, b, g, d, x, y, d1d, q1d);
   }
   else
   {
      TensorGradApply2D<QFn, T_D1D, T_Q1D, false>(
         NE, b, g, d, x, y, d1d, q1d);
   }
}

/** Entry: host lapack multi-RHS sum-fact when available, else device/Emulate. */
template <typename QFn, int DIM, int T_D1D, int T_Q1D>
inline void TensorGradApply(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
#ifdef MFEM_USE_LAPACK
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      if constexpr (DIM == 3)
      {
         if (mma::lapack::TryTensorGradHost3D<QFn, T_D1D, T_Q1D>(
                NE, symmetric, b, g, bt, gt, d, x, y))
         { return; }
      }
      else
      {
         if (mma::lapack::TryTensorGradHost2D<QFn, T_D1D, T_Q1D>(
                NE, symmetric, b, g, bt, gt, d, x, y))
         { return; }
      }
   }
#endif
   if constexpr (DIM == 3)
   {
      TensorGradApply3D_Dispatch<QFn, T_D1D, T_Q1D>(
         NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
   }
   else
   {
      TensorGradApply2D_Dispatch<QFn, T_D1D, T_Q1D>(
         NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
   }
}


} // namespace mfem::internal

/// \endcond
