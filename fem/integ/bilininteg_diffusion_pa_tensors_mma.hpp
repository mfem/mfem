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
#include "mma/mma.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{


namespace mma
{
namespace lapack
{
#ifdef MFEM_USE_LAPACK
/** Named slices into one host_Arena allocation for 2D diffusion tiles. */
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

/** One 2D diffusion tile: B/G forward, metric, Bt/Gt backward. */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensors2DTile(
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
            const real_t gx = ws.gX[qy + Q1D * (qx + Q1D * b)];
            const real_t gy = ws.gY[qy + Q1D * (qx + Q1D * b)];
            const real_t O11 = Dv[idx + Q1D * Q1D * (0 + PA_SIZE * (e0 + b))];
            const real_t O21 = Dv[idx + Q1D * Q1D * (1 + PA_SIZE * (e0 + b))];
            const real_t O12 = SYM ? O21
                               : Dv[idx + Q1D * Q1D * (2 + PA_SIZE * (e0 + b))];
            const real_t O22 = SYM
                               ? Dv[idx + Q1D * Q1D * (2 + PA_SIZE * (e0 + b))]
                               : Dv[idx + Q1D * Q1D * (3 + PA_SIZE * (e0 + b))];
            ws.gX[qy + Q1D * (qx + Q1D * b)] = O11 * gx + O12 * gy;
            ws.gY[qy + Q1D * (qx + Q1D * b)] = O21 * gx + O22 * gy;
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

template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensors2D(
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
      DiffusionApplyTensors2DTile<D1D, Q1D, SYM>(
         e0, nbe, NB, B, G, Bt, Gt, Dv, X, Y, ws);
   }
}

template <int D1D, int Q1D>
inline bool TryDiffusionApplyTensors2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!mma::host_PreferTensor(D1D, Q1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(), *Gt = gt.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (symmetric)
   {
      DiffusionApplyTensors2D<D1D, Q1D, true>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   else
   {
      DiffusionApplyTensors2D<D1D, Q1D, false>(
         NE, B, G, Bt, Gt, Dv, X, Y);
   }
   return true;
}

#endif // MFEM_USE_LAPACK

} // namespace lapack

namespace blas
{

/** Dense sum-fact diffusion 3D (PADiffusion contractions). */
template <int D1D, int Q1D, bool SYM>
inline void DiffusionApplyTensors3D(
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
   const int NB = mma::TensorTileNB3D(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   }
}

/** Host 3D diffusion: dense sum-fact (same policy as mass tensors). */
template <int D1D, int Q1D>
inline bool TryDiffusionApplyTensors3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> & /*bt*/, const Array<real_t> & /*gt*/,
   const Vector &d, const Vector &x, Vector &y)
{
   if (!mma::host_PreferTensor(D1D, Q1D, NE)) { return false; }
   const real_t *B = b.Read(), *G = g.Read();
   const real_t *Dv = d.Read(), *X = x.Read();
   real_t *Y = y.ReadWrite();
   if (symmetric)
   {
      DiffusionApplyTensors3D<D1D, Q1D, true>(NE, B, G, Dv, X, Y);
   }
   else
   {
      DiffusionApplyTensors3D<D1D, Q1D, false>(NE, B, G, Dv, X, Y);
   }
   return true;
}


} // namespace blas
} // namespace mma

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
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      if constexpr (DIM == 3)
      {
         // Host dense sum-fact (no LAPACK required).
         if (mma::blas::TryDiffusionApplyTensors3D<T_D1D, T_Q1D>(
                NE, symmetric, b, g, bt, gt, d, x, y))
         { return; }
      }
#ifdef MFEM_USE_LAPACK
      else
      {
         if (mma::lapack::TryDiffusionApplyTensors2D<T_D1D, T_Q1D>(
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
