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

#ifdef MFEM_USE_LAPACK
/** Grow a thread-local scratch buffer (GCD workers reuse across tiles). */
inline real_t *TensorScratch(std::vector<real_t> &buf, size_t n)
{
   if (buf.size() < n) { buf.resize(n); }
   return buf.data();
}
#endif

/** Prefer 1D multi-RHS GEMM for tensor SUM host apply (quad/hex mass). */
inline bool PreferTensorSumLapack(int D1D, int Q1D, int NE)
{
#ifdef MFEM_USE_LAPACK
   // Registered tensor MMA is p>=3 (D1D>=4). Always take Lapack on host for
   // those sizes when the mesh is not tiny — fat-N path beats Emulate shell.
   (void)Q1D;
   return NE >= 4 && D1D >= 4;
#else
   (void)D1D; (void)Q1D; (void)NE;
   return false;
#endif
}


// ---- Mass SUM BLAS (1D multi-RHS GEMM) -------------------------------------

#ifdef MFEM_USE_LAPACK
/** One 2D mass tile: fat 1D GEMMs; zero-copy X/Y when the tile is full. */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumLapack2DTile(
   const int e0, const int nbe, const int NB,
   const real_t *B, const real_t *Bt, const real_t *Dv,
   const real_t *X, real_t *Y,
   real_t *xloc, real_t *qq, real_t *qqt, real_t *U,
   real_t *T, real_t *Tt, real_t *ytmp)
{
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

   // InterpX: qq[qx, dy + D1D*b]
   mma::LapackGemm('N', 'N', Q1D, n_xy, D1D, real_t(1), B, Q1D,
                   Xsrc, D1D, real_t(0), qq, Q1D);

   // InterpY: small D1D → per-element 'N','T'; larger → pack + fat GEMM
   if (D1D <= 5)
   {
      for (int b = 0; b < nbe; ++b)
      {
         mma::LapackGemm('N', 'T', Q1D, Q1D, D1D, real_t(1), B, Q1D,
                         qq + Q1D * D1D * b, Q1D,
                         real_t(0), U + Q1D * Q1D * b, Q1D);
      }
      for (int b = nbe; b < NB; ++b)
      {
         std::fill(U + Q1D * Q1D * b, U + Q1D * Q1D * (b + 1), real_t(0));
      }
   }
   else
   {
      for (int b = 0; b < NB; ++b)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               qqt[dy + D1D * (qx + Q1D * b)] = qq[qx + Q1D * (dy + D1D * b)];
            }
         }
      }
      mma::LapackGemm('N', 'N', Q1D, n_qy, D1D, real_t(1), B, Q1D,
                      qqt, D1D, real_t(0), U, Q1D);
   }

   for (int b = 0; b < nbe; ++b)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            U[qy + Q1D * (qx + Q1D * b)] *=
               Dv[qx + Q1D * (qy + Q1D * (e0 + b))];
         }
      }
   }
   if (D1D > 5)
   {
      for (int b = nbe; b < NB; ++b)
      {
         for (int i = 0; i < Q1D * Q1D; ++i)
         {
            U[i + Q1D * Q1D * b] = real_t(0);
         }
      }
   }

   // InterpYt: T[dy, qx + Q1D*b]
   mma::LapackGemm('N', 'N', D1D, n_qy, Q1D, real_t(1), Bt, D1D,
                   U, Q1D, real_t(0), T, D1D);

   // InterpXt
   if (D1D <= 5)
   {
      if (nbe == NB)
      {
         real_t *Ydst = Y + static_cast<size_t>(ndof) * e0;
         for (int b = 0; b < NB; ++b)
         {
            mma::LapackGemm('N', 'T', D1D, D1D, Q1D, real_t(1), Bt, D1D,
                            T + D1D * Q1D * b, D1D,
                            real_t(1), Ydst + D1D * D1D * b, D1D);
         }
      }
      else
      {
         for (int b = 0; b < nbe; ++b)
         {
            mma::LapackGemm('N', 'T', D1D, D1D, Q1D, real_t(1), Bt, D1D,
                            T + D1D * Q1D * b, D1D,
                            real_t(0), ytmp + D1D * D1D * b, D1D);
         }
         for (int b = 0; b < nbe; ++b)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y[dx + D1D * (dy + D1D * (e0 + b))] +=
                     ytmp[dx + D1D * (dy + D1D * b)];
               }
            }
         }
      }
   }
   else
   {
      for (int b = 0; b < NB; ++b)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Tt[qx + Q1D * (dy + D1D * b)] = T[dy + D1D * (qx + Q1D * b)];
            }
         }
      }
      if (nbe == NB)
      {
         real_t *Ydst = Y + static_cast<size_t>(ndof) * e0;
         mma::LapackGemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Bt, D1D,
                         Tt, Q1D, real_t(1), Ydst, D1D);
      }
      else
      {
         mma::LapackGemm('N', 'N', D1D, n_xy, Q1D, real_t(1), Bt, D1D,
                         Tt, Q1D, real_t(0), ytmp, D1D);
         for (int b = 0; b < nbe; ++b)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y[dx + D1D * (dy + D1D * (e0 + b))] +=
                     ytmp[dx + D1D * (dy + D1D * b)];
               }
            }
         }
      }
   }
}

/** 2D mass SUM via fat 1D LapackGemm + Apple GCD over element tiles. */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumLapack2D(const int NE,
                                      const real_t *B, const real_t *Bt,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   const int NB = mma::TensorLapackNB(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   const int n_xy = D1D * NB;
   const int n_qy = Q1D * NB;

#if defined(__APPLE__)
   dispatch_apply(static_cast<size_t>(ntiles), DISPATCH_APPLY_AUTO,
                  ^(size_t tile)
   {
      const int e0 = static_cast<int>(tile) * NB;
      const int nbe = std::min(NB, NE - e0);
      thread_local std::vector<real_t> xloc, qq, qqt, U, T, Tt, ytmp;
      MassApplyTensorsSumLapack2DTile<D1D, Q1D>(
         e0, nbe, NB, B, Bt, Dv, X, Y,
         TensorScratch(xloc, static_cast<size_t>(D1D) * n_xy),
         TensorScratch(qq, static_cast<size_t>(Q1D) * n_xy),
         TensorScratch(qqt, static_cast<size_t>(D1D) * n_qy),
         TensorScratch(U, static_cast<size_t>(Q1D) * n_qy),
         TensorScratch(T, static_cast<size_t>(D1D) * n_qy),
         TensorScratch(Tt, static_cast<size_t>(Q1D) * n_xy),
         TensorScratch(ytmp, static_cast<size_t>(D1D) * n_xy));
   });
#else
   std::vector<real_t> xloc(static_cast<size_t>(D1D) * n_xy);
   std::vector<real_t> qq(static_cast<size_t>(Q1D) * n_xy);
   std::vector<real_t> qqt(static_cast<size_t>(D1D) * n_qy);
   std::vector<real_t> U(static_cast<size_t>(Q1D) * n_qy);
   std::vector<real_t> T(static_cast<size_t>(D1D) * n_qy);
   std::vector<real_t> Tt(static_cast<size_t>(Q1D) * n_xy);
   std::vector<real_t> ytmp(static_cast<size_t>(D1D) * n_xy);
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      MassApplyTensorsSumLapack2DTile<D1D, Q1D>(
         e0, nbe, NB, B, Bt, Dv, X, Y,
         xloc.data(), qq.data(), qqt.data(), U.data(),
         T.data(), Tt.data(), ytmp.data());
   }
#endif
}

/** One 3D mass tile over NB_e elements (RHS = D1D² * NB_e). */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumLapack3DTile(
   const int e0, const int nbe, const int NB,
   const real_t *B, const real_t *Bt, const real_t *Dv,
   const real_t *X, real_t *Y,
   real_t *xloc, real_t *t0, real_t *t0t, real_t *t1,
   real_t *Az, real_t *U, real_t *Tz, real_t *Ay, real_t *Ty,
   real_t *Tyt, real_t *ytmp)
{
   const int nd2 = D1D * D1D;
   const int ndof = D1D * nd2;
   const int nq2 = Q1D * Q1D;
   const int n_d2 = nd2 * NB;
   const int n_qd = Q1D * D1D * NB;
   const int n_q2 = nq2 * NB;

   const real_t *Xsrc;
   if (nbe == NB)
   {
      Xsrc = X + static_cast<size_t>(ndof) * e0;
   }
   else
   {
      std::fill(xloc, xloc + static_cast<size_t>(D1D) * n_d2, real_t(0));
      for (int b = 0; b < nbe; ++b)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  xloc[dx + D1D * (dy + D1D * (dz + D1D * b))] =
                     X[dx + D1D * (dy + D1D * (dz + D1D * (e0 + b)))];
               }
            }
         }
      }
      Xsrc = xloc;
   }

   // InterpX: t0[qx, dy + D1D*(dz + D1D*b)]
   mma::LapackGemm('N', 'N', Q1D, n_d2, D1D, real_t(1), B, Q1D,
                   Xsrc, D1D, real_t(0), t0, Q1D);

   // Pack + InterpY: t1[qy, qx + Q1D*(dz + D1D*b)]
   for (int b = 0; b < NB; ++b)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               t0t[dy + D1D * (qx + Q1D * (dz + D1D * b))] =
                  t0[qx + Q1D * (dy + D1D * (dz + D1D * b))];
            }
         }
      }
   }
   mma::LapackGemm('N', 'N', Q1D, n_qd, D1D, real_t(1), B, Q1D,
                   t0t, D1D, real_t(0), t1, Q1D);

   // Pack for InterpZ: Az[dz, qx + Q1D*(qy + Q1D*b)]
   for (int b = 0; b < NB; ++b)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               Az[dz + D1D * (qx + Q1D * (qy + Q1D * b))] =
                  t1[qy + Q1D * (qx + Q1D * (dz + D1D * b))];
            }
         }
      }
   }
   mma::LapackGemm('N', 'N', Q1D, n_q2, D1D, real_t(1), B, Q1D,
                   Az, D1D, real_t(0), U, Q1D);

   // Metric: U[qz, qx + Q1D*(qy + Q1D*b)] *= Dv[qx,qy,qz,e]
   for (int b = 0; b < nbe; ++b)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qz = 0; qz < Q1D; ++qz)
            {
               U[qz + Q1D * (qx + Q1D * (qy + Q1D * b))] *=
                  Dv[qx + Q1D * (qy + Q1D * (qz + Q1D * (e0 + b)))];
            }
         }
      }
   }
   for (int b = nbe; b < NB; ++b)
   {
      for (int i = 0; i < Q1D * nq2; ++i)
      {
         U[i + Q1D * nq2 * b] = real_t(0);
      }
   }

   // InterpZt: Tz[dz, qx + Q1D*(qy + Q1D*b)]
   mma::LapackGemm('N', 'N', D1D, n_q2, Q1D, real_t(1), Bt, D1D,
                   U, Q1D, real_t(0), Tz, D1D);

   // Pack + InterpYt: Ty[dy, qx + Q1D*(dz + D1D*b)]
   for (int b = 0; b < NB; ++b)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               Ay[qy + Q1D * (qx + Q1D * (dz + D1D * b))] =
                  Tz[dz + D1D * (qx + Q1D * (qy + Q1D * b))];
            }
         }
      }
   }
   mma::LapackGemm('N', 'N', D1D, n_qd, Q1D, real_t(1), Bt, D1D,
                   Ay, Q1D, real_t(0), Ty, D1D);

   // Pack + InterpXt: y[dx, dy + D1D*(dz + D1D*b)]
   for (int b = 0; b < NB; ++b)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Tyt[qx + Q1D * (dy + D1D * (dz + D1D * b))] =
                  Ty[dy + D1D * (qx + Q1D * (dz + D1D * b))];
            }
         }
      }
   }

   if (nbe == NB)
   {
      real_t *Ydst = Y + static_cast<size_t>(ndof) * e0;
      mma::LapackGemm('N', 'N', D1D, n_d2, Q1D, real_t(1), Bt, D1D,
                      Tyt, Q1D, real_t(1), Ydst, D1D);
   }
   else
   {
      mma::LapackGemm('N', 'N', D1D, n_d2, Q1D, real_t(1), Bt, D1D,
                      Tyt, Q1D, real_t(0), ytmp, D1D);
      for (int b = 0; b < nbe; ++b)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y[dx + D1D * (dy + D1D * (dz + D1D * (e0 + b)))] +=
                     ytmp[dx + D1D * (dy + D1D * (dz + D1D * b))];
               }
            }
         }
      }
   }
}

/** 3D mass SUM via batched-element fat 1D LapackGemm + Apple GCD. */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumLapack3D(const int NE,
                                      const real_t *B, const real_t *Bt,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   const int NB = mma::TensorLapackNB3D(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   const int nd2 = D1D * D1D;
   const int nq2 = Q1D * Q1D;
   const int n_d2 = nd2 * NB;
   const int n_qd = Q1D * D1D * NB;
   const int n_q2 = nq2 * NB;

#if defined(__APPLE__)
   dispatch_apply(static_cast<size_t>(ntiles), DISPATCH_APPLY_AUTO,
                  ^(size_t tile)
   {
      const int e0 = static_cast<int>(tile) * NB;
      const int nbe = std::min(NB, NE - e0);
      thread_local std::vector<real_t> xloc, t0, t0t, t1, Az, U, Tz, Ay, Ty,
         Tyt, ytmp;
      MassApplyTensorsSumLapack3DTile<D1D, Q1D>(
         e0, nbe, NB, B, Bt, Dv, X, Y,
         TensorScratch(xloc, static_cast<size_t>(D1D) * n_d2),
         TensorScratch(t0, static_cast<size_t>(Q1D) * n_d2),
         TensorScratch(t0t, static_cast<size_t>(D1D) * n_qd),
         TensorScratch(t1, static_cast<size_t>(Q1D) * n_qd),
         TensorScratch(Az, static_cast<size_t>(D1D) * n_q2),
         TensorScratch(U, static_cast<size_t>(Q1D) * n_q2),
         TensorScratch(Tz, static_cast<size_t>(D1D) * n_q2),
         TensorScratch(Ay, static_cast<size_t>(Q1D) * n_qd),
         TensorScratch(Ty, static_cast<size_t>(D1D) * n_qd),
         TensorScratch(Tyt, static_cast<size_t>(Q1D) * n_d2),
         TensorScratch(ytmp, static_cast<size_t>(D1D) * n_d2));
   });
#else
   std::vector<real_t> xloc(static_cast<size_t>(D1D) * n_d2);
   std::vector<real_t> t0(static_cast<size_t>(Q1D) * n_d2);
   std::vector<real_t> t0t(static_cast<size_t>(D1D) * n_qd);
   std::vector<real_t> t1(static_cast<size_t>(Q1D) * n_qd);
   std::vector<real_t> Az(static_cast<size_t>(D1D) * n_q2);
   std::vector<real_t> U(static_cast<size_t>(Q1D) * n_q2);
   std::vector<real_t> Tz(static_cast<size_t>(D1D) * n_q2);
   std::vector<real_t> Ay(static_cast<size_t>(Q1D) * n_qd);
   std::vector<real_t> Ty(static_cast<size_t>(D1D) * n_qd);
   std::vector<real_t> Tyt(static_cast<size_t>(Q1D) * n_d2);
   std::vector<real_t> ytmp(static_cast<size_t>(D1D) * n_d2);
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      MassApplyTensorsSumLapack3DTile<D1D, Q1D>(
         e0, nbe, NB, B, Bt, Dv, X, Y,
         xloc.data(), t0.data(), t0t.data(), t1.data(),
         Az.data(), U.data(), Tz.data(), Ay.data(), Ty.data(),
         Tyt.data(), ytmp.data());
   }
#endif
}
#endif // MFEM_USE_LAPACK

/** Host hand sum-fact mass 2D with Apple GCD over element tiles. */
template <int D1D, int Q1D>
inline void MassApplyTensorsHandGcd2D(const int NE, const real_t *B,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   auto apply_e = [&](int e)
   {
      real_t sol_xy[Q1D][Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx) { sol_xy[qy][qx] = real_t(0); }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t sol_x[Q1D];
         for (int qx = 0; qx < Q1D; ++qx) { sol_x[qx] = real_t(0); }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X[dx + D1D * (dy + D1D * e)];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B[qx + Q1D * dx] * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t d2q = B[qy + Q1D * dy];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] *= Dv[qx + Q1D * (qy + Q1D * e)];
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t sol_x[D1D];
         for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = real_t(0); }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t s = sol_xy[qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += B[qx + Q1D * dx] * s; // Bt(dx,qx)
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t q2d = B[qy + Q1D * dy]; // Bt(dy,qy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y[dx + D1D * (dy + D1D * e)] += q2d * sol_x[dx];
            }
         }
      }
   };
   // Tile GCD: quads are too cheap per element for one-task-per-e.
   const int NB = mma::TensorLapackNB(D1D, Q1D);
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

/** Host hand sum-fact mass 3D with Apple GCD over element tiles. */
template <int D1D, int Q1D>
inline void MassApplyTensorsHandGcd3D(const int NE, const real_t *B,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   auto apply_e = [&](int e)
   {
      real_t sol_xyz[Q1D][Q1D][Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xyz[qz][qy][qx] = real_t(0);

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t sol_xy[Q1D][Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xy[qy][qx] = real_t(0);
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t sol_x[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) { sol_x[qx] = real_t(0); }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X[dx + D1D * (dy + D1D * (dz + D1D * e))];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B[qx + Q1D * dx] * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = B[qy + Q1D * dy];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz = B[qz + Q1D * dz];
            for (int qy = 0; qy < Q1D; ++qy)
               for (int qx = 0; qx < Q1D; ++qx)
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xyz[qz][qy][qx] *=
                  Dv[qx + Q1D * (qy + Q1D * (qz + Q1D * e))];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t sol_xy[D1D][D1D];
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
               sol_xy[dy][dx] = real_t(0);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t sol_x[D1D];
            for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = real_t(0); }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += B[qx + Q1D * dx] * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy = B[qy + Q1D * dy];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz = B[qz + Q1D * dz];
            for (int dy = 0; dy < D1D; ++dy)
               for (int dx = 0; dx < D1D; ++dx)
                  Y[dx + D1D * (dy + D1D * (dz + D1D * e))] +=
                     wz * sol_xy[dy][dx];
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
inline bool TryMassApplyTensorsSumLapack2D(const int NE,
                                         const Array<real_t> &b,
                                         const Array<real_t> &bt,
                                         const Vector &d,
                                         const Vector &x,
                                         Vector &y)
{
   if (!PreferTensorSumLapack(D1D, Q1D, NE)) { return false; }
   // Tiled hand GCD: parallel SUM-quality kernels (quads need tiles, not 1-e tasks).
   if (D1D <= 6)
   {
      MassApplyTensorsHandGcd2D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                          x.Read(), y.ReadWrite());
      return true;
   }
#ifdef MFEM_USE_LAPACK
   MassApplyTensorsSumLapack2D<D1D, Q1D>(NE, b.Read(), bt.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#else
   (void)bt;
   MassApplyTensorsHandGcd2D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#endif
}

template <int D1D, int Q1D>
inline bool TryMassApplyTensorsSumLapack3D(const int NE,
                                         const Array<real_t> &b,
                                         const Array<real_t> &bt,
                                         const Vector &d,
                                         const Vector &x,
                                         Vector &y)
{
   if (!PreferTensorSumLapack(D1D, Q1D, NE)) { return false; }
   if (D1D <= 6)
   {
      MassApplyTensorsHandGcd3D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                          x.Read(), y.ReadWrite());
      return true;
   }
#ifdef MFEM_USE_LAPACK
   MassApplyTensorsSumLapack3D<D1D, Q1D>(NE, b.Read(), bt.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#else
   (void)bt;
   MassApplyTensorsHandGcd3D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#endif
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors3D(const int NE,
                                  const Array<real_t> &b,
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
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 3D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::MassNB3D<T_D1D, T_Q1D>()
                        : mma::MassNB3DRuntime(D1D);
   // Host forall_3D workers all see getThreadIdxX()==0; keep one thread to avoid
   // races on MFEM_SHARED (device uses full thread count + Emulate/MMA).
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::MassThreads3D<T_D1D, T_Q1D>()
                                 : mma::MassThreads3DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D * Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   // Serial multi-element batch: shared B once; one element smem at a time.
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpZMass<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1, D, e);
         MFEM_SYNC_THREAD;
         mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors2D(const int NE,
                                  const Array<real_t> &b,
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
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 2D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                        : mma::NB2DRuntime(D1D);
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                                 : mma::Threads2DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX2D<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         mma::InterpX2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpY2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
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
               sm1[idx] = sm0[idx] * D(idx, e);
            }
         }
         MFEM_SYNC_THREAD;

         mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpXt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

/** Runtime overloads for Fallback / unregistered (D1D,Q1D). */
inline void MmaMassApplyTensors2D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &bt,
                                  const Vector &d, const Vector &x,
                                  Vector &y,
                                  const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MmaMassApplyTensors2D<0, 0>(NE, b, d, x, y, d1d, q1d);
}

inline void MmaMassApplyTensors3D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &bt,
                                  const Vector &d, const Vector &x,
                                  Vector &y,
                                  const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MmaMassApplyTensors3D<0, 0>(NE, b, d, x, y, d1d, q1d);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaMassApplyTensors(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   // Host: 1D BLAS when profitable, else MMA shell (Interp/Grad Emulate).
   // Device: MMA shell (real MMA or fine-grained Emulate).
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      if constexpr (DIM == 3)
      {
         if (TryMassApplyTensorsSumLapack3D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
         { return; }
      }
      else
      {
         if (TryMassApplyTensorsSumLapack2D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
         { return; }
      }
   }
   if constexpr (DIM == 3)
   {
      MmaMassApplyTensors3D<T_D1D, T_Q1D>(NE, b, d, x, y, d1d, q1d);
   }
   else
   {
      MmaMassApplyTensors2D<T_D1D, T_Q1D>(NE, b, d, x, y, d1d, q1d);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

// Fallback defined in bilininteg_mass_pa_tensors_mma.cpp (MMA shell runtime).

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
