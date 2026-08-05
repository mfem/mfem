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

#include "mma/mma.hpp"
#include "../bilininteg.hpp"

#ifdef MFEM_USE_LAPACK
#include <vector>
#endif
namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

namespace mma
{

// ---------------------------------------------------------------------------
// Simplex diffusion PA helpers (NB / Q-tile / host dense / metrics)
// ---------------------------------------------------------------------------

/** Max diffusion NB for full-NQ X+DIM*U under a byte cap. */
template <int DIM, int D1D, int QND>
constexpr int DiffusionMmaNBFullNqAt(int bytes_cap)
{
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int per_batch_col = X_LD + DIM * U_LD;
   const int max_nb = bytes_cap / (int(sizeof(real_t)) * per_batch_col);
   if (D1D && QND)
   {
      if (NBATCH <= max_nb) { return NBATCH; }
#if defined(MFEM_USE_HIP)
      return max_nb > 0 ? max_nb : 1;
#else
      const int nb = (max_nb / mmaN) * mmaN;
      return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
#endif
   }
   if (mmaN <= max_nb) { return mmaN; }
   return max_nb > 0 ? max_nb : 1;
}

/** Diffusion full-NQ NB.
    CUDA: default shapes plan under Prefer (48KB). Exception: BP3 tet p=7
          (D1D==8, QND==123) uses PerBlock dynamic smem for full-NQ NBATCH=16
          (measured win on H100). Q-tile path uses Prefer budget. */
template <int DIM, int D1D, int QND>
constexpr int DiffusionMmaNBFullNq()
{
#if defined(MFEM_USE_CUDA)
   // BP3 tet p=7 (q=2p-2): allow dynamic smem to restore NBATCH=16.
   if (DIM == 3 && D1D == 8 && QND == 123)
   {
      return DiffusionMmaNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
   }
   return DiffusionMmaNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPrefer);
#else
   return DiffusionMmaNBFullNqAt<DIM, D1D, QND>(SharedMemBytesPerBlock);
#endif
}

/** True when diffusion should Q-tile.
    HIP: when full-NQ would force NB < NBATCH (16).
    CUDA: only when full-NQ cannot keep NB >= mmaN even with dynamic smem. */
template <int DIM, int D1D, int QND>
constexpr bool DiffusionUseQTile()
{
   if (!(DIM == 3 && D1D && QND)) { return false; }
#if defined(MFEM_USE_HIP)
   return DiffusionMmaNBFullNq<DIM, D1D, QND>() < NBATCH;
#else
   return DiffusionMmaNBFullNq<DIM, D1D, QND>() < mmaN;
#endif
}

/** Largest TQ (multiple of MMA M) that fits X + DIM·U at a given NB and byte cap. */
template <int DIM, int D1D, int QND, int NB>
constexpr int DiffusionQTileForNBAt(int bytes_cap)
{
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int step = mmaM;

   int best = step;
   for (int tq = step; tq <= MQ; tq += step)
   {
      const int U_LD = PadLdBank<MAP>(tq);
      const int bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

template <int DIM, int D1D, int QND, int NB>
constexpr int DiffusionQTileForNB()
{
   // Keep Q-tiles in the occupancy-friendly 48KB/Prefer budget (H100).
   return DiffusionQTileForNBAt<DIM, D1D, QND, NB>(SharedMemBytesPrefer);
}

/** Q-tile element batch: HIP keeps NBATCH; CUDA picks NB in {8,16} with fewer passes. */
template <int DIM, int D1D, int QND>
constexpr int DiffusionQTileNB()
{
#if defined(MFEM_USE_HIP)
   return NBATCH;
#else
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int tq16 = DiffusionQTileForNB<DIM, D1D, QND, NBATCH>();
   constexpr int tq8 = DiffusionQTileForNB<DIM, D1D, QND, mmaN>();
   constexpr int passes16 = (MQ + tq16 - 1) / tq16;
   constexpr int passes8 = (MQ + tq8 - 1) / tq8;
   return (passes8 < passes16) ? mmaN : NBATCH;
#endif
}

/** Largest TQ for the selected Q-tile NB. */
template <int DIM, int D1D, int QND>
constexpr int DiffusionQTileFor()
{
   return DiffusionQTileForNB<DIM, D1D, QND,
          DiffusionQTileNB<DIM, D1D, QND>()>();
}

/** @deprecated prefer DiffusionQTileFor — kept as MMA-M hint. */
constexpr int DiffusionQTile = mmaM;

template <int DIM, int D1D, int QND>
constexpr int DiffusionMmaNB()
{
   if constexpr (DiffusionUseQTile<DIM, D1D, QND>())
   {
      return DiffusionQTileNB<DIM, D1D, QND>();
   }
   return DiffusionMmaNBFullNq<DIM, D1D, QND>();
}

/** Runtime diffusion full-NQ NB under a byte cap. */
inline int DiffusionMmaNBFullNqAtRuntime(int ndof, int nq, int pa_comps,
                                         int bytes_cap)
{
   const int x_ld = PadLdBankRuntime(ndof);
   const int u_ld = PadLdBankRuntime(nq);
   const int denom = int(sizeof(real_t)) * (x_ld + pa_comps * u_ld);
   const int max_nb = (denom > 0) ? (bytes_cap / denom) : 0;
   if (NBATCH <= max_nb) { return NBATCH; }
#if defined(MFEM_USE_HIP)
   return max_nb > 0 ? max_nb : 1;
#else
   const int nb = (max_nb / mmaN) * mmaN;
   return nb > 0 ? nb : (max_nb > 0 ? max_nb : 1);
#endif
}

inline int DiffusionMmaNBFullNqRuntime(int /*dim*/, int ndof, int nq,
                                       int pa_comps)
{
#if defined(MFEM_USE_CUDA)
   return DiffusionMmaNBFullNqAtRuntime(ndof, nq, pa_comps,
                                        SharedMemBytesPrefer);
#else
   return DiffusionMmaNBFullNqAtRuntime(ndof, nq, pa_comps,
                                        SharedMemBytesPerBlock);
#endif
}

/** True when runtime diffusion should Q-tile (DIM==3 only). */
inline bool DiffusionUseQTileRuntime(int dim, int ndof, int nq, int pa_comps)
{
   if (dim != 3) { return false; }
#if defined(MFEM_USE_HIP)
   return DiffusionMmaNBFullNqRuntime(dim, ndof, nq, pa_comps) < NBATCH;
#else
   return DiffusionMmaNBFullNqRuntime(dim, ndof, nq, pa_comps) < mmaN;
#endif
}

inline int DiffusionQTileForNBAtRuntime(int ndof, int nq, int nb, int bytes_cap)
{
   const int x_ld = PadLdBankRuntime(ndof);
   const int step = mmaM;
   int best = step;
   for (int tq = step; tq <= nq; tq += step)
   {
      const int u_ld = PadLdBankRuntime(tq);
      const int bytes = int(sizeof(real_t)) * (x_ld + 3 * u_ld) * nb;
      if (bytes > bytes_cap) { break; }
      best = tq;
   }
   return best;
}

inline int DiffusionQTileNBRuntime(int ndof, int nq)
{
#if defined(MFEM_USE_HIP)
   MFEM_CONTRACT_VAR(ndof);
   MFEM_CONTRACT_VAR(nq);
   return NBATCH;
#else
   const int tq16 = DiffusionQTileForNBAtRuntime(ndof, nq, NBATCH,
                                                 SharedMemBytesPrefer);
   const int tq8 = DiffusionQTileForNBAtRuntime(ndof, nq, mmaN,
                                                SharedMemBytesPrefer);
   const int passes16 = (nq + tq16 - 1) / tq16;
   const int passes8 = (nq + tq8 - 1) / tq8;
   return (passes8 < passes16) ? mmaN : NBATCH;
#endif
}

inline int DiffusionQTileForRuntime(int ndof, int nq)
{
   const int nb = DiffusionQTileNBRuntime(ndof, nq);
   return DiffusionQTileForNBAtRuntime(ndof, nq, nb, SharedMemBytesPrefer);
}

/** Runtime diffusion NB (Q-tile or full-NQ), matching DiffusionMmaNB. */
inline int DiffusionMmaNBRuntime(int dim, int ndof, int nq, int pa_comps)
{
   if (DiffusionUseQTileRuntime(dim, ndof, nq, pa_comps))
   {
      return DiffusionQTileNBRuntime(ndof, nq);
   }
   return DiffusionMmaNBFullNqRuntime(dim, ndof, nq, pa_comps);
}

namespace blas
{

/** Multi-RHS NB for specialized diffusion. 2D uses NB=32. */
template <int DIM, int NQ>
constexpr int DiffusionNB()
{
   if constexpr (DIM == 2) { return 32; }
   return (NQ > 60) ? 2 : 4;
}

} // namespace blas

// ---- Diffusion metric at one quadrature point (vector length DIM) ----------

/** In-place metric: u[0:DIM) := O(D(q,e)) * u. PA D is (q, pa, e). */
template <int DIM, bool SYM>
MFEM_HOST_DEVICE inline void ApplyDiffusionMetricVec(real_t *u,
                                                     const real_t *Dv,
                                                     int q, int nq, int e,
                                                     int pa_size)
{
   if constexpr (DIM == 2)
   {
      const real_t u1 = u[0], u2 = u[1];
      const real_t O11 = Dv[q + nq * (0 + pa_size * e)];
      const real_t O21 = Dv[q + nq * (1 + pa_size * e)];
      if constexpr (SYM)
      {
         const real_t O22 = Dv[q + nq * (2 + pa_size * e)];
         u[0] = O11 * u1 + O21 * u2;
         u[1] = O21 * u1 + O22 * u2;
      }
      else
      {
         const real_t O12 = Dv[q + nq * (2 + pa_size * e)];
         const real_t O22 = Dv[q + nq * (3 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2;
         u[1] = O21 * u1 + O22 * u2;
      }
   }
   else
   {
      const real_t u1 = u[0], u2 = u[1], u3 = u[2];
      const real_t O11 = Dv[q + nq * (0 + pa_size * e)];
      const real_t O12 = Dv[q + nq * (1 + pa_size * e)];
      const real_t O13 = Dv[q + nq * (2 + pa_size * e)];
      if constexpr (SYM)
      {
         const real_t O22 = Dv[q + nq * (3 + pa_size * e)];
         const real_t O23 = Dv[q + nq * (4 + pa_size * e)];
         const real_t O33 = Dv[q + nq * (5 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2 + O13 * u3;
         u[1] = O12 * u1 + O22 * u2 + O23 * u3;
         u[2] = O13 * u1 + O23 * u2 + O33 * u3;
      }
      else
      {
         const real_t O21 = Dv[q + nq * (3 + pa_size * e)];
         const real_t O22 = Dv[q + nq * (4 + pa_size * e)];
         const real_t O23 = Dv[q + nq * (5 + pa_size * e)];
         const real_t O31 = Dv[q + nq * (6 + pa_size * e)];
         const real_t O32 = Dv[q + nq * (7 + pa_size * e)];
         const real_t O33 = Dv[q + nq * (8 + pa_size * e)];
         u[0] = O11 * u1 + O12 * u2 + O13 * u3;
         u[1] = O21 * u1 + O22 * u2 + O23 * u3;
         u[2] = O31 * u1 + O32 * u2 + O33 * u3;
      }
   }
}

namespace blas
{

/** Fused 2D diffusion for NQ=1 (BP3 p=1): per-element, no tile buffers. */
template <int NDOF, bool SYM>
inline void DiffusionApplyNQ1_2D(int NE, const real_t *G, const real_t *Dv,
                                 const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? 3 : 4;
   const real_t *G0 = G;
   const real_t *G1 = G + NDOF;
   for (int e = 0; e < NE; ++e)
   {
      const real_t *Xe = X + NDOF * e;
      real_t u0 = real_t(0), u1 = real_t(0);
      for (int i = 0; i < NDOF; ++i)
      {
         const real_t xi = Xe[i];
         u0 += G0[i] * xi;
         u1 += G1[i] * xi;
      }
      real_t u[2] = {u0, u1};
      ApplyDiffusionMetricVec<2, SYM>(u, Dv, 0, 1, e, PA_SIZE);
      real_t *Ye = Y + NDOF * e;
      for (int i = 0; i < NDOF; ++i)
      {
         Ye[i] += G0[i] * u[0] + G1[i] * u[1];
      }
   }
}

/** Metric on BLAS column-major U planes: uloc[d * nq * NB + q + nq * b]. */
template <int DIM, bool SYM>
inline void ApplyDiffusionMetricColMajor(real_t *uloc, const real_t *Dv,
                                         int nq, int e0, int NE, int NB,
                                         int pa_size)
{
   if constexpr (DIM == 2 && SYM)
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { break; }
         for (int q = 0; q < nq; ++q)
         {
            const real_t u1 = uloc[0 * nq * NB + q + nq * b];
            const real_t u2 = uloc[1 * nq * NB + q + nq * b];
            const real_t O11 = Dv[q + nq * (0 + pa_size * e)];
            const real_t O21 = Dv[q + nq * (1 + pa_size * e)];
            const real_t O22 = Dv[q + nq * (2 + pa_size * e)];
            uloc[0 * nq * NB + q + nq * b] = O11 * u1 + O21 * u2;
            uloc[1 * nq * NB + q + nq * b] = O21 * u1 + O22 * u2;
         }
      }
   }
   else
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { break; }
         for (int q = 0; q < nq; ++q)
         {
            real_t u[DIM];
            for (int d = 0; d < DIM; ++d)
            {
               u[d] = uloc[static_cast<size_t>(d) * nq * NB + q +
                           static_cast<size_t>(nq) * b];
            }
            ApplyDiffusionMetricVec<DIM, SYM>(u, Dv, q, nq, e, pa_size);
            for (int d = 0; d < DIM; ++d)
            {
               uloc[static_cast<size_t>(d) * nq * NB + q +
                    static_cast<size_t>(nq) * b] = u[d];
            }
         }
      }
   }
}

/** Metric on hand b-innermost U: uloc[(d * NQ + q) * NB + b]. */
template <int DIM, int NQ, int NB, bool SYM>
inline void DiffusionMetric(real_t *uloc, const real_t *Dv,
                            int e0, int NE, int pa_size)
{
   if constexpr (DIM == 2 && SYM)
   {
      for (int q = 0; q < NQ; ++q)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            const real_t u1 = uloc[(0 * NQ + q) * NB + b];
            const real_t u2 = uloc[(1 * NQ + q) * NB + b];
            const real_t O11 = Dv[q + NQ * (0 + pa_size * e)];
            const real_t O21 = Dv[q + NQ * (1 + pa_size * e)];
            const real_t O22 = Dv[q + NQ * (2 + pa_size * e)];
            uloc[(0 * NQ + q) * NB + b] = O11 * u1 + O21 * u2;
            uloc[(1 * NQ + q) * NB + b] = O21 * u1 + O22 * u2;
         }
      }
   }
   else
   {
      for (int q = 0; q < NQ; ++q)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            real_t u[DIM];
            for (int d = 0; d < DIM; ++d)
            {
               u[d] = uloc[(d * NQ + q) * NB + b];
            }
            ApplyDiffusionMetricVec<DIM, SYM>(u, Dv, q, NQ, e, pa_size);
            for (int d = 0; d < DIM; ++d)
            {
               uloc[(d * NQ + q) * NB + b] = u[d];
            }
         }
      }
   }
}

/** Diffusion hand: forward all GradP components into uloc[(d*NQ+q)*NB+b]. */
template <int DIM, int NDOF, int NQ, int NB>
inline void DiffusionForward(const real_t *G, const real_t *xloc,
                             real_t *uloc)
{
   for (int d = 0; d < DIM; ++d)
   {
      const real_t *Gd = G + static_cast<size_t>(d) * NQ * NDOF;
      blas::Gemm<NDOF, NQ, NB, false>(Gd, xloc, uloc + d * NQ * NB,
                                      nullptr, 0, 0);
   }
}

/** blas_: Y += sum_d G_d^T U_d. */
template <int DIM, int NDOF, int NQ, int NB>
inline void DiffusionBackward(const real_t *G, const real_t *uloc,
                              real_t *Y, int e0, int NE)
{
   for (int i = 0; i < NDOF; ++i)
   {
      real_t yb[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { yb[b] = real_t(0); }
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * NQ * NDOF;
         const real_t *Ud = uloc + d * NQ * NB;
         for (int q = 0; q < NQ; ++q)
         {
            const real_t gqi = Gd[q + NQ * i];
            MFEM_UNROLL(NB)
            for (int b = 0; b < NB; ++b)
            {
               yb[b] += gqi * Ud[q * NB + b];
            }
         }
      }
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e < NE) { Y[i + NDOF * e] += yb[b]; }
      }
   }
}

} // namespace blas

namespace lapack
{

#ifdef MFEM_USE_LAPACK
/** Diffusion: U_d = G_d X, metric, Y += sum G_d^T V_d.
    Full tiles GEMM against X/Y; partial trailing tile packs/scatters. */
template <int DIM, bool SYM>
inline void DiffusionApply(int NE, int nq, int ndof, const real_t *G,
                           const real_t *Dv, const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int NB = lapack::NB(nq, ndof);
   std::vector<real_t> uloc(static_cast<size_t>(DIM) * nq * NB);
   lapack::ElementTiles(NE, ndof, NB, X, Y,
                        [&](int e0, int /*nbe*/, int nb, const real_t *Xsrc,
                            real_t *Yout)
   {
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         real_t *Ud = uloc.data() + static_cast<size_t>(d) * nq * nb;
         lapack::Gemm('N', 'N', nq, nb, ndof, real_t(1), Gd, nq, Xsrc, ndof,
                      real_t(0), Ud, nq);
      }
      ApplyDiffusionMetricColMajor<DIM, SYM>(uloc.data(), Dv, nq, e0, NE, nb,
                                             PA_SIZE);
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         const real_t *Vd = uloc.data() + static_cast<size_t>(d) * nq * nb;
         lapack::Gemm('T', 'N', ndof, nb, nq, real_t(1), Gd, nq, Vd, nq,
                      real_t(1), Yout, ndof);
      }
   });
}
#endif // MFEM_USE_LAPACK

} // namespace lapack

namespace blas
{

template <int DIM, int NDOF, int NQ, bool SYM>
inline void DiffusionApply(int NE, const real_t *G,
                           const real_t *Dv, const real_t *X, real_t *Y)
{
   if constexpr (DIM == 2 && NQ == 1)
   {
      DiffusionApplyNQ1_2D<NDOF, SYM>(NE, G, Dv, X, Y);
      return;
   }
   constexpr int NB = DiffusionNB<DIM, NQ>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[DIM * NQ * NB];
      blas::PackX<NDOF, NB>(X, e0, NE, xloc);
      DiffusionForward<DIM, NDOF, NQ, NB>(G, xloc, uloc);
      DiffusionMetric<DIM, NQ, NB, SYM>(uloc, Dv, e0, NE, PA_SIZE);
      DiffusionBackward<DIM, NDOF, NQ, NB>(G, uloc, Y, e0, NE);
   }
}

/** Serial dense diffusion for one element; u_scratch holds DIM*nq reals. */
template <int DIM, bool SYM>
MFEM_HOST_DEVICE inline void DiffusionApplyDenseElement(
   const int nq, const int ndof, const real_t *G, const real_t *Dv_e,
   const real_t *X_e, real_t *Y_e, real_t *u_scratch)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   for (int d = 0; d < DIM; ++d)
   {
      for (int q = 0; q < nq; ++q)
      {
         real_t s = 0.0;
         for (int i = 0; i < ndof; ++i)
         {
            s += G[q + nq * (i + ndof * d)] * X_e[i];
         }
         u_scratch[d * nq + q] = s;
      }
   }
   for (int q = 0; q < nq; ++q)
   {
      real_t uv[DIM];
      for (int d = 0; d < DIM; ++d) { uv[d] = u_scratch[d * nq + q]; }
      // Dv layout (q, pa, e) with e=0 relative base.
      ApplyDiffusionMetricVec<DIM, SYM>(uv, Dv_e, q, nq, 0, PA_SIZE);
      for (int d = 0; d < DIM; ++d) { u_scratch[d * nq + q] = uv[d]; }
   }
   for (int i = 0; i < ndof; ++i)
   {
      real_t s = 0.0;
      for (int d = 0; d < DIM; ++d)
      {
         for (int q = 0; q < nq; ++q)
         {
            s += G[q + nq * (i + ndof * d)] * u_scratch[d * nq + q];
         }
      }
      Y_e[i] += s;
   }
}

template <int DIM, bool SYM>
inline void DiffusionApplyRuntime(int NE, int nq, int ndof, const real_t *G,
                                  const real_t *Dv, const real_t *X,
                                  real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(DIM * nq)));
      DiffusionApplyDenseElement<DIM, SYM>(
         nq, ndof, G, Dv + nq * PA_SIZE * e, X + ndof * e, Y + ndof * e, u);
   }
}

} // namespace blas

} // namespace mma

void PADiffusionSetupSimplexFromNodes(const int dim,
                                      const int coeffDim,
                                      const int NE,
                                      const int NQ,
                                      const int ND,
                                      const Array<real_t> &w,
                                      const Array<real_t> &g,
                                      const Vector &nodes_e,
                                      const Vector &c,
                                      Vector &d);

template <int DIM, int U_LD, int NB, bool SYM, int QND, typename TD>
MFEM_HOST_DEVICE inline
void ApplyDiffusionMetric(real_t *UV, TD D,
                          const int e0, const int NE,
                          const int tid,
                          const int nthreads)
{
   for (int i = tid; i < QND * NB; i += nthreads)
   {
      const int b = i / QND;
      const int q = i - b * QND;
      const int e = e0 + b;
      if (e >= NE || q >= QND) { continue; }
      if constexpr (DIM == 2)
      {
         const real_t u1 = UV[0 * U_LD * NB + q + U_LD * b];
         const real_t u2 = UV[1 * U_LD * NB + q + U_LD * b];
         const real_t O11 = D(q, 0, e);
         const real_t O21 = D(q, 1, e);
         if constexpr (SYM)
         {
            const real_t O22 = D(q, 2, e);
            UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O21 * u2;
            UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2;
         }
         else
         {
            const real_t O12 = D(q, 2, e);
            const real_t O22 = D(q, 3, e);
            UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2;
            UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2;
         }
      }
      else
      {
         const real_t u1 = UV[0 * U_LD * NB + q + U_LD * b];
         const real_t u2 = UV[1 * U_LD * NB + q + U_LD * b];
         const real_t u3 = UV[2 * U_LD * NB + q + U_LD * b];
         const real_t O11 = D(q, 0, e);
         const real_t O12 = D(q, 1, e);
         const real_t O13 = D(q, 2, e);
         if constexpr (SYM)
         {
            const real_t O22 = D(q, 3, e);
            const real_t O23 = D(q, 4, e);
            const real_t O33 = D(q, 5, e);
            UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * U_LD * NB + q + U_LD * b] = O12 * u1 + O22 * u2 + O23 * u3;
            UV[2 * U_LD * NB + q + U_LD * b] = O13 * u1 + O23 * u2 + O33 * u3;
         }
         else
         {
            const real_t O21 = D(q, 3, e);
            const real_t O22 = D(q, 4, e);
            const real_t O23 = D(q, 5, e);
            const real_t O31 = D(q, 6, e);
            const real_t O32 = D(q, 7, e);
            const real_t O33 = D(q, 8, e);
            UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2 + O23 * u3;
            UV[2 * U_LD * NB + q + U_LD * b] = O31 * u1 + O32 * u2 + O33 * u3;
         }
      }
   }
}

/** Q-tile metric: UV holds TQ local rows; D indexed at global q0+q. */
template <int DIM, int U_LD, int NB, bool SYM, int QND, typename TD>
MFEM_HOST_DEVICE inline
void ApplyDiffusionMetricQTile(real_t *UV, TD D,
                               const int e0, const int NE,
                               const int q0, const int nq_tile,
                               const int tid,
                               const int nthreads)
{
   for (int i = tid; i < nq_tile * NB; i += nthreads)
   {
      const int b = i / nq_tile;
      const int qloc = i - b * nq_tile;
      const int e = e0 + b;
      const int q = q0 + qloc;
      if (e >= NE || q >= QND) { continue; }
      if constexpr (DIM == 3)
      {
         const real_t u1 = UV[0 * U_LD * NB + qloc + U_LD * b];
         const real_t u2 = UV[1 * U_LD * NB + qloc + U_LD * b];
         const real_t u3 = UV[2 * U_LD * NB + qloc + U_LD * b];
         const real_t O11 = D(q, 0, e);
         const real_t O12 = D(q, 1, e);
         const real_t O13 = D(q, 2, e);
         if constexpr (SYM)
         {
            const real_t O22 = D(q, 3, e);
            const real_t O23 = D(q, 4, e);
            const real_t O33 = D(q, 5, e);
            UV[0 * U_LD * NB + qloc + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * U_LD * NB + qloc + U_LD * b] = O12 * u1 + O22 * u2 + O23 * u3;
            UV[2 * U_LD * NB + qloc + U_LD * b] = O13 * u1 + O23 * u2 + O33 * u3;
         }
         else
         {
            const real_t O21 = D(q, 3, e);
            const real_t O22 = D(q, 4, e);
            const real_t O23 = D(q, 5, e);
            const real_t O31 = D(q, 6, e);
            const real_t O32 = D(q, 7, e);
            const real_t O33 = D(q, 8, e);
            UV[0 * U_LD * NB + qloc + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * U_LD * NB + qloc + U_LD * b] = O21 * u1 + O22 * u2 + O23 * u3;
            UV[2 * U_LD * NB + qloc + U_LD * b] = O31 * u1 + O32 * u2 + O33 * u3;
         }
      }
   }
}

template<int DIM, int D1D, int QND, bool SYM>
MFEM_HOST_DEVICE inline
void MmaDiffusionApplySimplex_Batch(const int e0,
                                    const int NE,
                                    const real_t *g,
                                    const real_t *d,
                                    const real_t *x,
                                    real_t *y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int BASIS_DIM = mma::SimplexNdof<DIM, D1D>();
   constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = mma::PadLdBank<MAP>(BASIS_DIM);
   constexpr int NB = mma::DiffusionMmaNB<DIM, D1D, QND>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
   constexpr int ndof = BASIS_DIM;

   const auto D = Reshape(d, QND, PA_SIZE, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);

   const int tid = mma::getThreadIdx();
   const int nthreads = mma::getBlockNthreads();

   // Q-tiled path: keep NB=16 with TQ-row U planes (CUDA DMMA / HIP MFMA N util).
   if constexpr (mma::DiffusionUseQTile<DIM, D1D, QND>())
   {
      constexpr int TQ = mma::DiffusionQTileFor<DIM, D1D, QND>();
      constexpr int U_LD = mma::PadLdBank<MAP>(TQ);
      static_assert(sizeof(real_t) * (X_LD + DIM * U_LD) * NB <=
                    mma::SharedMemBytesPerBlock,
                    "Q-tiled diffusion smem exceeds SharedMemBytesPerBlock");
      struct alignas(16) SmemQ
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
      MFEM_SIMPLEX_MMA_SMEM(SmemQ, sm);

      if constexpr (mma::DeviceGemmEnabled())
      {
         mma::SmemMatAcc<X_LD> Xacc {sm.XY};
         mma::YBatchAcc Yacc{y, ndof, e0};
         mma::SmemMatAcc<U_LD> U0{sm.UV + 0 * U_LD * NB};
         mma::SmemMatAcc<U_LD> U1{sm.UV + 1 * U_LD * NB};
         mma::SmemMatAcc<U_LD> U2{sm.UV + 2 * U_LD * NB};

         mma::LoadXToSmem(sm.XY, X, e0, NE, ndof, X_LD, NB, tid, nthreads);
         MFEM_SYNC_THREAD;

         for (int q0 = 0; q0 < QND; q0 += TQ)
         {
            const int nq_tile = (QND - q0 < TQ) ? (QND - q0) : TQ;
            mma::GAccQTile A0{g, QND, ndof, 0, q0};
            mma::GAccQTile A1{g, QND, ndof, 1, q0};
            mma::GAccQTile A2{g, QND, ndof, 2, q0};
            mma::Gemm3<MAP>(nq_tile, ndof, NB, A0, A1, A2, Xacc,
                            U0, U1, U2, e0, NE);
            MFEM_SYNC_THREAD;
            ApplyDiffusionMetricQTile<DIM, U_LD, NB, SYM, QND>(
               sm.UV, D, e0, NE, q0, nq_tile, tid, nthreads);
            MFEM_SYNC_THREAD;
            mma::GemmT3<MAP>(nq_tile, ndof, NB, A0, A1, A2, U0, U1, U2,
                             Yacc, e0, NE);
            MFEM_SYNC_THREAD;
         }
      }
      else
      {
         auto Y = DeviceMatrix(y, ndof, NE);
         if (tid == 0)
         {
            for (int b = 0; b < NB; ++b)
            {
               const int e = e0 + b;
               if (e >= NE) { continue; }
               for (int i = 0; i < X_LD; ++i)
               {
                  sm.XY[i + X_LD * b] = (i < ndof) ? X(i, e) : real_t(0);
               }
               for (int q0 = 0; q0 < QND; q0 += TQ)
               {
                  const int nq_tile = (QND - q0 < TQ) ? (QND - q0) : TQ;
                  for (int c = 0; c < DIM; ++c)
                  {
                     for (int qloc = 0; qloc < nq_tile; ++qloc)
                     {
                        real_t u = 0.0;
                        const int q = q0 + qloc;
                        for (int i = 0; i < ndof; ++i)
                        {
                           u += g[q + QND * (i + ndof * c)] *
                                sm.XY[i + X_LD * b];
                        }
                        sm.UV[c * U_LD * NB + qloc + U_LD * b] = u;
                     }
                  }
                  ApplyDiffusionMetricQTile<DIM, U_LD, NB, SYM, QND>(
                     sm.UV, D, e0, NE, q0, nq_tile, 0, 1);
                  for (int i = 0; i < ndof; ++i)
                  {
                     real_t yi = 0.0;
                     for (int c = 0; c < DIM; ++c)
                     {
                        for (int qloc = 0; qloc < nq_tile; ++qloc)
                        {
                           const int q = q0 + qloc;
                           yi += g[q + QND * (i + ndof * c)] *
                                 sm.UV[c * U_LD * NB + qloc + U_LD * b];
                        }
                     }
                     Y(i, e) += yi;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else
   {
      // Full-nq path when Q-tiling is not needed.
      constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
      static_assert(sizeof(real_t) * (X_LD + DIM * U_LD) * NB <=
                    mma::SharedMemBytesPerBlock,
                    "Diffusion simplex MMA shared memory exceeds SharedMemBytesPerBlock");
      struct alignas(16) Smem
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
      MFEM_SIMPLEX_MMA_SMEM(Smem, sm);

      if constexpr (mma::DeviceGemmEnabled())
      {
         mma::SmemMatAcc<X_LD> Xacc {sm.XY};
         mma::YBatchAcc Yacc{y, ndof, e0};
         mma::NullDAcc nullD;

         mma::LoadXToSmem(sm.XY, X, e0, NE, ndof, X_LD, NB, tid, nthreads);
         MFEM_SYNC_THREAD;

         if constexpr (DIM == 2)
         {
            MFEM_UNROLL(2)
            for (int c = 0; c < 2; ++c)
            {
               mma::GAcc A{g, QND, ndof, c};
               mma::SmemMatAcc<U_LD> Uacc{sm.UV + c * U_LD * NB};
               mma::Gemm<MAP>(QND, ndof, NB, A, Xacc,
                              Uacc, nullD, e0, NE);
            }
            MFEM_SYNC_THREAD;
            ApplyDiffusionMetric<2, U_LD, NB, SYM, QND>(sm.UV, D, e0, NE, tid,
                                                        nthreads);
            MFEM_SYNC_THREAD;
            MFEM_UNROLL(2)
            for (int c = 0; c < 2; ++c)
            {
               mma::GAcc A{g, QND, ndof, c};
               mma::SmemMatAcc<U_LD> Vacc{sm.UV + c * U_LD * NB};
               mma::GemmT<MAP>(QND, ndof, NB, A, Vacc, Yacc, e0, NE);
            }
         }
         else if constexpr (DIM == 3)
         {
            for (int c = 0; c < 3; ++c)
            {
               mma::GAcc A{g, QND, ndof, c};
               mma::SmemMatAcc<U_LD> Uacc{sm.UV + c * U_LD * NB};
               mma::Gemm<MAP>(QND, ndof, NB, A, Xacc,
                              Uacc, nullD, e0, NE);
            }
            MFEM_SYNC_THREAD;
            ApplyDiffusionMetric<3, U_LD, NB, SYM, QND>(sm.UV, D, e0, NE, tid,
                                                        nthreads);
            MFEM_SYNC_THREAD;
            for (int c = 0; c < 3; ++c)
            {
               mma::GAcc A{g, QND, ndof, c};
               mma::SmemMatAcc<U_LD> Vacc{sm.UV + c * U_LD * NB};
               mma::GemmT<MAP>(QND, ndof, NB, A, Vacc, Yacc, e0, NE);
            }
         }
      }
      else
      {
         auto Y = DeviceMatrix(y, ndof, NE);
         if (tid == 0)
         {
            for (int b = 0; b < NB; ++b)
            {
               const int e = e0 + b;
               if (e >= NE) { continue; }
               for (int i = 0; i < X_LD; ++i)
               {
                  sm.XY[i + X_LD * b] = (i < ndof) ? X(i, e) : real_t(0);
               }
               for (int c = 0; c < DIM; ++c)
               {
                  for (int q = 0; q < QND; ++q)
                  {
                     real_t u = 0.0;
                     for (int i = 0; i < ndof; ++i)
                     {
                        u += g[q + QND * (i + ndof * c)] * sm.XY[i + X_LD * b];
                     }
                     sm.UV[c * U_LD * NB + q + U_LD * b] = u;
                  }
               }
               for (int q = 0; q < QND; ++q)
               {
                  if constexpr (DIM == 2)
                  {
                     const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
                     const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
                     const real_t O11 = D(q, 0, e);
                     const real_t O21 = D(q, 1, e);
                     real_t O12, O22;
                     if constexpr (SYM)
                     {
                        O12 = O21;
                        O22 = D(q, 2, e);
                     }
                     else
                     {
                        O12 = D(q, 2, e);
                        O22 = D(q, 3, e);
                     }
                     sm.UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2;
                     sm.UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2;
                  }
                  else
                  {
                     const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
                     const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
                     const real_t u3 = sm.UV[2 * U_LD * NB + q + U_LD * b];
                     const real_t O11 = D(q, 0, e);
                     const real_t O12 = D(q, 1, e);
                     const real_t O13 = D(q, 2, e);
                     real_t O21, O22, O23, O31, O32, O33;
                     if constexpr (SYM)
                     {
                        O21 = O12;
                        O22 = D(q, 3, e);
                        O23 = D(q, 4, e);
                        O31 = O13;
                        O32 = O23;
                        O33 = D(q, 5, e);
                     }
                     else
                     {
                        O21 = D(q, 3, e);
                        O22 = D(q, 4, e);
                        O23 = D(q, 5, e);
                        O31 = D(q, 6, e);
                        O32 = D(q, 7, e);
                        O33 = D(q, 8, e);
                     }
                     sm.UV[0 * U_LD * NB + q + U_LD * b] =
                        O11 * u1 + O12 * u2 + O13 * u3;
                     sm.UV[1 * U_LD * NB + q + U_LD * b] =
                        O21 * u1 + O22 * u2 + O23 * u3;
                     sm.UV[2 * U_LD * NB + q + U_LD * b] =
                        O31 * u1 + O32 * u2 + O33 * u3;
                  }
               }
               for (int i = 0; i < ndof; ++i)
               {
                  real_t yi = 0.0;
                  for (int c = 0; c < DIM; ++c)
                  {
                     for (int q = 0; q < QND; ++q)
                     {
                        yi += g[q + QND * (i + ndof * c)] *
                              sm.UV[c * U_LD * NB + q + U_LD * b];
                     }
                  }
                  Y(i, e) += yi;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
}

/** Host dense diffusion (Lapack or Blas):
    U_d = G_d x,  V = O(D) U,  y += sum_d G_d^T V_d.
    Large (nq,ndof): BLAS multi-RHS when MFEM_USE_LAPACK is on.
    Specialized sizes: Blas multi-RHS tiles; else runtime single-element. */
template<int DIM, int D1D, int QND, bool SYM>
inline void DiffusionApplySimplex(const int NE,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int ndof = mma::SimplexNdof<DIM, D1D>();

   const real_t *G = g.Read();
   const real_t *Dv = d.Read();
   const real_t *X = x.Read();
   real_t *Y = y.ReadWrite();

#ifdef MFEM_USE_LAPACK
   if (mma::lapack::PreferDiffusion(QND, ndof, NE))
   {
      mma::lapack::DiffusionApply<DIM, SYM>(NE, QND, ndof, G, Dv, X, Y);
      return;
   }
#endif

   mma::blas::DiffusionApply<DIM, ndof, QND, SYM>(
      NE, G, Dv, X, Y);
}

template<int DIM, int D1D, int QND, bool SYM>
inline void MmaDiffusionApplySimplex(const int NE,
                                     const Array<real_t> &g,
                                     const Vector &d,
                                     const Vector &x,
                                     Vector &y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int NB = mma::DiffusionMmaNB<DIM, D1D, QND>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int ndof = mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == PA_SIZE * QND * NE, "");
   MFEM_VERIFY(g.Size() == QND * ndof * DIM, "");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      DiffusionApplySimplex<DIM, D1D, QND, SYM>(
         NE, g, d, x, y);
      return;
   }

   constexpr int BASIS = ndof;
   constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = mma::PadLdBank<MAP>(BASIS);
   int smem_bytes = 0;
   int nthreads = 0;
   if constexpr (mma::DiffusionUseQTile<DIM, D1D, QND>())
   {
      constexpr int TQ = mma::DiffusionQTileFor<DIM, D1D, QND>();
      constexpr int U_LD = mma::PadLdBank<MAP>(TQ);
      smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      nthreads = mma::LaunchNthreads<TQ>(TQ, ndof);
   }
   else
   {
      constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
      constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
      smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      nthreads = mma::LaunchNthreads<QND>(QND, ndof);
   }
   mma::VerifySharedMemBytes(smem_bytes);

   const auto G = g.Read(), D = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   const int nbatches = (NE + NB - 1) / NB;
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaDiffusionApplySimplex_Batch<DIM, D1D, QND, SYM>(
         batch * NB, NE, G, D, X, Y);
   });
}

/** Host dispatch matching ApplySimplexMmaKernelType (runtime symmetric flag). */
template<int DIM, int D1D, int QND>
inline void MmaDiffusionApplySimplex_Dispatch(const int NE,
                                              const bool symmetric,
                                              const Array<real_t> &g,
                                              const Vector &d,
                                              const Vector &x,
                                              Vector &y)
{
   using Fn = decltype(&MmaDiffusionApplySimplex<DIM, D1D, QND, true>);
   const Fn apply = symmetric ? &MmaDiffusionApplySimplex<DIM, D1D, QND, true>
                    : &MmaDiffusionApplySimplex<DIM, D1D, QND, false>;
   apply(NE, g, d, x, y);
}

/** Runtime dense diffusion (host BLAS/hand; device per-element). */
template<int DIM, bool SYM>
inline void PADiffusionApplySimplexDenseRuntime(const int NE,
                                                const Array<real_t> &g,
                                                const Vector &d,
                                                const Vector &x,
                                                Vector &y)
{
   MFEM_VERIFY(NE > 0, "");
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   MFEM_VERIFY(d.Size() % (PA_SIZE * NE) == 0, "");
   const int nq = d.Size() / (PA_SIZE * NE);
   MFEM_VERIFY(nq > 0 && g.Size() % (nq * DIM) == 0, "");
   const int ndof = g.Size() / (nq * DIM);
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA diffusion runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      const real_t *G = g.Read();
      const real_t *Dv = d.Read();
      const real_t *X = x.Read();
      real_t *Y = y.ReadWrite();
#ifdef MFEM_USE_LAPACK
      if (mma::lapack::PreferDiffusion(nq, ndof, NE))
      {
         mma::lapack::DiffusionApply<DIM, SYM>(NE, nq, ndof, G, Dv, X, Y);
         return;
      }
#endif
      mma::blas::DiffusionApplyRuntime<DIM, SYM>(NE, nq, ndof, G, Dv, X, Y);
      return;
   }

   const auto G = g.Read();
   const auto Dv = d.Read();
   const auto X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[DIM * max_nq];
      mma::blas::DiffusionApplyDenseElement<DIM, SYM>(
         nq, ndof, G, Dv + nq * PA_SIZE * e,
         X + ndof * e, Y + ndof * e, u);
   });
}

template <int DIM, bool SYM, typename TD>
MFEM_HOST_DEVICE inline
void ApplyDiffusionMetricRuntime(real_t *UV, TD D,
                                 const int e0, const int NE,
                                 const int nq, const int u_ld, const int nb,
                                 const int tid, const int nthreads)
{
   for (int i = tid; i < nq * nb; i += nthreads)
   {
      const int b = i / nq;
      const int q = i - b * nq;
      const int e = e0 + b;
      if (e >= NE || q >= nq) { continue; }
      if constexpr (DIM == 2)
      {
         const real_t u1 = UV[0 * u_ld * nb + q + u_ld * b];
         const real_t u2 = UV[1 * u_ld * nb + q + u_ld * b];
         const real_t O11 = D(q, 0, e);
         const real_t O21 = D(q, 1, e);
         if constexpr (SYM)
         {
            const real_t O22 = D(q, 2, e);
            UV[0 * u_ld * nb + q + u_ld * b] = O11 * u1 + O21 * u2;
            UV[1 * u_ld * nb + q + u_ld * b] = O21 * u1 + O22 * u2;
         }
         else
         {
            const real_t O12 = D(q, 2, e);
            const real_t O22 = D(q, 3, e);
            UV[0 * u_ld * nb + q + u_ld * b] = O11 * u1 + O12 * u2;
            UV[1 * u_ld * nb + q + u_ld * b] = O21 * u1 + O22 * u2;
         }
      }
      else
      {
         const real_t u1 = UV[0 * u_ld * nb + q + u_ld * b];
         const real_t u2 = UV[1 * u_ld * nb + q + u_ld * b];
         const real_t u3 = UV[2 * u_ld * nb + q + u_ld * b];
         const real_t O11 = D(q, 0, e);
         const real_t O12 = D(q, 1, e);
         const real_t O13 = D(q, 2, e);
         if constexpr (SYM)
         {
            const real_t O22 = D(q, 3, e);
            const real_t O23 = D(q, 4, e);
            const real_t O33 = D(q, 5, e);
            UV[0 * u_ld * nb + q + u_ld * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * u_ld * nb + q + u_ld * b] = O12 * u1 + O22 * u2 + O23 * u3;
            UV[2 * u_ld * nb + q + u_ld * b] = O13 * u1 + O23 * u2 + O33 * u3;
         }
         else
         {
            const real_t O21 = D(q, 3, e);
            const real_t O22 = D(q, 4, e);
            const real_t O23 = D(q, 5, e);
            const real_t O31 = D(q, 6, e);
            const real_t O32 = D(q, 7, e);
            const real_t O33 = D(q, 8, e);
            UV[0 * u_ld * nb + q + u_ld * b] = O11 * u1 + O12 * u2 + O13 * u3;
            UV[1 * u_ld * nb + q + u_ld * b] = O21 * u1 + O22 * u2 + O23 * u3;
            UV[2 * u_ld * nb + q + u_ld * b] = O31 * u1 + O32 * u2 + O33 * u3;
         }
      }
   }
}

/** Full-NQ runtime batch (Fallback). Q-tile sizes use dense element path. */
template<int DIM, bool SYM>
MFEM_HOST_DEVICE inline
void MmaDiffusionApplySimplex_Batch(const int e0,
                                    const int NE,
                                    const int nq,
                                    const int ndof,
                                    const int x_ld,
                                    const int u_ld,
                                    const int nb,
                                    const real_t *g,
                                    const real_t *d,
                                    const real_t *x,
                                    real_t *y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();
   constexpr int max_x_ld = mma::PadLdBank<mma::MmaMapDefault>(
                               max_ndof);
   constexpr int max_u_ld = mma::PadLdBank<mma::MmaMapDefault>(
                               max_nq);
   constexpr int max_nb = mma::NBATCH;

   const auto D = Reshape(d, nq, PA_SIZE, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);
   const int tid = mma::getThreadIdx();
   [[maybe_unused]] const int nthreads = mma::getBlockNthreads();

   // Dyn layout matches launch smem_bytes; static uses caps.
#if defined(__CUDA_ARCH__)
   real_t *XY = reinterpret_cast<real_t *>(mma::SimplexMmaDynSmem());
   real_t *UV = XY + x_ld * nb;
   MFEM_CONTRACT_VAR(max_x_ld);
   MFEM_CONTRACT_VAR(max_u_ld);
   MFEM_CONTRACT_VAR(max_nb);
#else
   MFEM_SHARED real_t XY[max_x_ld * max_nb];
   MFEM_SHARED real_t UV[DIM * max_u_ld * max_nb];
#endif

   if constexpr (mma::DeviceGemmEnabled())
   {
      constexpr int MAP = mma::MmaMapDefault;
      mma::SmemMatAccRt Xacc{XY, x_ld};
      mma::YBatchAcc Yacc{y, ndof, e0};
      mma::NullDAcc nullD;

      mma::LoadXToSmem(XY, X, e0, NE, ndof, x_ld, nb, tid, nthreads);
      MFEM_SYNC_THREAD;

      for (int c = 0; c < DIM; ++c)
      {
         mma::GAcc A{g, nq, ndof, c};
         mma::SmemMatAccRt Uacc{UV + c * u_ld * nb, u_ld};
         mma::Gemm<MAP>(nq, ndof, nb, A, Xacc,
                        Uacc, nullD, e0, NE);
      }
      MFEM_SYNC_THREAD;
      ApplyDiffusionMetricRuntime<DIM, SYM>(UV, D, e0, NE, nq, u_ld, nb,
                                            tid, nthreads);
      MFEM_SYNC_THREAD;
      for (int c = 0; c < DIM; ++c)
      {
         mma::GAcc A{g, nq, ndof, c};
         mma::SmemMatAccRt Vacc{UV + c * u_ld * nb, u_ld};
         mma::GemmT<MAP>(nq, ndof, nb, A, Vacc, Yacc, e0, NE);
      }
   }
   else
   {
      auto Y = DeviceMatrix(y, ndof, NE);
      if (tid == 0)
      {
         for (int b = 0; b < nb; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            real_t u_scratch[DIM * max_nq];
            for (int i = 0; i < ndof; ++i)
            {
               XY[i + x_ld * b] = X(i, e);
            }
            mma::blas::DiffusionApplyDenseElement<DIM, SYM>(
               nq, ndof, g, &D(0, 0, e), &XY[x_ld * b], &Y(0, e), u_scratch);
            MFEM_CONTRACT_VAR(UV);
            MFEM_CONTRACT_VAR(u_ld);
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/** Runtime Fallback: host dense; device full-NQ batch or dense if Q-tile. */
template<int DIM, bool SYM>
inline void MmaDiffusionApplySimplex(const int NE,
                                     const Array<real_t> &g,
                                     const Vector &d,
                                     const Vector &x,
                                     Vector &y)
{
   MFEM_VERIFY(NE > 0, "");
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   MFEM_VERIFY(d.Size() % (PA_SIZE * NE) == 0, "");
   const int nq = d.Size() / (PA_SIZE * NE);
   MFEM_VERIFY(nq > 0 && g.Size() % (nq * DIM) == 0, "");
   const int ndof = g.Size() / (nq * DIM);
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA diffusion runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      PADiffusionApplySimplexDenseRuntime<DIM, SYM>(NE, g, d, x, y);
      return;
   }

   // Q-tile budget: keep correctness via dense element path (rare Fallback).
   if (mma::DiffusionUseQTileRuntime(DIM, ndof, nq, DIM))
   {
      PADiffusionApplySimplexDenseRuntime<DIM, SYM>(NE, g, d, x, y);
      return;
   }

   const int x_ld = mma::PadLdBankRuntime(ndof);
   const int u_ld = mma::PadLdBankRuntime(nq);
   const int nb = mma::DiffusionMmaNBRuntime(DIM, ndof, nq, DIM);
   MFEM_VERIFY(x_ld <= mma::PadLdBank<mma::MmaMapDefault>(max_ndof) &&
               u_ld <= mma::PadLdBank<mma::MmaMapDefault>(max_nq) &&
               nb <= mma::NBATCH,
               "Simplex MMA diffusion runtime Fallback smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * (x_ld + DIM * u_ld) * nb;
   mma::VerifySharedMemBytes(smem_bytes);

   const auto G = g.Read(), Dv = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   const int nthreads = mma::LaunchNthreads(nq, ndof);
   const int nbatches = (NE + nb - 1) / nb;
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaDiffusionApplySimplex_Batch<DIM, SYM>(
         batch * nb, NE, nq, ndof, x_ld, u_ld, nb, G, Dv, X, Y);
   });
}

template<int DIM>
inline void MmaDiffusionApplySimplex_Dispatch(const int NE,
                                              const bool symmetric,
                                              const Array<real_t> &g,
                                              const Vector &d,
                                              const Vector &x,
                                              Vector &y)
{
   using Fn = decltype(&MmaDiffusionApplySimplex<DIM, true>);
   const Fn apply = symmetric ? &MmaDiffusionApplySimplex<DIM, true>
                    : &MmaDiffusionApplySimplex<DIM, false>;
   apply(NE, g, d, x, y);
}

} // namespace internal

template<int DIM, int D1D, int QND>
DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::MmaDiffusionApplySimplex_Dispatch<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::MmaDiffusionApplySimplex_Dispatch<3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA diffusion only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA diffusion PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return internal::MmaDiffusionApplySimplex_Dispatch<2>;
   }
   return internal::MmaDiffusionApplySimplex_Dispatch<3>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
