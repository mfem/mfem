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

#include "bilininteg_pa_simplices_mma.hpp"

#include <algorithm>
#include <cstdlib> // alloca
#include <vector>

namespace mfem
{

namespace internal
{

namespace simplex_mma
{

// ---------------------------------------------------------------------------
// Shared host dense multi-RHS helpers (mass + diffusion)
//
// Hand path layout (b-innermost, good for SIMD across elements):
//   xloc[i * NB + b], uloc[q * NB + b]
// BLAS path layout (column-major Fortran / dgemm):
//   xloc[i + ndof * b], uloc[q + nq * b]
// ---------------------------------------------------------------------------

/** Hand multi-RHS NB for specialized mass (keep U in L1 on large 3D). */
template <int DIM, int NQ>
constexpr int HandMassNB()
{
   return (DIM == 3 && NQ > 80) ? 4 : 8;
}

/** Hand multi-RHS NB for specialized diffusion. */
template <int DIM, int NQ>
constexpr int HandDiffusionNB()
{
   return (DIM == 3 && NQ > 60) ? 2 : 4;
}

// ---- Pack / scatter (column-major, BLAS) ------------------------------------

/** Pack X(:, e0:e0+NB) into column-major xloc[ndof * NB]; pad zeros. */
inline void PackXColMajor(const real_t *X, int ndof, int e0, int NE, int NB,
                          real_t *xloc)
{
   std::fill(xloc, xloc + static_cast<size_t>(ndof) * NB, real_t(0));
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int i = 0; i < ndof; ++i)
      {
         xloc[static_cast<size_t>(i) + static_cast<size_t>(ndof) * b] =
            X[i + ndof * e];
      }
   }
}

/** Y(:, e0:e0+NB) += column-major ytmp[ndof * NB]. */
inline void ScatterAddYColMajor(const real_t *ytmp, int ndof, int e0, int NE,
                                int NB, real_t *Y)
{
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int i = 0; i < ndof; ++i)
      {
         Y[i + ndof * e] +=
            ytmp[static_cast<size_t>(i) + static_cast<size_t>(ndof) * b];
      }
   }
}

/** Scale U columns by mass PA weights D(q,e): U ⊙= D. Column-major U. */
inline void ScaleUByMassD(real_t *uloc, const real_t *D, int nq, int e0, int NE,
                          int NB)
{
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { break; }
      for (int q = 0; q < nq; ++q)
      {
         uloc[static_cast<size_t>(q) + static_cast<size_t>(nq) * b] *=
            D[q + nq * e];
      }
   }
}

// ---- Diffusion metric at one quadrature point (vector length DIM) ----------

/** In-place metric: u[0:DIM) := O(D(q,e)) * u. PA D is (q, pa, e). */
template <int DIM, bool SYM>
inline void ApplyDiffusionMetricVec(real_t *u, const real_t *Dv,
                                    int q, int nq, int e, int pa_size)
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

/** Metric on BLAS column-major U planes: uloc[d * nq * NB + q + nq * b]. */
template <int DIM, bool SYM>
inline void ApplyDiffusionMetricColMajor(real_t *uloc, const real_t *Dv,
                                         int nq, int e0, int NE, int NB,
                                         int pa_size)
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

/** Metric on hand b-innermost U: uloc[(d * NQ + q) * NB + b]. */
template <int DIM, int NQ, int NB, bool SYM>
inline void ApplyDiffusionMetricHand(real_t *uloc, const real_t *Dv,
                                     int e0, int NE, int pa_size)
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

// ---- Hand multi-RHS GEMM (b-innermost) -------------------------------------

/** Load X tile: xloc[i*NB+b], pad zeros. */
template <int NDOF, int NB>
inline void PackXHand(const real_t *X, int e0, int NE, real_t *xloc)
{
   for (int i = 0; i < NDOF; ++i)
   {
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         xloc[i * NB + b] = (e < NE) ? X[i + NDOF * e] : real_t(0);
      }
   }
}

/** U = B * X with optional mass scale: U_qb = (sum_i B_qi X_ib) * [D_qe].
    B is column-major nq×ndof: B[q + NQ*i].
    If scale_mass is false, D may be null and scale is 1. */
template <int NDOF, int NQ, int NB, bool SCALE_MASS>
inline void HandGemmForward(const real_t *B, const real_t *xloc, real_t *uloc,
                            const real_t *D, int e0, int NE)
{
   for (int q = 0; q < NQ; ++q)
   {
      real_t ub[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { ub[b] = real_t(0); }
      for (int i = 0; i < NDOF; ++i)
      {
         const real_t bqi = B[q + NQ * i];
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            ub[b] += bqi * xloc[i * NB + b];
         }
      }
      if constexpr (SCALE_MASS)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            const real_t dq = (e < NE) ? D[q + NQ * e] : real_t(0);
            uloc[q * NB + b] = ub[b] * dq;
         }
      }
      else
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            uloc[q * NB + b] = ub[b];
         }
      }
   }
}

/** Y(e0+b) += B^T * U. B column-major nq×ndof. */
template <int NDOF, int NQ, int NB>
inline void HandGemmBackward(const real_t *B, const real_t *uloc, real_t *Y,
                             int e0, int NE)
{
   for (int i = 0; i < NDOF; ++i)
   {
      real_t yb[NB];
      MFEM_UNROLL(NB)
      for (int b = 0; b < NB; ++b) { yb[b] = real_t(0); }
      for (int q = 0; q < NQ; ++q)
      {
         const real_t bqi = B[q + NQ * i];
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            yb[b] += bqi * uloc[q * NB + b];
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

/** Diffusion hand: forward all GradP components into uloc[(d*NQ+q)*NB+b]. */
template <int DIM, int NDOF, int NQ, int NB>
inline void HandDiffusionForward(const real_t *G, const real_t *xloc,
                                 real_t *uloc)
{
   for (int d = 0; d < DIM; ++d)
   {
      const real_t *Gd = G + static_cast<size_t>(d) * NQ * NDOF;
      HandGemmForward<NDOF, NQ, NB, false>(Gd, xloc, uloc + d * NQ * NB,
                                           nullptr, 0, 0);
   }
}

/** Diffusion hand: Y += sum_d G_d^T U_d. */
template <int DIM, int NDOF, int NQ, int NB>
inline void HandDiffusionBackward(const real_t *G, const real_t *uloc,
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

// ---- BLAS multi-RHS mass / diffusion tiles ---------------------------------

#ifdef MFEM_USE_LAPACK
/** Mass: serial tiles, reused buffers. U = P X, scale D, Y += P^T U. */
inline void MassApplyBlas(int NE, int nq, int ndof, const real_t *P,
                          const real_t *D, const real_t *X, real_t *Y)
{
   const int NB = HostBlasNB(nq, ndof);
   const int ntiles = (NE + NB - 1) / NB;
   std::vector<real_t> xloc(static_cast<size_t>(ndof) * NB);
   std::vector<real_t> uloc(static_cast<size_t>(nq) * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);

   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      PackXColMajor(X, ndof, e0, NE, NB, xloc.data());
      HostGemm('N', 'N', nq, NB, ndof, real_t(1), P, nq, xloc.data(), ndof,
               real_t(0), uloc.data(), nq);
      ScaleUByMassD(uloc.data(), D, nq, e0, NE, NB);
      HostGemm('T', 'N', ndof, NB, nq, real_t(1), P, nq, uloc.data(), nq,
               real_t(0), ytmp.data(), ndof);
      ScatterAddYColMajor(ytmp.data(), ndof, e0, NE, NB, Y);
   }
}

/** Diffusion: U_d = G_d X, metric, Y += sum G_d^T V_d. */
template <int DIM, bool SYM>
inline void DiffusionApplyBlas(int NE, int nq, int ndof, const real_t *G,
                               const real_t *Dv, const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int NB = HostBlasNB(nq, ndof);
   const int ntiles = (NE + NB - 1) / NB;
   std::vector<real_t> xloc(static_cast<size_t>(ndof) * NB);
   std::vector<real_t> uloc(static_cast<size_t>(DIM) * nq * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);

   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      PackXColMajor(X, ndof, e0, NE, NB, xloc.data());

      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         real_t *Ud = uloc.data() + static_cast<size_t>(d) * nq * NB;
         HostGemm('N', 'N', nq, NB, ndof, real_t(1), Gd, nq, xloc.data(), ndof,
                  real_t(0), Ud, nq);
      }

      ApplyDiffusionMetricColMajor<DIM, SYM>(uloc.data(), Dv, nq, e0, NE, NB,
                                             PA_SIZE);

      std::fill(ytmp.begin(), ytmp.end(), real_t(0));
      for (int d = 0; d < DIM; ++d)
      {
         const real_t *Gd = G + static_cast<size_t>(d) * nq * ndof;
         const real_t *Vd = uloc.data() + static_cast<size_t>(d) * nq * NB;
         HostGemm('T', 'N', ndof, NB, nq, real_t(1), Gd, nq, Vd, nq,
                  real_t(1), ytmp.data(), ndof);
      }
      ScatterAddYColMajor(ytmp.data(), ndof, e0, NE, NB, Y);
   }
}
#endif // MFEM_USE_LAPACK

// ---- Specialized hand tiles ------------------------------------------------

template <int DIM, int NDOF, int NQ>
inline void MassApplyHandSpecialized(int NE, const real_t *P, const real_t *D,
                                     const real_t *X, real_t *Y)
{
   constexpr int NB = HandMassNB<DIM, NQ>();
   const int ntiles = (NE + NB - 1) / NB;
   mfem::forall(ntiles, [=](int tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[NQ * NB];
      PackXHand<NDOF, NB>(X, e0, NE, xloc);
      HandGemmForward<NDOF, NQ, NB, true>(P, xloc, uloc, D, e0, NE);
      HandGemmBackward<NDOF, NQ, NB>(P, uloc, Y, e0, NE);
   });
}

template <int DIM, int NDOF, int NQ, bool SYM>
inline void DiffusionApplyHandSpecialized(int NE, const real_t *G,
                                          const real_t *Dv, const real_t *X,
                                          real_t *Y)
{
   constexpr int NB = HandDiffusionNB<DIM, NQ>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int ntiles = (NE + NB - 1) / NB;
   mfem::forall(ntiles, [=](int tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[DIM * NQ * NB];
      PackXHand<NDOF, NB>(X, e0, NE, xloc);
      HandDiffusionForward<DIM, NDOF, NQ, NB>(G, xloc, uloc);
      ApplyDiffusionMetricHand<DIM, NQ, NB, SYM>(uloc, Dv, e0, NE, PA_SIZE);
      HandDiffusionBackward<DIM, NDOF, NQ, NB>(G, uloc, Y, e0, NE);
   });
}

// ---- Runtime (unspecialized) single-element fallbacks ----------------------

inline void MassApplyHandRuntime(int NE, int nq, int ndof, const real_t *P,
                                 const real_t *D, const real_t *X, real_t *Y)
{
   mfem::forall(NE, [=](int e)
   {
      real_t *u = static_cast<real_t *>(
                     alloca(sizeof(real_t) * static_cast<size_t>(nq)));
      for (int q = 0; q < nq; ++q)
      {
         real_t s = 0.0;
         for (int i = 0; i < ndof; ++i)
         {
            s += P[q + nq * i] * X[i + ndof * e];
         }
         u[q] = s * D[q + nq * e];
      }
      for (int i = 0; i < ndof; ++i)
      {
         real_t s = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            s += P[q + nq * i] * u[q];
         }
         Y[i + ndof * e] += s;
      }
   });
}

template <int DIM, bool SYM>
inline void DiffusionApplyHandRuntime(int NE, int nq, int ndof, const real_t *G,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   mfem::forall(NE, [=](int e)
   {
      real_t *u = static_cast<real_t *>(
                     alloca(sizeof(real_t) * static_cast<size_t>(DIM * nq)));
      for (int d = 0; d < DIM; ++d)
      {
         for (int q = 0; q < nq; ++q)
         {
            real_t s = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               s += G[q + nq * (i + ndof * d)] * X[i + ndof * e];
            }
            u[d * nq + q] = s;
         }
      }
      for (int q = 0; q < nq; ++q)
      {
         real_t uv[DIM];
         for (int d = 0; d < DIM; ++d) { uv[d] = u[d * nq + q]; }
         ApplyDiffusionMetricVec<DIM, SYM>(uv, Dv, q, nq, e, PA_SIZE);
         for (int d = 0; d < DIM; ++d) { u[d * nq + q] = uv[d]; }
      }
      for (int i = 0; i < ndof; ++i)
      {
         real_t s = 0.0;
         for (int d = 0; d < DIM; ++d)
         {
            for (int q = 0; q < nq; ++q)
            {
               s += G[q + nq * (i + ndof * d)] * u[d * nq + q];
            }
         }
         Y[i + ndof * e] += s;
      }
   });
}

} // namespace simplex_mma

} // namespace internal

} // namespace mfem
