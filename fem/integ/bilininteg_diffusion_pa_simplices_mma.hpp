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

#include "bilininteg_pa_mma.hpp"
// #include "bilininteg_pa_simplices_mma_host.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

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
void MmaPADiffusionApplySimplex_Batch(const int e0,
                                      const int NE,
                                      const real_t *g_,
                                      const real_t *d_,
                                      const real_t *x_,
                                      real_t *y_)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int BASIS_DIM = simplex_mma::SimplexNdof<DIM, D1D>();
   constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAP>(BASIS_DIM);
   constexpr int NB = simplex_mma::DiffusionMmaNB<DIM, D1D, QND>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
   constexpr int ndof = BASIS_DIM;

   const auto D = Reshape(d_, QND, PA_SIZE, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   const int tid = simplex_mma::getThreadIdx();
   [[maybe_unused]] const int nthreads = simplex_mma::getBlockNthreads();

   // Q-tiled path: keep NB=16 with TQ-row U planes (CUDA DMMA / HIP MFMA N util).
   if constexpr (simplex_mma::DiffusionUseQTile<DIM, D1D, QND>())
   {
      constexpr int TQ = simplex_mma::DiffusionQTileFor<DIM, D1D, QND>();
#if defined(MFEM_USE_HIP)
      constexpr int U_LD = simplex_mma::PadLdBankHip(TQ);
#else
      constexpr int U_LD = simplex_mma::PadLdBank<MAP>(TQ);
#endif
      static_assert(sizeof(real_t) * (X_LD + DIM * U_LD) * NB <=
                    simplex_mma::SharedMemBytesPerBlock,
                    "Q-tiled diffusion smem exceeds SharedMemBytesPerBlock");
      struct alignas(16) SmemQ
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
#if defined(__CUDA_ARCH__)
      SmemQ &sm = *reinterpret_cast<SmemQ *>(simplex_mma::SimplexMmaDynSmem());
#else
      MFEM_SHARED SmemQ sm;
#endif

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    !defined(MFEM_USE_SINGLE)
      simplex_mma::SmemMatAcc<X_LD> Xacc {sm.XY};
      simplex_mma::YBatchAcc Yacc{y_, ndof, e0};
      simplex_mma::SmemMatAcc<U_LD> U0{sm.UV + 0 * U_LD * NB};
      simplex_mma::SmemMatAcc<U_LD> U1{sm.UV + 1 * U_LD * NB};
      simplex_mma::SmemMatAcc<U_LD> U2{sm.UV + 2 * U_LD * NB};

      simplex_mma::LoadXToSmem(sm.XY, x, e0, NE, ndof, X_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;

      for (int q0 = 0; q0 < QND; q0 += TQ)
      {
         const int nq_tile = (QND - q0 < TQ) ? (QND - q0) : TQ;
         simplex_mma::GAccQTile A0{g_, QND, ndof, 0, q0};
         simplex_mma::GAccQTile A1{g_, QND, ndof, 1, q0};
         simplex_mma::GAccQTile A2{g_, QND, ndof, 2, q0};
         simplex_mma::BasisGemmForward3<MAP>(nq_tile, ndof, NB, A0, A1, A2, Xacc,
                                             U0, U1, U2, e0, NE);
         MFEM_SYNC_THREAD;
         ApplyDiffusionMetricQTile<DIM, U_LD, NB, SYM, QND>(
            sm.UV, D, e0, NE, q0, nq_tile, tid, nthreads);
         MFEM_SYNC_THREAD;
         simplex_mma::BasisGemmT3<MAP>(nq_tile, ndof, NB, A0, A1, A2, U0, U1, U2,
                                       Yacc, e0, NE);
         MFEM_SYNC_THREAD;
      }
#else
      auto Y = DeviceMatrix(y_, ndof, NE);
      if (tid == 0)
      {
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < X_LD; ++i)
            {
               sm.XY[i + X_LD * b] = (i < ndof) ? x(i, e) : real_t(0);
            }
            for (int q0 = 0; q0 < QND; q0 += TQ)
            {
               const int nq_tile = (QND - q0 < TQ) ? (QND - q0) : TQ;
               for (int d = 0; d < DIM; ++d)
               {
                  for (int qloc = 0; qloc < nq_tile; ++qloc)
                  {
                     real_t u = 0.0;
                     const int q = q0 + qloc;
                     for (int i = 0; i < ndof; ++i)
                     {
                        u += g_[q + QND * (i + ndof * d)] *
                             sm.XY[i + X_LD * b];
                     }
                     sm.UV[d * U_LD * NB + qloc + U_LD * b] = u;
                  }
               }
               ApplyDiffusionMetricQTile<DIM, U_LD, NB, SYM, QND>(
                  sm.UV, D, e0, NE, q0, nq_tile, 0, 1);
               for (int i = 0; i < ndof; ++i)
               {
                  real_t yi = 0.0;
                  for (int d = 0; d < DIM; ++d)
                  {
                     for (int qloc = 0; qloc < nq_tile; ++qloc)
                     {
                        const int q = q0 + qloc;
                        yi += g_[q + QND * (i + ndof * d)] *
                              sm.UV[d * U_LD * NB + qloc + U_LD * b];
                     }
                  }
                  Y(i, e) += yi;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
#endif
   }
   else
   {
      // Full-nq path when Q-tiling is not needed.
      constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
      static_assert(sizeof(real_t) * (X_LD + DIM * U_LD) * NB <=
                    simplex_mma::SharedMemBytesPerBlock,
                    "Diffusion simplex MMA shared memory exceeds SharedMemBytesPerBlock");
      struct alignas(16) Smem
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
#if defined(__CUDA_ARCH__)
      Smem &sm = *reinterpret_cast<Smem *>(simplex_mma::SimplexMmaDynSmem());
#else
      MFEM_SHARED Smem sm;
#endif

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    !defined(MFEM_USE_SINGLE)
      simplex_mma::SmemMatAcc<X_LD> Xacc {sm.XY};
      simplex_mma::YBatchAcc Yacc{y_, ndof, e0};
      simplex_mma::NullDAcc nullD;

      simplex_mma::LoadXToSmem(sm.XY, x, e0, NE, ndof, X_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;

      if constexpr (DIM == 2)
      {
         MFEM_UNROLL(2)
         for (int d = 0; d < 2; ++d)
         {
            simplex_mma::GAcc A{g_, QND, ndof, d};
            simplex_mma::SmemMatAcc<U_LD> Uacc{sm.UV + d * U_LD * NB};
            simplex_mma::BasisGemmForward<MAP, false>(QND, ndof, NB, A, Xacc,
                                                      Uacc, nullD, e0, NE);
         }
         MFEM_SYNC_THREAD;
         ApplyDiffusionMetric<2, U_LD, NB, SYM, QND>(sm.UV, D, e0, NE, tid,
                                                     nthreads);
         MFEM_SYNC_THREAD;
         MFEM_UNROLL(2)
         for (int d = 0; d < 2; ++d)
         {
            simplex_mma::GAcc A{g_, QND, ndof, d};
            simplex_mma::SmemMatAcc<U_LD> Vacc{sm.UV + d * U_LD * NB};
            simplex_mma::BasisGemmT<MAP>(QND, ndof, NB, A, Vacc, Yacc, e0, NE);
         }
      }
      else if constexpr (DIM == 3)
      {
         for (int d = 0; d < 3; ++d)
         {
            simplex_mma::GAcc A{g_, QND, ndof, d};
            simplex_mma::SmemMatAcc<U_LD> Uacc{sm.UV + d * U_LD * NB};
            simplex_mma::BasisGemmForward<MAP, false>(QND, ndof, NB, A, Xacc,
                                                      Uacc, nullD, e0, NE);
         }
         MFEM_SYNC_THREAD;
         ApplyDiffusionMetric<3, U_LD, NB, SYM, QND>(sm.UV, D, e0, NE, tid,
                                                     nthreads);
         MFEM_SYNC_THREAD;
         for (int d = 0; d < 3; ++d)
         {
            simplex_mma::GAcc A{g_, QND, ndof, d};
            simplex_mma::SmemMatAcc<U_LD> Vacc{sm.UV + d * U_LD * NB};
            simplex_mma::BasisGemmT<MAP>(QND, ndof, NB, A, Vacc, Yacc, e0, NE);
         }
      }
#else
      auto Y = DeviceMatrix(y_, ndof, NE);
      if (tid == 0)
      {
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < X_LD; ++i)
            {
               sm.XY[i + X_LD * b] = (i < ndof) ? x(i, e) : real_t(0);
            }
            for (int d = 0; d < DIM; ++d)
            {
               for (int q = 0; q < QND; ++q)
               {
                  real_t u = 0.0;
                  for (int i = 0; i < ndof; ++i)
                  {
                     u += g_[q + QND * (i + ndof * d)] * sm.XY[i + X_LD * b];
                  }
                  sm.UV[d * U_LD * NB + q + U_LD * b] = u;
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
               for (int d = 0; d < DIM; ++d)
               {
                  for (int q = 0; q < QND; ++q)
                  {
                     yi += g_[q + QND * (i + ndof * d)] *
                           sm.UV[d * U_LD * NB + q + U_LD * b];
                  }
               }
               Y(i, e) += yi;
            }
         }
      }
      MFEM_SYNC_THREAD;
#endif
   }
}

/** Host-optimized dense diffusion:
    U_d = G_d x,  V = O(D) U,  y += sum_d G_d^T V_d.
    Large (nq,ndof): BLAS multi-RHS when MFEM_USE_LAPACK is on.
    Specialized sizes: hand multi-RHS tiles; else runtime single-element. */
template<int DIM, int D1D, int QND, bool SYM>
inline void PADiffusionApplySimplexDenseHost(const int NE,
                                             const Array<real_t> &g_,
                                             const Vector &d_,
                                             const Vector &x_,
                                             Vector &y_)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();

   const real_t *G = g_.Read();
   const real_t *Dv = d_.Read();
   const real_t *X = x_.Read();
   real_t *Y = y_.ReadWrite();

#ifdef MFEM_USE_LAPACK
   if (simplex_mma::PreferHostBlas(QND, ndof))
   {
      simplex_mma::DiffusionApplyBlas<DIM, SYM>(NE, QND, ndof, G, Dv, X, Y);
      return;
   }
#endif

   simplex_mma::DiffusionApplyHandSpecialized<DIM, ndof, QND, SYM>(
      NE, G, Dv, X, Y);
}

template<int DIM, int D1D, int QND, bool SYM>
inline void MmaPADiffusionApplySimplex(const int NE,
                                       const Array<real_t> &g_,
                                       const Vector &d_,
                                       const Vector &x_,
                                       Vector &y_)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA diffusion requires specialized D1D/QND");
   constexpr int NB = simplex_mma::DiffusionMmaNB<DIM, D1D, QND>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d_.Size() == PA_SIZE * QND * NE, "");
   MFEM_VERIFY(g_.Size() == QND * ndof * DIM, "");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      PADiffusionApplySimplexDenseHost<DIM, D1D, QND, SYM>(
         NE, g_, d_, x_, y_);
      return;
   }

   {
      constexpr int BASIS = ndof;
      constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
      constexpr int X_LD = simplex_mma::PadLdBank<MAP>(BASIS);
      if constexpr (simplex_mma::DiffusionUseQTile<DIM, D1D, QND>())
      {
         constexpr int TQ = simplex_mma::DiffusionQTileFor<DIM, D1D, QND>();
#if defined(MFEM_USE_HIP)
         constexpr int U_LD = simplex_mma::PadLdBankHip(TQ);
#else
         constexpr int U_LD = simplex_mma::PadLdBank<MAP>(TQ);
#endif
         simplex_mma::VerifySharedMemBytes(
            int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB);
      }
      else
      {
         constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
         constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
         simplex_mma::VerifySharedMemBytes(
            int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB);
      }
   }

   const auto G = g_.Read(), D = d_.Read(), X = x_.Read();
   auto Y = y_.ReadWrite();

   // Match thread count to Q-tile M when Q-tiling (avoids idle warps/waves at TQ<<nq).
   int nthreads;
   if constexpr (simplex_mma::DiffusionUseQTile<DIM, D1D, QND>())
   {
      constexpr int TQ = simplex_mma::DiffusionQTileFor<DIM, D1D, QND>();
      nthreads = simplex_mma::LaunchNthreads<TQ>(TQ, ndof);
   }
   else
   {
      nthreads = simplex_mma::LaunchNthreads<QND>(QND, ndof);
   }
   const int nbatches = (NE + NB - 1) / NB;
   int smem_bytes = 0;
   {
      constexpr int BASIS = ndof;
      constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
      constexpr int X_LD = simplex_mma::PadLdBank<MAP>(BASIS);
      if constexpr (simplex_mma::DiffusionUseQTile<DIM, D1D, QND>())
      {
         constexpr int TQ = simplex_mma::DiffusionQTileFor<DIM, D1D, QND>();
#if defined(MFEM_USE_HIP)
         constexpr int U_LD = simplex_mma::PadLdBankHip(TQ);
#else
         constexpr int U_LD = simplex_mma::PadLdBank<MAP>(TQ);
#endif
         smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      }
      else
      {
         constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
         constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
         smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      }
   }
#if defined(MFEM_USE_CUDA)
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaPADiffusionApplySimplex_Batch<DIM, D1D, QND, SYM>(
         batch * NB, NE, G, D, X, Y);
   });
#else
   MFEM_CONTRACT_VAR(smem_bytes);
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaPADiffusionApplySimplex_Batch<DIM, D1D, QND, SYM>(
         batch * NB, NE, G, D, X, Y);
   });
#endif
}

/** Host dispatch matching ApplySimplexMmaKernelType (runtime symmetric flag). */
template<int DIM, int D1D, int QND>
inline void SmemPADiffusionApplySimplexMmaDispatch(const int NE,
                                                   const bool symmetric,
                                                   const Array<real_t> &g_,
                                                   const Vector &d_,
                                                   const Vector &x_,
                                                   Vector &y_)
{
   if (symmetric)
   {
      MmaPADiffusionApplySimplex<DIM, D1D, QND, true>(NE, g_, d_, x_, y_);
   }
   else
   {
      MmaPADiffusionApplySimplex<DIM, D1D, QND, false>(NE, g_, d_, x_, y_);
   }
}

} // namespace internal

template<int DIM, int D1D, int QND>
DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPADiffusionApplySimplexMmaDispatch<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPADiffusionApplySimplexMmaDispatch<3, D1D, QND>;
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
   MFEM_ABORT("No fallback for Simplex MMA diffusion PA");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
