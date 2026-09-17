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

#include "common.hpp"

// ======================================================================
// CUDA (dmma) — Gemm + SUMF
// ======================================================================

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::dmma
{

#if defined(MFEM_USE_CUDA) && !defined(MFEM_USE_SINGLE)

MFEM_HOST_DEVICE inline void Sync(double aReg[1],
                                  double bReg[1],
                                  double cReg[2])
{
#ifdef __CUDA_ARCH__
   asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+d"(cReg[0]), "+d"(cReg[1]) : "d"(aReg[0]), "d"(bReg[0]));
#endif
}

/** C = A * B with fused D-scale on the C store (U *= D from registers). */
template<int MAP, bool SCALE,
         typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Gemm8(const int M, const int K, const int N,
                                   TA A, TB B, TC C, TD D,
                                   const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (M + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {};

      for (int mK = 0; mK < (K + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aRow = row0 + groupId;
         const int aColumn = threadIdInGroup + mK * mmaK;
         aReg[0] = (aRow < M && aColumn < K)
                   ? static_cast<double>(A(aRow, aColumn)) : 0.0;

         MFEM_UNROLL(2)
         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MapCol<MAP>(groupId);
            bReg[0] = (bRow < K && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            Sync(aReg, bReg, cReg[nt]);
         }
      }
      MFEM_UNROLL(2)
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MapCol<MAP>(threadIdInGroup * 2 + i);
            if (cRow < M && cColumn < nTile)
            {
               real_t v = static_cast<real_t>(cReg[nt][i]);
               if constexpr (SCALE)
               {
                  const int e = e0 + n0 + cColumn;
                  v = (e < NE) ? v * D(cRow, e) : real_t(0);
               }
               C(cRow, n0 + cColumn) = v;
            }
         }
      }
   }
}

template <int MAP, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void GemmT8(const int M, const int K, const int N,
                                    TA A, TB B, TC C,
                                    const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (K + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {};

      for (int mK = 0; mK < (M + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aT_row = row0 + groupId;
         const int aT_col = threadIdInGroup + mK * mmaK;
         aReg[0] = (aT_row < K && aT_col < M)
                   ? static_cast<double>(A(aT_col, aT_row)) : 0.0;

         MFEM_UNROLL(2)
         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MapCol<MAP>(groupId);
            bReg[0] = (bRow < M && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            Sync(aReg, bReg, cReg[nt]);
         }
      }
      MFEM_UNROLL(2)
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MapCol<MAP>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[nt][i]);
            }
         }
      }
   }
}

/** Fused 3-comp forward: U_d = G_d * X (shared X loads). */
template <int MAP, typename TA0, typename TA1, typename TA2,
          typename TB, typename TC0, typename TC1, typename TC2>
MFEM_HOST_DEVICE inline void Gemm8_Fwd3(const int M, const int K,
                                        const int N,
                                        TA0 A0, TA1 A1, TA2 A2, TB B,
                                        TC0 C0, TC1 C1, TC2 C2)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (M + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         double c0[2] = {}, c1[2] = {}, c2[2] = {};

         for (int mK = 0; mK < (K + mmaK - 1) / mmaK; mK++)
         {
            const int aRow = row0 + groupId;
            const int aColumn = threadIdInGroup + mK * mmaK;
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MapCol<MAP>(groupId);
            const double bV = (bRow < K && bColumn < nTile)
                              ? static_cast<double>(B(bRow, n0 + bColumn))
                              : 0.0;
            double aReg[1], bReg[1] = {bV};
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A0(aRow, aColumn)) : 0.0;
            Sync(aReg, bReg, c0);
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A1(aRow, aColumn)) : 0.0;
            Sync(aReg, bReg, c1);
            aReg[0] = (aRow < M && aColumn < K)
                      ? static_cast<double>(A2(aRow, aColumn)) : 0.0;
            Sync(aReg, bReg, c2);
         }
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MapCol<MAP>(threadIdInGroup * 2 + i);
            if (cRow < M && cColumn < nTile)
            {
               C0(cRow, n0 + cColumn) = static_cast<real_t>(c0[i]);
               C1(cRow, n0 + cColumn) = static_cast<real_t>(c1[i]);
               C2(cRow, n0 + cColumn) = static_cast<real_t>(c2[i]);
            }
         }
      }
   }
}

/** Fused 3-comp GemmT: Y += G_d^T * U_d (shared Y accumulate). */
template <int MAP, typename TA0, typename TA1, typename TA2,
          typename TB0, typename TB1, typename TB2, typename TC>
MFEM_HOST_DEVICE inline void GemmT8_3(const int M, const int K,
                                      const int N,
                                      TA0 A0, TA1 A1, TA2 A2,
                                      TB0 B0, TB1 B1, TB2 B2, TC C,
                                      const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (K + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         double cReg[2] = {};

         for (int mK = 0; mK < (M + mmaK - 1) / mmaK; mK++)
         {
            const int aT_row = row0 + groupId;
            const int aT_col = threadIdInGroup + mK * mmaK;
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MapCol<MAP>(groupId);
            const bool a_ok = (aT_row < K && aT_col < M);
            const bool b_ok = (bRow < M && bColumn < nTile);
            double aReg[1], bReg[1];
            aReg[0] = a_ok ? static_cast<double>(A0(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B0(bRow, n0 + bColumn)) : 0.0;
            Sync(aReg, bReg, cReg);
            aReg[0] = a_ok ? static_cast<double>(A1(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B1(bRow, n0 + bColumn)) : 0.0;
            Sync(aReg, bReg, cReg);
            aReg[0] = a_ok ? static_cast<double>(A2(aT_col, aT_row)) : 0.0;
            bReg[0] = b_ok ? static_cast<double>(B2(bRow, n0 + bColumn)) : 0.0;
            Sync(aReg, bReg, cReg);
         }
         MFEM_UNROLL(2)
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MapCol<MAP>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** CUDA m8n8k4 entry points (single tile shape). */
template<int MAP, bool SCALE,
         typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Gemm(const int M, const int K, const int N,
                                  TA A, TB B, TC C, TD D,
                                  const int e0, const int NE)
{
   Gemm8<MAP, SCALE>(M, K, N, A, B, C, D, e0, NE);
}

template <int MAP, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void GemmT(const int M, const int K, const int N,
                                   TA A, TB B, TC C,
                                   const int e0, const int NE)
{
   GemmT8<MAP>(M, K, N, A, B, C, e0, NE);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradX(const int m, const int n, const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdxX();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int bankMap = BankMap();

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            double gReg[1];
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MappedNCol(bankMap, bColumnInWarp, n0);
            if (bColumn < n && bRow < k)
            {
               bReg[0] = B(bRow, bColumn);
               gReg[0] = G(bRow, bColumn);
            }
            else
            {
               bReg[0] = 0;
               gReg[0] = 0;
            }
            double aReg[1];
            int aRow = MapM(aRowInWarp, mM);
            int aColumn = aColumnInWarp + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix aA(A[0], k, m);
               aReg[0] = aA(aColumn, aRow);
            }
            else
            {
               aReg[0] = 0;
            }
            Sync(aReg, gReg, &cReg[0]);
            Sync(aReg, bReg, &cReg[2]);
         }
         for (int d = 0; d < 2; d++)
         {
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
               int cRow = MapM(groupId, mM);
               int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradY(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   ConstDeviceMatrix B(BG[0], k, n);
   ConstDeviceMatrix G(BG[1], k, n);

   int thread = getThreadIdxX();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int bankMap = BankMap();

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[6] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            double gReg[1];
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MappedNCol(bankMap, bColumnInWarp, n0);
            if (bColumn < n && bRow < k)
            {
               bReg[0] = B(bRow, bColumn);
               gReg[0] = G(bRow, bColumn);
            }
            else
            {
               bReg[0] = 0;
               gReg[0] = 0;
            }
            double agReg[1];
            double abReg[1];
            int aRow = MapM(aRowInWarp, mM);
            int aColumn = aColumnInWarp + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix gA(A[0], k, m);
               ConstDeviceMatrix bA(A[1], k, m);
               agReg[0] = gA(aColumn, aRow);
               abReg[0] = bA(aColumn, aRow);
            }
            else
            {
               agReg[0] = 0;
               abReg[0] = 0;
            }
            Sync(agReg, bReg, &cReg[0]);
            Sync(abReg, gReg, &cReg[2]);
            Sync(abReg, bReg, &cReg[4]);
         }
         for (int d = 0; d < 3; d++)
         {
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
               int cRow = MapM(groupId, mM);
               int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// Grad strip-mine shared by GradZ / GradZt / GradYt (gIdx selects G vs B).
/// BG rows are B then G (or Bt/Gt); A[d] as (k,m); C[d] as (m,n).
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZtLike(const int m, const int n,
                                        const int k, const int gIdx,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   ConstDeviceMatrix Bt(BG[0], k, n);
   ConstDeviceMatrix Gt(BG[1], k, n);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   const int bankMap = BankMap();
   const int aRowInWarp = groupId;
   const int aColumnInWarp = threadIdInGroup;
   const int bRowInWarp = threadIdInGroup;
   const int bColumnInWarp = groupId;

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[6] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1];
            double GtReg[1];
            const int bRow = bRowInWarp + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, bColumnInWarp, n0);
            if (bColumn < n && bRow < k)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else
            {
               BtReg[0] = 0;
               GtReg[0] = 0;
            }
            for (int d = 0; d < 3; d++)
            {
               double aReg[1];
               const int aRow = MapM(aRowInWarp, mM);
               const int aColumn = aColumnInWarp + mK * mmaK;
               if (aRow < m && aColumn < k)
               {
                  ConstDeviceMatrix aA(A[d], k, m);
                  aReg[0] = aA(aColumn, aRow);
               }
               else
               {
                  aReg[0] = 0;
               }
               Sync(aReg, d == gIdx ? GtReg : BtReg, &cReg[d * 2]);
            }
         }
         for (int d = 0; d < 3; d++)
         {
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
               const int cRow = MapM(groupId, mM);
               const int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
               if (cRow < m && cColumn < n)
               {
                  DeviceMatrix cC(C[d], m, n);
                  cC(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZ(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF],
                                   int gIdx)
{
   GradZtLike<MD1, MQ1, BUF>(m, n, k, gIdx, BG, A, C);
}

/** Mass interp core: strip-mined 1-comp B·A → C.
 *  ScaleAtStore: C *= D(q,e) (fused mass Q-fn). */
template<int MD1, int MQ1, bool ScaleAtStore = false>
MFEM_HOST_DEVICE inline void InterpAx(const int m, const int n,
                                      const int k,
                                      const real_t *B1d,
                                      const real_t *A, real_t *C,
                                      const DeviceTensor<2, const real_t> *D = nullptr,
                                      const int e = 0)
{
   ConstDeviceMatrix B(B1d, k, n);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int bankMap = BankMap(n);
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            bReg[0] = (bColumn < n && bRow < k) ? B(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM);
            const int aColumn = threadIdInGroup + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix aA(A, k, m);
               aReg[0] = aA(aColumn, aRow);
            }
            else { aReg[0] = 0; }
            Sync(aReg, bReg, cReg);
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM);
            const int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
            if (cRow < m && cColumn < n)
            {
               DeviceMatrix cC(C, m, n);
               if constexpr (ScaleAtStore)
               {
                  cC(cRow, cColumn) = cReg[i] * (*D)(cRow + m * cColumn, e);
               }
               else
               {
                  cC(cRow, cColumn) = cReg[i];
               }
            }
         }
      }
   }
}

/// 3D Transposed Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sDDQ)[3][MDQ*MDQ*MDQ],
                                    const DeviceTensor<4> &Y, // output
                                    const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   int thread = getThreadIdxX();
   int warpId = getWarpId(thread);
   int laneId = getLaneId(thread);
   int groupId = getGroupId(laneId);
   int threadIdInGroup = getThreadIdInGroup(laneId);

   // dx (D1D), dy (D1D) === M, dz (D1D) === N, qz (Q1D) === K
   int mPass = (D1D * D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   int aRowInWarp = groupId;
   int aColumnInWarp = threadIdInGroup;
   int bRowInWarp = threadIdInGroup;
   int bColumnInWarp = groupId;
   const int bankMap = BankMap();

   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double BtReg[1];
         double GtReg[1];
         double cReg[2] = {};

         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            int bRow = bRowInWarp + mK * mmaK;
            int bColumn = MappedNCol(bankMap, bColumnInWarp, n0);
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else
            {
               BtReg[0] = 0;
               GtReg[0] = 0;
            }
            for (int d = 0; d < 3; d++)
            {
               double aReg[1];
               int aRow = MapM(aRowInWarp, mM);
               int aColumn = aColumnInWarp + mK * mmaK;
               if (aRow < D1D * D1D && aColumn < Q1D)
               {
                  ConstDeviceMatrix Xx(sDDQ[d], Q1D, D1D * D1D); // qz, dx, dy
                  aReg[0] = Xx(aColumn, aRow);
               }
               else
               {
                  aReg[0] = 0;
               }

               Sync(aReg, d == 2 ? GtReg : BtReg, cReg);
            }
         }
#pragma unroll
         for (int i = 0; i < 2; i++)
         {
            int cRow = MapM(groupId, mM);
            int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
            if (cRow < D1D * D1D && cColumn < D1D)
            {
               int dx = cRow % D1D;
               int dy = cRow / D1D;
               int dz = cColumn;
               Y(dx,dy,dz,e) += cReg[i];
            }
         }
      }
   }
}

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int bankMap = BankMap(D1D);
   const int m = D1D * D1D, n = D1D, k = Q1D;
   const int mPass = (m + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < n; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (k + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            bReg[0] = (bColumn < n && bRow < k) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM);
            const int aColumn = threadIdInGroup + mK * mmaK;
            if (aRow < m && aColumn < k)
            {
               ConstDeviceMatrix Xx(sDDQ, k, m);
               aReg[0] = Xx(aColumn, aRow);
            }
            else { aReg[0] = 0; }
            Sync(aReg, bReg, cReg);
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM);
            const int cColumn = MappedNCol(bankMap, threadIdInGroup * 2 + i, n0);
            if (cRow < m && cColumn < n)
            {
               Y(cRow % D1D, cRow / D1D, cColumn, e) += cReg[i];
            }
         }
      }
   }
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int bankMap = BankMap();
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < Q1D; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (D1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1], gReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            if (bColumn < Q1D && bRow < D1D)
            {
               bReg[0] = B(bRow, bColumn);
               gReg[0] = G(bRow, bColumn);
            }
            else { bReg[0] = gReg[0] = 0; }
            double a0[1], a1[1];
            const int aRow = MapM(groupId, mM);
            const int aColumn = tinG + mK * mmaK;
            if (aRow < Q1D && aColumn < D1D)
            {
               ConstDeviceMatrix A0(sDQ[0], D1D, Q1D);
               ConstDeviceMatrix A1(sDQ[1], D1D, Q1D);
               a0[0] = A0(aColumn, aRow);
               a1[0] = A1(aColumn, aRow);
            }
            else { a0[0] = a1[0] = 0; }
            Sync(a0, bReg, &cReg[0]);
            Sync(a1, gReg, &cReg[2]);
         }
         for (int d = 0; d < 2; d++)
         {
            for (int i = 0; i < 2; i++)
            {
               const int cRow = MapM(groupId, mM);
               const int cColumn = MappedNCol(bankMap, tinG * 2 + i, n0);
               if (cRow < Q1D && cColumn < Q1D)
               {
                  DeviceMatrix C(sQQ[d], Q1D, Q1D);
                  C(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int bankMap = BankMap();
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[4] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1], GtReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else { BtReg[0] = GtReg[0] = 0; }
            for (int d = 0; d < 2; d++)
            {
               double aReg[1];
               const int aRow = MapM(groupId, mM);
               const int aColumn = tinG + mK * mmaK;
               if (aRow < Q1D && aColumn < Q1D)
               {
                  ConstDeviceMatrix A(sQQ[d], Q1D, Q1D);
                  aReg[0] = A(aRow, aColumn);
               }
               else { aReg[0] = 0; }
               Sync(aReg, d == 1 ? GtReg : BtReg, &cReg[d * 2]);
            }
         }
         for (int d = 0; d < 2; d++)
         {
            for (int i = 0; i < 2; i++)
            {
               const int cRow = MapM(groupId, mM);
               const int cColumn = MappedNCol(bankMap, tinG * 2 + i, n0);
               if (cRow < Q1D && cColumn < D1D)
               {
                  DeviceMatrix C(sQD[d], Q1D, D1D); // (qx, dy)
                  C(cRow, cColumn) = cReg[d * 2 + i];
               }
            }
         }
      }
   }
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int bankMap = BankMap();
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double BtReg[1], GtReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            if (bColumn < D1D && bRow < Q1D)
            {
               BtReg[0] = Bt(bRow, bColumn);
               GtReg[0] = Gt(bRow, bColumn);
            }
            else { BtReg[0] = GtReg[0] = 0; }
            for (int d = 0; d < 2; d++)
            {
               double aReg[1];
               const int aRow = MapM(groupId, mM);
               const int aColumn = tinG + mK * mmaK;
               if (aRow < D1D && aColumn < Q1D)
               {
                  ConstDeviceMatrix A(sQD[d], Q1D, D1D); // (qx, dy)
                  aReg[0] = A(aColumn, aRow);
               }
               else { aReg[0] = 0; }
               Sync(aReg, d == 0 ? GtReg : BtReg, cReg);
            }
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM); // dy
            const int cColumn = MappedNCol(bankMap, tinG * 2 + i, n0); // dx
            if (cRow < D1D && cColumn < D1D)
            {
               Y(cColumn, cRow, e) += cReg[i];
            }
         }
      }
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQQ, real_t *sQD)
{
   // K=qy fastest in A(qx,qy); N=dy — not the same A layout as InterpAx.
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int bankMap = BankMap(D1D);
   const int mPass = (Q1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM);
            const int aColumn = tinG + mK * mmaK;
            if (aRow < Q1D && aColumn < Q1D)
            {
               ConstDeviceMatrix A(sQQ, Q1D, Q1D);
               aReg[0] = A(aRow, aColumn);
            }
            else { aReg[0] = 0; }
            Sync(aReg, bReg, cReg);
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM);
            const int cColumn = MappedNCol(bankMap, tinG * 2 + i, n0);
            if (cRow < Q1D && cColumn < D1D)
            {
               DeviceMatrix C(sQD, Q1D, D1D);
               C(cRow, cColumn) = cReg[i];
            }
         }
      }
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQD,
                                        const DeviceTensor<3> &Y, const int e)
{
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int tinG = getThreadIdInGroup(laneId);
   const int bankMap = BankMap(D1D);
   const int mPass = (D1D + mmaM - 1) / mmaM;
   const int nWarps = NWarps(mPass);
   for (int mM = warpId; mM < mPass; mM += nWarps)
   {
      for (int n0 = 0; n0 < D1D; n0 += mmaN)
      {
         double cReg[2] = {};
         for (int mK = 0; mK < (Q1D + mmaK - 1) / mmaK; mK++)
         {
            double bReg[1];
            const int bRow = tinG + mK * mmaK;
            const int bColumn = MappedNCol(bankMap, groupId, n0);
            bReg[0] = (bColumn < D1D && bRow < Q1D) ? Bt(bRow, bColumn) : 0.0;
            double aReg[1];
            const int aRow = MapM(groupId, mM);
            const int aColumn = tinG + mK * mmaK;
            if (aRow < D1D && aColumn < Q1D)
            {
               ConstDeviceMatrix A(sQD, Q1D, D1D);
               aReg[0] = A(aColumn, aRow);
            }
            else { aReg[0] = 0; }
            Sync(aReg, bReg, cReg);
         }
         for (int i = 0; i < 2; i++)
         {
            const int cRow = MapM(groupId, mM);
            const int cColumn = MappedNCol(bankMap, tinG * 2 + i, n0);
            if (cRow < D1D && cColumn < D1D)
            {
               Y(cColumn, cRow, e) += cReg[i];
            }
         }
      }
   }
}

#endif // MFEM_USE_CUDA && !MFEM_USE_SINGLE


} // namespace mfem::internal::mma::dmma

/// \endcond DO_NOT_DOCUMENT

