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

#include "common.hpp" // IWYU pragma: export

// ======================================================================
// HIP (mfma) — Gemm + SUMF
// ======================================================================

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::mfma
{

#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)

using double4 =
   __attribute__((__vector_size__(4 * sizeof(double)))) double;

MFEM_HOST_DEVICE inline void Sync16(double a, double b, double4 &c)
{
   c = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}

MFEM_HOST_DEVICE inline void Sync4(double a, double b, double &c)
{
   c = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}

/** C = A * B via MFMA 16x16x4 (CDNA3). Lane L: A[L%16][L/16], B[L/16][L%16],
    C[(L/16)+4*i][L%16] = cReg[i]. */
template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Gemm16(const int M, const int K, const int N,
                                    TA A, TB B, TC C, TD D,
                                    const int e0, const int NE)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM; // also B's K index
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + aRow;
            const int aC = k0 + aColK;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + aColK;
            const double bV = (bR < K && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            Sync16(aV, bV, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            if (cRow < M && cCol < nTile)
            {
               real_t v = static_cast<real_t>(cReg[i]);
               if constexpr (SCALE)
               {
                  const int e = e0 + n0 + cCol;
                  v = (e < NE) ? v * D(cRow, e) : real_t(0);
               }
               C(cRow, n0 + cCol) = v;
            }
         }
      }
   }
}

/** Fused 3-component forward: U_d = G_d * X for d=0..2, loading each X fragment once. */
template <typename TA0, typename TA1, typename TA2, typename TB,
          typename TC0, typename TC1, typename TC2>
MFEM_HOST_DEVICE inline void Gemm16_Fwd3(const int M, const int K,
                                         const int N,
                                         TA0 A0, TA1 A1, TA2 A2, TB B,
                                         TC0 C0, TC1 C1, TC2 C2)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM;
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         double4 c0 = {0, 0, 0, 0};
         double4 c1 = {0, 0, 0, 0};
         double4 c2 = {0, 0, 0, 0};

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + aRow;
            const int aC = k0 + aColK;
            const int bR = k0 + aColK;
            const double bV = (bR < K && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            const double a0V = (aR < M && aC < K)
                               ? static_cast<double>(A0(aR, aC)) : 0.0;
            const double a1V = (aR < M && aC < K)
                               ? static_cast<double>(A1(aR, aC)) : 0.0;
            const double a2V = (aR < M && aC < K)
                               ? static_cast<double>(A2(aR, aC)) : 0.0;
            Sync16(a0V, bV, c0);
            Sync16(a1V, bV, c1);
            Sync16(a2V, bV, c2);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            if (cRow < M && cCol < nTile)
            {
               C0(cRow, n0 + cCol) = static_cast<real_t>(c0[i]);
               C1(cRow, n0 + cCol) = static_cast<real_t>(c1[i]);
               C2(cRow, n0 + cCol) = static_cast<real_t>(c2[i]);
            }
         }
      }
   }
}

/** Fused 3-component GemmT: Y += G_d^T * U_d for d=0..2 (shared Y accumulate). */
template <typename TA0, typename TA1, typename TA2, typename TB0,
          typename TB1, typename TB2, typename TC>
MFEM_HOST_DEVICE inline void GemmT16_3(const int M, const int K,
                                       const int N,
                                       TA0 A0, TA1 A1, TA2 A2,
                                       TB0 B0, TB1 B1, TB2 B2, TC C,
                                       const int e0, const int NE)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM;
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aT_row = row0 + aRow;
            const int aT_col = k0 + aColK;
            const int bR = k0 + aColK;
            const bool a_ok = (aT_row < K && aT_col < M);
            const bool b_ok = (bR < M && bCol < nTile);
            const double a0V = a_ok ? static_cast<double>(A0(aT_col, aT_row))
                               : 0.0;
            const double a1V = a_ok ? static_cast<double>(A1(aT_col, aT_row))
                               : 0.0;
            const double a2V = a_ok ? static_cast<double>(A2(aT_col, aT_row))
                               : 0.0;
            const double b0V = b_ok ? static_cast<double>(B0(bR, n0 + bCol))
                               : 0.0;
            const double b1V = b_ok ? static_cast<double>(B1(bR, n0 + bCol))
                               : 0.0;
            const double b2V = b_ok ? static_cast<double>(B2(bR, n0 + bCol))
                               : 0.0;
            // Accumulate all three components into one C tile.
            Sync16(a0V, b0V, cReg);
            Sync16(a1V, b1V, cReg);
            Sync16(a2V, b2V, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            const int e = e0 + n0 + cCol;
            if (cRow < K && cCol < nTile && e < NE)
            {
               C(cRow, n0 + cCol) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** C += A^T * B via MFMA 16x16x4. Loads A as A^T fragments. */
template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void GemmT16(const int M, const int K, const int N,
                                     TA A, TB B, TC C,
                                     const int e0, const int NE)
{
   // GemmT: out rows = K (ndof), reduce over M (nq). Tile out-rows with TM.
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;   // output row fragment within tile
   const int aColK = lane / TM;  // reduction K of A^T (= row of A)
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            // A^T(row,col) = A(col,row): row in K(ndof), col in M(nq)
            const int aT_row = row0 + aRow;
            const int aT_col = k0 + aColK;
            const double aV = (aT_row < K && aT_col < M)
                              ? static_cast<double>(A(aT_col, aT_row)) : 0.0;
            const int bR = k0 + aColK;
            const double bV = (bR < M && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            Sync16(aV, bV, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            const int e = e0 + n0 + cCol;
            if (cRow < K && cCol < nTile && e < NE)
            {
               C(cRow, n0 + cCol) += static_cast<real_t>(cReg[i]);
            }
         }
      }
   }
}

/** C = A * B via MFMA 4x4x4 with 4 blocks covering N=16 columns.
    Lane L: block=(L%16)/4, m=(L%16)%4, k=L/16. */
template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Gemm4(const int M, const int K, const int N,
                                   TA A, TB B, TC C, TD D,
                                   const int e0, const int NE)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int block = (lane % 16) / 4;
   const int mLoc = (lane % 16) % 4;
   const int kLoc = lane / 16;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + N_EFF - 1) / N_EFF;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * N_EFF;
         double cReg = 0.0;

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + mLoc;
            const int aC = k0 + kLoc;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + kLoc;
            const int bC = n0 + TN_BLK * block + mLoc; // n within block
            const double bV = (bR < K && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            Sync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc; // D layout: row = lane/16
         const int cCol = n0 + TN_BLK * block + mLoc;
         if (cRow < M && cCol < N)
         {
            real_t v = static_cast<real_t>(cReg);
            if constexpr (SCALE)
            {
               const int e = e0 + cCol;
               v = (e < NE) ? v * D(cRow, e) : real_t(0);
            }
            C(cRow, cCol) = v;
         }
      }
   }
}

template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void GemmT4(const int M, const int K, const int N,
                                    TA A, TB B, TC C,
                                    const int e0, const int NE)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int lane = getLaneId(thread);
   const int block = (lane % 16) / 4;
   const int mLoc = (lane % 16) % 4;
   const int kLoc = lane / 16;
   const int mPass = (K + TM - 1) / TM;
   const int nTiles = (N + N_EFF - 1) / N_EFF;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * N_EFF;
         double cReg = 0.0;

         for (int mK = 0; mK < (M + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aT_row = row0 + mLoc;
            const int aT_col = k0 + kLoc;
            const double aV = (aT_row < K && aT_col < M)
                              ? static_cast<double>(A(aT_col, aT_row)) : 0.0;
            const int bR = k0 + kLoc;
            const int bC = n0 + TN_BLK * block + mLoc;
            const double bV = (bR < M && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            Sync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc;
         const int cCol = n0 + TN_BLK * block + mLoc;
         const int e = e0 + cCol;
         if (cRow < K && cCol < N && e < NE)
         {
            C(cRow, cCol) += static_cast<real_t>(cReg);
         }
      }
   }
}

template <bool SCALE, typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Gemm(const int M, const int K, const int N,
                                  TA A, TB B, TC C, TD D,
                                  const int e0, const int NE)
{
   using Fn = decltype(&Gemm16<SCALE, TA, TB, TC, TD>);
   const Fn gemm = PreferMfma4(M, K)
                   ? &Gemm4<SCALE, TA, TB, TC, TD>
                   : &Gemm16<SCALE, TA, TB, TC, TD>;
   gemm(M, K, N, A, B, C, D, e0, NE);
}

template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void GemmT(const int M, const int K, const int N,
                                   TA A, TB B, TC C,
                                   const int e0, const int NE)
{
   // PreferMfma4 on (nq=M, ndof=K) — same as forward dims.
   using Fn = decltype(&GemmT16<TA, TB, TC>);
   const Fn gemm = PreferMfma4(M, K)
                   ? &GemmT4<TA, TB, TC>
                   : &GemmT16<TA, TB, TC>;
   gemm(M, K, N, A, B, C, e0, NE);
}

/** SUMF layout: storage is DeviceMatrix(p, k, m); A_fwd(m_idx,k_idx)=storage(k_idx,m_idx). */
struct SumfAFromKbyM
{
   const real_t *p;
   int k, m;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, k, m)(col, row);
   }
};

/** Already (M,K) layout: DeviceMatrix(p, m, k); A_fwd(row,col)=storage(row,col). */
struct SumfAFromMbyK
{
   const real_t *p;
   int m, k;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, m, k)(row, col);
   }
};

struct SumfBFromKbyN
{
   const real_t *p;
   int k, n;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return ConstDeviceMatrix(p, k, n)(row, col);
   }
};

struct SumfCToMbyN
{
   real_t *p;
   int m, n;
   MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
   {
      return DeviceMatrix(p, m, n)(row, col);
   }
};

struct SumfNullD
{
   MFEM_HOST_DEVICE inline real_t operator()(int, int) const { return real_t(1); }
};

struct SumfMassD
{
   const DeviceTensor<2, const real_t> *D;
   int m, e;
   MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
   {
      return (*D)(row + m * col, e);
   }
};

/** C = A * B via MFMA 16x16x4. A is (M,K), B is (K,N), C is (M,N). */
template <bool SCALE, bool ACCUM,
          typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Sumf16(const int M, const int K,
                                    const int N, TA A, TB B, TC C, TD D)
{
   constexpr int TM = 16, TN = 16, TK = 4;
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int nWarps = NWarps(1);
   const int lane = getLaneId(thread);
   const int aRow = lane % TM;
   const int aColK = lane / TM;
   const int bCol = lane % TN;
   const int cRowBase = lane / TN;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + TN - 1) / TN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * TN;
         const int nTile = (N - n0 < TN) ? (N - n0) : TN;
         mma::double4 cReg = {0, 0, 0, 0};

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + aRow;
            const int aC = k0 + aColK;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + aColK;
            const double bV = (bR < K && bCol < nTile)
                              ? static_cast<double>(B(bR, n0 + bCol)) : 0.0;
            Sync16(aV, bV, cReg);
         }

         for (int i = 0; i < 4; ++i)
         {
            const int cRow = row0 + cRowBase + 4 * i;
            const int cCol = bCol;
            if (cRow < M && cCol < nTile)
            {
               real_t v = static_cast<real_t>(cReg[i]);
               if constexpr (SCALE) { v *= D(cRow, n0 + cCol); }
               if constexpr (ACCUM) { C(cRow, n0 + cCol) += v; }
               else { C(cRow, n0 + cCol) = v; }
            }
         }
      }
   }
}

/** C = A * B via MFMA 4x4x4_4b covering N=16. */
template <bool SCALE, bool ACCUM,
          typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Sumf4(const int M, const int K,
                                   const int N, TA A, TB B, TC C, TD D)
{
   constexpr int TM = 4, TN_BLK = 4, N_EFF = 16, TK = 4;
   const int thread = getThreadIdxX();
   const int warpId = getWarpId(thread);
   const int nWarps = NWarps(1);
   const int lane = getLaneId(thread);
   const int block = (lane % 16) / 4;
   const int mLoc = (lane % 16) % 4;
   const int kLoc = lane / 16;
   const int mPass = (M + TM - 1) / TM;
   const int nTiles = (N + N_EFF - 1) / N_EFF;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * TM;
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * N_EFF;
         double cReg = 0.0;

         for (int mK = 0; mK < (K + TK - 1) / TK; ++mK)
         {
            const int k0 = mK * TK;
            const int aR = row0 + mLoc;
            const int aC = k0 + kLoc;
            const double aV = (aR < M && aC < K)
                              ? static_cast<double>(A(aR, aC)) : 0.0;
            const int bR = k0 + kLoc;
            const int bC = n0 + TN_BLK * block + mLoc;
            const double bV = (bR < K && bC < N)
                              ? static_cast<double>(B(bR, bC)) : 0.0;
            Sync4(aV, bV, cReg);
         }

         const int cRow = row0 + kLoc;
         const int cCol = n0 + TN_BLK * block + mLoc;
         if (cRow < M && cCol < N)
         {
            real_t v = static_cast<real_t>(cReg);
            if constexpr (SCALE) { v *= D(cRow, cCol); }
            if constexpr (ACCUM) { C(cRow, cCol) += v; }
            else { C(cRow, cCol) = v; }
         }
      }
   }
}

template <bool SCALE, bool ACCUM,
          typename TA, typename TB, typename TC, typename TD>
MFEM_HOST_DEVICE inline void Sumf(const int M, const int K, const int N,
                                  TA A, TB B, TC C, TD D)
{
   using Fn = decltype(&Sumf16<SCALE, ACCUM, TA, TB, TC, TD>);
   const Fn sumf = mma::PreferMfma4(M, N)
                   ? &Sumf4<SCALE, ACCUM, TA, TB, TC, TD>
                   : &Sumf16<SCALE, ACCUM, TA, TB, TC, TD>;
   sumf(M, K, N, A, B, C, D);
}

/** SUMF contraction C(m,n) = sum_k storageA(k,m) * storageB(k,n). */
template <bool SCALE, bool ACCUM, typename TD>
MFEM_HOST_DEVICE inline void Sumf(const int m, const int n,
                                  const int k, const real_t *A,
                                  const real_t *B1d, real_t *C,
                                  TD D)
{
   Sumf<SCALE, ACCUM>(m, k, n, SumfAFromKbyM{A, k, m},
                      SumfBFromKbyN{B1d, k, n}, SumfCToMbyN{C, m, n}, D);
}

// SUMF backend helpers (mfma)
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradX(const int m, const int n, const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   SumfNullD nd;
   // C[0] from G, C[1] from B (matches CUDA dmma::GradX store order).
   Sumf<false, false>(m, n, k, A[0], BG[1], C[0], nd);
   Sumf<false, false>(m, n, k, A[0], BG[0], C[1], nd);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradY(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   SumfNullD nd;
   Sumf<false, false>(m, n, k, A[0], BG[0], C[0], nd); // A0*B
   Sumf<false, false>(m, n, k, A[1], BG[1], C[1], nd); // A1*G
   Sumf<false, false>(m, n, k, A[1], BG[0], C[2], nd); // A1*B
}

/// Grad strip-mine shared by GradZ / GradZt / GradYt (gIdx selects G vs B).
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZtLike(const int m, const int n,
                                        const int k, const int gIdx,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   SumfNullD nd;
   for (int d = 0; d < 3; d++)
   {
      const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
      Sumf<false, false>(m, n, k, A[d], B1d, C[d], nd);
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
   if constexpr (ScaleAtStore)
   {
      SumfMassD Dd{D, m, e};
      Sumf<true, false>(m, n, k, A, B1d, C, Dd);
   }
   else
   {
      SumfNullD nd;
      Sumf<false, false>(m, n, k, A, B1d, C, nd);
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
   const int m = D1D * D1D, n = D1D, k = Q1D;
   struct Y3Acc
   {
      const DeviceTensor<4> *Y;
      int D1D, e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(row % D1D, row / D1D, col, e);
      }
   };
   SumfNullD nd;
   Y3Acc Yacc{&Y, D1D, e};
   // Y += A0*Bt + A1*Bt + A2*Gt
   Sumf<false, true>(m, k, n, SumfAFromKbyM{sDDQ[0], k, m},
                     SumfBFromKbyN{sBG[0], k, n}, Yacc, nd);
   Sumf<false, true>(m, k, n, SumfAFromKbyM{sDDQ[1], k, m},
                     SumfBFromKbyN{sBG[0], k, n}, Yacc, nd);
   Sumf<false, true>(m, k, n, SumfAFromKbyM{sDDQ[2], k, m},
                     SumfBFromKbyN{sBG[1], k, n}, Yacc, nd);
}

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   const int m = D1D * D1D, n = D1D, k = Q1D;
   struct Y3Acc
   {
      const DeviceTensor<4> *Y;
      int D1D, e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(row % D1D, row / D1D, col, e);
      }
   };
   SumfNullD nd;
   Sumf<false, true>(m, k, n, SumfAFromKbyM{sDDQ, k, m},
                     SumfBFromKbyN{sBt, k, n}, Y3Acc{&Y, D1D, e}, nd);
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ])
{
   SumfNullD nd;
   Sumf<false, false>(Q1D, Q1D, D1D, sDQ[0], sBG[0], sQQ[0], nd);
   Sumf<false, false>(Q1D, Q1D, D1D, sDQ[1], sBG[1], sQQ[1], nd);
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ])
{
   SumfNullD nd;
   // A is (qx,qy)=(M,K); B is Bt/Gt (K,N)=(Q,D)
   Sumf<false, false>(Q1D, Q1D, D1D, SumfAFromMbyK{sQQ[0], Q1D, Q1D},
                      SumfBFromKbyN{sBG[0], Q1D, D1D},
                      SumfCToMbyN{sQD[0], Q1D, D1D}, nd);
   Sumf<false, false>(Q1D, Q1D, D1D, SumfAFromMbyK{sQQ[1], Q1D, Q1D},
                      SumfBFromKbyN{sBG[1], Q1D, D1D},
                      SumfCToMbyN{sQD[1], Q1D, D1D}, nd);
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   // A storage (qx,dy)=(K,M); C row=dy, col=dx → Y(dx,dy) = Y(col,row)
   struct Y2Acc
   {
      const DeviceTensor<3> *Y;
      int e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(col, row, e);
      }
   };
   SumfNullD nd;
   Y2Acc Yacc{&Y, e};
   Sumf<false, true>(D1D, Q1D, D1D, SumfAFromKbyM{sQD[0], Q1D, D1D},
                     SumfBFromKbyN{sBG[1], Q1D, D1D}, Yacc, nd); // Gt
   Sumf<false, true>(D1D, Q1D, D1D, SumfAFromKbyM{sQD[1], Q1D, D1D},
                     SumfBFromKbyN{sBG[0], Q1D, D1D}, Yacc, nd); // Bt
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQQ, real_t *sQD)
{
   // K=qy fastest in A(qx,qy); N=dy — (M,K) layout, not InterpAx's (K,M).
   SumfNullD nd;
   Sumf<false, false>(Q1D, Q1D, D1D, SumfAFromMbyK{sQQ, Q1D, Q1D},
                      SumfBFromKbyN{sBt, Q1D, D1D},
                      SumfCToMbyN{sQD, Q1D, D1D}, nd);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQD,
                                        const DeviceTensor<3> &Y, const int e)
{
   struct Y2Acc
   {
      const DeviceTensor<3> *Y;
      int e;
      MFEM_HOST_DEVICE inline real_t &operator()(int row, int col) const
      {
         return (*Y)(col, row, e);
      }
   };
   SumfNullD nd;
   Sumf<false, true>(D1D, Q1D, D1D, SumfAFromKbyM{sQD, Q1D, D1D},
                     SumfBFromKbyN{sBt, Q1D, D1D}, Y2Acc{&Y, e}, nd);
}

#endif // __HIP_DEVICE_COMPILE__ && !MFEM_USE_SINGLE


} // namespace mfem::internal::mma::mfma

/// \endcond DO_NOT_DOCUMENT

