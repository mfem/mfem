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

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::blas
{

// Cooperative dense GEMM (device Emulate / host)

/** Dense cooperative GEMM (no MMA): U(q,b) = sum_i B(q,i)*X(i,b) [, *D].
    Sibling of dmma::Gemm / mfma::Gemm for CPU, single, and pre-sm_80 paths. */
template <bool SCALE, typename BasisAcc,
          typename XAcc, typename UAcc, typename DAcc>
MFEM_HOST_DEVICE inline void Gemm(const int M, const int ndof,
                                  const int NB, BasisAcc B,
                                  XAcc X, UAcc U, DAcc D,
                                  const int e0, const int NE)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   for (int idx = tid; idx < M * NB; idx += nthreads)
   {
      const int b = idx / M;
      const int q = idx - b * M;
      real_t s = 0.0;
      for (int i = 0; i < ndof; ++i)
      {
         s += B(q, i) * X(i, b);
      }
      if constexpr (SCALE)
      {
         const int e = e0 + b;
         s = (e < NE) ? s * D(q, e) : real_t(0);
      }
      U(q, b) = s;
   }
}

/** Dense cooperative GEMM^T: Y(i,b) += sum_q B(q,i)*U(q,b). */
template <typename BasisAcc, typename UAcc, typename YAcc>
MFEM_HOST_DEVICE inline void GemmT(const int M, const int ndof,
                                   const int NB, BasisAcc B,
                                   UAcc U, YAcc Y,
                                   const int e0, const int NE)
{
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();
   for (int idx = tid; idx < ndof * NB; idx += nthreads)
   {
      const int b = idx / ndof;
      const int i = idx - b * ndof;
      const int e = e0 + b;
      if (e >= NE) { continue; }
      real_t s = 0.0;
      for (int q = 0; q < M; ++q)
      {
         s += B(q, i) * U(q, b);
      }
      Y(i, b) += s;
   }
}


/** Load X tile: xloc[i*NB+b], pad zeros. */
template <int NDOF, int NB>
inline void PackX(const real_t *X, int e0, int NE, real_t *xloc)
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
    If SCALE is false, D may be null and scale is 1. */
template <int NDOF, int NQ, int NB, bool SCALE>
inline void Gemm(const real_t *B, const real_t *xloc, real_t *uloc,
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
      if constexpr (SCALE)
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

/** Like Gemm but reads X as column-major X[i + NDOF*(e0+b)] (no pack).
    Requires a full tile: e0+NB <= NE. */
template <int NDOF, int NQ, int NB, bool SCALE>
inline void GemmFromColMajor(const real_t *B, const real_t *X, int e0,
                             real_t *uloc, const real_t *D)
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
            ub[b] += bqi * X[i + NDOF * (e0 + b)];
         }
      }
      if constexpr (SCALE)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            uloc[q * NB + b] = ub[b] * D[q + NQ * (e0 + b)];
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

/** Y(e0+b) += B^T * U. B column-major nq×ndof.
    FULL_TILE: e0+NB <= NE, no bounds checks. */
template <int NDOF, int NQ, int NB, bool FULL_TILE = false>
inline void GemmT(const real_t *B, const real_t *uloc, real_t *Y,
                  int e0, int NE = 0)
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
      if constexpr (FULL_TILE)
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            Y[i + NDOF * (e0 + b)] += yb[b];
         }
      }
      else
      {
         MFEM_UNROLL(NB)
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e < NE) { Y[i + NDOF * e] += yb[b]; }
         }
      }
   }
}

/** Full-tile wrapper: e0+NB <= NE. */
template <int NDOF, int NQ, int NB>
inline void GemmTFull(const real_t *B, const real_t *uloc, real_t *Y,
                      int e0)
{
   GemmT<NDOF, NQ, NB, true>(B, uloc, Y, e0, 0);
}


// ======================================================================


/** Dense SUMF: C(m,n) =[/+=] sum_k A_storage(k,m)*B(k,n) [, *D].
    A is stored as DeviceMatrix(k,m); B as DeviceMatrix(k,n). */
template <bool SCALE, bool ACCUM>
MFEM_HOST_DEVICE inline void Sumf(const int m, const int n,
                                  const int k, const real_t *A,
                                  const real_t *B1d, real_t *C,
                                  const DeviceTensor<2, const real_t> *D = nullptr,
                                  const int e = 0)
{
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix B(B1d, k, n);
   ConstDeviceMatrix aA(A, k, m);
   DeviceMatrix cC(C, m, n);
   for (int idx = tid; idx < m * n; idx += nthreads)
   {
      const int col = idx / m;
      const int row = idx - col * m;
      real_t s = 0.0;
      for (int p = 0; p < k; ++p)
      {
         s += aA(p, row) * B(p, col);
      }
      if constexpr (SCALE)
      {
         s *= (*D)(row + m * col, e);
      }
      if constexpr (ACCUM)
      {
         cC(row, col) += s;
      }
      else
      {
         cC(row, col) = s;
      }
   }
}

/** Dense SUMF with A already in (M,K) layout. */
template <bool ACCUM>
MFEM_HOST_DEVICE inline void GemmMbyK(const int M, const int K,
                                      const int N, const real_t *A,
                                      const real_t *B1d, real_t *C)
{
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix aA(A, M, K);
   ConstDeviceMatrix B(B1d, K, N);
   DeviceMatrix cC(C, M, N);
   for (int idx = tid; idx < M * N; idx += nthreads)
   {
      const int col = idx / M;
      const int row = idx - col * M;
      real_t s = 0.0;
      for (int p = 0; p < K; ++p)
      {
         s += aA(row, p) * B(p, col);
      }
      if constexpr (ACCUM)
      {
         cC(row, col) += s;
      }
      else
      {
         cC(row, col) = s;
      }
   }
}

/** Dense GradXt: A* from GradYt as qx + Q*(dy + D*dz). */
MFEM_HOST_DEVICE inline void GradXt3D(const int D1D, const int Q1D,
                                      const real_t *Bt, const real_t *Gt,
                                      const real_t *A0, const real_t *A1,
                                      const real_t *A2,
                                      const DeviceTensor<4> &Y, const int e)
{
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix Btm(Bt, Q1D, D1D);
   ConstDeviceMatrix Gtm(Gt, Q1D, D1D);
   const int nout = D1D * D1D * D1D;
   for (int idx = tid; idx < nout; idx += nthreads)
   {
      const int dx = idx % D1D;
      const int t = idx / D1D;
      const int dy = t % D1D;
      const int dz = t / D1D;
      real_t s = 0.0;
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const int a = qx + Q1D * (dy + D1D * dz);
         // Match Element SUM: gX*Gt + gY*Bt + gZ*Bt
         s += A0[a] * Gtm(qx, dx);
         s += A1[a] * Btm(qx, dx);
         s += A2[a] * Btm(qx, dx);
      }
      Y(dx, dy, dz, e) += s;
   }
}


// SUMF backend helpers (blas)
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradX(const int m, const int n, const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   Sumf<false, false>(m, n, k, A[0], BG[1], C[0]);
   Sumf<false, false>(m, n, k, A[0], BG[0], C[1]);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradY(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   Sumf<false, false>(m, n, k, A[0], BG[0], C[0]);
   Sumf<false, false>(m, n, k, A[1], BG[1], C[1]);
   Sumf<false, false>(m, n, k, A[1], BG[0], C[2]);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZ(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF],
                                   int gIdx)
{
   for (int d = 0; d < 3; d++)
   {
      const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
      Sumf<false, false>(m, n, k, A[d], B1d, C[d]);
   }
}

/// Transposed Grad strip-mine shared by GradZt (gIdx=0) and GradYt (gIdx=1).
/// BG is BGt layout (Q,D); A[d] viewed as (k,m); C[d] as (m,n).
template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZtLike(const int m, const int n,
                                        const int k, const int gIdx,
                                        const real_t (&BG)[2][MQ1*MD1],
                                        const real_t (*A)[BUF],
                                        real_t (*C)[BUF])
{
   // Forward Grad* stores C as DeviceMatrix(m,n)=(M,K); transpose reads (M,K).
   for (int d = 0; d < 3; d++)
   {
      const real_t *B1d = (d == gIdx) ? BG[1] : BG[0];
      GemmMbyK<false>(m, k, n, A[d], B1d, C[d]);
   }
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
   Sumf<ScaleAtStore, false>(m, n, k, A, B1d, C, D, e);
}

/// 3D Transposed Gradient, 2/3 (blas): Gt on component 1; layout for GradXt.
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   // sDQQ from GradZt GemmMbyK: (qx+Q*qy)+Q*Q*dz. Contract qy; store
   // qx + Q*(dy + D*dz) for GradXt3D. Gt on component 1.
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   const int nout = Q1D * D1D * D1D;
   for (int idx = tid; idx < nout; idx += nthreads)
   {
      const int qx = idx % Q1D;
      const int t = idx / Q1D;
      const int dy = t % D1D;
      const int dz = t / D1D;
      for (int d = 0; d < 3; ++d)
      {
         real_t s = 0.0;
         const real_t *B1d = (d == 1) ? Gt : Bt;
         ConstDeviceMatrix Bq(B1d, Q1D, D1D);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            s += sDQQ[d][(qx + Q1D * qy) + Q1D * Q1D * dz] * Bq(qy, dy);
         }
         sDDQ[d][qx + Q1D * (dy + D1D * dz)] = s;
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
   GradXt3D(D1D, Q1D, sBG[0], sBG[1], sDDQ[0], sDDQ[1], sDDQ[2], Y, e);
}

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   // sDDQ from InterpYt Emulate: qx + Q*(dy + D*dz)
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   const int nout = D1D * D1D * D1D;
   for (int idx = tid; idx < nout; idx += nthreads)
   {
      const int dx = idx % D1D;
      const int t = idx / D1D;
      const int dy = t % D1D;
      const int dz = t / D1D;
      real_t s = 0.0;
      for (int qx = 0; qx < Q1D; ++qx)
      {
         s += sDDQ[qx + Q1D * (dy + D1D * dz)] * Bt(qx, dx);
      }
      Y(dx, dy, dz, e) += s;
   }
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ])
{
   Sumf<false, false>(Q1D, Q1D, D1D, sDQ[0], sBG[0], sQQ[0]);
   Sumf<false, false>(Q1D, Q1D, D1D, sDQ[1], sBG[1], sQQ[1]);
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ])
{
   GemmMbyK<false>(Q1D, Q1D, D1D, sQQ[0], sBG[0], sQD[0]);
   GemmMbyK<false>(Q1D, Q1D, D1D, sQQ[1], sBG[1], sQD[1]);
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceMatrix A0(sQD[0], Q1D, D1D);
   ConstDeviceMatrix A1(sQD[1], Q1D, D1D);
   for (int idx = tid; idx < D1D * D1D; idx += nthreads)
   {
      const int dy = idx / D1D; // row
      const int dx = idx - dy * D1D; // col
      real_t s = 0.0;
      for (int q = 0; q < Q1D; ++q)
      {
         s += A0(q, dy) * Gt(q, dx);
         s += A1(q, dy) * Bt(q, dx);
      }
      Y(dx, dy, e) += s;
   }
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQQ, real_t *sQD)
{
   GemmMbyK<false>(Q1D, Q1D, D1D, sQQ, sBt, sQD);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQD,
                                        const DeviceTensor<3> &Y, const int e)
{
   const int tid = getThreadIdxX();
   const int nthreads = getBlockNthreadsX();
   ConstDeviceMatrix Bt(sBt, Q1D, D1D);
   ConstDeviceMatrix A(sQD, Q1D, D1D);
   for (int idx = tid; idx < D1D * D1D; idx += nthreads)
   {
      const int dy = idx / D1D;
      const int dx = idx - dy * D1D;
      real_t s = 0.0;
      for (int q = 0; q < Q1D; ++q)
      {
         s += A(q, dy) * Bt(q, dx);
      }
      Y(dx, dy, e) += s;
   }
}

} // namespace mfem::internal::mma::blas

/// \endcond DO_NOT_DOCUMENT

