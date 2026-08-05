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

#include "dmma.hpp"
#include "mfma.hpp"
#include "blas.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma
{

// ======================================================================
// Dispatch — pick dmma / mfma / blas
// ======================================================================

/** Select backend function pointer: tensor MMA when enabled, else dense blas.
    Macro (not a template): host-only builds never form &dmma:: / &mfma::, which
    exist only under MFEM_USE_CUDA / HIP device compile.
    Callers must parenthesize each arg (commas in template args):
      MMA_BACKEND_PICK((&dmma::Foo<A,B>), (&mfma::Foo<A,B>), (&blas::Foo<A,B>)) */
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
#define MMA_BACKEND_PICK(dmma_fn, mfma_fn, blas_fn) \
   (TensorMmaEnabled() ? static_cast<decltype(blas_fn)>(dmma_fn) : (blas_fn))
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
#define MMA_BACKEND_PICK(dmma_fn, mfma_fn, blas_fn) \
   (TensorMmaEnabled() ? static_cast<decltype(blas_fn)>(mfma_fn) : (blas_fn))
#else
#define MMA_BACKEND_PICK(dmma_fn, mfma_fn, blas_fn) (blas_fn)
#endif

// ---- Gemm* + public SUMF -----------------------------------------------


/** One-component forward: U = B * X [, * D if SCALE].
    M is the GEMM row count (full QND or nq_tile). */
template <int MAP = 0, bool SCALE = false, typename BasisAcc, typename XAcc,
          typename UAcc, typename DAcc>
MFEM_HOST_DEVICE inline void Gemm(const int M, const int ndof,
                                  const int NB, BasisAcc B,
                                  XAcc X, UAcc U, DAcc D,
                                  const int e0, const int NE)
{
   const auto gemm =
      MMA_BACKEND_PICK(
         (&dmma::Gemm<MAP, SCALE, BasisAcc, XAcc, UAcc, DAcc>),
         (&mfma::Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>),
         (&blas::Gemm<SCALE, BasisAcc, XAcc, UAcc, DAcc>));
   gemm(M, ndof, NB, B, X, U, D, e0, NE);
}

/** One-component transpose accumulate: Y += B^T * U. */
template <int MAP = 0, typename BasisAcc, typename UAcc, typename YAcc>
MFEM_HOST_DEVICE inline void GemmT(const int M, const int ndof,
                                   const int NB, BasisAcc B,
                                   UAcc U, YAcc Y,
                                   const int e0, const int NE)
{
   const auto gemm =
      MMA_BACKEND_PICK(
         (&dmma::GemmT<MAP, BasisAcc, UAcc, YAcc>),
         (&mfma::GemmT<BasisAcc, UAcc, YAcc>),
         (&blas::GemmT<BasisAcc, UAcc, YAcc>));
   gemm(M, ndof, NB, B, U, Y, e0, NE);
}

/** Fused 3D GradP forward: U0,U1,U2 = G0,G1,G2 * X. */
template <int MAP = 0, typename Basis0, typename Basis1, typename Basis2,
          typename XAcc, typename U0, typename U1, typename U2>
MFEM_HOST_DEVICE inline void Gemm3(const int M, const int ndof,
                                   const int NB,
                                   Basis0 B0, Basis1 B1, Basis2 B2,
                                   XAcc X, U0 U0a, U1 U1a, U2 U2a,
                                   const int e0, const int NE)
{
   if (TensorMmaEnabled())
   {
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
      (void)e0; (void)NE;
      dmma::Gemm8_Fwd3<MAP>(M, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
      if (PreferMfma4(M, ndof))
      {
         NullDAcc nullD;
         Gemm(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
         Gemm(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
         Gemm(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
      }
      else
      {
         (void)e0; (void)NE;
         mfma::Gemm16_Fwd3(M, ndof, NB, B0, B1, B2, X, U0a, U1a, U2a);
      }
#else
      NullDAcc nullD;
      Gemm(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
      Gemm(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
      Gemm(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
#endif
   }
   else
   {
      NullDAcc nullD;
      Gemm(M, ndof, NB, B0, X, U0a, nullD, e0, NE);
      Gemm(M, ndof, NB, B1, X, U1a, nullD, e0, NE);
      Gemm(M, ndof, NB, B2, X, U2a, nullD, e0, NE);
   }
}


/** Fused 3D GradP^T accumulate into Y. */
template <int MAP = 0, typename Basis0, typename Basis1, typename Basis2,
          typename U0, typename U1, typename U2, typename YAcc>
MFEM_HOST_DEVICE inline void GemmT3(const int M, const int ndof,
                                    const int NB,
                                    Basis0 B0, Basis1 B1, Basis2 B2,
                                    U0 U0a, U1 U1a, U2 U2a, YAcc Y,
                                    const int e0, const int NE)
{
   if (TensorMmaEnabled())
   {
#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
      dmma::GemmT8_3<MAP>(M, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
      if (PreferMfma4(M, ndof))
      {
         GemmT(M, ndof, NB, B0, U0a, Y, e0, NE);
         GemmT(M, ndof, NB, B1, U1a, Y, e0, NE);
         GemmT(M, ndof, NB, B2, U2a, Y, e0, NE);
      }
      else
      {
         mfma::GemmT16_3(M, ndof, NB, B0, B1, B2, U0a, U1a, U2a, Y, e0, NE);
      }
#else
      GemmT(M, ndof, NB, B0, U0a, Y, e0, NE);
      GemmT(M, ndof, NB, B1, U1a, Y, e0, NE);
      GemmT(M, ndof, NB, B2, U2a, Y, e0, NE);
#endif
   }
   else
   {
      GemmT(M, ndof, NB, B0, U0a, Y, e0, NE);
      GemmT(M, ndof, NB, B1, U1a, Y, e0, NE);
      GemmT(M, ndof, NB, B2, U2a, Y, e0, NE);
   }
}


template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradX(const int m, const int n, const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradX<MD1, MQ1, BUF>),
                      (&mfma::GradX<MD1, MQ1, BUF>),
                      (&blas::GradX<MD1, MQ1, BUF>));
   fn(m, n, k, BG, A, C);
}

/// 3D Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDD)[MDQ*MDQ*MDQ],
                                   real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   GradX<MD1, MQ1, MDQ*MDQ*MDQ>(D1D * D1D, Q1D, D1D, sBG, sDDD, sDDQ);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradY(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF])
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradY<MD1, MQ1, BUF>),
                      (&mfma::GradY<MD1, MQ1, BUF>),
                      (&blas::GradY<MD1, MQ1, BUF>));
   fn(m, n, k, BG, A, C);
}

/// 3D Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDQ)[MDQ*MDQ*MDQ],
                                   real_t (*sDQQ)[MDQ*MDQ*MDQ])
{
   GradY<MD1, MQ1, MDQ*MDQ*MDQ>(D1D * Q1D, Q1D, D1D, sBG, sDDQ, sDQQ);
}

template<int MD1, int MQ1, int BUF>
MFEM_HOST_DEVICE inline void GradZ(const int m, const int n,
                                   const int k,
                                   const real_t (&BG)[2][MQ1*MD1],
                                   const real_t (*A)[BUF],
                                   real_t (*C)[BUF],
                                   int gIdx)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradZ<MD1, MQ1, BUF>),
                      (&mfma::GradZ<MD1, MQ1, BUF>),
                      (&blas::GradZ<MD1, MQ1, BUF>));
   fn(m, n, k, BG, A, C, gIdx);
}

/// 3D Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZ(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                   real_t (*sQQQ)[MDQ*MDQ*MDQ])
{
   GradZ<MD1, MQ1, MDQ*MDQ*MDQ>(Q1D * Q1D, Q1D, D1D, sBG, sDQQ, sQQQ, 2);
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
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradZtLike<MD1, MQ1, BUF>),
                      (&mfma::GradZtLike<MD1, MQ1, BUF>),
                      (&blas::GradZtLike<MD1, MQ1, BUF>));
   fn(m, n, k, gIdx, BG, A, C);
}

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sQQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDQQ)[MDQ*MDQ*MDQ])
{
   // Blas uses physical gZ (d==2); MMA fragment convention uses gIdx=0.
   constexpr int BUF = MDQ * MDQ * MDQ;
   const auto fn =
      MMA_BACKEND_PICK(
         (&dmma::GradZtLike<MD1, MQ1, BUF>),
         (&mfma::GradZtLike<MD1, MQ1, BUF>),
         (&blas::GradZtLike<MD1, MQ1, BUF>));
   const int gIdx = TensorMmaEnabled() ? 0 : 2;
   fn(Q1D * Q1D, D1D, Q1D, gIdx, sBG, sQQQ, sDQQ);
}

/// 3D Transposed Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDDQ)[MDQ*MDQ*MDQ])
{
   if (!mma::TensorMmaEnabled())
   {
      blas::GradYt<MD1, MQ1, MDQ>(D1D, Q1D, sBG, sDQQ, sDDQ);
      return;
   }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(MFEM_USE_SINGLE)
   mfma::GradZtLike<MD1, MQ1, MDQ*MDQ*MDQ>(
      D1D * Q1D, D1D, Q1D, 1, sBG, sDQQ, sDDQ);
#elif defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   dmma::GradZtLike<MD1, MQ1, MDQ*MDQ*MDQ>(
      D1D * Q1D, D1D, Q1D, 1, sBG, sDQQ, sDDQ);
#else
   blas::GradYt<MD1, MQ1, MDQ>(D1D, Q1D, sBG, sDQQ, sDDQ);
#endif
}

/// 3D Transposed Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sDDQ)[3][MDQ*MDQ*MDQ],
                                    const DeviceTensor<4> &Y, // output
                                    const int e)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradXt<MD1, MQ1, MDQ>),
                      (&mfma::GradXt<MD1, MQ1, MDQ>),
                      (&blas::GradXt<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBG, sDDQ, Y, e);
}

/// Load forward (D,Q) and transpose (Q,D) B with one global read.
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBBoth(const int D1D, const int Q1D,
                                       const ConstDeviceMatrix &b,
                                       real_t (&sB)[MQ1*MD1],
                                       real_t (&sBt)[MQ1*MD1])
{
   DeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix Bt(sBt, Q1D, D1D);
   const int tid = getThreadIdxX();
   const int n = D1D * Q1D;
   const int stride = getBlockNthreadsX();
   for (int t = tid; t < n; t += stride)
   {
      const int q = t / D1D;
      const int d = t % D1D;
      const real_t bv = b(q, d);
      B(d, q) = bv;
      Bt(q, d) = bv;
   }
}

/** Mass interp core: strip-mined 1-comp B·A → C.
 *  ScaleAtStore: C *= D(q,e) (fused mass Q-fn). */
template<int MD1, int MQ1, bool ScaleAtStore = false>
MFEM_HOST_DEVICE inline void InterpAx(const int m, const int n, const int k,
                                      const real_t *B1d,
                                      const real_t *A, real_t *C,
                                      const DeviceTensor<2, const real_t> *D = nullptr,
                                      const int e = 0)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::InterpAx<MD1, MQ1, ScaleAtStore>),
                      (&mfma::InterpAx<MD1, MQ1, ScaleAtStore>),
                      (&blas::InterpAx<MD1, MQ1, ScaleAtStore>));
   fn(m, n, k, B1d, A, C, D, e);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpX(const int D1D, const int Q1D,
                                     const real_t *sB,
                                     const real_t *sDDD, real_t *sDDQ)
{
   InterpAx<MD1, MQ1>(D1D * D1D, Q1D, D1D, sB, sDDD, sDDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpY(const int D1D, const int Q1D,
                                     const real_t *sB,
                                     const real_t *sDDQ, real_t *sDQQ)
{
   InterpAx<MD1, MQ1>(D1D * Q1D, Q1D, D1D, sB, sDDQ, sDQQ);
}

/** InterpZ + mass Q-fn fused at DMMA store. */
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpZMass(
   const int D1D, const int Q1D, const real_t *sB,
   const real_t *sDQQ, real_t *sQQQ,
   const DeviceTensor<2, const real_t> &D, const int e)
{
   InterpAx<MD1, MQ1, true>(Q1D * Q1D, Q1D, D1D, sB, sDQQ, sQQQ, &D, e);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpZt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sQQQ, real_t *sDQQ)
{
   if (!mma::TensorMmaEnabled())
   {
      // Forward InterpZ stored (M,K)=(Q*Q,Q); transpose is GemmMbyK.
      blas::GemmMbyK<false>(Q1D * Q1D, Q1D, D1D, sQQQ, sBt, sDQQ);
      return;
   }
   InterpAx<MD1, MQ1>(Q1D * Q1D, D1D, Q1D, sBt, sQQQ, sDQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDQQ, real_t *sDDQ)
{
   if (!mma::TensorMmaEnabled())
   {
      // sDQQ from InterpZt: (qx+Q*qy)+Q*Q*dz. Contract qy -> dy; store
      // sDDQ as qx + Q*(dy + D*dz) for InterpXt Emulate.
      const int tid = getThreadIdxX();
      const int nthreads = getBlockNthreadsX();
      ConstDeviceMatrix Bt(sBt, Q1D, D1D);
      const int nout = Q1D * D1D * D1D;
      for (int idx = tid; idx < nout; idx += nthreads)
      {
         const int qx = idx % Q1D;
         const int t = idx / Q1D;
         const int dy = t % D1D;
         const int dz = t / D1D;
         real_t s = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            s += sDQQ[(qx + Q1D * qy) + Q1D * Q1D * dz] * Bt(qy, dy);
         }
         sDDQ[qx + Q1D * (dy + D1D * dz)] = s;
      }
      return;
   }
   InterpAx<MD1, MQ1>(D1D * Q1D, D1D, Q1D, sBt, sDQQ, sDDQ);
}

/** InterpAx store to global Y (3D mass): Y(dx,dy,dz,e) += C. */
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void InterpXt(const int D1D, const int Q1D,
                                      const real_t *sBt,
                                      const real_t *sDDQ,
                                      const DeviceTensor<4> &Y, const int e)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::InterpXt<MD1, MQ1>),
                      (&mfma::InterpXt<MD1, MQ1>),
                      (&blas::InterpXt<MD1, MQ1>));
   fn(D1D, Q1D, sBt, sDDQ, Y, e);
}


// ---- 2D (quad) helpers ----

template<int MQ1>
MFEM_HOST_DEVICE inline void LoadX2D(const int e, const int D1D,
                                     const DeviceTensor<3, const real_t> &x,
                                     real_t *sm)
{
   DeviceMatrix X(sm, D1D, D1D);
   const int tid = getThreadIdxX();
   const int n = D1D * D1D;
   const int stride = getBlockNthreadsX();
   for (int t = tid; t < n; t += stride)
   {
      const int dx = t % D1D;
      const int dy = t / D1D;
      X(dx, dy) = x(dx, dy, e);
   }
}

/// 2D GradX: M=D1D (dy), N=Q1D, K=D1D → 2 comps (G, B)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDD)[MDQ*MDQ],
                                     real_t (*sDQ)[MDQ*MDQ])
{
   GradX<MD1, MQ1, MDQ*MDQ>(D1D, Q1D, D1D, sBG, sDD, sDQ);
}

/// 2D GradY: M=Q1D (qx), N=Q1D (qy), K=D1D → gX=A0*B, gY=A1*G
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY2D(const int D1D, const int Q1D,
                                     const real_t (&sBG)[2][MQ1*MD1],
                                     const real_t (*sDQ)[MDQ*MDQ],
                                     real_t (*sQQ)[MDQ*MDQ])
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradY2D<MD1, MQ1, MDQ>),
                      (&mfma::GradY2D<MD1, MQ1, MDQ>),
                      (&blas::GradY2D<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBG, sDQ, sQQ);
}

/// Undo GradY: K=qy, M=qx, N=dy; Gt on gY (d==1)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQQ)[MDQ*MDQ],
                                      real_t (*sQD)[MDQ*MDQ])
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradYt2D<MD1, MQ1, MDQ>),
                      (&mfma::GradYt2D<MD1, MQ1, MDQ>),
                      (&blas::GradYt2D<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBG, sQQ, sQD);
}

/// Undo GradX: K=qx, M=dy, N=dx; Gt on gX (d==0); accumulate both comps
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt2D(const int D1D, const int Q1D,
                                      const real_t (&sBG)[2][MQ1*MD1],
                                      const real_t (*sQD)[MDQ*MDQ],
                                      const DeviceTensor<3> &Y, const int e)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::GradXt2D<MD1, MQ1, MDQ>),
                      (&mfma::GradXt2D<MD1, MQ1, MDQ>),
                      (&blas::GradXt2D<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBG, sQD, Y, e);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpX2D(const int D1D, const int Q1D,
                                       const real_t *sB,
                                       const real_t *sDD, real_t *sDQ)
{
   InterpAx<MD1, MQ1>(D1D, Q1D, D1D, sB, sDD, sDQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpY2D(const int D1D, const int Q1D,
                                       const real_t *sB,
                                       const real_t *sDQ, real_t *sQQ)
{
   InterpAx<MD1, MQ1>(Q1D, Q1D, D1D, sB, sDQ, sQQ);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpYt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQQ, real_t *sQD)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::InterpYt2D<MD1, MQ1, MDQ>),
                      (&mfma::InterpYt2D<MD1, MQ1, MDQ>),
                      (&blas::InterpYt2D<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBt, sQQ, sQD);
}

template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void InterpXt2D(const int D1D, const int Q1D,
                                        const real_t *sBt,
                                        const real_t *sQD,
                                        const DeviceTensor<3> &Y, const int e)
{
   const auto fn = MMA_BACKEND_PICK(
                      (&dmma::InterpXt2D<MD1, MQ1, MDQ>),
                      (&mfma::InterpXt2D<MD1, MQ1, MDQ>),
                      (&blas::InterpXt2D<MD1, MQ1, MDQ>));
   fn(D1D, Q1D, sBt, sQD, Y, e);
}


} // namespace mfem::internal::mma

/// \endcond DO_NOT_DOCUMENT

