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

/** @file tensors.hpp
    Tensor-product form apply (integrator-agnostic).

    Contents:
      - PackPaMetric, ApplyGradQFnVec/Smem
      - Eval×Eval: host dense + device shell (TensorEvalApply / *Device; DIM-templated)
      - Grad×Grad: host multi-RHS tiles + device shell (TensorGradApply / *Device)
      - form::ApplyTensor<QFn, …> (SFINAE; Grad / H(curl) mass+curl / H(div) mass+div)
      - H(curl)/H(div) smem InterpAx/Grad* shells
      - 2D/3D: outer entries unified; Tile/Element/Ws stay dim-specific

    Physics QFns live under fem/integ/ only.
*/

#include "../mma.hpp"
#include "fields.hpp"
#include "../../../../general/array.hpp"
#include "../../../../linalg/vector.hpp"
#include "../../../../linalg/tensor.hpp"
#include <algorithm>
#include <type_traits>

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

using mfem::future::tensor;

// ---------------------------------------------------------------------------
// PA metric pack + Grad QFn helpers
// ---------------------------------------------------------------------------

using mfem::future::tensor;

template <int DIM, bool SYM>
MFEM_HOST_DEVICE inline void PackPaMetric(tensor<real_t, DIM, DIM> &A,
                                          const real_t *O)
{
   if constexpr (DIM == 2)
   {
      const real_t O11 = O[0], O21 = O[1];
      if constexpr (SYM)
      {
         const real_t O22 = O[2];
         A(0, 0) = O11; A(0, 1) = O21;
         A(1, 0) = O21; A(1, 1) = O22;
      }
      else
      {
         const real_t O12 = O[2], O22 = O[3];
         A(0, 0) = O11; A(0, 1) = O12;
         A(1, 0) = O21; A(1, 1) = O22;
      }
   }
   else
   {
      const real_t O11 = O[0], O12 = O[1], O13 = O[2];
      if constexpr (SYM)
      {
         const real_t O22 = O[3], O23 = O[4], O33 = O[5];
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O12; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O13; A(2, 1) = O23; A(2, 2) = O33;
      }
      else
      {
         const real_t O21 = O[3], O22 = O[4], O23 = O[5];
         const real_t O31 = O[6], O32 = O[7], O33 = O[8];
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O21; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O31; A(2, 1) = O32; A(2, 2) = O33;
      }
   }
}

/** Apply Grad×Grad QFn at one qp: g[] in/out, O packed PA.
    DIM / SYM come from qfn_traits<QFn> (not extra template args). */
template <typename QFn>
MFEM_HOST_DEVICE inline void ApplyGradQFnVec(QFn qfn, real_t *g,
                                             const real_t *O)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::trial_is_grad &&
                 Tr::has_trial, "ApplyGradQFnVec needs Grad×Grad QFn");
   constexpr int DIM = Tr::spatial_dim;
   constexpr bool SYM = Tr::symmetric_pa;
   grad_t<DIM> u, y;
   for (int c = 0; c < DIM; ++c) { u[c] = g[c]; }
   tensor<real_t, DIM, DIM> A{};
   PackPaMetric<DIM, SYM>(A, O);
   InvokeQFn(qfn, u, y, A);
   for (int c = 0; c < DIM; ++c) { g[c] = y[c]; }
}

/** Device smem: planes g_in/g_out[c * plane_ld + q], D(q,c,e). */
template <typename QFn, typename TD>
MFEM_HOST_DEVICE inline
void ApplyGradQFnSmem(QFn qfn, real_t *g_in, real_t *g_out, const int plane_ld,
                      TD D, const int e, const int Q1D,
                      const int tid, const int stride)
{
   using Tr = qfn_traits<QFn>;
   constexpr int DIM = Tr::spatial_dim;
   constexpr bool SYM = Tr::symmetric_pa;
   constexpr int PA = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int nq = (DIM == 2) ? Q1D * Q1D : Q1D * Q1D * Q1D;
   for (int q = tid; q < nq; q += stride)
   {
      real_t g[DIM];
      for (int c = 0; c < DIM; ++c) { g[c] = g_in[c * plane_ld + q]; }
      real_t O[PA];
      for (int c = 0; c < PA; ++c) { O[c] = D(q, c, e); }
      ApplyGradQFnVec(qfn, g, O);
      for (int c = 0; c < DIM; ++c) { g_out[c * plane_ld + q] = g[c]; }
   }
}

// ApplyEvalQFn is in fields.hpp (trait-driven arity).

} // namespace mfem::internal::mma::form

namespace mfem::internal
{

// ---------------------------------------------------------------------------
// Eval×Eval sum-fact host + device shells
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Host: PreferTensorDense hand sum-fact (B-only + scalar QFn)
// ---------------------------------------------------------------------------
namespace mma::blas
{

/** Dense sum-fact host mass 2D (serial over elements in outer tiles). */
template <typename QFn, int D1D, int Q1D>
inline void TensorEvalHost2D(const int NE, const real_t *B,
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
            mma::form::ApplyEvalQFn<QFn>(sol_xy[qy][qx],
                                         Dv[qx + Q1D * (qy + Q1D * e)]);
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
   const int NB = mma::TensorTileNB(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   }
}

/** Dense sum-fact host mass 3D (serial over elements in outer tiles). */
template <typename QFn, int D1D, int Q1D>
inline void TensorEvalHost3D(const int NE, const real_t *B,
                             const real_t *Dv, const real_t *X,
                             real_t *Y)
{
   auto apply_e = [&](int e)
   {
      real_t sol_xyz[Q1D][Q1D][Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] = real_t(0);
            }

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t sol_xy[Q1D][Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = real_t(0);
            }
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
               {
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
               }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mma::form::ApplyEvalQFn<QFn>(
                  sol_xyz[qz][qy][qx],
                  Dv[qx + Q1D * (qy + Q1D * (qz + Q1D * e))]);
            }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t sol_xy[D1D][D1D];
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = real_t(0);
            }
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
   const int NB = mma::TensorTileNB3D(D1D);
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   }
}

/** PreferTensorDense → hand sum-fact (2D/3D).
    Runtime Fallback (D1D=Q1D=0) skips host path — avoids zero-size VLA. */
template <typename QFn, int DIM, int D1D, int Q1D>
inline bool TryTensorEvalHost(const int NE,
                              const Array<real_t> &b,
                              const Vector &d,
                              const Vector &x,
                              Vector &y)
{
   if constexpr (D1D == 0 || Q1D == 0)
   {
      return false;
   }
   else
   {
      if (!mma::PreferTensorDense(D1D, NE)) { return false; }
      if constexpr (DIM == 3)
      {
         TensorEvalHost3D<QFn, D1D, Q1D>(NE, b.Read(), d.Read(),
                                         x.Read(), y.ReadWrite());
      }
      else
      {
         TensorEvalHost2D<QFn, D1D, Q1D>(NE, b.Read(), d.Read(),
                                         x.Read(), y.ReadWrite());
      }
      return true;
   }
}

} // namespace mma::blas

// ---------------------------------------------------------------------------
// Device (or host Emulate) smem shells — sum-fact Interp* + QFn
// Host entry uses Try* first (see TensorEvalApply).
// ---------------------------------------------------------------------------

/** Named forall bodies — avoid nvcc extended-lambda host stubs that can be
    null when the same specialization is instantiated in multiple TUs
    (scalar Mass + VectorMass).
    MQ=true: matrix coeff (ucomp smem). Scalar/VQ Q use MQ=false (coeff_vdim). */
template <typename QFn, int MD1, int MQ1, int MDQ, bool MQ>
struct TensorEvalKernel2D
{
   int NE, D1D, Q1D, NB, vdim, coeff_vdim;
   DeviceTensor<2, const real_t> B;
   DeviceTensor<3, const real_t> D; // (nq, coeff_vdim, NE)
   DeviceTensor<4, const real_t> X;
   DeviceTensor<4, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];
      MFEM_SHARED real_t ucomp[MQ ? (3 * MDQ * MDQ) : 1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      if constexpr (MQ)
      {
         for (int i = 0; i < NB; i++)
         {
            const int e = b * NB + i;
            if (e >= NE) { break; }

            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int n = D1D * D1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < n; t += stride)
                  {
                     const int dx = t % D1D;
                     const int dy = t / D1D;
                     sm0[dx + D1D * dy] = X(dx, dy, vc, e);
                  }
               }
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
                     ucomp[t + nq * vc] = sm0[t];
                  }
               }
               MFEM_SYNC_THREAD;
            }
            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int nq = Q1D * Q1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < nq; t += stride)
                  {
                     real_t s = 0.0;
                     for (int j = 0; j < vdim; ++j)
                     {
                        s += D(t, j + vdim * vc, e) * ucomp[t + nq * j];
                     }
                     sm1[t] = s;
                  }
               }
               MFEM_SYNC_THREAD;
               mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm1, sm0);
               MFEM_SYNC_THREAD;
               {
                  const int tid = mma::getThreadIdxX();
                  const int nthreads_x = mma::getBlockNthreadsX();
                  for (int idx = tid; idx < D1D * D1D; idx += nthreads_x)
                  {
                     const int dy = idx / D1D;
                     const int dx = idx - dy * D1D;
                     real_t s = 0.0;
                     for (int q = 0; q < Q1D; ++q)
                     {
                        s += sm0[q + Q1D * dy] * sBt[q + Q1D * dx];
                     }
                     Y(dx, dy, vc, e) += s;
                  }
               }
               MFEM_SYNC_THREAD;
            }
         }
      }
      else
      {
         const bool vector_coeff = (coeff_vdim == vdim);

         for (int i = 0; i < NB; i++)
         {
            const int e = b * NB + i;
            if (e >= NE) { break; }

            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int n = D1D * D1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < n; t += stride)
                  {
                     const int dx = t % D1D;
                     const int dy = t / D1D;
                     sm0[dx + D1D * dy] = X(dx, dy, vc, e);
                  }
               }
               MFEM_SYNC_THREAD;

               mma::InterpX2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::InterpY2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
               MFEM_SYNC_THREAD;

               {
                  const int tid = mma::getThreadIdxX();
                  const int nq = Q1D * Q1D;
                  const int stride = mma::getBlockNthreadsX();
                  const int dc = vector_coeff ? vc : 0;
                  for (int t = tid; t < nq; t += stride)
                  {
                     real_t u = sm0[t];
                     mma::form::ApplyEvalQFn<QFn>(u, D(t, dc, e));
                     sm1[t] = u;
                  }
               }
               MFEM_SYNC_THREAD;

               mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm1, sm0);
               MFEM_SYNC_THREAD;
               {
                  const int tid = mma::getThreadIdxX();
                  const int nthreads_x = mma::getBlockNthreadsX();
                  for (int idx = tid; idx < D1D * D1D; idx += nthreads_x)
                  {
                     const int dy = idx / D1D;
                     const int dx = idx - dy * D1D;
                     real_t s = 0.0;
                     for (int q = 0; q < Q1D; ++q)
                     {
                        s += sm0[q + Q1D * dy] * sBt[q + Q1D * dx];
                     }
                     Y(dx, dy, vc, e) += s;
                  }
               }
               MFEM_SYNC_THREAD;
            }
         }
      }
   }
};

/** True scalar Eval 3D (vdim==1): pre-vector layout via LoadX / InterpXt. */
template <typename QFn, int MD1, int MQ1>
struct TensorEvalKernel3DScalar
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> B;
   DeviceTensor<2, const real_t> D;
   DeviceTensor<4, const real_t> X;
   DeviceTensor<4, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
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
         mma::InterpAx<MD1, MQ1, false>(Q1D * Q1D, Q1D, D1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         {
            const int tid = mma::getThreadIdxX();
            const int nq = Q1D * Q1D * Q1D;
            const int stride = mma::getBlockNthreadsX();
            for (int t = tid; t < nq; t += stride)
            {
               mma::form::ApplyEvalQFn<QFn>(sm1[t], D(t, e));
            }
         }
         MFEM_SYNC_THREAD;
         mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y, e);
         MFEM_SYNC_THREAD;
      }
   }
};

template <typename QFn, int MD1, int MQ1, bool MQ>
struct TensorEvalKernel3D
{
   int NE, D1D, Q1D, NB, vdim, coeff_vdim;
   DeviceTensor<2, const real_t> B;
   DeviceTensor<3, const real_t> D;
   DeviceTensor<5, const real_t> X;
   DeviceTensor<4, real_t> Y4;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];
      MFEM_SHARED real_t ucomp[MQ ? (3 * MQ1 * MQ1 * MQ1) : 1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      if constexpr (MQ)
      {
         for (int i = 0; i < NB; i++)
         {
            const int e = b * NB + i;
            if (e >= NE) { break; }

            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int DDD = D1D * D1D * D1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < DDD; t += stride)
                  {
                     const int dx = t % D1D;
                     const int div = t / D1D;
                     const int dy = div % D1D;
                     const int dz = div / D1D;
                     sm0[t] = X(dx, dy, dz, vc, e);
                  }
               }
               MFEM_SYNC_THREAD;
               mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
               MFEM_SYNC_THREAD;
               mma::InterpAx<MD1, MQ1, false>(Q1D * Q1D, Q1D, D1D, sB, sm0, sm1);
               MFEM_SYNC_THREAD;
               {
                  const int tid = mma::getThreadIdxX();
                  const int nq = Q1D * Q1D * Q1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < nq; t += stride)
                  {
                     ucomp[t + nq * vc] = sm1[t];
                  }
               }
               MFEM_SYNC_THREAD;
            }
            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int nq = Q1D * Q1D * Q1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < nq; t += stride)
                  {
                     real_t s = 0.0;
                     for (int j = 0; j < vdim; ++j)
                     {
                        s += D(t, j + vdim * vc, e) * ucomp[t + nq * j];
                     }
                     sm1[t] = s;
                  }
               }
               MFEM_SYNC_THREAD;
               mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
               MFEM_SYNC_THREAD;
               mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y4, vc + vdim * e);
               MFEM_SYNC_THREAD;
            }
         }
      }
      else
      {
         const bool vector_coeff = (coeff_vdim == vdim);

         for (int i = 0; i < NB; i++)
         {
            const int e = b * NB + i;
            if (e >= NE) { break; }

            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int DDD = D1D * D1D * D1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < DDD; t += stride)
                  {
                     const int dx = t % D1D;
                     const int div = t / D1D;
                     const int dy = div % D1D;
                     const int dz = div / D1D;
                     sm0[t] = X(dx, dy, dz, vc, e);
                  }
               }
               MFEM_SYNC_THREAD;

               mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
               MFEM_SYNC_THREAD;
               mma::InterpAx<MD1, MQ1, false>(Q1D * Q1D, Q1D, D1D, sB, sm0, sm1);
               MFEM_SYNC_THREAD;
               {
                  const int tid = mma::getThreadIdxX();
                  const int nq = Q1D * Q1D * Q1D;
                  const int stride = mma::getBlockNthreadsX();
                  const int dc = vector_coeff ? vc : 0;
                  for (int t = tid; t < nq; t += stride)
                  {
                     mma::form::ApplyEvalQFn<QFn>(sm1[t], D(t, dc, e));
                  }
               }
               MFEM_SYNC_THREAD;
               mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
               MFEM_SYNC_THREAD;
               mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y4, vc + vdim * e);
               MFEM_SYNC_THREAD;
            }
         }
      }
   }
};

/** Device/Emulate Eval shell (2D or 3D).
    PA: (nq, coeff_vdim, NE) with coeff_vdim in {1, vdim, vdim²}. */
template <typename QFn, int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void TensorEvalApplyDevice(const int NE,
                                  const Array<real_t> &b,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0,
                                  const int vdim = 1)
{
   static_assert(DIM == 2 || DIM == 3, "TensorEvalApplyDevice: DIM 2 or 3");
   MFEM_VERIFY(vdim >= 1, "TensorEvalApplyDevice: vdim >= 1");
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   const int nq = (DIM == 2) ? (Q1D * Q1D) : (Q1D * Q1D * Q1D);
   MFEM_VERIFY(d.Size() % (nq * NE) == 0, "");
   const int coeff_vdim = d.Size() / (nq * NE);
   MFEM_VERIFY(coeff_vdim == 1 || coeff_vdim == vdim ||
               coeff_vdim == vdim * vdim, "");

   if constexpr (DIM == 2)
   {
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      dq.Verify(NE, "Tensor Eval MMA 2D D1D/Q1D exceeds shell cap");

      const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                     : mma::NB2DRuntime(D1D);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                              : mma::Threads2DRuntime(D1D, Q1D));

      const auto B = Reshape(b.Read(), Q1D, D1D);
      const auto X = Reshape(x.Read(), D1D, D1D, vdim, NE);
      auto Y = Reshape(y.ReadWrite(), D1D, D1D, vdim, NE);

      const int nblocks = (NE + NB - 1) / NB;
      // vdim==1: coeff_vdim==1 == vdim² is scalar Q, not MQ.
      const bool use_mq = (vdim > 1 && coeff_vdim == vdim * vdim);
      const auto D = Reshape(d.Read(), nq, coeff_vdim, NE);
      if (use_mq)
      {
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorEvalKernel2D<QFn, MD1, MQ1, MDQ, true>
         {NE, D1D, Q1D, NB, vdim, coeff_vdim, B, D, X, Y});
      }
      else
      {
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorEvalKernel2D<QFn, MD1, MQ1, MDQ, false>
         {NE, D1D, Q1D, NB, vdim, coeff_vdim, B, D, X, Y});
      }
   }
   else
   {
      dq.Verify(NE, "Tensor Eval MMA 3D D1D/Q1D exceeds shell cap");

      const int nthreads = mma::TensorShellNthreads(
                              T_D1D
                              ? mma::TensorThreads3D<T_D1D, T_Q1D,
                              mma::kTensorCostLight>()
                              : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                            mma::kTensorCostLight));

      const auto B = Reshape(b.Read(), Q1D, D1D);

      const bool use_mq = (vdim > 1 && coeff_vdim == vdim * vdim);
      if (vdim == 1 && coeff_vdim == 1)
      {
         const int NB = mma::TensorEvalNB3D(Q1D);
         const int nblocks = (NE + NB - 1) / NB;
         const auto D = Reshape(d.Read(), nq, NE);
         const auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
         auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorEvalKernel3DScalar<QFn, MD1, MQ1>
         {NE, D1D, Q1D, NB, B, D, X, Y});
      }
      else
      {
         const int NB = T_D1D
                        ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostLight>()
                        : mma::TensorNB3DRuntime(D1D, mma::kTensorCostLight);
         const int nblocks = (NE + NB - 1) / NB;
         const auto D = Reshape(d.Read(), nq, coeff_vdim, NE);
         const auto X = Reshape(x.Read(), D1D, D1D, D1D, vdim, NE);
         auto Y4 = Reshape(y.ReadWrite(), D1D, D1D, D1D, vdim * NE);
         if (use_mq)
         {
            mfem::forall_3D(nblocks, nthreads, 1, 1,
                            TensorEvalKernel3D<QFn, MD1, MQ1, true>
            {NE, D1D, Q1D, NB, vdim, coeff_vdim, B, D, X, Y4});
         }
         else
         {
            mfem::forall_3D(nblocks, nthreads, 1, 1,
                            TensorEvalKernel3D<QFn, MD1, MQ1, false>
            {NE, D1D, Q1D, NB, vdim, coeff_vdim, B, D, X, Y4});
         }
      }
   }
}

/** Entry: host PreferTensorDense sum-fact (vdim==1), else device/Emulate shell. */
template <typename QFn, int DIM, int T_D1D, int T_Q1D>
inline void TensorEvalApply(
   const int NE,
   const Array<real_t> &b,
   [[maybe_unused]] const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d,
   const int vdim = 1)
{
   if (vdim == 1 && !Device::Allows(Backend::DEVICE_MASK))
   {
      if (mma::blas::TryTensorEvalHost<QFn, DIM, T_D1D, T_Q1D>(NE, b, d, x, y))
      { return; }
   }
   TensorEvalApplyDevice<QFn, DIM, T_D1D, T_Q1D>(
      NE, b, d, x, y, d1d, q1d, vdim);
}

// ---------------------------------------------------------------------------
// Grad×Grad sum-fact host + device shells
// ---------------------------------------------------------------------------

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
            mma::form::ApplyGradQFnVec(QFn{}, gv, O);
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
               mma::form::ApplyGradQFnVec(QFn{}, gv, O);
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

/** Host multi-RHS Grad: tile over elements; dim-specific Diff*Ws + *Tile. */
template <typename QFn, int DIM, int D1D, int Q1D, bool SYM>
inline void TensorGradHost(
   const int NE,
   const real_t *B, const real_t *G, const real_t *Bt, const real_t *Gt,
   const real_t *Dv, const real_t *X, real_t *Y)
{
   static_assert(DIM == 2 || DIM == 3, "TensorGradHost: DIM 2 or 3");
   const int NB = (DIM == 2) ? mma::TensorTileNB(D1D, Q1D)
                  : mma::TensorTileNB3D(D1D);
   const int ntiles = (NE + NB - 1) / NB;
   mma::host_Arena arena;
   if constexpr (DIM == 2)
   {
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
   else
   {
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
}

/** PreferTensorDense → multi-RHS host Grad (2D/3D). SYM from qfn_traits.
    Runtime Fallback (D1D=Q1D=0) skips host path — avoids zero-size VLA. */
template <typename QFn, int DIM, int D1D, int Q1D>
inline bool TryTensorGradHost(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y)
{
   if constexpr (D1D == 0 || Q1D == 0)
   {
      return false;
   }
   else
   {
      using Tr = mma::form::qfn_traits<QFn>;
      constexpr bool SYM = Tr::symmetric_pa;
      if (!mma::PreferTensorDense(D1D, NE)) { return false; }
      const real_t *B = b.Read(), *G = g.Read(), *Bt = bt.Read(),
                    *Gt = gt.Read();
      const real_t *Dv = d.Read(), *X = x.Read();
      real_t *Y = y.ReadWrite();
      TensorGradHost<QFn, DIM, D1D, Q1D, SYM>(NE, B, G, Bt, Gt, Dv, X, Y);
      return true;
   }
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

   mma::form::ApplyGradQFnSmem(QFn{},
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

   mma::form::ApplyGradQFnSmem(QFn{},
                               sm0[0], sm1[0], plane_ld, D, e, Q1D,
                               mma::getThreadIdxX(), mma::getBlockNthreadsX());
   MFEM_SYNC_THREAD;

   mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
   MFEM_SYNC_THREAD;
   mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y, e);
   MFEM_SYNC_THREAD;
}

/** Named Grad forall bodies — same nvcc multi-TU stub rationale as Eval. */
template <typename QFn, int MD1, int MQ1, int MDQ, bool SYM>
struct TensorGradKernel2DScalar
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> B, G;
   DeviceTensor<3, const real_t> D;
   DeviceTensor<3, const real_t> Xs;
   DeviceTensor<3, real_t> Ys;

   MFEM_HOST_DEVICE void operator()(int b) const
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
            D1D, Q1D, e, BG, BGt, sm0, sm1, D, Xs, Ys);
      }
   }
};

template <typename QFn, int MD1, int MQ1, int MDQ>
struct TensorGradKernel2DVector
{
   int NE, D1D, Q1D, NB, vdim, ncomp;
   DeviceTensor<2, const real_t> B, G;
   DeviceTensor<5, const real_t> D;
   DeviceTensor<4, const real_t> X;
   DeviceTensor<3, real_t> Y3;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      MFEM_SHARED real_t sm0[2][MDQ * MDQ];
      MFEM_SHARED real_t sm1[2][MDQ * MDQ];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];
      MFEM_SHARED real_t gcomp[2][3 * MDQ * MDQ];
      MFEM_SHARED real_t gout[2][3 * MDQ * MDQ];

      mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, B, G, BG, BGt);
      MFEM_SYNC_THREAD;

      const bool matrix_coeff = (ncomp == vdim * 2);

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         if (matrix_coeff)
         {
            const int nq = Q1D * Q1D;
            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int n = D1D * D1D;
                  const int stride = mma::getBlockNthreadsX();
                  for (int t = tid; t < n; t += stride)
                  {
                     const int dx = t % D1D;
                     const int dy = t / D1D;
                     sm0[0][dx + D1D * dy] = X(dx, dy, vc, e);
                  }
               }
               MFEM_SYNC_THREAD;
               mma::GradX2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm0, sm1);
               MFEM_SYNC_THREAD;
               mma::GradY2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm1, sm0);
               MFEM_SYNC_THREAD;
               {
                  const int tid = mma::getThreadIdxX();
                  const int stride = mma::getBlockNthreadsX();
                  for (int q = tid; q < nq; q += stride)
                  {
                     gcomp[0][q + nq * vc] = sm0[0][q];
                     gcomp[1][q + nq * vc] = sm0[1][q];
                     gout[0][q + nq * vc] = 0.0;
                     gout[1][q + nq * vc] = 0.0;
                  }
               }
               MFEM_SYNC_THREAD;
            }
            for (int ii = 0; ii < vdim; ++ii)
            {
               for (int jj = 0; jj < vdim; ++jj)
               {
                  const int k = jj + ii * vdim;
                  {
                     const int tid = mma::getThreadIdxX();
                     const int stride = mma::getBlockNthreadsX();
                     for (int q = tid; q < nq; q += stride)
                     {
                        const int qx = q % Q1D;
                        const int qy = q / Q1D;
                        real_t gv[2] = {gcomp[0][q + nq * ii],
                                        gcomp[1][q + nq * ii]
                                       };
                        const real_t Osym[3] =
                        {
                           D(qx, qy, 0, k, e),
                           D(qx, qy, 2, k, e),
                           D(qx, qy, 3, k, e)
                        };
                        mma::form::ApplyGradQFnVec(QFn{}, gv, Osym);
                        gout[0][q + nq * jj] += gv[0];
                        gout[1][q + nq * jj] += gv[1];
                     }
                  }
                  MFEM_SYNC_THREAD;
               }
            }
            for (int vc = 0; vc < vdim; ++vc)
            {
               {
                  const int tid = mma::getThreadIdxX();
                  const int stride = mma::getBlockNthreadsX();
                  for (int q = tid; q < nq; q += stride)
                  {
                     sm1[0][q] = gout[0][q + nq * vc];
                     sm1[1][q] = gout[1][q + nq * vc];
                  }
               }
               MFEM_SYNC_THREAD;
               mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
               MFEM_SYNC_THREAD;
               mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y3,
                                            vc + vdim * e);
               MFEM_SYNC_THREAD;
            }
            continue;
         }

         for (int vc = 0; vc < vdim; ++vc)
         {
            {
               const int tid = mma::getThreadIdxX();
               const int n = D1D * D1D;
               const int stride = mma::getBlockNthreadsX();
               for (int t = tid; t < n; t += stride)
               {
                  const int dx = t % D1D;
                  const int dy = t / D1D;
                  sm0[0][dx + D1D * dy] = X(dx, dy, vc, e);
               }
            }
            MFEM_SYNC_THREAD;

            mma::GradX2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::GradY2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm1, sm0);
            MFEM_SYNC_THREAD;

            {
               const int tid = mma::getThreadIdxX();
               const int nq = Q1D * Q1D;
               const int stride = mma::getBlockNthreadsX();
               for (int q = tid; q < nq; q += stride)
               {
                  const int qx = q % Q1D;
                  const int qy = q / Q1D;
                  real_t gv[2] = {sm0[0][q], sm0[1][q]};
                  const real_t Osym[3] =
                  {
                     D(qx, qy, 0, vc, e),
                     D(qx, qy, 2, vc, e),
                     D(qx, qy, 3, vc, e)
                  };
                  mma::form::ApplyGradQFnVec(QFn{}, gv, Osym);
                  sm1[0][q] = gv[0];
                  sm1[1][q] = gv[1];
               }
            }
            MFEM_SYNC_THREAD;

            mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
            MFEM_SYNC_THREAD;
            mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y3,
                                         vc + vdim * e);
            MFEM_SYNC_THREAD;
         }
      }
   }
};

template <typename QFn, int MD1, int MQ1, bool SYM>
struct TensorGradKernel3DScalar
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> B, G;
   DeviceTensor<3, const real_t> D;
   DeviceTensor<4, const real_t> Xs;
   DeviceTensor<4, real_t> Ys;

   MFEM_HOST_DEVICE void operator()(int b) const
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
            D1D, Q1D, e, BG, BGt, sm0, sm1, D, Xs, Ys);
      }
   }
};

template <typename QFn, int MD1, int MQ1>
struct TensorGradKernel3DVector
{
   int NE, D1D, Q1D, NB, vdim, ncomp;
   DeviceTensor<2, const real_t> B, G;
   DeviceTensor<6, const real_t> D;
   DeviceTensor<5, const real_t> X;
   DeviceTensor<4, real_t> Y4;

   MFEM_HOST_DEVICE void operator()(int b) const
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

         // MQ (ncomp == vdim*3) is handled outside this smem kernel.
         for (int vc = 0; vc < vdim; ++vc)
         {
            {
               const int tid = mma::getThreadIdxX();
               const int DDD = D1D * D1D * D1D;
               const int stride = mma::getBlockNthreadsX();
               for (int t = tid; t < DDD; t += stride)
               {
                  const int dx = t % D1D;
                  const int div = t / D1D;
                  const int dy = div % D1D;
                  const int dz = div / D1D;
                  sm0[0][t] = X(dx, dy, dz, vc, e);
               }
            }
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
               for (int q = tid; q < nq; q += stride)
               {
                  const int qx = q % Q1D;
                  const int t1 = q / Q1D;
                  const int qy = t1 % Q1D;
                  const int qz = t1 / Q1D;
                  real_t gv[3] = {sm1[0][q], sm1[1][q], sm1[2][q]};
                  real_t O[6];
                  for (int c = 0; c < 6; ++c)
                  {
                     O[c] = D(qx, qy, qz, c, vc, e);
                  }
                  mma::form::ApplyGradQFnVec(QFn{}, gv, O);
                  sm0[0][q] = gv[0];
                  sm0[1][q] = gv[1];
                  sm0[2][q] = gv[2];
               }
            }
            MFEM_SYNC_THREAD;

            mma::GradZt<MD1, MQ1>(D1D, Q1D, BGt, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::GradYt<MD1, MQ1>(D1D, Q1D, BGt, sm1, sm0);
            MFEM_SYNC_THREAD;
            mma::GradXt<MD1, MQ1>(D1D, Q1D, BGt, sm0, Y4, vc + vdim * e);
            MFEM_SYNC_THREAD;
         }
      }
   }
};

/** Device/Emulate Grad shell (2D or 3D). Element kernels stay dim-specific.
    vdim==1: scalar PA layout (nq, PA, NE) from DiffusionIntegrator.
    vdim>1: VectorDiffusion-style PA (nq, PA_full, vdim, NE), block-diagonal
    components. 2D vector PA is full 4-pack; packed to SYM (3) for QFn.
    3D vector PA is stock dim*dim=9 pack (SYM values in slots 0..5). */
template <typename QFn, int DIM, int T_D1D = 0, int T_Q1D = 0, bool SYM = true>
inline void TensorGradApplyDevice(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0,
                                  const int vdim = 1)
{
   static_assert(DIM == 2 || DIM == 3, "TensorGradApplyDevice: DIM 2 or 3");
   MFEM_VERIFY(vdim >= 1, "TensorGradApplyDevice: vdim >= 1");
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;

   if constexpr (DIM == 2)
   {
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      // Scalar: SYM 3 or full 4. VectorDiffusion stores full 4 per component.
      constexpr int PA_SCALAR = SYM ? 3 : 4;
      constexpr int PA_VEC = 4;
      dq.Verify(NE, "Tensor Grad MMA 2D D1D/Q1D exceeds shell cap");

      const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                     : mma::NB2DRuntime(D1D);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                              : mma::Threads2DRuntime(D1D, Q1D));

      const auto B = Reshape(b.Read(), Q1D, D1D);
      const auto G = Reshape(g.Read(), Q1D, D1D);

      if (vdim == 1)
      {
         const auto D = Reshape(d.Read(), Q1D * Q1D, PA_SCALAR, NE);
         const auto Xs = Reshape(x.Read(), D1D, D1D, NE);
         auto Ys = Reshape(y.ReadWrite(), D1D, D1D, NE);

         const int nblocks = (NE + NB - 1) / NB;
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorGradKernel2DScalar<QFn, MD1, MQ1, MDQ, SYM>
         {NE, D1D, Q1D, NB, B, G, D, Xs, Ys});
      }
      else
      {
         MFEM_VERIFY(d.Size() % (PA_VEC * Q1D * Q1D * NE) == 0, "");
         const int ncomp = d.Size() / (PA_VEC * Q1D * Q1D * NE);
         MFEM_VERIFY(ncomp == vdim || ncomp == vdim * DIM, "");
         // VectorDiffusion PA: (Q,Q, 4, ncomp, NE); X/Y: (D,D, vdim, NE)
         const auto D = Reshape(d.Read(), Q1D, Q1D, PA_VEC, ncomp, NE);
         const auto X = Reshape(x.Read(), D1D, D1D, vdim, NE);
         auto Y3 = Reshape(y.ReadWrite(), D1D, D1D, vdim * NE);

         const int nblocks = (NE + NB - 1) / NB;
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorGradKernel2DVector<QFn, MD1, MQ1, MDQ>
         {NE, D1D, Q1D, NB, vdim, ncomp, B, G, D, X, Y3});
      }
   }
   else
   {
      constexpr int PA_SCALAR = SYM ? 6 : 9;
      // VectorDiffusion AssemblePA uses pa_size=dim*dim (9); SYM in 0..5.
      constexpr int PA_VEC = 9;
      dq.Verify(NE, "Tensor Grad MMA 3D D1D/Q1D exceeds shell cap");

      const int NB = T_D1D
                     ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                     : mma::TensorNB3DRuntime(D1D, mma::kTensorCostHeavy);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D
                              ? mma::TensorThreads3D<T_D1D, T_Q1D,
                              mma::kTensorCostHeavy>()
                              : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                            mma::kTensorCostHeavy));

      const auto B = Reshape(b.Read(), Q1D, D1D);
      const auto G = Reshape(g.Read(), Q1D, D1D);

      if (vdim == 1)
      {
         MFEM_VERIFY(d.Size() == PA_SCALAR * Q1D * Q1D * Q1D * NE, "");
         const auto D = Reshape(d.Read(), Q1D * Q1D * Q1D, PA_SCALAR, NE);
         const auto Xs = Reshape(x.Read(), D1D, D1D, D1D, NE);
         auto Ys = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

         const int nblocks = (NE + NB - 1) / NB;
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorGradKernel3DScalar<QFn, MD1, MQ1, SYM>
         {NE, D1D, Q1D, NB, B, G, D, Xs, Ys});
      }
      else
      {
         MFEM_VERIFY(d.Size() % (PA_VEC * Q1D * Q1D * Q1D * NE) == 0, "");
         const int ncomp = d.Size() / (PA_VEC * Q1D * Q1D * Q1D * NE);
         // MQ (ncomp == vdim*DIM) is dispatched to stock apply in AddMultPA.
         MFEM_VERIFY(ncomp == vdim, "Tensor Grad MMA 3D vector expects VQ/Q");
         const auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, PA_VEC, ncomp, NE);
         const auto X = Reshape(x.Read(), D1D, D1D, D1D, vdim, NE);
         auto Y4 = Reshape(y.ReadWrite(), D1D, D1D, D1D, vdim * NE);

         const int nblocks = (NE + NB - 1) / NB;
         mfem::forall_3D(nblocks, nthreads, 1, 1,
                         TensorGradKernel3DVector<QFn, MD1, MQ1>
         {NE, D1D, Q1D, NB, vdim, ncomp, B, G, D, X, Y4});
      }
   }
}

/** Entry: host lapack multi-RHS when available (vdim==1), else device shell.
    SYM from qfn_traits; vdim>1 is block-diagonal VectorDiffusion layout. */
template <typename QFn, int DIM, int T_D1D, int T_Q1D>
inline void TensorGradApply(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d,
   const int vdim = 1)
{
   MFEM_CONTRACT_VAR(bt);
   MFEM_CONTRACT_VAR(gt);
#ifdef MFEM_USE_LAPACK
   if (vdim == 1 && !Device::Allows(Backend::DEVICE_MASK))
   {
      if (mma::lapack::TryTensorGradHost<QFn, DIM, T_D1D, T_Q1D>(
             NE, b, g, bt, gt, d, x, y))
      { return; }
   }
#endif
   constexpr bool SYM = mma::form::qfn_traits<QFn>::symmetric_pa;
   TensorGradApplyDevice<QFn, DIM, T_D1D, T_Q1D, SYM>(
      NE, b, g, d, x, y, d1d, q1d, vdim);
}

} // namespace mfem::internal

namespace mfem::internal::mma::form
{

// ---------------------------------------------------------------------------
// ApplyTensor — unified entry (Eval or Grad via qfn_traits)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Eval×Eval — B basis only (mass-like)
// ---------------------------------------------------------------------------

/** Tensor Eval apply. bt is accepted for API symmetry with registration; unused.
    vdim>1: block-diagonal multi-component (layout D… × vdim × NE). */
template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<
   !qfn_traits<QFn>::trial_is_grad &&
   !qfn_traits<QFn>::trial_is_curl &&
   !qfn_traits<QFn>::trial_is_div &&
   !qfn_traits<QFn>::trial_is_vec_eval, void>
ApplyTensor(const int NE,
            const Array<real_t> &b,
            const Array<real_t> &bt,
            const Vector &d,
            const Vector &x,
            Vector &y,
            const int d1d = 0,
            const int q1d = 0,
            const int vdim = 1)
{
   TensorEvalApply<QFn, DIM, D1D, Q1D>(NE, b, bt, d, x, y, d1d, q1d, vdim);
}

// ---------------------------------------------------------------------------
// Grad×Grad — B/G bases; SYM from qfn_traits<QFn>::symmetric_pa
// ---------------------------------------------------------------------------

/** Tensor Grad apply. Packed PA from QFn traits; vdim>1 = block-diag vector. */
template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
ApplyTensor(const int NE,
            const Array<real_t> &b,
            const Array<real_t> &g,
            const Array<real_t> &bt,
            const Array<real_t> &gt,
            const Vector &d,
            const Vector &x,
            Vector &y,
            const int d1d = 0,
            const int q1d = 0,
            const int vdim = 1)
{
   TensorGradApply<QFn, DIM, D1D, Q1D>(
      NE, b, g, bt, gt, d, x, y, d1d, q1d, vdim);
}

// H(curl)/H(div) tensor PA — smem InterpAx/Grad* engines + ApplyTensor
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// H(curl) / H(div) helpers (ND / RT component sizes + packed O)
// ---------------------------------------------------------------------------

/** ND / H(curl): open along the component axis. */
MFEM_HOST_DEVICE inline void CurlCompSizes2D(const int D1D, const int c,
                                             int &Dx, int &Dy)
{
   Dx = (c == 0) ? D1D - 1 : D1D;
   Dy = (c == 1) ? D1D - 1 : D1D;
}

MFEM_HOST_DEVICE inline void CurlCompSizes3D(const int D1D, const int c,
                                             int &Dx, int &Dy, int &Dz)
{
   Dx = (c == 0) ? D1D - 1 : D1D;
   Dy = (c == 1) ? D1D - 1 : D1D;
   Dz = (c == 2) ? D1D - 1 : D1D;
}

MFEM_HOST_DEVICE inline bool CurlAxisOpen(const int axis, const int c)
{
   return axis == c;
}

/** RT / H(div): closed along the component axis. */
MFEM_HOST_DEVICE inline void DivCompSizes2D(const int D1D, const int c,
                                            int &Dx, int &Dy)
{
   Dx = (c == 1) ? D1D - 1 : D1D;
   Dy = (c == 0) ? D1D - 1 : D1D;
}

MFEM_HOST_DEVICE inline void DivCompSizes3D(const int D1D, const int c,
                                            int &Dx, int &Dy, int &Dz)
{
   Dx = (c == 0) ? D1D : D1D - 1;
   Dy = (c == 1) ? D1D : D1D - 1;
   Dz = (c == 2) ? D1D : D1D - 1;
}

MFEM_HOST_DEVICE inline bool DivAxisOpen(const int axis, const int c)
{
   return axis != c;
}

template <bool SYM, bool OPEN>
MFEM_HOST_DEVICE inline void ApplyVecEvalO2D(const real_t u0, const real_t u1,
                                             const real_t *O,
                                             real_t &y0, real_t &y1)
{
   const real_t O11 = O[0];
   if constexpr (SYM)
   {
      const real_t Ooff = O[1], O22 = O[2];
      y0 = O11 * u0 + Ooff * u1;
      y1 = Ooff * u0 + O22 * u1;
   }
   else if constexpr (OPEN)
   {
      const real_t O21 = O[1], O12 = O[2], O22 = O[3];
      y0 = O11 * u0 + O12 * u1;
      y1 = O21 * u0 + O22 * u1;
   }
   else
   {
      const real_t O12 = O[1], O21 = O[2], O22 = O[3];
      y0 = O11 * u0 + O12 * u1;
      y1 = O21 * u0 + O22 * u1;
   }
}

template <bool SYM>
MFEM_HOST_DEVICE inline void ApplyVecEvalO3D(const real_t u[3],
                                             const real_t *O, real_t y[3])
{
   tensor<real_t, 3, 3> A{};
   PackPaMetric<3, SYM>(A, O);
   y[0] = A(0, 0) * u[0] + A(0, 1) * u[1] + A(0, 2) * u[2];
   y[1] = A(1, 0) * u[0] + A(1, 1) * u[1] + A(1, 2) * u[2];
   y[2] = A(2, 0) * u[0] + A(2, 1) * u[1] + A(2, 2) * u[2];
}

/** Y += sign * (Byt ⊗ Bxt) qq. qq is qx + Q*qy. */
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void CurlTranspAdd2D(
   const int Dx, const int Dy, const int Q1D,
   const real_t *Bxt, const real_t *Byt,
   const real_t *qq, real_t *sm0, real_t *sm1,
   const DeviceTensor<2> &Y, const int osc, const int e,
   const real_t sign, const int tid, const int stride)
{
   mma::GemmMbyK<false>(Q1D, Q1D, Dy, qq, Byt, sm0);
   MFEM_SYNC_THREAD;
   mma::InterpAx<MD1, MQ1>(Dx, Dy, Q1D, sm0, Bxt, sm1);
   MFEM_SYNC_THREAD;
   const int n = Dx * Dy;
   for (int t = tid; t < n; t += stride)
   {
      const int dx = t % Dx;
      const int dy = t / Dx;
      Y(dx + dy * Dx + osc, e) += sign * sm1[dx + Dx * dy];
   }
   MFEM_SYNC_THREAD;
}

/** Y += sign * (Bzt ⊗ Byt ⊗ Bxt) qqq. qqq is (qx+Q*qy)+Q*Q*qz. */
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void CurlTranspAdd3D(
   const int Dx, const int Dy, const int Dz, const int Q1D,
   const real_t *Bxt, const real_t *Byt, const real_t *Bzt,
   const real_t *qqq, real_t *sm0, real_t *sm1,
   const DeviceTensor<2> &Y, const int osc, const int e,
   const real_t sign, const int tid, const int stride)
{
   mma::GemmMbyK<false>(Q1D * Q1D, Q1D, Dz, qqq, Bzt, sm0);
   MFEM_SYNC_THREAD;
   for (int dz = 0; dz < Dz; ++dz)
   {
      mma::GemmMbyK<false>(Q1D, Q1D, Dy, sm0 + Q1D * Q1D * dz, Byt,
                           sm1 + Q1D * Dy * dz);
   }
   MFEM_SYNC_THREAD;
   mma::InterpAx<MD1, MQ1>(Dx, Dy * Dz, Q1D, sm1, Bxt, sm0);
   MFEM_SYNC_THREAD;
   const int n = Dx * Dy * Dz;
   for (int t = tid; t < n; t += stride)
   {
      const int dx = t % Dx;
      const int rest = t / Dx;
      const int dy = rest % Dy;
      const int dz = rest / Dy;
      Y(dx + (dy + dz * Dy) * Dx + osc, e) +=
         sign * sm0[dx + Dx * (dy + Dy * dz)];
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE inline void CurlAccum(real_t *dst, const real_t *src,
                                      const int n, const real_t sign,
                                      const int tid, const int stride)
{
   for (int t = tid; t < n; t += stride)
   {
      dst[t] += sign * src[t];
   }
}

template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void DivTranspAdd3D(
   const int Dx, const int Dy, const int Dz, const int Q1D,
   const real_t *Bxt, const real_t *Byt, const real_t *Bzt,
   const real_t *qqq, real_t *sm0, real_t *sm1,
   const DeviceTensor<2> &Y, const int osc, const int e,
   const real_t sign, const int tid, const int stride)
{
   CurlTranspAdd3D<MD1, MQ1>(Dx, Dy, Dz, Q1D, Bxt, Byt, Bzt, qqq, sm0, sm1,
                             Y, osc, e, sign, tid, stride);
}

MFEM_HOST_DEVICE inline void DivAccum(real_t *dst, const real_t *src,
                                      const int n, const real_t sign,
                                      const int tid, const int stride)
{
   CurlAccum(dst, src, n, sign, tid, stride);
}


// ---------------------------------------------------------------------------
// Vector FE mass — Eval×Eval with Bo/Bc (Hcurl OPEN=true, Hdiv OPEN=false)
// ---------------------------------------------------------------------------

template <int MD1, int MQ1, int MDQ, bool OPEN, bool SYM>
struct TensorVecEvalKernel2D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Bc;
   DeviceTensor<3, const real_t> D; // (nq, ncomp, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      constexpr int VDIM = 2;
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t sBc[MD1 * MQ1];
      MFEM_SHARED real_t sBct[MD1 * MQ1];
      MFEM_SHARED real_t ucomp[VDIM * MDQ * MDQ];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, Bc, sBc, sBct);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D;
      const int osc_c = (D1D - 1) * D1D;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy;
            if constexpr (OPEN) { CurlCompSizes2D(D1D, c, Dx, Dy); }
            else { DivCompSizes2D(D1D, c, Dx, Dy); }
            const int n = Dx * Dy;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int dy = t / Dx;
               sm0[dx + Dx * dy] = X(dx + dy * Dx + c * osc_c, e);
            }
            MFEM_SYNC_THREAD;

            const real_t *Bx = (OPEN ? CurlAxisOpen(0, c) : DivAxisOpen(0, c)) ? sBo : sBc;
            const real_t *By = (OPEN ? CurlAxisOpen(1, c) : DivAxisOpen(1, c)) ? sBo : sBc;
            mma::InterpAx<MD1, MQ1>(Dy, Q1D, Dx, Bx, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D, Q1D, Dy, By, sm1, sm0);
            MFEM_SYNC_THREAD;

            for (int t = tid; t < nq; t += stride)
            {
               ucomp[t + nq * c] = sm0[t];
            }
            MFEM_SYNC_THREAD;
         }

         constexpr int PA = SYM ? 3 : 4;
         for (int t = tid; t < nq; t += stride)
         {
            real_t O[PA];
            for (int k = 0; k < PA; ++k) { O[k] = D(t, k, e); }
            real_t y0, y1;
            ApplyVecEvalO2D<SYM, OPEN>(ucomp[t], ucomp[t + nq], O, y0, y1);
            ucomp[t] = y0;
            ucomp[t + nq] = y1;
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy;
            if constexpr (OPEN) { CurlCompSizes2D(D1D, c, Dx, Dy); }
            else { DivCompSizes2D(D1D, c, Dx, Dy); }
            const real_t *Byt = (OPEN ? CurlAxisOpen(1, c) : DivAxisOpen(1, c)) ? sBot : sBct;
            const real_t *Bxt = (OPEN ? CurlAxisOpen(0, c) : DivAxisOpen(0, c)) ? sBot : sBct;
            CurlTranspAdd2D<MD1, MQ1>(Dx, Dy, Q1D, Bxt, Byt, ucomp + nq * c,
                                      sm0, sm1, Y, c * osc_c, e, 1.0,
                                      tid, stride);
         }
      }
   }
};

template <int MD1, int MQ1, bool OPEN, bool SYM>
struct TensorVecEvalKernel3D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Bc;
   DeviceTensor<3, const real_t> D; // (nq, ncomp, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      constexpr int VDIM = 3;
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t sBc[MD1 * MQ1];
      MFEM_SHARED real_t sBct[MD1 * MQ1];
      MFEM_SHARED real_t ucomp[VDIM * MQ1 * MQ1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, Bc, sBc, sBct);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D * Q1D;
      const int osc_c = OPEN ? ((D1D - 1) * D1D * D1D)
                        : ((D1D - 1) * (D1D - 1) * D1D);

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy, Dz;
            if constexpr (OPEN) { CurlCompSizes3D(D1D, c, Dx, Dy, Dz); }
            else { DivCompSizes3D(D1D, c, Dx, Dy, Dz); }
            const int n = Dx * Dy * Dz;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int rest = t / Dx;
               const int dy = rest % Dy;
               const int dz = rest / Dy;
               sm0[dx + Dx * (dy + Dy * dz)] =
                  X(dx + (dy + dz * Dy) * Dx + c * osc_c, e);
            }
            MFEM_SYNC_THREAD;

            const real_t *Bx = (OPEN ? CurlAxisOpen(0, c) : DivAxisOpen(0, c)) ? sBo : sBc;
            const real_t *By = (OPEN ? CurlAxisOpen(1, c) : DivAxisOpen(1, c)) ? sBo : sBc;
            const real_t *Bz = (OPEN ? CurlAxisOpen(2, c) : DivAxisOpen(2, c)) ? sBo : sBc;
            mma::InterpAx<MD1, MQ1>(Dy * Dz, Q1D, Dx, Bx, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, By, sm1, sm0);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, Bz, sm0, sm1);
            MFEM_SYNC_THREAD;

            for (int t = tid; t < nq; t += stride)
            {
               ucomp[t + nq * c] = sm1[t];
            }
            MFEM_SYNC_THREAD;
         }

         constexpr int PA = SYM ? 6 : 9;
         for (int t = tid; t < nq; t += stride)
         {
            real_t O[PA];
            for (int k = 0; k < PA; ++k) { O[k] = D(t, k, e); }
            const real_t u[3] = {ucomp[t], ucomp[t + nq], ucomp[t + 2 * nq]};
            real_t yv[3];
            ApplyVecEvalO3D<SYM>(u, O, yv);
            ucomp[t] = yv[0];
            ucomp[t + nq] = yv[1];
            ucomp[t + 2 * nq] = yv[2];
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy, Dz;
            if constexpr (OPEN) { CurlCompSizes3D(D1D, c, Dx, Dy, Dz); }
            else { DivCompSizes3D(D1D, c, Dx, Dy, Dz); }
            const real_t *Bzt = (OPEN ? CurlAxisOpen(2, c) : DivAxisOpen(2, c)) ? sBot : sBct;
            const real_t *Byt = (OPEN ? CurlAxisOpen(1, c) : DivAxisOpen(1, c)) ? sBot : sBct;
            const real_t *Bxt = (OPEN ? CurlAxisOpen(0, c) : DivAxisOpen(0, c)) ? sBot : sBct;
            CurlTranspAdd3D<MD1, MQ1>(Dx, Dy, Dz, Q1D, Bxt, Byt, Bzt,
                                      ucomp + nq * c, sm0, sm1, Y,
                                      c * osc_c, e, 1.0, tid, stride);
         }
      }
   }
};

/** Device/Emulate vector-mass shell (ND/RT). bot/bct unused: LoadBBoth. */
template <typename QFn, int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void TensorVecEvalApplyDevice(const int NE,
                                     const Array<real_t> &bo,
                                     const Array<real_t> &bc,
                                     const Array<real_t> &,
                                     const Array<real_t> &,
                                     const Vector &d,
                                     const Vector &x,
                                     Vector &y,
                                     const int d1d = 0,
                                     const int q1d = 0)
{
   using Tr = qfn_traits<QFn>;
   static_assert(DIM == 2 || DIM == 3, "TensorVecEvalApplyDevice: DIM 2 or 3");
   static_assert(Tr::trial_is_vec_eval, "");
   constexpr bool OPEN = Tr::open_on_component;
   constexpr bool SYM = Tr::symmetric_pa;
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   dq.Verify(NE, "Tensor H(curl)/H(div) mass MMA D1D/Q1D exceeds shell cap");

   const int nq = (DIM == 2) ? (Q1D * Q1D) : (Q1D * Q1D * Q1D);
   const int ncomp = SYM ? ((DIM == 2) ? 3 : 6) : ((DIM == 2) ? 4 : 9);
   MFEM_VERIFY(d.Size() == nq * ncomp * NE, "");

   const int ndof = (DIM == 2)
                    ? (2 * (D1D - 1) * D1D)
                    : (OPEN ? (3 * (D1D - 1) * D1D * D1D)
                       : (3 * (D1D - 1) * (D1D - 1) * D1D));
   MFEM_VERIFY(x.Size() == ndof * NE && y.Size() == ndof * NE, "");

   const auto Bo = Reshape(bo.Read(), Q1D, D1D - 1);
   const auto Bc = Reshape(bc.Read(), Q1D, D1D);
   const auto Dv = Reshape(d.Read(), nq, ncomp, NE);
   const auto X = Reshape(x.Read(), ndof, NE);
   auto Y = Reshape(y.ReadWrite(), ndof, NE);

   if constexpr (DIM == 2)
   {
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                     : mma::NB2DRuntime(D1D);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                              : mma::Threads2DRuntime(D1D, Q1D));
      const int nblocks = (NE + NB - 1) / NB;
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorVecEvalKernel2D<MD1, MQ1, MDQ, OPEN, SYM>
      {NE, D1D, Q1D, NB, Bo, Bc, Dv, X, Y});
   }
   else
   {
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D
                              ? mma::TensorThreads3D<T_D1D, T_Q1D,
                              mma::kTensorCostHeavy>()
                              : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                            mma::kTensorCostHeavy));
      const int NB = T_D1D
                     ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                     : mma::TensorNB3DRuntime(D1D, mma::kTensorCostHeavy);
      const int nblocks = (NE + NB - 1) / NB;
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorVecEvalKernel3D<MD1, MQ1, OPEN, SYM>
      {NE, D1D, Q1D, NB, Bo, Bc, Dv, X, Y});
   }
}

// ---------------------------------------------------------------------------
// Curl×Curl — Bo / Bc / Gc (2D: Bo+Gc only)
// ---------------------------------------------------------------------------

template <int MD1, int MQ1, int MDQ>
struct TensorCurlCurlKernel2D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Gc;
   DeviceTensor<2, const real_t> D; // (nq, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t sGc[MD1 * MQ1];
      MFEM_SHARED real_t sGct[MD1 * MQ1];
      MFEM_SHARED real_t curl[MDQ * MDQ];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, Gc, sGc, sGct);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D;
      const int osc_c = (D1D - 1) * D1D;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int t = tid; t < nq; t += stride) { curl[t] = 0.0; }
         MFEM_SYNC_THREAD;

         // ux: Bo_x, Gc_y → curl -=
         {
            const int Dx = D1D - 1, Dy = D1D;
            const int n = Dx * Dy;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int dy = t / Dx;
               sm0[dx + Dx * dy] = X(dx + dy * Dx, e);
            }
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dy, Q1D, Dx, sBo, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D, Q1D, Dy, sGc, sm1, sm0);
            MFEM_SYNC_THREAD;
            CurlAccum(curl, sm0, nq, -1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }
         // uy: Gc_x, Bo_y → curl +=
         {
            const int Dx = D1D, Dy = D1D - 1;
            const int n = Dx * Dy;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int dy = t / Dx;
               sm0[dx + Dx * dy] = X(dx + dy * Dx + osc_c, e);
            }
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dy, Q1D, Dx, sGc, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D, Q1D, Dy, sBo, sm1, sm0);
            MFEM_SYNC_THREAD;
            CurlAccum(curl, sm0, nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         for (int t = tid; t < nq; t += stride)
         {
            curl[t] *= D(t, e);
         }
         MFEM_SYNC_THREAD;

         // ux adjoint: Yt Gct, Xt Bot, minus
         CurlTranspAdd2D<MD1, MQ1>(D1D - 1, D1D, Q1D, sBot, sGct, curl,
                                   sm0, sm1, Y, 0, e, -1.0, tid, stride);
         // uy adjoint: Yt Bot, Xt Gct
         CurlTranspAdd2D<MD1, MQ1>(D1D, D1D - 1, Q1D, sGct, sBot, curl,
                                   sm0, sm1, Y, osc_c, e, 1.0, tid, stride);
      }
   }
};

template <int MD1, int MQ1, bool SYM>
struct TensorCurlCurlKernel3D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Bc, Gc;
   DeviceTensor<3, const real_t> D; // (nq, ncomp, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      constexpr int BUF = MQ1 * MQ1 * MQ1;
      MFEM_SHARED real_t smA[2][BUF];
      MFEM_SHARED real_t smC[2][BUF];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];
      MFEM_SHARED real_t curl[3][BUF];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, Bc, Gc, BG, BGt);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D * Q1D;
      const int osc_c = (D1D - 1) * D1D * D1D;
      const real_t *sBc = BG[0];
      const real_t *sGc = BG[1];
      const real_t *sBct = BGt[0];
      const real_t *sGct = BGt[1];

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int t = tid; t < nq; t += stride)
         {
            curl[0][t] = curl[1][t] = curl[2][t] = 0.0;
         }
         MFEM_SYNC_THREAD;

         // ---- u0 (open x): Bo_x, (Gc,Bc)_y, then curl_y += Bc_y*Gc_z,
         //      curl_z -= Gc_y*Bc_z
         {
            const int Dx = D1D - 1, Dy = D1D, Dz = D1D;
            const int n = Dx * Dy * Dz;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int rest = t / Dx;
               const int dy = rest % Dy;
               const int dz = rest / Dy;
               smC[0][dx + Dx * (dy + Dy * dz)] =
                  X(dx + (dy + dz * Dy) * Dx, e);
            }
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dy * Dz, Q1D, Dx, sBo, smC[0], smA[0]);
            MFEM_SYNC_THREAD;
            mma::GradX<MD1, MQ1, BUF>(Dz * Q1D, Q1D, Dy, BG, smA, smC);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sGc, smC[1], smA[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[1], smA[0], nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sBc, smC[0], smA[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[2], smA[0], nq, -1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         // ---- u1 (open y): (Gc,Bc)_x, Bo_y, curl_x -= Bc_x*Gc_z,
         //      curl_z += Gc_x*Bc_z
         {
            const int Dx = D1D, Dy = D1D - 1, Dz = D1D;
            const int n = Dx * Dy * Dz;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int rest = t / Dx;
               const int dy = rest % Dy;
               const int dz = rest / Dy;
               smA[0][dx + Dx * (dy + Dy * dz)] =
                  X(dx + (dy + dz * Dy) * Dx + osc_c, e);
            }
            MFEM_SYNC_THREAD;
            mma::GradX<MD1, MQ1, BUF>(Dy * Dz, Q1D, Dx, BG, smA, smC);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, sBo, smC[0], smA[0]);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, sBo, smC[1], smA[1]);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sGc, smA[1], smC[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[0], smC[0], nq, -1.0, tid, stride);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sBc, smA[0], smC[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[2], smC[0], nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         // ---- u2 (open z): (Gc,Bc)_x, then Gc_y on Bc_x / Bc_y on Gc_x,
         //      Bo_z; curl_x += Gc_y*Bc_x, curl_y -= Bc_y*Gc_x
         {
            const int Dx = D1D, Dy = D1D, Dz = D1D - 1;
            const int n = Dx * Dy * Dz;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int rest = t / Dx;
               const int dy = rest % Dy;
               const int dz = rest / Dy;
               smA[0][dx + Dx * (dy + Dy * dz)] =
                  X(dx + (dy + dz * Dy) * Dx + 2 * osc_c, e);
            }
            MFEM_SYNC_THREAD;
            mma::GradX<MD1, MQ1, BUF>(Dy * Dz, Q1D, Dx, BG, smA, smC);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, sGc, smC[1], smA[0]);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, sBc, smC[0], smA[1]);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sBo, smA[0], smC[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[0], smC[0], nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, sBo, smA[1], smC[0]);
            MFEM_SYNC_THREAD;
            CurlAccum(curl[1], smC[0], nq, -1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         constexpr int PA = SYM ? 6 : 9;
         for (int t = tid; t < nq; t += stride)
         {
            real_t O[PA];
            for (int k = 0; k < PA; ++k) { O[k] = D(t, k, e); }
            const real_t u[3] = {curl[0][t], curl[1][t], curl[2][t]};
            real_t yv[3];
            ApplyVecEvalO3D<SYM>(u, O, yv);
            curl[0][t] = yv[0];
            curl[1][t] = yv[1];
            curl[2][t] = yv[2];
         }
         MFEM_SYNC_THREAD;

         // u0 adjoint: curl_y → (Bot, Bct, Gct); curl_z → -(Bot, Gct, Bct)
         CurlTranspAdd3D<MD1, MQ1>(D1D - 1, D1D, D1D, Q1D, sBot, sBct, sGct,
                                   curl[1], smA[0], smC[0], Y, 0, e, 1.0,
                                   tid, stride);
         CurlTranspAdd3D<MD1, MQ1>(D1D - 1, D1D, D1D, Q1D, sBot, sGct, sBct,
                                   curl[2], smA[0], smC[0], Y, 0, e, -1.0,
                                   tid, stride);
         // u1: curl_x → -(Bct, Bot, Gct); curl_z → (Gct, Bot, Bct)
         CurlTranspAdd3D<MD1, MQ1>(D1D, D1D - 1, D1D, Q1D, sBct, sBot, sGct,
                                   curl[0], smA[0], smC[0], Y, osc_c, e, -1.0,
                                   tid, stride);
         CurlTranspAdd3D<MD1, MQ1>(D1D, D1D - 1, D1D, Q1D, sGct, sBot, sBct,
                                   curl[2], smA[0], smC[0], Y, osc_c, e, 1.0,
                                   tid, stride);
         // u2: curl_x → (Bct, Gct, Bot); curl_y → -(Gct, Bct, Bot)
         CurlTranspAdd3D<MD1, MQ1>(D1D, D1D, D1D - 1, Q1D, sBct, sGct, sBot,
                                   curl[0], smA[0], smC[0], Y, 2 * osc_c, e, 1.0,
                                   tid, stride);
         CurlTranspAdd3D<MD1, MQ1>(D1D, D1D, D1D - 1, Q1D, sGct, sBct, sBot,
                                   curl[1], smA[0], smC[0], Y, 2 * osc_c, e, -1.0,
                                   tid, stride);
      }
   }
};

template <typename QFn, int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void TensorCurlCurlApplyDevice(const int NE,
                                      const Array<real_t> &bo,
                                      const Array<real_t> &bc,
                                      const Array<real_t> &,
                                      const Array<real_t> &,
                                      const Array<real_t> &gc,
                                      const Array<real_t> &,
                                      const Vector &d,
                                      const Vector &x,
                                      Vector &y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   using Tr = qfn_traits<QFn>;
   static_assert(DIM == 2 || DIM == 3, "TensorCurlCurlApplyDevice: DIM 2 or 3");
   static_assert(Tr::trial_is_curl, "");
   constexpr bool SYM = Tr::symmetric_pa;
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   dq.Verify(NE, "Tensor curl-curl MMA D1D/Q1D exceeds shell cap");

   const int nq = (DIM == 2) ? (Q1D * Q1D) : (Q1D * Q1D * Q1D);
   const int ndof = (DIM == 2) ? (2 * (D1D - 1) * D1D)
                    : (3 * (D1D - 1) * D1D * D1D);
   MFEM_VERIFY(x.Size() == ndof * NE && y.Size() == ndof * NE, "");

   const auto Bo = Reshape(bo.Read(), Q1D, D1D - 1);
   const auto X = Reshape(x.Read(), ndof, NE);
   auto Y = Reshape(y.ReadWrite(), ndof, NE);

   if constexpr (DIM == 2)
   {
      MFEM_VERIFY(d.Size() == nq * NE, "");
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                     : mma::NB2DRuntime(D1D);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                              : mma::Threads2DRuntime(D1D, Q1D));
      const int nblocks = (NE + NB - 1) / NB;
      const auto Gc = Reshape(gc.Read(), Q1D, D1D);
      const auto Dv = Reshape(d.Read(), nq, NE);
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorCurlCurlKernel2D<MD1, MQ1, MDQ>
      {NE, D1D, Q1D, NB, Bo, Gc, Dv, X, Y});
   }
   else
   {
      constexpr int ncomp = SYM ? 6 : 9;
      MFEM_VERIFY(d.Size() == nq * ncomp * NE, "");
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D
                              ? mma::TensorThreads3D<T_D1D, T_Q1D,
                              mma::kTensorCostHeavy>()
                              : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                            mma::kTensorCostHeavy));
      const int NB = T_D1D
                     ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                     : mma::TensorNB3DRuntime(D1D, mma::kTensorCostHeavy);
      const int nblocks = (NE + NB - 1) / NB;
      const auto Bc = Reshape(bc.Read(), Q1D, D1D);
      const auto Gc = Reshape(gc.Read(), Q1D, D1D);
      const auto Dv = Reshape(d.Read(), nq, ncomp, NE);
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorCurlCurlKernel3D<MD1, MQ1, SYM>
      {NE, D1D, Q1D, NB, Bo, Bc, Gc, Dv, X, Y});
   }
}

// ---------------------------------------------------------------------------
// Div×Div — Bo on open faces, Gc on the closed/normal axis
// ---------------------------------------------------------------------------

template <int MD1, int MQ1, int MDQ>
struct TensorDivDivKernel2D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Gc;
   DeviceTensor<2, const real_t> D; // (nq, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      constexpr int VDIM = 2;
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t sGc[MD1 * MQ1];
      MFEM_SHARED real_t sGct[MD1 * MQ1];
      MFEM_SHARED real_t divq[MDQ * MDQ];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, Gc, sGc, sGct);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D;
      const int osc_c = (D1D - 1) * D1D;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int t = tid; t < nq; t += stride) { divq[t] = 0.0; }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy;
            DivCompSizes2D(D1D, c, Dx, Dy);
            const int n = Dx * Dy;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int dy = t / Dx;
               sm0[dx + Dx * dy] = X(dx + dy * Dx + c * osc_c, e);
            }
            MFEM_SYNC_THREAD;
            const real_t *Bx = (c == 0) ? sGc : sBo;
            const real_t *By = (c == 1) ? sGc : sBo;
            mma::InterpAx<MD1, MQ1>(Dy, Q1D, Dx, Bx, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D, Q1D, Dy, By, sm1, sm0);
            MFEM_SYNC_THREAD;
            DivAccum(divq, sm0, nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         for (int t = tid; t < nq; t += stride)
         {
            divq[t] *= D(t, e);
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy;
            DivCompSizes2D(D1D, c, Dx, Dy);
            const real_t *Byt = (c == 1) ? sGct : sBot;
            const real_t *Bxt = (c == 0) ? sGct : sBot;
            CurlTranspAdd2D<MD1, MQ1>(Dx, Dy, Q1D, Bxt, Byt, divq, sm0, sm1,
                                      Y, c * osc_c, e, 1.0, tid, stride);
         }
      }
   }
};

template <int MD1, int MQ1>
struct TensorDivDivKernel3D
{
   int NE, D1D, Q1D, NB;
   DeviceTensor<2, const real_t> Bo, Gc;
   DeviceTensor<2, const real_t> D; // (nq, NE)
   DeviceTensor<2, const real_t> X;
   DeviceTensor<2, real_t> Y;

   MFEM_HOST_DEVICE void operator()(int b) const
   {
      constexpr int VDIM = 3;
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sBo[MD1 * MQ1];
      MFEM_SHARED real_t sBot[MD1 * MQ1];
      MFEM_SHARED real_t sGc[MD1 * MQ1];
      MFEM_SHARED real_t sGct[MD1 * MQ1];
      MFEM_SHARED real_t divq[MQ1 * MQ1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D - 1, Q1D, Bo, sBo, sBot);
      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, Gc, sGc, sGct);
      MFEM_SYNC_THREAD;

      const int tid = mma::getThreadIdxX();
      const int stride = mma::getBlockNthreadsX();
      const int nq = Q1D * Q1D * Q1D;
      const int osc_c = (D1D - 1) * (D1D - 1) * D1D;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         for (int t = tid; t < nq; t += stride) { divq[t] = 0.0; }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy, Dz;
            DivCompSizes3D(D1D, c, Dx, Dy, Dz);
            const int n = Dx * Dy * Dz;
            for (int t = tid; t < n; t += stride)
            {
               const int dx = t % Dx;
               const int rest = t / Dx;
               const int dy = rest % Dy;
               const int dz = rest / Dy;
               sm0[dx + Dx * (dy + Dy * dz)] =
                  X(dx + (dy + dz * Dy) * Dx + c * osc_c, e);
            }
            MFEM_SYNC_THREAD;
            const real_t *Bx = (c == 0) ? sGc : sBo;
            const real_t *By = (c == 1) ? sGc : sBo;
            const real_t *Bz = (c == 2) ? sGc : sBo;
            mma::InterpAx<MD1, MQ1>(Dy * Dz, Q1D, Dx, Bx, sm0, sm1);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Dz * Q1D, Q1D, Dy, By, sm1, sm0);
            MFEM_SYNC_THREAD;
            mma::InterpAx<MD1, MQ1>(Q1D * Q1D, Q1D, Dz, Bz, sm0, sm1);
            MFEM_SYNC_THREAD;
            DivAccum(divq, sm1, nq, 1.0, tid, stride);
            MFEM_SYNC_THREAD;
         }

         for (int t = tid; t < nq; t += stride)
         {
            divq[t] *= D(t, e);
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < VDIM; ++c)
         {
            int Dx, Dy, Dz;
            DivCompSizes3D(D1D, c, Dx, Dy, Dz);
            const real_t *Bxt = (c == 0) ? sGct : sBot;
            const real_t *Byt = (c == 1) ? sGct : sBot;
            const real_t *Bzt = (c == 2) ? sGct : sBot;
            DivTranspAdd3D<MD1, MQ1>(Dx, Dy, Dz, Q1D, Bxt, Byt, Bzt,
                                     divq, sm0, sm1, Y, c * osc_c, e, 1.0,
                                     tid, stride);
         }
      }
   }
};

template <typename QFn, int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void TensorDivDivApplyDevice(const int NE,
                                    const Array<real_t> &bo,
                                    const Array<real_t> &gc,
                                    const Array<real_t> &,
                                    const Array<real_t> &,
                                    const Vector &d,
                                    const Vector &x,
                                    Vector &y,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   static_assert(DIM == 2 || DIM == 3, "TensorDivDivApplyDevice: DIM 2 or 3");
   const mma::TensorShellDims<T_D1D, T_Q1D> dq(d1d, q1d);
   const int D1D = dq.D1D, Q1D = dq.Q1D;
   constexpr int MD1 = mma::TensorShellDims<T_D1D, T_Q1D>::MD1;
   constexpr int MQ1 = mma::TensorShellDims<T_D1D, T_Q1D>::MQ1;
   dq.Verify(NE, "Tensor div-div MMA D1D/Q1D exceeds shell cap");

   const int nq = (DIM == 2) ? (Q1D * Q1D) : (Q1D * Q1D * Q1D);
   const int ndof = (DIM == 2) ? (2 * (D1D - 1) * D1D)
                    : (3 * (D1D - 1) * (D1D - 1) * D1D);
   MFEM_VERIFY(d.Size() == nq * NE, "");
   MFEM_VERIFY(x.Size() == ndof * NE && y.Size() == ndof * NE, "");

   const auto Bo = Reshape(bo.Read(), Q1D, D1D - 1);
   const auto Gc = Reshape(gc.Read(), Q1D, D1D);
   const auto Dv = Reshape(d.Read(), nq, NE);
   const auto X = Reshape(x.Read(), ndof, NE);
   auto Y = Reshape(y.ReadWrite(), ndof, NE);

   if constexpr (DIM == 2)
   {
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                     : mma::NB2DRuntime(D1D);
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                              : mma::Threads2DRuntime(D1D, Q1D));
      const int nblocks = (NE + NB - 1) / NB;
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorDivDivKernel2D<MD1, MQ1, MDQ>
      {NE, D1D, Q1D, NB, Bo, Gc, Dv, X, Y});
   }
   else
   {
      const int nthreads = mma::TensorShellNthreads(
                              T_D1D
                              ? mma::TensorThreads3D<T_D1D, T_Q1D,
                              mma::kTensorCostHeavy>()
                              : mma::TensorThreads3DRuntime(D1D, Q1D,
                                                            mma::kTensorCostHeavy));
      const int NB = T_D1D
                     ? mma::TensorNB3D<T_D1D, T_Q1D, mma::kTensorCostHeavy>()
                     : mma::TensorNB3DRuntime(D1D, mma::kTensorCostHeavy);
      const int nblocks = (NE + NB - 1) / NB;
      mfem::forall_3D(nblocks, nthreads, 1, 1,
                      TensorDivDivKernel3D<MD1, MQ1>
      {NE, D1D, Q1D, NB, Bo, Gc, Dv, X, Y});
   }
}


// ---------------------------------------------------------------------------
// ApplyTensor — Vector FE mass (Bo/Bc)
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_vec_eval, void>
ApplyTensor(const int NE,
            const Array<real_t> &bo, const Array<real_t> &bc,
            const Array<real_t> &bot, const Array<real_t> &bct,
            const Vector &d, const Vector &x, Vector &y,
            const int d1d = 0, const int q1d = 0)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "QFn DIM must match ApplyTensor DIM");
   TensorVecEvalApplyDevice<QFn, DIM, D1D, Q1D>(
      NE, bo, bc, bot, bct, d, x, y, d1d, q1d);
}

// ---------------------------------------------------------------------------
// ApplyTensor — Curl×Curl (Bo/Bc/Gc)
// Host: InterpAx / GradX / GemmMbyK → blas (LAPACK GEMM when enabled).
// CUDA/HIP: same calls → MMA_BACKEND_PICK (DMMA / MFMA), including adjoints.
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_curl, void>
ApplyTensor(const int NE,
            const Array<real_t> &bo, const Array<real_t> &bc,
            const Array<real_t> &bot, const Array<real_t> &bct,
            const Array<real_t> &gc, const Array<real_t> &gct,
            const Vector &d, const Vector &x, Vector &y,
            const int d1d = 0, const int q1d = 0)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "QFn DIM must match ApplyTensor DIM");
   TensorCurlCurlApplyDevice<QFn, DIM, D1D, Q1D>(
      NE, bo, bc, bot, bct, gc, gct, d, x, y, d1d, q1d);
}

// ---------------------------------------------------------------------------
// ApplyTensor — Div×Div (Bo/Gc)
// Host: InterpAx / GemmMbyK → blas (LAPACK GEMM when enabled).
// CUDA/HIP: same calls → MMA_BACKEND_PICK (DMMA / MFMA), including adjoints.
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_div, void>
ApplyTensor(const int NE,
            const Array<real_t> &bo, const Array<real_t> &gc,
            const Array<real_t> &bot, const Array<real_t> &gct,
            const Vector &d, const Vector &x, Vector &y,
            const int d1d = 0, const int q1d = 0)
{
   TensorDivDivApplyDevice<QFn, DIM, D1D, Q1D>(
      NE, bo, gc, bot, gct, d, x, y, d1d, q1d);
}

} // namespace mfem::internal::mma::form

/// \endcond
