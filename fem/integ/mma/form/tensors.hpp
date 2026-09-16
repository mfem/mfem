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
      - form::ApplyTensor<QFn, …> (SFINAE; Grad / De Rham vec-eval curl div)
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

// De Rham tensor PA — Bo/Bc/(Gc) engines + ApplyTensor overloads
// (formerly tensors_derham.hpp)
// ---------------------------------------------------------------------------

namespace detail
{

inline void TensorHcurlMassApply2D(const int NE, const bool symmetric,
                        [[maybe_unused]] const bool scalar_coeff,
                        const Array<real_t> &bo, const Array<real_t> &bc,
                        const Array<real_t> &bot, const Array<real_t> &bct,
                        const Vector &pa_data, const Vector &x, Vector &y,
                        const int D1D, [[maybe_unused]] const int TestD1D,
                        const int Q1D)
{
   MFEM_ASSERT(D1D == TestD1D,
               "Trial and Test space must have the same number of dofs");
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t O11 = op(qx,qy,0,e);
            const real_t O21 = op(qx,qy,1,e);
            const real_t O12 = symmetric ? O21 : op(qx,qy,2,e);
            const real_t O22 = symmetric ? op(qx,qy,2,e) : op(qx,qy,3,e);
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O21*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

inline void TensorHcurlMassApply3D(const int NE, const bool symmetric,
                        [[maybe_unused]] const bool scalar_coeff,
                        const Array<real_t> &bo, const Array<real_t> &bc,
                        const Array<real_t> &bot, const Array<real_t> &bct,
                        const Vector &pa_data, const Vector &x, Vector &y,
                        const int D1D, [[maybe_unused]] const int TestD1D,
                        const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MAX_D1D][MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

inline void TensorHdivMassApply2D(const int NE, const bool symmetric, const bool,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int TestD1D, const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t O11 = op(qx,qy,0,e);
            const real_t O12 = op(qx,qy,1,e);
            const real_t O21 = symmetric ? O12 : op(qx,qy,2,e);
            const real_t O22 = symmetric ? op(qx,qy,2,e) : op(qx,qy,3,e);
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O21*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dx = (c == 1) ? D1D - 1 : D1D;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bct(dx,qx) :
                                                  Bot(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

inline void TensorHdivMassApply3D(const int NE, const bool symmetric, const bool,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int TestD1D, const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t mass[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[DofQuadLimits::HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D : D1D - 1;
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dx = (c == 0) ? D1D : D1D - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[DofQuadLimits::HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] *
                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

inline void TensorCurlCurlApply2D(const int D1D, const int Q1D, const bool, const int NE,
                       const Array<real_t> &bo, const Array<real_t> &,
                       const Array<real_t> &bot, const Array<real_t> &,
                       const Array<real_t> &gc, const Array<real_t> &gct,
                       const Vector &pa_data, const Vector &x, Vector &y,
                       const bool useAbs)
{

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t curl[MAX_Q1D][MAX_Q1D];

      // curl[qy][qx] will be computed as du_y/dx - du_x/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] = 0.0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int sign = useAbs ? 1 : -1;
               const real_t wy = (c == 0) ? (sign*Gc(qy,dy)) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  curl[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += curl[qy][qx] * ((c == 0) ? Bot(dx,qx) : Gct(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const int sign = useAbs ? 1 : -1;
               const real_t wy = (c == 0) ? (sign*Gct(dy,qy)) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

inline void TensorCurlCurlApply3D(const int d1d,
                              const int q1d,
                              const bool symmetric,
                              const int NE,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Array<real_t> &gc,
                              const Array<real_t> &gct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y,
                              const bool useAbs)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = d1d;
   const int Q1D = q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk),
      // we get:
      // (\nabla\times u) \cdot (\nabla\times v)
      //     = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = d1d;
      const int Q1D = q1d;

      real_t curl[MQ1D][MQ1D][MQ1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MQ1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     if (useAbs)
                     {
                        // +(u_0)_{x_1}
                        curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;
                     }
                     else
                     {
                        // -(u_0)_{x_1}
                        curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;
                     }
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t massY[MQ1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = Bc(qx,dx);
                  const real_t wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     if (useAbs)
                     {
                        // +(u_1)_{x_2}
                        curl[qz][qy][qx][0] += gradXY[qy][qx][1] * wDz;
                     }
                     else
                     {
                        // -(u_1)_{x_2}
                        curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz;
                     }
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            real_t gradYZ[MQ1D][MQ1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massZ[MQ1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx = Bc(qx,dx);
               const real_t wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     if (useAbs)
                     {
                        // +(u_2)_{x_0}
                        curl[qz][qy][qx][1] += gradYZ[qz][qy][0] * wDx;
                     }
                     else
                     {
                        // -(u_2)_{x_0}
                        curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx;
                     }
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const real_t c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const real_t c2 = (O21 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const real_t c3 = (O31 * curl[qz][qy][qx][0]) + (O32 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY12[MD1D][MD1D];
            real_t gradXY21[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MD1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const real_t wx = Bot(dx,qx);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     if (useAbs)
                     {
                        // (u_0)_{x_2} * (op * curl)_1 +
                        // (u_0)_{x_1} * (op * curl)_2
                        Y(idx, e) += (gradXY21[dy][dx] * wDz) +
                                     (gradXY12[dy][dx] * wz);
                     }
                     else
                     {
                        // (u_0)_{x_2} * (op * curl)_1 -
                        // (u_0)_{x_1} * (op * curl)_2
                        Y(idx, e) += (gradXY21[dy][dx] * wDz) -
                                     (gradXY12[dy][dx] * wz);
                     }
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY02[MD1D][MD1D];
            real_t gradXY20[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               real_t massY[MD1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const real_t wy = Bot(dy,qy);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t wx = Bct(dx,qx);
                  const real_t wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     if (useAbs)
                     {
                        // +(u_1)_{x_2} * (op * curl)_0 +
                        //  (u_1)_{x_0} * (op * curl)_2
                        Y(idx, e) += (gradXY20[dy][dx] * wDz) +
                                     (gradXY02[dy][dx] * wz);
                     }
                     else
                     {
                        // -(u_1)_{x_2} * (op * curl)_0 +
                        //  (u_1)_{x_0} * (op * curl)_2
                        Y(idx, e) += (-gradXY20[dy][dx] * wDz) +
                                     (gradXY02[dy][dx] * wz);
                     }
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            real_t gradYZ01[MD1D][MD1D];
            real_t gradYZ10[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massZ[MD1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const real_t wz = Bot(dz,qz);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t wx = Bct(dx,qx);
               const real_t wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     if (useAbs)
                     {
                        // (u_2)_{x_1} * (op * curl)_0 +
                        // (u_2)_{x_0} * (op * curl)_1
                        Y(idx, e) += (gradYZ10[dz][dy] * wx) +
                                     (gradYZ01[dz][dy] * wDx);
                     }
                     else
                     {
                        // (u_2)_{x_1} * (op * curl)_0 -
                        // (u_2)_{x_0} * (op * curl)_1
                        Y(idx, e) += (gradYZ10[dz][dy] * wx) -
                                     (gradYZ01[dz][dy] * wDx);
                     }
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
}

inline void TensorDivDivApply2D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_)
{
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t div[MAX_Q1D][MAX_Q1D];

      // div[qy][qx] will be computed as du_x/dx + du_y/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] = 0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 0) ? Bo(qy,dy) : Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  div[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dx = (c == 1) ? D1D - 1 : D1D;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            real_t gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += div[qy][qx] * (c == 0 ? Gct(dx,qx) : Bot(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 0) ? Bot(dy,qy) : Gct(dy,qy);
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

inline void TensorDivDivApply3D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t div[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qz][qy][qx] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t aXY[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t aX[DofQuadLimits::HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     aX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = aX[qx];
                     aXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     div[qz][qy][qx] += aXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t aXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D : D1D - 1;
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dx = (c == 0) ? D1D : D1D - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t aX[DofQuadLimits::HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aX[dx] += div[qz][qy][qx] *
                               (c == 0 ? Gct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aXY[dy][dx] += aX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Gct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        aXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

} // namespace detail

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
   constexpr bool SYM = Tr::symmetric_pa;
   const int D = D1D ? D1D : d1d;
   const int Q = Q1D ? Q1D : q1d;
   const bool scalar_coeff = true; // unused in engines
   if constexpr (Tr::open_on_component)
   {
      if constexpr (DIM == 2)
      {
         detail::TensorHcurlMassApply2D(NE, SYM, scalar_coeff, bo, bc, bot, bct,
                                        d, x, y, D, D, Q);
      }
      else
      {
         detail::TensorHcurlMassApply3D(NE, SYM, scalar_coeff, bo, bc, bot, bct,
                                        d, x, y, D, D, Q);
      }
   }
   else
   {
      if constexpr (DIM == 2)
      {
         detail::TensorHdivMassApply2D(NE, SYM, scalar_coeff, bo, bc, bot, bct,
                                       d, x, y, D, D, Q);
      }
      else
      {
         detail::TensorHdivMassApply3D(NE, SYM, scalar_coeff, bo, bc, bot, bct,
                                       d, x, y, D, D, Q);
      }
   }
}

// ---------------------------------------------------------------------------
// ApplyTensor — Curl×Curl (Bo/Bc/Gc)
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
   constexpr bool SYM = Tr::symmetric_pa;
   const int D = D1D ? D1D : d1d;
   const int Q = Q1D ? Q1D : q1d;
   if constexpr (DIM == 2)
   {
      detail::TensorCurlCurlApply2D(D, Q, SYM, NE, bo, bc, bot, bct, gc, gct,
                                    d, x, y, false);
   }
   else
   {
      detail::TensorCurlCurlApply3D(D, Q, SYM, NE, bo, bc, bot, bct, gc, gct,
                                    d, x, y, false);
   }
}

// ---------------------------------------------------------------------------
// ApplyTensor — Div×Div (Bo/Gc)
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_div, void>
ApplyTensor(const int NE,
            const Array<real_t> &bo, const Array<real_t> &gc,
            const Array<real_t> &bot, const Array<real_t> &gct,
            const Vector &d, const Vector &x, Vector &y,
            const int d1d = 0, const int q1d = 0)
{
   const int D = D1D ? D1D : d1d;
   const int Q = Q1D ? Q1D : q1d;
   if constexpr (DIM == 2)
   {
      detail::TensorDivDivApply2D(D, Q, NE, bo, gc, bot, gct, d, x, y);
   }
   else
   {
      detail::TensorDivDivApply3D(D, Q, NE, bo, gc, bot, gct, d, x, y);
   }
}

} // namespace mfem::internal::mma::form

/// \endcond
