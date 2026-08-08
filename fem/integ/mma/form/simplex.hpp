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

/** @file simplex.hpp
    Simplex dense form apply (integrator-agnostic).

    Eval×Eval / Grad×Grad / ApplyLF pipelines on dense P/G basis.
    Companion to tensors.hpp (tensor-product sum-fact ApplyTensor).
*/

#include "../dispatch.hpp"
#include "plan.hpp"

#include "../../../../general/array.hpp"
#include "../../../../linalg/vector.hpp"

#include <type_traits>
#ifdef MFEM_USE_LAPACK
#include <vector>
#include "../lapack.hpp"
#endif

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

// ---------------------------------------------------------------------------
// Pointwise Eval Q-phase on smem U(q,b) with scalar PA density D(q,e)
// ---------------------------------------------------------------------------

/** Apply QFn to each (q,b) in U smem. Layout U(q,b) = Us[q + u_ld*b].
    QFn: operator()(const eval_t&, eval_t&, real_t). */
template <typename QFn, typename DAcc>
MFEM_HOST_DEVICE inline void ApplyEvalQFnSmem(
   QFn qfn, real_t *Us, DAcc D,
   const int e0, const int NE,
   const int nq, const int u_ld, const int nb,
   const int tid, const int nthreads)
{
   for (int idx = tid; idx < nq * nb; idx += nthreads)
   {
      const int b = idx / nq;
      const int q = idx - b * nq;
      const int e = e0 + b;
      if (e >= NE) { continue; }
      eval_t u(Us[q + u_ld * b]);
      eval_t y;
      InvokeQFn(qfn, u, y, D(q, e));
      Us[q + u_ld * b] = real_t(y);
   }
}

/** Serial dense element: Y += P^T QFn(P X, D). u_scratch holds nq reals. */
template <typename QFn>
MFEM_HOST_DEVICE inline void EvalApplyDenseElement(
   QFn qfn,
   const int nq, const int ndof,
   const real_t *P, const real_t *D_e,
   const real_t *X_e, real_t *Y_e, real_t *u_scratch)
{
   for (int q = 0; q < nq; ++q)
   {
      real_t s = 0.0;
      for (int i = 0; i < ndof; ++i)
      {
         s += P[q + nq * i] * X_e[i];
      }
      eval_t u(s), y;
      InvokeQFn(qfn, u, y, D_e[q]);
      u_scratch[q] = real_t(y);
   }
   for (int i = 0; i < ndof; ++i)
   {
      real_t s = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         s += P[q + nq * i] * u_scratch[q];
      }
      Y_e[i] += s;
   }
}

// ---------------------------------------------------------------------------
// Device batch body — Eval × Eval (unfused Gemm + QFn + GemmT)
// ---------------------------------------------------------------------------

template <typename QFn, int MAP, int QND, int NDOF, int X_LD, int U_LD, int NB>
MFEM_HOST_DEVICE inline void EvalBatchBody(
   QFn qfn,
   const int e0, const int NE,
   const real_t *p, const real_t *d, const real_t *x, real_t *y,
   real_t *XY, real_t *Us,
   const int tid, const int nthreads)
{
   const auto D = ConstDeviceMatrix(d, QND, NE);
   const auto X = ConstDeviceMatrix(x, NDOF, NE);
   SmemMatAcc<X_LD> Xacc{XY};
   SmemMatAcc<U_LD> Uacc{Us};
   YBatchAcc Yacc{y, NDOF, e0};
   PAcc A{p, QND, NDOF};

   if (DeviceGemmEnabled())
   {
      LoadXToSmem(XY, X, e0, NE, NDOF, X_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;

      // Forward without fused SCALE — QFn owns the pointwise math.
      Gemm<MAP, false>(QND, NDOF, NB, A, Xacc, Uacc, D, e0, NE);
      MFEM_SYNC_THREAD;

      ApplyEvalQFnSmem(qfn, Us, D, e0, NE, QND, U_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;

      GemmT<MAP>(QND, NDOF, NB, A, Uacc, Yacc, e0, NE);
   }
   else
   {
      // Host/device Emulate serial: only tid 0.
      if (tid == 0)
      {
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < X_LD; ++i)
            {
               XY[i + X_LD * b] = (i < NDOF) ? X(i, e) : real_t(0);
            }
            EvalApplyDenseElement(qfn, QND, NDOF, p, &D(0, e),
                                  &XY[X_LD * b],
                                  y + NDOF * e, &Us[U_LD * b]);
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template <typename QFn>
MFEM_HOST_DEVICE inline void EvalBatchBodyRuntime(
   QFn qfn,
   const int e0, const int NE,
   const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const real_t *p, const real_t *d, const real_t *x, real_t *y,
   real_t *XY, real_t *Us,
   const int tid, const int nthreads)
{
   constexpr int MAP = MmaMapDefault;
   const auto D = ConstDeviceMatrix(d, nq, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);
   SmemMatAccRt Xacc{XY, x_ld};
   SmemMatAccRt Uacc{Us, u_ld};
   YBatchAcc Yacc{y, ndof, e0};
   PAcc A{p, nq, ndof};

   if (DeviceGemmEnabled())
   {
      LoadXToSmem(XY, X, e0, NE, ndof, x_ld, nb, tid, nthreads);
      MFEM_SYNC_THREAD;
      Gemm<MAP, false>(nq, ndof, nb, A, Xacc, Uacc, D, e0, NE);
      MFEM_SYNC_THREAD;
      ApplyEvalQFnSmem(qfn, Us, D, e0, NE, nq, u_ld, nb, tid, nthreads);
      MFEM_SYNC_THREAD;
      GemmT<MAP>(nq, ndof, nb, A, Uacc, Yacc, e0, NE);
   }
   else if (tid == 0)
   {
      for (int b = 0; b < nb; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int i = 0; i < x_ld; ++i)
         {
            XY[i + x_ld * b] = (i < ndof) ? X(i, e) : real_t(0);
         }
         EvalApplyDenseElement(qfn, nq, ndof, p, &D(0, e),
                               &XY[x_ld * b],
                               y + ndof * e, &Us[u_ld * b]);
      }
   }
   MFEM_SYNC_THREAD;
}

// ---------------------------------------------------------------------------
// Host apply (dense element with QFn)
// ---------------------------------------------------------------------------

template <typename QFn>
inline void HostEvalApply(QFn qfn, const int NE, const int nq, const int ndof,
                          const real_t *P, const real_t *D,
                          const real_t *X, real_t *Y)
{
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(nq)));
      EvalApplyDenseElement(qfn, nq, ndof, P, D + nq * e,
                            X + ndof * e, Y + ndof * e, u);
   }
}

/** Block-diag multi-component: X/Y layout (ndof × vdim × NE), shared D. */
template <typename QFn>
inline void HostEvalApply(QFn qfn, const int NE, const int nq, const int ndof,
                          const real_t *P, const real_t *D,
                          const real_t *X, real_t *Y, const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      HostEvalApply(qfn, NE, nq, ndof, P, D, X, Y);
      return;
   }
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(nq)));
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = ndof * (vc + vdim * e);
         EvalApplyDenseElement(qfn, nq, ndof, P, D + nq * e,
                               X + off, Y + off, u);
      }
   }
}

// ---------------------------------------------------------------------------
// Public Apply entry points
// ---------------------------------------------------------------------------

/** Specialized simplex Eval×Eval Apply (mass-like).
    Signature matches MassIntegrator::ApplySimplexMmaKernelType. */
template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<!qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::load_x && !Tr::test_is_grad,
                 "Eval Apply requires load_x and Eval test");
   static_assert(D1D > 0 && QND > 0, "requires specialized D1D/QND");

   constexpr int NDOF = SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == QND * NE, "");
   MFEM_VERIFY(basis.Size() == QND * NDOF, "");
   MFEM_VERIFY(x.Size() >= NDOF * NE && y.Size() >= NDOF * NE, "");

   DumpFormApply<QFn, DIM, D1D, QND>("Apply", NE, QND, NDOF);

   QFn qfn{};

   // ---- host ---------------------------------------------------------------
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostEvalApply(qfn, NE, QND, NDOF,
                    basis.Read(), d.Read(), x.Read(), y.ReadWrite());
      return;
   }

   // ---- device smem batch --------------------------------------------------
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int X_LD = PadLdBank<MAP>(NDOF);
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int NB = MassLikeNB<DIM, D1D, QND>();
   constexpr int smem_bytes = int(sizeof(real_t)) * (X_LD + U_LD) * NB;
   VerifySharedMemBytes(smem_bytes);

   const int nbatches = (NE + NB - 1) / NB;
   const int nthreads = LaunchNthreads<QND>(QND, NDOF);
   const auto P = basis.Read(), D = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();

   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      struct alignas(16) Smem
      {
         real_t XY[X_LD * NB];
         real_t Us[U_LD * NB];
      };
      MFEM_SIMPLEX_MMA_SMEM(Smem, sm);
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      EvalBatchBody<QFn, MAP, QND, NDOF, X_LD, U_LD, NB>(
         qfn, batch * NB, NE, P, D, X, Y, sm.XY, sm.Us, tid, nthr);
   });
}

/** Specialized Eval×Eval with block-diag vdim (X/Y: ndof × vdim × NE). */
template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<!qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y,
      const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      Apply<QFn, DIM, D1D, QND>(NE, basis, d, x, y);
      return;
   }

   using Tr = qfn_traits<QFn>;
   static_assert(Tr::load_x && !Tr::test_is_grad, "");
   static_assert(D1D > 0 && QND > 0, "");

   constexpr int NDOF = SimplexNdof<DIM, D1D>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   MFEM_VERIFY(NE > 0 && d.Size() == QND * NE, "");
   MFEM_VERIFY(basis.Size() == QND * NDOF, "");
   MFEM_VERIFY(x.Size() >= NDOF * vdim * NE && y.Size() >= NDOF * vdim * NE, "");

   DumpFormApply<QFn, DIM, D1D, QND>("Apply", NE, QND, NDOF);

   QFn qfn{};
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostEvalApply(qfn, NE, QND, NDOF,
                    basis.Read(), d.Read(), x.Read(), y.ReadWrite(), vdim);
      return;
   }

   // Device: dense per-element × component (shared PA).
   const auto P = basis.Read(), D = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[MQ];
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = NDOF * (vc + vdim * e);
         EvalApplyDenseElement(qfn, QND, NDOF, P, D + QND * e,
                               X + off, Y + off, u);
      }
   });
}

/** Runtime Fallback Eval×Eval Apply. */
template <typename QFn, int DIM>
inline std::enable_if_t<!qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::load_x && !Tr::test_is_grad,
                 "Eval Apply runtime requires load_x and Eval test");

   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && basis.Size() % nq == 0, "");
   const int ndof = basis.Size() / nq;
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int MAX_NQ = SimplexMaxNq<DIM, 0>();
   constexpr int MAX_NDOF = SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= MAX_NQ && ndof <= MAX_NDOF,
               "simplex Eval Apply runtime exceeds size caps");

   DumpFormApplyRuntime<QFn, DIM>("Apply", NE, nq, ndof);

   QFn qfn{};

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostEvalApply(qfn, NE, nq, ndof,
                    basis.Read(), d.Read(), x.Read(), y.ReadWrite());
      return;
   }

   const SmemPlan plan = MakeEvalPlanRuntime(ndof, nq, true);
   const int x_ld = plan.x_ld;
   const int u_ld = plan.u_ld;
   const int nb = plan.nb;
   MFEM_VERIFY(x_ld <= PadLdBank<MmaMapDefault>(MAX_NDOF) &&
               u_ld <= PadLdBank<MmaMapDefault>(MAX_NQ) &&
               nb <= NBATCH,
               "simplex Eval Apply runtime smem layout exceeds caps");
   VerifySharedMemBytes(plan.smem_bytes);

   const auto P = basis.Read();
   const auto D = d.Read();
   const auto X = x.Read();
   auto Y = y.ReadWrite();
   const int nthreads = plan.nthreads;
   const int nbatches = (NE + nb - 1) / nb;

   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, plan.smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
#if defined(__CUDA_ARCH__)
      real_t *XY = reinterpret_cast<real_t *>(SimplexMmaDynSmem());
      real_t *Us = XY + x_ld * nb;
#else
      constexpr int max_nb = NBATCH;
      constexpr int max_x_ld = PadLdBank<MmaMapDefault>(MAX_NDOF);
      constexpr int max_u_ld = PadLdBank<MmaMapDefault>(MAX_NQ);
      MFEM_SHARED real_t XY[max_x_ld * max_nb];
      MFEM_SHARED real_t Us[max_u_ld * max_nb];
#endif
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      EvalBatchBodyRuntime(qfn, batch * nb, NE, nq, ndof, x_ld, u_ld, nb,
                           P, D, X, Y, XY, Us, tid, nthr);
   });
}

/** Runtime Eval×Eval with block-diag vdim. */
template <typename QFn, int DIM>
inline std::enable_if_t<!qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y,
      const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      Apply<QFn, DIM>(NE, basis, d, x, y);
      return;
   }

   using Tr = qfn_traits<QFn>;
   static_assert(Tr::load_x && !Tr::test_is_grad, "");

   MFEM_VERIFY(NE > 0 && d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && basis.Size() % nq == 0, "");
   const int ndof = basis.Size() / nq;
   MFEM_VERIFY(x.Size() >= ndof * vdim * NE && y.Size() >= ndof * vdim * NE, "");

   constexpr int MAX_NQ = SimplexMaxNq<DIM, 0>();
   constexpr int MAX_NDOF = SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= MAX_NQ && ndof <= MAX_NDOF, "");

   DumpFormApplyRuntime<QFn, DIM>("Apply", NE, nq, ndof);

   QFn qfn{};
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostEvalApply(qfn, NE, nq, ndof,
                    basis.Read(), d.Read(), x.Read(), y.ReadWrite(), vdim);
      return;
   }

   const auto P = basis.Read(), D = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[MAX_NQ];
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = ndof * (vc + vdim * e);
         EvalApplyDenseElement(qfn, nq, ndof, P, D + nq * e,
                               X + off, Y + off, u);
      }
   });
}

// ---------------------------------------------------------------------------
// Linear-form ApplyLF — trial none_t (DomainLF): QFn fills U from D, then GemmT
// ---------------------------------------------------------------------------

/** Fill U(q,b) via QFn(test_t, coeff) — no trial DOFs. */
template <typename QFn, typename DAcc>
MFEM_HOST_DEVICE inline void ApplyNoneQFnSmem(
   QFn qfn, real_t *Us, DAcc D,
   const int e0, const int NE,
   const int nq, const int u_ld, const int nb,
   const int tid, const int nthreads)
{
   for (int idx = tid; idx < nq * nb; idx += nthreads)
   {
      const int b = idx / nq;
      const int q = idx - b * nq;
      const int e = e0 + b;
      if (e >= NE) { continue; }
      eval_t y;
      InvokeQFn(qfn, y, D(q, e));
      Us[q + u_ld * b] = real_t(y);
   }
}

/** Host: Y_vc += P^T QFn(D). */
template <typename QFn>
inline void HostLFApply(QFn qfn, const int NE, const int nq, const int ndof,
                        const real_t *P, const real_t *D, real_t *Y,
                        const int vdim, const int vc)
{
#ifdef MFEM_USE_LAPACK
   // Size-based: multi-RHS GEMM after pointwise QFn fill of U.
   if (lapack::PreferMultiRhs(nq, ndof, NE))
   {
      const int NB = lapack::NB(nq, ndof);
      const int ntiles = (NE + NB - 1) / NB;
      std::vector<real_t> uloc(static_cast<size_t>(nq) * NB);
      std::vector<real_t> ytmp(static_cast<size_t>(ndof) * NB);
      for (int tile = 0; tile < ntiles; ++tile)
      {
         const int e0 = tile * NB;
         std::fill(uloc.begin(), uloc.end(), real_t(0));
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { break; }
            for (int q = 0; q < nq; ++q)
            {
               eval_t ye;
               InvokeQFn(qfn, ye, D[q + nq * e]);
               uloc[static_cast<size_t>(q) + static_cast<size_t>(nq) * b] =
                  real_t(ye);
            }
         }
         lapack::Gemm('T', 'N', ndof, NB, nq, real_t(1), P, nq,
                      uloc.data(), nq, real_t(0), ytmp.data(), ndof);
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { break; }
            for (int i = 0; i < ndof; ++i)
            {
               Y[i + ndof * (vc + vdim * e)] +=
                  ytmp[static_cast<size_t>(i) +
                                              static_cast<size_t>(ndof) * b];
            }
         }
      }
      return;
   }
#endif
   for (int e = 0; e < NE; ++e)
   {
      for (int i = 0; i < ndof; ++i)
      {
         real_t yi = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            eval_t ye;
            InvokeQFn(qfn, ye, D[q + nq * e]);
            yi += P[q + nq * i] * real_t(ye);
         }
         Y[i + ndof * (vc + vdim * e)] += yi;
      }
   }
}

template <typename QFn, int MAP, int QND, int NDOF, int U_LD, int NB>
MFEM_HOST_DEVICE inline void LFBatchBody(
   QFn qfn,
   const int e0, const int NE,
   const real_t *p, const real_t *d, real_t *y,
   real_t *Us, const int vdim, const int vc,
   const int tid, const int nthreads)
{
   const auto D = ConstDeviceMatrix(d, QND, NE);
   SmemMatAcc<U_LD> Uacc{Us};
   YVdimAcc Yacc{y, NDOF, vdim, vc, e0};
   PAcc A{p, QND, NDOF};

   if (DeviceGemmEnabled())
   {
      // QFn fills U from point-local coeff (e.g. load D, or transform).
      ApplyNoneQFnSmem(qfn, Us, D, e0, NE, QND, U_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;
      GemmT<MAP>(QND, NDOF, NB, A, Uacc, Yacc, e0, NE);
   }
   else if (tid == 0)
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int q = 0; q < QND; ++q)
         {
            eval_t ye;
            InvokeQFn(qfn, ye, D(q, e));
            Us[q + U_LD * b] = real_t(ye);
         }
         for (int i = 0; i < NDOF; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < QND; ++q)
            {
               yi += p[q + QND * i] * Us[q + U_LD * b];
            }
            y[i + NDOF * (vc + vdim * e)] += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template <typename QFn>
MFEM_HOST_DEVICE inline void LFBatchBodyRuntime(
   QFn qfn,
   const int e0, const int NE,
   const int nq, const int ndof, const int u_ld, const int nb,
   const real_t *p, const real_t *d, real_t *y,
   real_t *Us, const int vdim, const int vc,
   const int tid, const int nthreads)
{
   constexpr int MAP = MmaMapDefault;
   const auto D = ConstDeviceMatrix(d, nq, NE);
   SmemMatAccRt Uacc{Us, u_ld};
   YVdimAcc Yacc{y, ndof, vdim, vc, e0};
   PAcc A{p, nq, ndof};

   if (DeviceGemmEnabled())
   {
      ApplyNoneQFnSmem(qfn, Us, D, e0, NE, nq, u_ld, nb, tid, nthreads);
      MFEM_SYNC_THREAD;
      GemmT<MAP>(nq, ndof, nb, A, Uacc, Yacc, e0, NE);
   }
   else if (tid == 0)
   {
      for (int b = 0; b < nb; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int q = 0; q < nq; ++q)
         {
            eval_t ye;
            InvokeQFn(qfn, ye, D(q, e));
            Us[q + u_ld * b] = real_t(ye);
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < nq; ++q)
            {
               yi += p[q + nq * i] * Us[q + u_ld * b];
            }
            y[i + ndof * (vc + vdim * e)] += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/** Specialized DomainLF-style Apply. Matches AssembleSimplexMmaKernelType.
    Launch: forall_3D + MFEM_SHARED U-only (preserves today's specialized quirk). */
template <typename QFn, int DIM, int D1D, int QND>
inline void ApplyLF(const int NE,
                    const Array<real_t> &basis,
                    const Vector &d,
                    real_t *y,
                    const int vdim,
                    const int vc)
{
   using Tr = qfn_traits<QFn>;
   static_assert(!Tr::load_x && !Tr::test_is_grad,
                 "ApplyLF requires none trial + Eval test");
   static_assert(D1D > 0 && QND > 0, "requires specialized D1D/QND");

   constexpr int NDOF = SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == QND * NE, "");
   MFEM_VERIFY(basis.Size() == QND * NDOF, "");
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   DumpFormApply<QFn, DIM, D1D, QND>("ApplyLF", NE, QND, NDOF);

   QFn qfn{};

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostLFApply(qfn, NE, QND, NDOF, basis.Read(), d.Read(), y, vdim, vc);
      return;
   }

   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int U_LD = PadLdBank<MAP>(MQ);
   constexpr int NB = MassLikeNB<DIM, D1D, QND>();
   VerifySharedMemBytes(int(sizeof(real_t)) * U_LD * NB);

   const auto P = basis.Read();
   const auto D = d.Read();
   const int nthreads = LaunchNthreads<QND>(QND, NDOF);
   const int nbatches = (NE + NB - 1) / NB;

   // Specialized: forall_3D + MFEM_SHARED (not forall_3D_smem) — DomainLF quirk.
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      struct alignas(16) Smem
      {
         real_t Us[U_LD * NB];
      };
      MFEM_SHARED Smem sm;
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      LFBatchBody<QFn, MAP, QND, NDOF, U_LD, NB>(
         qfn, batch * NB, NE, P, D, y, sm.Us, vdim, vc, tid, nthr);
   });
}

/** Runtime Fallback ApplyLF — forall_3D_smem + MassLikeNBRuntime. */
template <typename QFn, int DIM>
inline void ApplyLF(const int NE,
                    const Array<real_t> &basis,
                    const Vector &d,
                    real_t *y,
                    const int vdim,
                    const int vc)
{
   using Tr = qfn_traits<QFn>;
   static_assert(!Tr::load_x && !Tr::test_is_grad,
                 "ApplyLF runtime requires none trial + Eval test");

   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && basis.Size() % nq == 0, "");
   const int ndof = basis.Size() / nq;
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   constexpr int MAX_NQ = SimplexMaxNq<DIM, 0>();
   constexpr int MAX_NDOF = SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= MAX_NQ && ndof <= MAX_NDOF,
               "ApplyLF runtime exceeds size caps");

   DumpFormApplyRuntime<QFn, DIM>("ApplyLF", NE, nq, ndof);

   QFn qfn{};

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostLFApply(qfn, NE, nq, ndof, basis.Read(), d.Read(), y, vdim, vc);
      return;
   }

   const int u_ld = PadLdBankRuntime(nq);
   const int nb = MassLikeNBRuntime(ndof, nq);
   MFEM_VERIFY(u_ld <= PadLdBank<MmaMapDefault>(MAX_NQ) && nb <= NBATCH,
               "ApplyLF runtime smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * u_ld * nb;
   VerifySharedMemBytes(smem_bytes);

   const auto P = basis.Read();
   const auto D = d.Read();
   const int nthreads = LaunchNthreads(nq, ndof);
   const int nbatches = (NE + nb - 1) / nb;

   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
#if defined(__CUDA_ARCH__)
      real_t *Us = reinterpret_cast<real_t *>(SimplexMmaDynSmem());
#else
      constexpr int max_nb = NBATCH;
      constexpr int max_u_ld = PadLdBank<MmaMapDefault>(MAX_NQ);
      MFEM_SHARED real_t Us[max_u_ld * max_nb];
#endif
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      LFBatchBodyRuntime(qfn, batch * nb, NE, nq, ndof, u_ld, nb,
                         P, D, y, Us, vdim, vc, tid, nthr);
   });
}


// ===========================================================================
// Grad×Grad Apply
// ===========================================================================


// ---------------------------------------------------------------------------
// PA metric → full tensor (SYM packed or full DIM×DIM)
// ---------------------------------------------------------------------------

template <int DIM, bool SYM, typename TD>
MFEM_HOST_DEVICE inline void LoadMetricTensor(
   tensor<real_t, DIM, DIM> &A, TD D, int q, int e)
{
   if constexpr (DIM == 2)
   {
      const real_t O11 = D(q, 0, e);
      const real_t O21 = D(q, 1, e);
      if constexpr (SYM)
      {
         const real_t O22 = D(q, 2, e);
         A(0, 0) = O11; A(0, 1) = O21;
         A(1, 0) = O21; A(1, 1) = O22;
      }
      else
      {
         const real_t O12 = D(q, 2, e);
         const real_t O22 = D(q, 3, e);
         A(0, 0) = O11; A(0, 1) = O12;
         A(1, 0) = O21; A(1, 1) = O22;
      }
   }
   else
   {
      const real_t O11 = D(q, 0, e);
      const real_t O12 = D(q, 1, e);
      const real_t O13 = D(q, 2, e);
      if constexpr (SYM)
      {
         const real_t O22 = D(q, 3, e);
         const real_t O23 = D(q, 4, e);
         const real_t O33 = D(q, 5, e);
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O12; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O13; A(2, 1) = O23; A(2, 2) = O33;
      }
      else
      {
         const real_t O21 = D(q, 3, e);
         const real_t O22 = D(q, 4, e);
         const real_t O23 = D(q, 5, e);
         const real_t O31 = D(q, 6, e);
         const real_t O32 = D(q, 7, e);
         const real_t O33 = D(q, 8, e);
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O21; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O31; A(2, 1) = O32; A(2, 2) = O33;
      }
   }
}

/** Pointwise Grad QFn on smem UV[c*u_ld*nb + q_loc + u_ld*b]. */
template <int DIM, bool SYM, typename QFn, typename TD>
MFEM_HOST_DEVICE inline void ApplyGradQFnSmem(
   QFn qfn, real_t *UV, TD D,
   const int e0, const int NE,
   const int q0, const int nq_span, const int nq_total,
   const int u_ld, const int nb,
   const int tid, const int nthreads)
{
   for (int i = tid; i < nq_span * nb; i += nthreads)
   {
      const int b = i / nq_span;
      const int q_loc = i - b * nq_span;
      const int e = e0 + b;
      const int q_g = q0 + q_loc;
      if (e >= NE || q_g >= nq_total) { continue; }

      grad_t<DIM> u, y;
      for (int c = 0; c < DIM; ++c)
      {
         u[c] = UV[c * u_ld * nb + q_loc + u_ld * b];
      }
      tensor<real_t, DIM, DIM> A{};
      LoadMetricTensor<DIM, SYM>(A, D, q_g, e);
      InvokeQFn(qfn, u, y, A);
      for (int c = 0; c < DIM; ++c)
      {
         UV[c * u_ld * nb + q_loc + u_ld * b] = y[c];
      }
   }
}

// ---------------------------------------------------------------------------
// Dense element Grad apply with QFn (host / Emulate)
// ---------------------------------------------------------------------------

template <int DIM, bool SYM, typename QFn>
MFEM_HOST_DEVICE inline void GradApplyDenseElement(
   QFn qfn,
   const int nq, const int ndof,
   const real_t *G, const real_t *Dv_e,
   const real_t *X_e, real_t *Y_e, real_t *u_scratch)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   // U = G X
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
   // QFn
   for (int q = 0; q < nq; ++q)
   {
      grad_t<DIM> u, y;
      for (int d = 0; d < DIM; ++d) { u[d] = u_scratch[d * nq + q]; }
      // Wrap single-element PA as Reshape-compatible via raw pointer access
      struct D1
      {
         const real_t *base;
         int nq_, pa_;
         MFEM_HOST_DEVICE real_t operator()(int q, int pa, int /*e*/) const
         {
            return base[q + nq_ * pa];
         }
      };
      D1 Dacc{Dv_e, nq, PA_SIZE};
      tensor<real_t, DIM, DIM> A{};
      LoadMetricTensor<DIM, SYM>(A, Dacc, q, 0);
      InvokeQFn(qfn, u, y, A);
      for (int d = 0; d < DIM; ++d) { u_scratch[d * nq + q] = y[d]; }
   }
   // Y += G^T U
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

template <int DIM, bool SYM, typename QFn>
inline void HostGradApply(QFn qfn, const int NE, const int nq, const int ndof,
                          const real_t *G, const real_t *Dv,
                          const real_t *X, real_t *Y)
{
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(DIM * nq)));
      GradApplyDenseElement<DIM, SYM>(
         qfn, nq, ndof, G, Dv + nq * PA_SIZE * e,
         X + ndof * e, Y + ndof * e, u);
   }
}

template <int DIM, bool SYM, typename QFn>
inline void HostGradApply(QFn qfn, const int NE, const int nq, const int ndof,
                          const real_t *G, const real_t *Dv,
                          const real_t *X, real_t *Y, const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      HostGradApply<DIM, SYM>(qfn, NE, nq, ndof, G, Dv, X, Y);
      return;
   }
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(DIM * nq)));
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = ndof * (vc + vdim * e);
         GradApplyDenseElement<DIM, SYM>(
            qfn, nq, ndof, G, Dv + nq * PA_SIZE * e,
            X + off, Y + off, u);
      }
   }
}

// ---------------------------------------------------------------------------
// Device full-NQ / Q-tile batch bodies
// ---------------------------------------------------------------------------

template <int DIM, bool SYM, int MAP, typename QFn, typename TD, typename XMat>
MFEM_HOST_DEVICE inline void GradFullNqGemm(
   QFn qfn, real_t *XY, real_t *UV, TD D,
   const real_t *g, real_t *y, XMat X,
   const int e0, const int NE,
   const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const int tid, const int nthreads)
{
   SmemMatAccRt Xacc{XY, x_ld};
   YBatchAcc Yacc{y, ndof, e0};
   NullDAcc nullD;

   LoadXToSmem(XY, X, e0, NE, ndof, x_ld, nb, tid, nthreads);
   MFEM_SYNC_THREAD;

   for (int c = 0; c < DIM; ++c)
   {
      GAcc A{g, nq, ndof, c};
      SmemMatAccRt Uacc{UV + c * u_ld * nb, u_ld};
      Gemm<MAP, false>(nq, ndof, nb, A, Xacc, Uacc, nullD, e0, NE);
   }
   MFEM_SYNC_THREAD;
   ApplyGradQFnSmem<DIM, SYM>(qfn, UV, D, e0, NE, 0, nq, nq, u_ld, nb,
                              tid, nthreads);
   MFEM_SYNC_THREAD;
   for (int c = 0; c < DIM; ++c)
   {
      GAcc A{g, nq, ndof, c};
      SmemMatAccRt Vacc{UV + c * u_ld * nb, u_ld};
      GemmT<MAP>(nq, ndof, nb, A, Vacc, Yacc, e0, NE);
   }
}

template <bool SYM, int MAP, int X_LD, int U_LD, int NB, int QND, int TQ,
          typename QFn, typename TD, typename XMat>
MFEM_HOST_DEVICE inline void GradQTileGemm(
   QFn qfn, real_t *XY, real_t *UV, TD D,
   const real_t *g, real_t *y, XMat X,
   const int e0, const int NE, const int ndof,
   const int tid, const int nthreads)
{
   SmemMatAcc<X_LD> Xacc{XY};
   YBatchAcc Yacc{y, ndof, e0};
   SmemMatAcc<U_LD> U0{UV + 0 * U_LD * NB};
   SmemMatAcc<U_LD> U1{UV + 1 * U_LD * NB};
   SmemMatAcc<U_LD> U2{UV + 2 * U_LD * NB};

   LoadXToSmem(XY, X, e0, NE, ndof, X_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   for (int q0 = 0; q0 < QND; q0 += TQ)
   {
      const int nq_tile = (QND - q0 < TQ) ? (QND - q0) : TQ;
      GAccQTile A0{g, QND, ndof, 0, q0};
      GAccQTile A1{g, QND, ndof, 1, q0};
      GAccQTile A2{g, QND, ndof, 2, q0};
      Gemm3<MAP>(nq_tile, ndof, NB, A0, A1, A2, Xacc, U0, U1, U2, e0, NE);
      MFEM_SYNC_THREAD;
      ApplyGradQFnSmem<3, SYM>(qfn, UV, D, e0, NE, q0, nq_tile, QND, U_LD, NB,
                               tid, nthreads);
      MFEM_SYNC_THREAD;
      GemmT3<MAP>(nq_tile, ndof, NB, A0, A1, A2, U0, U1, U2, Yacc, e0, NE);
      MFEM_SYNC_THREAD;
   }
}

template <int DIM, int D1D, int QND, bool SYM, typename QFn>
MFEM_HOST_DEVICE inline void GradBatchBody(
   QFn qfn,
   const int e0, const int NE,
   const real_t *g, const real_t *d, const real_t *x, real_t *y)
{
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int NB = BatchNB<DIM, D1D, QND>();
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int ndof = BASIS;

   const auto D = Reshape(d, QND, PA_SIZE, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);
   const int tid = getThreadIdx();
   const int nthreads = getBlockNthreads();

   if constexpr (BatchUseQTile<DIM, D1D, QND>())
   {
      constexpr int TQ = BatchQTileFor<DIM, D1D, QND>();
      constexpr int U_LD = PadLdBank<MAP>(TQ);
      struct alignas(16) SmemQ
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
      MFEM_SIMPLEX_MMA_SMEM(SmemQ, sm);

      if (DeviceGemmEnabled())
      {
         GradQTileGemm<SYM, MAP, X_LD, U_LD, NB, QND, TQ>(
            qfn, sm.XY, sm.UV, D, g, y, X, e0, NE, ndof, tid, nthreads);
      }
      else if (tid == 0)
      {
         real_t u_scratch[DIM * MQ];
         auto Y = DeviceMatrix(y, ndof, NE);
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < ndof; ++i)
            {
               sm.XY[i + X_LD * b] = X(i, e);
            }
            GradApplyDenseElement<DIM, SYM>(
               qfn, QND, ndof, g, &D(0, 0, e),
               &sm.XY[X_LD * b], &Y(0, e), u_scratch);
         }
      }
      MFEM_SYNC_THREAD;
   }
   else
   {
      constexpr int U_LD = PadLdBank<MAP>(MQ);
      struct alignas(16) Smem
      {
         real_t XY[X_LD * NB];
         real_t UV[DIM * U_LD * NB];
      };
      MFEM_SIMPLEX_MMA_SMEM(Smem, sm);

      if (DeviceGemmEnabled())
      {
         GradFullNqGemm<DIM, SYM, MAP>(
            qfn, sm.XY, sm.UV, D, g, y, X, e0, NE, QND, ndof,
            X_LD, U_LD, NB, tid, nthreads);
      }
      else if (tid == 0)
      {
         real_t u_scratch[DIM * MQ];
         auto Y = DeviceMatrix(y, ndof, NE);
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < ndof; ++i)
            {
               sm.XY[i + X_LD * b] = X(i, e);
            }
            GradApplyDenseElement<DIM, SYM>(
               qfn, QND, ndof, g, &D(0, 0, e),
               &sm.XY[X_LD * b], &Y(0, e), u_scratch);
         }
      }
      MFEM_SYNC_THREAD;
   }
}

// ---------------------------------------------------------------------------
// Public Apply for Grad×Grad QFns (generic; integrator supplies QFn + traits)
// ---------------------------------------------------------------------------

/** Specialized Grad×Grad Apply. */
template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::test_is_grad, "Grad Apply requires Grad×Grad QFn");
   static_assert(D1D > 0 && QND > 0, "requires specialized D1D/QND");
   static_assert(Tr::spatial_dim == DIM, "QFn DIM must match Apply DIM");

   constexpr bool SYM = Tr::symmetric_pa;
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int ndof = SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == PA_SIZE * QND * NE, "");
   MFEM_VERIFY(basis.Size() == QND * ndof * DIM, "");
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   DumpFormApply<QFn, DIM, D1D, QND>("ApplyGrad", NE, QND, ndof);

   QFn qfn{};

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostGradApply<DIM, SYM>(qfn, NE, QND, ndof,
                              basis.Read(), d.Read(), x.Read(), y.ReadWrite());
      return;
   }

   const SmemPlan plan = MakeGradPlan<DIM, D1D, QND>();
   VerifySharedMemBytes(plan.smem_bytes);

   const auto G = basis.Read(), Dv = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   const int nbatches = (NE + plan.nb - 1) / plan.nb;

   mfem::forall_3D_smem(nbatches, plan.nthreads, 1, 1, plan.smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      GradBatchBody<DIM, D1D, QND, SYM>(
         qfn, batch * plan.nb, NE, G, Dv, X, Y);
   });
}

/** Specialized Grad×Grad with block-diag vdim. */
template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y,
      const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      Apply<QFn, DIM, D1D, QND>(NE, basis, d, x, y);
      return;
   }

   using Tr = qfn_traits<QFn>;
   static_assert(Tr::test_is_grad, "");
   static_assert(D1D > 0 && QND > 0, "");
   static_assert(Tr::spatial_dim == DIM, "");

   constexpr bool SYM = Tr::symmetric_pa;
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   constexpr int ndof = SimplexNdof<DIM, D1D>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   MFEM_VERIFY(NE > 0 && d.Size() == PA_SIZE * QND * NE, "");
   MFEM_VERIFY(basis.Size() == QND * ndof * DIM, "");
   MFEM_VERIFY(x.Size() >= ndof * vdim * NE && y.Size() >= ndof * vdim * NE, "");

   DumpFormApply<QFn, DIM, D1D, QND>("ApplyGrad", NE, QND, ndof);

   QFn qfn{};
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostGradApply<DIM, SYM>(qfn, NE, QND, ndof,
                              basis.Read(), d.Read(), x.Read(), y.ReadWrite(),
                              vdim);
      return;
   }

   const auto G = basis.Read(), Dv = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[DIM * MQ];
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = ndof * (vc + vdim * e);
         GradApplyDenseElement<DIM, SYM>(
            qfn, QND, ndof, G, Dv + QND * PA_SIZE * e,
            X + off, Y + off, u);
      }
   });
}

/** Runtime Fallback Grad Apply — full-NQ only; Q-tile falls back to dense. */
template <typename QFn, int DIM>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::test_is_grad, "Grad Apply runtime requires Grad×Grad QFn");
   static_assert(Tr::spatial_dim == DIM, "QFn DIM must match Apply DIM");

   constexpr bool SYM = Tr::symmetric_pa;
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;

   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % (PA_SIZE * NE) == 0, "");
   const int nq = d.Size() / (PA_SIZE * NE);
   MFEM_VERIFY(nq > 0 && basis.Size() % (nq * DIM) == 0, "");
   const int ndof = basis.Size() / (nq * DIM);
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int max_nq = SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Grad Apply runtime exceeds size caps");

   DumpFormApplyRuntime<QFn, DIM>("ApplyGrad", NE, nq, ndof);

   QFn qfn{};

   if (!Device::Allows(Backend::DEVICE_MASK) ||
       BatchUseQTileRuntime(DIM, ndof, nq, DIM))
   {
      // Host or runtime Q-tile: dense per-element path.
      if (!Device::Allows(Backend::DEVICE_MASK))
      {
         HostGradApply<DIM, SYM>(qfn, NE, nq, ndof,
                                 basis.Read(), d.Read(), x.Read(),
                                 y.ReadWrite());
         return;
      }
      const auto G = basis.Read();
      const auto Dv = d.Read();
      const auto X = x.Read();
      auto Y = y.ReadWrite();
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         real_t u[DIM * max_nq];
         GradApplyDenseElement<DIM, SYM>(
            qfn, nq, ndof, G, Dv + nq * PA_SIZE * e,
            X + ndof * e, Y + ndof * e, u);
      });
      return;
   }

   const int x_ld = PadLdBankRuntime(ndof);
   const int u_ld = PadLdBankRuntime(nq);
   const int nb = BatchNBRuntime(DIM, ndof, nq, DIM);
   MFEM_VERIFY(x_ld <= PadLdBank<MmaMapDefault>(max_ndof) &&
               u_ld <= PadLdBank<MmaMapDefault>(max_nq) &&
               nb <= NBATCH,
               "Grad Apply runtime smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * (x_ld + DIM * u_ld) * nb;
   VerifySharedMemBytes(smem_bytes);

   const auto G = basis.Read(), Dv = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   const int nthreads = LaunchNthreads(nq, ndof);
   const int nbatches = (NE + nb - 1) / nb;

   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
#if defined(__CUDA_ARCH__)
      real_t *XY = reinterpret_cast<real_t *>(SimplexMmaDynSmem());
      real_t *UV = XY + x_ld * nb;
#else
      constexpr int max_nb = NBATCH;
      constexpr int max_x_ld = PadLdBank<MmaMapDefault>(max_ndof);
      constexpr int max_u_ld = PadLdBank<MmaMapDefault>(max_nq);
      MFEM_SHARED real_t XY[max_x_ld * max_nb];
      MFEM_SHARED real_t UV[DIM * max_u_ld * max_nb];
#endif
      const int tid = getThreadIdx();
      const int nthr = getBlockNthreads();
      const auto D = Reshape(Dv, nq, PA_SIZE, NE);
      const auto Xm = ConstDeviceMatrix(X, ndof, NE);
      if (DeviceGemmEnabled())
      {
         GradFullNqGemm<DIM, SYM, MmaMapDefault>(
            qfn, XY, UV, D, G, Y, Xm, batch * nb, NE, nq, ndof,
            x_ld, u_ld, nb, tid, nthr);
      }
      else if (tid == 0)
      {
         real_t u_scratch[DIM * max_nq];
         auto Ym = DeviceMatrix(Y, ndof, NE);
         for (int b = 0; b < nb; ++b)
         {
            const int e = batch * nb + b;
            if (e >= NE) { continue; }
            for (int i = 0; i < ndof; ++i)
            {
               XY[i + x_ld * b] = Xm(i, e);
            }
            GradApplyDenseElement<DIM, SYM>(
               qfn, nq, ndof, G, &D(0, 0, e),
               &XY[x_ld * b], &Ym(0, e), u_scratch);
         }
      }
      MFEM_SYNC_THREAD;
   });
}

/** Runtime Grad×Grad with block-diag vdim. */
template <typename QFn, int DIM>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
Apply(const int NE,
      const Array<real_t> &basis,
      const Vector &d,
      const Vector &x,
      Vector &y,
      const int vdim)
{
   MFEM_VERIFY(vdim >= 1, "");
   if (vdim == 1)
   {
      Apply<QFn, DIM>(NE, basis, d, x, y);
      return;
   }

   using Tr = qfn_traits<QFn>;
   static_assert(Tr::test_is_grad, "");
   static_assert(Tr::spatial_dim == DIM, "");

   constexpr bool SYM = Tr::symmetric_pa;
   constexpr int PA_SIZE = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;

   MFEM_VERIFY(NE > 0 && d.Size() % (PA_SIZE * NE) == 0, "");
   const int nq = d.Size() / (PA_SIZE * NE);
   MFEM_VERIFY(nq > 0 && basis.Size() % (nq * DIM) == 0, "");
   const int ndof = basis.Size() / (nq * DIM);
   MFEM_VERIFY(x.Size() >= ndof * vdim * NE && y.Size() >= ndof * vdim * NE, "");

   constexpr int max_nq = SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof, "");

   DumpFormApplyRuntime<QFn, DIM>("ApplyGrad", NE, nq, ndof);

   QFn qfn{};
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      HostGradApply<DIM, SYM>(qfn, NE, nq, ndof,
                              basis.Read(), d.Read(), x.Read(), y.ReadWrite(),
                              vdim);
      return;
   }

   const auto G = basis.Read(), Dv = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[DIM * max_nq];
      for (int vc = 0; vc < vdim; ++vc)
      {
         const int off = ndof * (vc + vdim * e);
         GradApplyDenseElement<DIM, SYM>(
            qfn, nq, ndof, G, Dv + nq * PA_SIZE * e,
            X + off, Y + off, u);
      }
   });
}


} // namespace mfem::internal::mma::form

/// \endcond
