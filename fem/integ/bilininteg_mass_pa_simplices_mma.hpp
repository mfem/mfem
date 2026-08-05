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

#include "../bilininteg.hpp"
#include "bilininteg_pa_mma.hpp"

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
// Simplex mass PA helpers (host dense / device batch)
// ---------------------------------------------------------------------------

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

namespace lapack
{

#ifdef MFEM_USE_LAPACK
/** Mass: serial tiles, reused buffers. U = P X, scale D, Y += P^T U.
    Full tiles GEMM against X/Y; partial trailing tile packs/scatters. */
inline void MassApply(int NE, int nq, int ndof, const real_t *P,
                             const real_t *D, const real_t *X, real_t *Y)
{
   const int NB = lapack::NB(nq, ndof);
   std::vector<real_t> uloc(static_cast<size_t>(nq) * NB);
   lapack::ElementTiles(NE, ndof, NB, X, Y,
                       [&](int e0, int /*nbe*/, int nb, const real_t *Xsrc,
                           real_t *Yout)
   {
      lapack::Gemm('N', 'N', nq, nb, ndof, real_t(1), P, nq, Xsrc, ndof,
                  real_t(0), uloc.data(), nq);
      ScaleUByMassD(uloc.data(), D, nq, e0, NE, nb);
      lapack::Gemm('T', 'N', ndof, nb, nq, real_t(1), P, nq, uloc.data(), nq,
                  real_t(1), Yout, ndof);
   });
}

#endif // MFEM_USE_LAPACK

/** Host entry: runs MassApply when LAPACK is on and Prefer is true. */
inline bool TryMassApply(int NE, int nq, int ndof, const real_t *P,
                         const real_t *D, const real_t *X, real_t *Y)
{
#ifdef MFEM_USE_LAPACK
   if (!Prefer(nq, ndof, NE)) { return false; }
   MassApply(NE, nq, ndof, P, D, X, Y);
   return true;
#else
   (void)NE; (void)nq; (void)ndof; (void)P; (void)D; (void)X; (void)Y;
   return false;
#endif
}


} // namespace lapack

namespace blas
{

/** Multi-RHS NB for specialized mass (keep U in L1 on large 3D). */
template <int DIM, int NQ>
constexpr int MassNB()
{
   if constexpr (DIM == 2) { return (NQ == 12) ? 16 : 32; }
   return (NQ > 80) ? 4 : 8;
}

template <int DIM, int NDOF, int NQ>
inline void MassApply(int NE, const real_t *P, const real_t *D,
                      const real_t *X, real_t *Y)
{
   constexpr int NB = MassNB<DIM, NQ>();
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t uloc[NQ * NB];
      if (e0 + NB <= NE)
      {
         GemmFromColMajor<NDOF, NQ, NB, true>(P, X, e0, uloc, D);
         GemmTFull<NDOF, NQ, NB>(P, uloc, Y, e0);
      }
      else
      {
         alignas(64) real_t xloc[NDOF * NB];
         PackX<NDOF, NB>(X, e0, NE, xloc);
         Gemm<NDOF, NQ, NB, true>(P, xloc, uloc, D, e0, NE);
         GemmT<NDOF, NQ, NB>(P, uloc, Y, e0, NE);
      }
   }
}

} // namespace blas

// ---- Portable dense mass (device stub / single / runtime Fallback) ---------

/** Serial dense: Y_e += P^T (D_e ⊙ (P X_e)). u_scratch must hold nq reals. */
MFEM_HOST_DEVICE inline void MassApplyDenseElement(const int nq, const int ndof,
                                                   const real_t *P,
                                                   const real_t *D_e,
                                                   const real_t *X_e,
                                                   real_t *Y_e,
                                                   real_t *u_scratch)
{
   for (int q = 0; q < nq; ++q)
   {
      real_t s = 0.0;
      for (int i = 0; i < ndof; ++i)
      {
         s += P[q + nq * i] * X_e[i];
      }
      u_scratch[q] = s * D_e[q];
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

/** Host multi-element driver over MassApplyDenseElement. */
inline void MassApplyRuntime(int NE, int nq, int ndof, const real_t *P,
                                  const real_t *D, const real_t *X, real_t *Y)
{
   for (int e = 0; e < NE; ++e)
   {
      auto *u = static_cast<real_t *>(
                   alloca(sizeof(real_t) * static_cast<size_t>(nq)));
      MassApplyDenseElement(nq, ndof, P, D + nq * e, X + ndof * e,
                            Y + ndof * e, u);
   }
}

/** Batch dense mass using padded smem tiles XY[X_LD*NB], Us[U_LD*NB].
    Only tid == 0 performs work; callers must MFEM_SYNC_THREAD afterward. */
template <int QND, int NDOF, int X_LD, int U_LD, int NB>
MFEM_HOST_DEVICE inline void MassBatchApplyDense(const int e0, const int NE,
                                                 const real_t *p,
                                                 const real_t *d,
                                                 const real_t *x,
                                                 real_t *y,
                                                 real_t *XY, real_t *Us,
                                                 const int tid)
{
   if (tid != 0) { return; }
   const auto D = ConstDeviceMatrix(d, QND, NE);
   const auto X = ConstDeviceMatrix(x, NDOF, NE);
   auto Y = DeviceMatrix(y, NDOF, NE);
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { continue; }
      for (int i = 0; i < X_LD; ++i)
      {
         XY[i + X_LD * b] = (i < NDOF) ? X(i, e) : real_t(0);
      }
      MassApplyDenseElement(QND, NDOF, p, &D(0, e), &XY[X_LD * b],
                            &Y(0, e), &Us[U_LD * b]);
   }
}

/** Tensor MMA / MFMA batch mass: LoadX + Gemm + GemmT. */
template <int MAP, int QND, int NDOF, int X_LD, int U_LD, int NB>
MFEM_HOST_DEVICE inline void MmaMassBatchApply(const int e0, const int NE,
                                               const real_t *p,
                                               const real_t *d,
                                               const real_t *x,
                                               real_t *y,
                                               real_t *XY, real_t *Us,
                                               const int tid,
                                               const int nthreads)
{
   const auto D = ConstDeviceMatrix(d, QND, NE);
   const auto X = ConstDeviceMatrix(x, NDOF, NE);
   SmemMatAcc<X_LD> Xacc{XY};
   SmemMatAcc<U_LD> Uacc{Us};
   YBatchAcc Yacc{y, NDOF, e0};

   LoadXToSmem(XY, X, e0, NE, NDOF, X_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   PAcc A{p, QND, NDOF};
   Gemm<MAP, true>(QND, NDOF, NB, A, Xacc, Uacc, D, e0, NE);
   MFEM_SYNC_THREAD;
   GemmT<MAP>(QND, NDOF, NB, A, Uacc, Yacc, e0, NE);
}

/** Runtime-sized dense batch mass (tid==0); sync afterward. */
MFEM_HOST_DEVICE inline void MassBatchApplyRuntime(
   const int e0, const int NE, const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const real_t *p, const real_t *d, const real_t *x, real_t *y,
   real_t *XY, real_t *Us, const int tid)
{
   if (tid != 0) { return; }
   const auto D = ConstDeviceMatrix(d, nq, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);
   auto Y = DeviceMatrix(y, ndof, NE);
   for (int b = 0; b < nb; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { continue; }
      for (int i = 0; i < x_ld; ++i)
      {
         XY[i + x_ld * b] = (i < ndof) ? X(i, e) : real_t(0);
      }
      MassApplyDenseElement(nq, ndof, p, &D(0, e), &XY[x_ld * b],
                            &Y(0, e), &Us[u_ld * b]);
   }
}

/** Runtime-sized MMA / Emulate batch mass (MmaMapDefault). */
MFEM_HOST_DEVICE inline void MmaMassBatchApplyRuntime(
   const int e0, const int NE, const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const real_t *p, const real_t *d, const real_t *x, real_t *y,
   real_t *XY, real_t *Us, const int tid, const int nthreads)
{
   constexpr int MAP = MmaMapDefault;
   const auto D = ConstDeviceMatrix(d, nq, NE);
   const auto X = ConstDeviceMatrix(x, ndof, NE);
   SmemMatAccRt Xacc{XY, x_ld};
   SmemMatAccRt Uacc{Us, u_ld};
   YBatchAcc Yacc{y, ndof, e0};

   LoadXToSmem(XY, X, e0, NE, ndof, x_ld, nb, tid, nthreads);
   MFEM_SYNC_THREAD;

   PAcc A{p, nq, ndof};
   Gemm<MAP, true>(nq, ndof, nb, A, Xacc, Uacc, D, e0, NE);
   MFEM_SYNC_THREAD;
   GemmT<MAP>(nq, ndof, nb, A, Uacc, Yacc, e0, NE);
}

} // namespace mma

/** Host dense mass (Lapack or Blas): y += P^T ( D ⊙ (P x) ) per element.
    Large (QND,ndof): BLAS multi-RHS when profitable; else hand tiles. */
template<int DIM, int D1D, int QND>
inline void MassApplySimplex(const int NE,
                                  const Array<real_t> &p,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA mass requires specialized D1D/QND");
   constexpr int ndof = mma::SimplexNdof<DIM, D1D>();
   const real_t *P = p.Read();
   const real_t *D = d.Read();
   const real_t *X = x.Read();
   real_t *Y = y.ReadWrite();

   if (mma::lapack::TryMassApply(NE, QND, ndof, P, D, X, Y)) { return; }
   mma::blas::MassApply<DIM, ndof, QND>(NE, P, D, X, Y);
}

/** Portable runtime dense mass for unspecialized (D1D,QND). Works on CPU and
    GPU; sizes inferred from P/D; bounded by SimplexMaxNq/SimplexNdof caps. */
template<int DIM>
inline void MassApplySimplexRuntime(const int NE,
                                         const Array<real_t> &p,
                                         const Vector &d,
                                         const Vector &x,
                                         Vector &y)
{
   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && p.Size() % nq == 0, "");
   const int ndof = p.Size() / nq;
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA mass runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      const real_t *P = p.Read();
      const real_t *D = d.Read();
      const real_t *X = x.Read();
      real_t *Y = y.ReadWrite();
      if (mma::lapack::TryMassApply(NE, nq, ndof, P, D, X, Y)) { return; }
      mma::MassApplyRuntime(NE, nq, ndof, P, D, X, Y);
      return;
   }

   const auto P = p.Read();
   const auto D = d.Read();
   const auto X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[max_nq];
      mma::MassApplyDenseElement(nq, ndof, P, D + nq * e,
                                 X + ndof * e, Y + ndof * e, u);
   });
}

template<int DIM, int D1D, int QND>
MFEM_HOST_DEVICE inline
void MmaMassApplySimplex_Batch(const int e0,
                               const int NE,
                               const real_t *p,
                               const real_t *d,
                               const real_t *x,
                               real_t *y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA mass requires specialized D1D/QND");
   constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
   constexpr int BASIS_DIM = mma::SimplexNdof<DIM, D1D>();
   constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = mma::PadLdBank<MAP>(BASIS_DIM);
   constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
   constexpr int NB = mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = BASIS_DIM;

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
   MFEM_SIMPLEX_MMA_SMEM(Smem, sm);

   const int tid = mma::getThreadIdx();
   const int nthreads = mma::getBlockNthreads();

   if constexpr (mma::TensorMmaEnabled())
   {
      mma::MmaMassBatchApply<MAP, QND, ndof, X_LD, U_LD, NB>(
         e0, NE, p, d, x, y, sm.XY, sm.Us, tid, nthreads);
   }
   else
   {
      mma::MassBatchApplyDense<QND, ndof, X_LD, U_LD, NB>(
         e0, NE, p, d, x, y, sm.XY, sm.Us, tid);
      MFEM_SYNC_THREAD;
   }
}

/** Runtime-sized batch body for Fallback (MmaMapDefault, dyn/static smem). */
template<int DIM> MFEM_HOST_DEVICE inline
void MmaMassApplySimplex_Batch(const int e0,
                               const int NE,
                               const int nq,
                               const int ndof,
                               const int x_ld,
                               const int u_ld,
                               const int nb,
                               const real_t *p,
                               const real_t *d,
                               const real_t *x,
                               real_t *y)
{
   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();

   // Dyn layout matches launch smem_bytes=(x_ld+u_ld)*nb; static uses caps.
#if defined(__CUDA_ARCH__)
   real_t *XY = reinterpret_cast<real_t *>(mma::SimplexMmaDynSmem());
   real_t *Us = XY + x_ld * nb;
#else
   constexpr int max_nb = mma::NBATCH;
   constexpr int max_x_ld =
      mma::PadLdBank<mma::MmaMapDefault>(max_ndof);
   constexpr int max_u_ld =
      mma::PadLdBank<mma::MmaMapDefault>(max_nq);
   MFEM_SHARED real_t XY[max_x_ld * max_nb];
   MFEM_SHARED real_t Us[max_u_ld * max_nb];
#endif

   const int tid = mma::getThreadIdx();
   const int nthreads = mma::getBlockNthreads();

   if constexpr (mma::TensorMmaEnabled())
   {
      mma::MmaMassBatchApplyRuntime(
         e0, NE, nq, ndof, x_ld, u_ld, nb, p, d, x, y,
         XY, Us, tid, nthreads);
   }
   else
   {
      mma::MassBatchApplyRuntime(
         e0, NE, nq, ndof, x_ld, u_ld, nb, p, d, x, y, XY, Us, tid);
      MFEM_SYNC_THREAD;
   }
}

template<int DIM, int D1D, int QND>
inline void MmaMassApplySimplex(const int NE,
                                const Array<real_t> &p,
                                const Vector &d,
                                const Vector &x,
                                Vector &y)
{
   static_assert(D1D > 0 && QND > 0,
                 "Simplex MMA mass requires specialized D1D/QND");
   constexpr int NB = mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int NDOF = mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == QND * NE, "");
   MFEM_VERIFY(p.Size() == QND * NDOF, "");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      MassApplySimplex<DIM, D1D, QND>(NE, p, d, x, y);
      return;
   }

   constexpr int BASIS = NDOF;
   constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
   constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = mma::PadLdBank<MAP>(BASIS);
   constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
   constexpr int smem_bytes = int(sizeof(real_t)) * (X_LD + U_LD) * NB;
   mma::VerifySharedMemBytes(smem_bytes);

   const int nbatches = (NE + NB - 1) / NB;
   const int nthreads = mma::LaunchNthreads<QND>(QND, NDOF);

   const auto P = p.Read(), D = d.Read(), X = x.Read();
   auto Y = y.ReadWrite();

   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaMassApplySimplex_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, X, Y);
   });
}

/** Runtime Fallback shell: host dense BLAS/hand; device batched MMA/Dense. */
template<int DIM>
inline void MmaMassApplySimplex(const int NE,
                                const Array<real_t> &p,
                                const Vector &d,
                                const Vector &x,
                                Vector &y)
{
   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && p.Size() % nq == 0, "");
   const int ndof = p.Size() / nq;
   MFEM_VERIFY(x.Size() >= ndof * NE && y.Size() >= ndof * NE, "");

   constexpr int MAX_NQ = mma::SimplexMaxNq<DIM, 0>();
   constexpr int MAX_NDOF = mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= MAX_NQ && ndof <= MAX_NDOF,
               "Simplex MMA mass runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      const real_t *P = p.Read();
      const real_t *D = d.Read();
      const real_t *X = x.Read();
      real_t *Y = y.ReadWrite();
      if (mma::lapack::TryMassApply(NE, nq, ndof, P, D, X, Y)) { return; }
      mma::MassApplyRuntime(NE, nq, ndof, P, D, X, Y);
      return;
   }

   const int x_ld = mma::PadLdBankRuntime(ndof);
   const int u_ld = mma::PadLdBankRuntime(nq);
   const int nb = mma::MassLikeNBRuntime(ndof, nq);
   MFEM_VERIFY(x_ld <= mma::PadLdBank<mma::MmaMapDefault>
               (MAX_NDOF) &&
               u_ld <= mma::PadLdBank<mma::MmaMapDefault>(MAX_NQ) &&
               nb <= mma::NBATCH,
               "Simplex MMA mass runtime Fallback smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * (x_ld + u_ld) * nb;
   mma::VerifySharedMemBytes(smem_bytes);

   const auto P = p.Read();
   const auto D = d.Read();
   const auto X = x.Read();
   auto Y = y.ReadWrite();

   const int nthreads = mma::LaunchNthreads(nq, ndof);
   const int nbatches = (NE + nb - 1) / nb;
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaMassApplySimplex_Batch<DIM>(
         batch * nb, NE, nq, ndof, x_ld, u_ld, nb, P, D, X, Y);
   });
}

} // namespace internal

template<int DIM, int D1D, int QND>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::MmaMassApplySimplex<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::MmaMassApplySimplex<3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA mass only supports DIM 2 or 3");
      return nullptr;
   }
}

inline MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA mass PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return internal::MmaMassApplySimplex<2>;
   }
   return internal::MmaMassApplySimplex<3>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
