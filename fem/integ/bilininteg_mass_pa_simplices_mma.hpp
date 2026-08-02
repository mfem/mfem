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

namespace simplex_mma
{

// ---------------------------------------------------------------------------
// Simplex mass PA helpers (host dense / device batch)
// ---------------------------------------------------------------------------

/** Hand multi-RHS NB for specialized mass (keep U in L1 on large 3D). */
template <int DIM, int NQ>
constexpr int HandMassNB()
{
   return (DIM == 3 && NQ > 80) ? 4 : 8;
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

#endif // MFEM_USE_LAPACK

/** Always-available host BLAS entry: runs MassApplyBlas when LAPACK is on and
    PreferHostBlas is true. Returns whether the BLAS path ran. */
inline bool TryMassApplyBlas(int NE, int nq, int ndof, const real_t *P,
                             const real_t *D, const real_t *X, real_t *Y)
{
#ifdef MFEM_USE_LAPACK
   if (!PreferHostBlas(nq, ndof)) { return false; }
   MassApplyBlas(NE, nq, ndof, P, D, X, Y);
   return true;
#else
   (void)NE; (void)nq; (void)ndof; (void)P; (void)D; (void)X; (void)Y;
   return false;
#endif
}


template <int DIM, int NDOF, int NQ>
inline void MassApplyHandSpecialized(int NE, const real_t *P, const real_t *D,
                                     const real_t *X, real_t *Y)
{
   constexpr int NB = HandMassNB<DIM, NQ>();
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      alignas(64) real_t xloc[NDOF * NB];
      alignas(64) real_t uloc[NQ * NB];
      PackXHand<NDOF, NB>(X, e0, NE, xloc);
      HandGemmForward<NDOF, NQ, NB, true>(P, xloc, uloc, D, e0, NE);
      HandGemmBackward<NDOF, NQ, NB>(P, uloc, Y, e0, NE);
   }
}


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
inline void MassApplyHandRuntime(int NE, int nq, int ndof, const real_t *P,
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
                                                 const real_t *p_,
                                                 const real_t *d_,
                                                 const real_t *x_,
                                                 real_t *y_,
                                                 real_t *XY, real_t *Us,
                                                 const int tid)
{
   if (tid != 0) { return; }
   const auto D = ConstDeviceMatrix(d_, QND, NE);
   const auto x = ConstDeviceMatrix(x_, NDOF, NE);
   auto Y = DeviceMatrix(y_, NDOF, NE);
   for (int b = 0; b < NB; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { continue; }
      for (int i = 0; i < X_LD; ++i)
      {
         XY[i + X_LD * b] = (i < NDOF) ? x(i, e) : real_t(0);
      }
      MassApplyDenseElement(QND, NDOF, p_, &D(0, e), &XY[X_LD * b],
                            &Y(0, e), &Us[U_LD * b]);
   }
}

/** Tensor MMA / MFMA batch mass: LoadX + BasisGemmForward + BasisGemmT. */
template <int MAP, int QND, int NDOF, int X_LD, int U_LD, int NB>
MFEM_HOST_DEVICE inline void MassBatchApplyMma(const int e0, const int NE,
                                               const real_t *p_,
                                               const real_t *d_,
                                               const real_t *x_,
                                               real_t *y_,
                                               real_t *XY, real_t *Us,
                                               const int tid,
                                               const int nthreads)
{
   const auto D = ConstDeviceMatrix(d_, QND, NE);
   const auto x = ConstDeviceMatrix(x_, NDOF, NE);
   SmemMatAcc<X_LD> Xacc{XY};
   SmemMatAcc<U_LD> Uacc{Us};
   YBatchAcc Yacc{y_, NDOF, e0};

   LoadXToSmem(XY, x, e0, NE, NDOF, X_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   PAcc A{p_, QND, NDOF};
   BasisGemmForward<MAP, true>(QND, NDOF, NB, A, Xacc, Uacc, D, e0, NE);
   MFEM_SYNC_THREAD;
   BasisGemmT<MAP>(QND, NDOF, NB, A, Uacc, Yacc, e0, NE);
}

/** Runtime-sized dense batch mass (tid==0); sync afterward. */
MFEM_HOST_DEVICE inline void MassBatchApplyDenseRuntime(
   const int e0, const int NE, const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const real_t *p_, const real_t *d_, const real_t *x_, real_t *y_,
   real_t *XY, real_t *Us, const int tid)
{
   if (tid != 0) { return; }
   const auto D = ConstDeviceMatrix(d_, nq, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   auto Y = DeviceMatrix(y_, ndof, NE);
   for (int b = 0; b < nb; ++b)
   {
      const int e = e0 + b;
      if (e >= NE) { continue; }
      for (int i = 0; i < x_ld; ++i)
      {
         XY[i + x_ld * b] = (i < ndof) ? x(i, e) : real_t(0);
      }
      MassApplyDenseElement(nq, ndof, p_, &D(0, e), &XY[x_ld * b],
                            &Y(0, e), &Us[u_ld * b]);
   }
}

/** Runtime-sized MMA / Emulate batch mass (MmaMapDefault). */
MFEM_HOST_DEVICE inline void MassBatchApplyMmaRuntime(
   const int e0, const int NE, const int nq, const int ndof,
   const int x_ld, const int u_ld, const int nb,
   const real_t *p_, const real_t *d_, const real_t *x_, real_t *y_,
   real_t *XY, real_t *Us, const int tid, const int nthreads)
{
   constexpr int MAP = MmaMapDefault;
   const auto D = ConstDeviceMatrix(d_, nq, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   SmemMatAccRt Xacc{XY, x_ld};
   SmemMatAccRt Uacc{Us, u_ld};
   YBatchAcc Yacc{y_, ndof, e0};

   LoadXToSmem(XY, x, e0, NE, ndof, x_ld, nb, tid, nthreads);
   MFEM_SYNC_THREAD;

   PAcc A{p_, nq, ndof};
   BasisGemmForward<MAP, true>(nq, ndof, nb, A, Xacc, Uacc, D, e0, NE);
   MFEM_SYNC_THREAD;
   BasisGemmT<MAP>(nq, ndof, nb, A, Uacc, Yacc, e0, NE);
}


} // namespace simplex_mma

/** Host-optimized dense mass: y += P^T ( D ⊙ (P x) ) per element.
    Large (QND,ndof): BLAS multi-RHS when profitable; else hand tiles. */
template<int DIM, int D1D, int QND>
inline void PAMassApplySimplexDenseHost(const int NE,
                                        const Array<real_t> &p_,
                                        const Vector &d_,
                                        const Vector &x_,
                                        Vector &y_)
{
   static_assert(D1D > 0 && QND > 0, "Simplex MMA mass requires specialized D1D/QND");
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();
   const real_t *P = p_.Read();
   const real_t *D = d_.Read();
   const real_t *X = x_.Read();
   real_t *Y = y_.ReadWrite();

   if (simplex_mma::TryMassApplyBlas(NE, QND, ndof, P, D, X, Y)) { return; }
   simplex_mma::MassApplyHandSpecialized<DIM, ndof, QND>(NE, P, D, X, Y);
}

/** Portable runtime dense mass for unspecialized (D1D,QND). Works on CPU and
    GPU; sizes inferred from P/D; bounded by SimplexMaxNq/SimplexNdof caps. */
template<int DIM>
inline void PAMassApplySimplexDenseRuntime(const int NE,
                                           const Array<real_t> &p_,
                                           const Vector &d_,
                                           const Vector &x_,
                                           Vector &y_)
{
   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d_.Size() % NE == 0, "");
   const int nq = d_.Size() / NE;
   MFEM_VERIFY(nq > 0 && p_.Size() % nq == 0, "");
   const int ndof = p_.Size() / nq;
   MFEM_VERIFY(x_.Size() >= ndof * NE && y_.Size() >= ndof * NE, "");

   constexpr int max_nq = simplex_mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = simplex_mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA mass runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      const real_t *P = p_.Read();
      const real_t *D = d_.Read();
      const real_t *X = x_.Read();
      real_t *Y = y_.ReadWrite();
      if (simplex_mma::TryMassApplyBlas(NE, nq, ndof, P, D, X, Y)) { return; }
      simplex_mma::MassApplyHandRuntime(NE, nq, ndof, P, D, X, Y);
      return;
   }

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t u[max_nq];
      simplex_mma::MassApplyDenseElement(nq, ndof, P, D + nq * e,
                                         X + ndof * e, Y + ndof * e, u);
   });
}

template<int DIM, int D1D, int QND>
MFEM_HOST_DEVICE inline
void MmaMassApplySimplex_Batch(const int e0,
                               const int NE,
                               const real_t *p_,
                               const real_t *d_,
                               const real_t *x_,
                               real_t *y_)
{
   static_assert(D1D > 0 && QND > 0, "Simplex MMA mass requires specialized D1D/QND");
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
   constexpr int BASIS_DIM = simplex_mma::SimplexNdof<DIM, D1D>();
   constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAP>(BASIS_DIM);
   constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
   constexpr int NB = simplex_mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = BASIS_DIM;

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
   MFEM_SIMPLEX_MMA_SMEM(Smem, sm);

   const int tid = simplex_mma::getThreadIdx();
   [[maybe_unused]] const int nthreads = simplex_mma::getBlockNthreads();

   if constexpr (simplex_mma::TensorMmaEnabled())
   {
      simplex_mma::MassBatchApplyMma<MAP, QND, ndof, X_LD, U_LD, NB>(
         e0, NE, p_, d_, x_, y_, sm.XY, sm.Us, tid, nthreads);
   }
   else
   {
      simplex_mma::MassBatchApplyDense<QND, ndof, X_LD, U_LD, NB>(
         e0, NE, p_, d_, x_, y_, sm.XY, sm.Us, tid);
      MFEM_SYNC_THREAD;
   }
}

/** Runtime-sized batch body for Fallback (MmaMapDefault, dyn/static smem). */
template<int DIM>
MFEM_HOST_DEVICE inline
void MmaMassApplySimplex_Batch(const int e0,
                               const int NE,
                               const int nq,
                               const int ndof,
                               const int x_ld,
                               const int u_ld,
                               const int nb,
                               const real_t *p_,
                               const real_t *d_,
                               const real_t *x_,
                               real_t *y_)
{
   constexpr int max_nq = simplex_mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = simplex_mma::SimplexNdof<DIM, 0>();
   constexpr int max_x_ld = simplex_mma::PadLdBank<simplex_mma::MmaMapDefault>(
                               max_ndof);
   constexpr int max_u_ld = simplex_mma::PadLdBank<simplex_mma::MmaMapDefault>(
                               max_nq);
   constexpr int max_nb = simplex_mma::NBATCH;

   // Dyn layout matches launch smem_bytes=(x_ld+u_ld)*nb; static uses caps.
#if defined(__CUDA_ARCH__)
   real_t *XY = reinterpret_cast<real_t *>(simplex_mma::SimplexMmaDynSmem());
   real_t *Us = XY + x_ld * nb;
   MFEM_CONTRACT_VAR(max_x_ld);
   MFEM_CONTRACT_VAR(max_u_ld);
   MFEM_CONTRACT_VAR(max_nb);
#else
   MFEM_SHARED real_t XY[max_x_ld * max_nb];
   MFEM_SHARED real_t Us[max_u_ld * max_nb];
#endif

   const int tid = simplex_mma::getThreadIdx();
   [[maybe_unused]] const int nthreads = simplex_mma::getBlockNthreads();

   if constexpr (simplex_mma::TensorMmaEnabled())
   {
      simplex_mma::MassBatchApplyMmaRuntime(
         e0, NE, nq, ndof, x_ld, u_ld, nb, p_, d_, x_, y_,
         XY, Us, tid, nthreads);
   }
   else
   {
      simplex_mma::MassBatchApplyDenseRuntime(
         e0, NE, nq, ndof, x_ld, u_ld, nb, p_, d_, x_, y_, XY, Us, tid);
      MFEM_SYNC_THREAD;
   }
}

template<int DIM, int D1D, int QND>
inline void MmaMassApplySimplex(const int NE,
                                const Array<real_t> &p_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_)
{
   static_assert(D1D > 0 && QND > 0, "Simplex MMA mass requires specialized D1D/QND");
   constexpr int NB = simplex_mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d_.Size() == QND * NE, "");
   MFEM_VERIFY(p_.Size() == QND * ndof, "");

   // Dedicated host path: multi-RHS dense GEMM without GPU batch/smem layout.
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      PAMassApplySimplexDenseHost<DIM, D1D, QND>(NE, p_, d_, x_, y_);
      return;
   }

   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
   constexpr int BASIS = ndof;
   constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAP>(BASIS);
   constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
   constexpr int smem_bytes = int(sizeof(real_t)) * (X_LD + U_LD) * NB;
   simplex_mma::VerifySharedMemBytes(smem_bytes);

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int nthreads = simplex_mma::LaunchNthreads<QND>(QND, ndof);
   const int nbatches = (NE + NB - 1) / NB;
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
                                const Array<real_t> &p_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_)
{
   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d_.Size() % NE == 0, "");
   const int nq = d_.Size() / NE;
   MFEM_VERIFY(nq > 0 && p_.Size() % nq == 0, "");
   const int ndof = p_.Size() / nq;
   MFEM_VERIFY(x_.Size() >= ndof * NE && y_.Size() >= ndof * NE, "");

   constexpr int max_nq = simplex_mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = simplex_mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA mass runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      const real_t *P = p_.Read();
      const real_t *D = d_.Read();
      const real_t *X = x_.Read();
      real_t *Y = y_.ReadWrite();
      if (simplex_mma::TryMassApplyBlas(NE, nq, ndof, P, D, X, Y)) { return; }
      simplex_mma::MassApplyHandRuntime(NE, nq, ndof, P, D, X, Y);
      return;
   }

   const int x_ld = simplex_mma::PadLdBankRuntime(ndof);
   const int u_ld = simplex_mma::PadLdBankRuntime(nq);
   const int nb = simplex_mma::MassLikeNBRuntime(ndof, nq);
   MFEM_VERIFY(x_ld <= simplex_mma::PadLdBank<simplex_mma::MmaMapDefault>(max_ndof) &&
               u_ld <= simplex_mma::PadLdBank<simplex_mma::MmaMapDefault>(max_nq) &&
               nb <= simplex_mma::NBATCH,
               "Simplex MMA mass runtime Fallback smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * (x_ld + u_ld) * nb;
   simplex_mma::VerifySharedMemBytes(smem_bytes);

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int nthreads = simplex_mma::LaunchNthreads(nq, ndof);
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
