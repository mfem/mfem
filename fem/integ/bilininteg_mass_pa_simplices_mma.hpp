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

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

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
void SmemPAMassApplySimplexMma_Batch(const int e0,
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
   // Dyn smem on CUDA device; static/automatic otherwise (must stay local).
#if defined(__CUDA_ARCH__)
   Smem &sm = *reinterpret_cast<Smem *>(simplex_mma::SimplexMmaDynSmem());
#else
   MFEM_SHARED Smem sm;
#endif

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

template<int DIM, int D1D, int QND>
inline void SmemPAMassApplySimplexMma(const int NE,
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
      SmemPAMassApplySimplexMma_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, X, Y);
   });
}

} // namespace internal

template<int DIM, int D1D, int QND>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPAMassApplySimplexMma<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAMassApplySimplexMma<3, D1D, QND>;
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
      return internal::PAMassApplySimplexDenseRuntime<2>;
   }
   return internal::PAMassApplySimplexDenseRuntime<3>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
