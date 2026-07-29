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
// #include "bilininteg_pa_simplices_mma_host.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Host-optimized dense mass: y += P^T ( D ⊙ (P x) ) per element.
    Large (QND,ndof): BLAS multi-RHS when MFEM_USE_LAPACK is on.
    Specialized sizes: hand multi-RHS tiles. */
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

#ifdef MFEM_USE_LAPACK
   if (simplex_mma::PreferHostBlas(QND, ndof))
   {
      simplex_mma::MassApplyBlas(NE, QND, ndof, P, D, X, Y);
      return;
   }
#endif
   simplex_mma::MassApplyHandSpecialized<DIM, ndof, QND>(NE, P, D, X, Y);
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

   const auto D = ConstDeviceMatrix(d_, QND, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
#if defined(__CUDA_ARCH__)
   Smem &sm = *reinterpret_cast<Smem *>(simplex_mma::SimplexMmaDynSmem());
#else
   MFEM_SHARED Smem sm; // HIP static / host automatic
#endif

   const int tid = simplex_mma::getThreadIdx();
   [[maybe_unused]] const int nthreads = simplex_mma::getBlockNthreads();

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    !defined(MFEM_USE_SINGLE)
   simplex_mma::SmemMatAcc<X_LD> Xacc {sm.XY};
   simplex_mma::SmemMatAcc<U_LD> Uacc{sm.Us};
   simplex_mma::YBatchAcc Yacc{y_, ndof, e0};

   simplex_mma::LoadXToSmem(sm.XY, x, e0, NE, ndof, X_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   simplex_mma::PAcc A{p_, QND, ndof};
   simplex_mma::BasisGemmForward<MAP, true>(QND, ndof, NB, A, Xacc, Uacc,
                                            D, e0, NE);
   MFEM_SYNC_THREAD;
   simplex_mma::BasisGemmT<MAP>(QND, ndof, NB, A, Uacc, Yacc, e0, NE);
#else
   // Device-compiled fallback (e.g. single precision): serial dense per batch.
   auto Y = DeviceMatrix(y_, ndof, NE);
   if (tid == 0)
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int i = 0; i < X_LD; ++i)
         {
            sm.XY[i + X_LD * b] = (i < ndof) ? x(i, e) : real_t(0);
         }
         for (int q = 0; q < QND; ++q)
         {
            real_t u = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               u += p_[q + QND * i] * sm.XY[i + X_LD * b];
            }
            sm.Us[q + U_LD * b] = u * D(q, e);
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < QND; ++q)
            {
               yi += p_[q + QND * i] * sm.Us[q + U_LD * b];
            }
            Y(i, e) += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
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
#if defined(MFEM_USE_CUDA)
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemPAMassApplySimplexMma_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, X, Y);
   });
#else
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemPAMassApplySimplexMma_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, X, Y);
   });
#endif
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
   MFEM_ABORT("No fallback for Simplex MMA mass PA");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
