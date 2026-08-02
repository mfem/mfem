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

#include "bilininteg_pa_mma.hpp"
#include "../lininteg.hpp"

#ifdef MFEM_USE_LAPACK
#include <vector>
#endif

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

void DLFEvalAssembleSimplexMma(const FiniteElementSpace &fes,
                               const IntegrationRule *ir,
                               const Array<int> &markers,
                               const Vector &coeff,
                               Vector &y);

namespace internal
{

/** DomainLF simplex MMA batch: y += P^T D for one vdim component.
    E-vector layout is (ndof x vdim x NE); component c is selected via vc. */
template<int DIM, int D1D, int QND>
MFEM_HOST_DEVICE inline
void MmaDLFAssembleSimplex_Batch(const int e0,
                                 const int NE,
                                 const real_t *p,
                                 const real_t *d,
                                 real_t *y,
                                 const int vdim,
                                 const int vc)
{
   static_assert(D1D > 0 &&
                 QND > 0, "Simplex MMA DomainLF requires specialized D1D/QND");
   constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
   constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
   constexpr int NB = mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = mma::SimplexNdof<DIM, D1D>();

   const auto D = ConstDeviceMatrix(d, QND, NE);

   struct alignas(16) Smem
   {
      real_t Us[U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   const int tid = mma::getThreadIdx();
   const int nthreads = mma::getBlockNthreads();

   if constexpr (mma::DeviceGemmEnabled())
   {
      mma::SmemMatAcc<U_LD> Uacc {sm.Us};
      mma::YVdimAcc Yacc{y, ndof, vdim, vc, e0};

      mma::LoadDToSmem(sm.Us, D, e0, NE, QND, U_LD, NB, tid, nthreads);
      MFEM_SYNC_THREAD;

      mma::PAcc A{p, QND, ndof};
      mma::BasisGemmT<MAP>(QND, ndof, NB, A, Uacc, Yacc, e0, NE);
   }
   else
   {
      if (tid == 0)
      {
         for (int b = 0; b < NB; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int q = 0; q < QND; ++q)
            {
               sm.Us[q + U_LD * b] = D(q, e);
            }
            for (int i = 0; i < ndof; ++i)
            {
               real_t yi = 0.0;
               for (int q = 0; q < QND; ++q)
               {
                  yi += p[q + QND * i] * sm.Us[q + U_LD * b];
               }
               y[i + ndof * (vc + vdim * e)] += yi;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template<int DIM, int D1D, int QND>
inline void MmaDLFAssembleSimplex(const int NE,
                                  const Array<real_t> &p,
                                  const Vector &d,
                                  real_t *y,
                                  const int vdim,
                                  const int vc)
{
   static_assert(D1D > 0 &&
                 QND > 0, "Simplex MMA DomainLF requires specialized D1D/QND");
   constexpr int NB = mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d.Size() == QND * NE, "");
   MFEM_VERIFY(p.Size() == QND * ndof, "");
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   {
      constexpr int MQ = mma::SimplexMaxNq<DIM, QND>();
      constexpr int MAP = mma::MmaMapFor<DIM, D1D, QND>();
      constexpr int U_LD = mma::PadLdBank<MAP>(MQ);
      // DomainLF only stages Us[U_LD * NB] in shared memory.
      mma::VerifySharedMemBytes(int(sizeof(real_t)) * U_LD * NB);
   }

   const auto P = p.Read();
   const auto D = d.Read();

   const int nthreads = mma::LaunchNthreads<QND>(QND, ndof);
   const int nbatches = (NE + NB - 1) / NB;
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaDLFAssembleSimplex_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, y, vdim, vc);
   });
}

// ---- Runtime Fallback: host dense + device batched MMA/Dense -------------

/** Host: Y_vc += P^T D. Uses multi-RHS GEMM when LAPACK is on and sizes
    pass PreferLapack; otherwise a dense per-element loop.
    PreferLapack is already false without MFEM_USE_LAPACK; the ifdef only
    gates LapackGemm / std::vector. */
inline void DLFAssembleBlas(int NE, int nq, int ndof, const real_t *P,
                            const real_t *D, real_t *Y, int vdim, int vc)
{
#ifdef MFEM_USE_LAPACK
   if (mma::PreferLapack(nq, ndof))
   {
      const int NB = mma::LapackNB(nq, ndof);
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
               uloc[static_cast<size_t>(q) + static_cast<size_t>(nq) * b] =
                  D[q + nq * e];
            }
         }
         mma::LapackGemm('T', 'N', ndof, NB, nq, real_t(1), P, nq,
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
            yi += P[q + nq * i] * D[q + nq * e];
         }
         Y[i + ndof * (vc + vdim * e)] += yi;
      }
   }
}

/** Runtime-sized batch body for Fallback (MmaMapDefault, Us-only smem). */
template<int DIM>
MFEM_HOST_DEVICE inline
void MmaDLFAssembleSimplex_Batch(const int e0,
                                 const int NE,
                                 const int nq,
                                 const int ndof,
                                 const int u_ld,
                                 const int nb,
                                 const real_t *p,
                                 const real_t *d,
                                 real_t *y,
                                 const int vdim,
                                 const int vc)
{
   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_u_ld = mma::PadLdBank<mma::MmaMapDefault>(
                               max_nq);
   constexpr int max_nb = mma::NBATCH;

#if defined(__CUDA_ARCH__)
   real_t *Us = reinterpret_cast<real_t *>(mma::SimplexMmaDynSmem());
   MFEM_CONTRACT_VAR(max_u_ld);
   MFEM_CONTRACT_VAR(max_nb);
#else
   MFEM_SHARED real_t Us[max_u_ld * max_nb];
#endif

   const int tid = mma::getThreadIdx();
   const int nthreads = mma::getBlockNthreads();
   const auto D = ConstDeviceMatrix(d, nq, NE);

   if constexpr (mma::DeviceGemmEnabled())
   {
      constexpr int MAP = mma::MmaMapDefault;
      mma::SmemMatAccRt Uacc{Us, u_ld};
      mma::YVdimAcc Yacc{y, ndof, vdim, vc, e0};

      mma::LoadDToSmem(Us, D, e0, NE, nq, u_ld, nb, tid, nthreads);
      MFEM_SYNC_THREAD;

      mma::PAcc A{p, nq, ndof};
      mma::BasisGemmT<MAP>(nq, ndof, nb, A, Uacc, Yacc, e0, NE);
   }
   else
   {
      if (tid == 0)
      {
         for (int b = 0; b < nb; ++b)
         {
            const int e = e0 + b;
            if (e >= NE) { continue; }
            for (int q = 0; q < nq; ++q)
            {
               Us[q + u_ld * b] = D(q, e);
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
}

/** Runtime Fallback shell: host dense BLAS/hand; device batched MMA/Dense. */
template<int DIM>
inline void MmaDLFAssembleSimplex(const int NE,
                                  const Array<real_t> &p,
                                  const Vector &d,
                                  real_t *y,
                                  const int vdim,
                                  const int vc)
{
   MFEM_VERIFY(NE > 0, "");
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   MFEM_VERIFY(nq > 0 && p.Size() % nq == 0, "");
   const int ndof = p.Size() / nq;
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   constexpr int max_nq = mma::SimplexMaxNq<DIM, 0>();
   constexpr int max_ndof = mma::SimplexNdof<DIM, 0>();
   MFEM_VERIFY(nq <= max_nq && ndof <= max_ndof,
               "Simplex MMA DomainLF runtime Fallback exceeds size caps");

   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      DLFAssembleBlas(NE, nq, ndof, p.Read(), d.Read(), y, vdim, vc);
      return;
   }

   const int u_ld = mma::PadLdBankRuntime(nq);
   const int nb = mma::MassLikeNBRuntime(ndof, nq);
   MFEM_VERIFY(u_ld <= mma::PadLdBank<mma::MmaMapDefault>
               (max_nq) &&
               nb <= mma::NBATCH,
               "Simplex MMA DomainLF runtime Fallback smem layout exceeds caps");
   const int smem_bytes = int(sizeof(real_t)) * u_ld * nb;
   mma::VerifySharedMemBytes(smem_bytes);

   const auto P = p.Read();
   const auto D = d.Read();

   const int nthreads = mma::LaunchNthreads(nq, ndof);
   const int nbatches = (NE + nb - 1) / nb;
   mfem::forall_3D_smem(nbatches, nthreads, 1, 1, smem_bytes,
                        [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaDLFAssembleSimplex_Batch<DIM>(
         batch * nb, NE, nq, ndof, u_ld, nb, P, D, y, vdim, vc);
   });
}

} // namespace internal

template<int DIM, int D1D, int QND>
DomainLFIntegrator::AssembleSimplexMmaKernelType
DomainLFIntegrator::AssembleSimplexMmaKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::MmaDLFAssembleSimplex<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::MmaDLFAssembleSimplex<3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA DomainLF only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DomainLFIntegrator::AssembleSimplexMmaKernelType
DomainLFIntegrator::AssembleSimplexMmaKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA DomainLF is only implemented for triangles/tets");
   if (dim == 2)
   {
      return internal::MmaDLFAssembleSimplex<2>;
   }
   return internal::MmaDLFAssembleSimplex<3>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
