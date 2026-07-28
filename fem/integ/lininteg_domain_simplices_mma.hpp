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

#include "bilininteg_pa_simplices_mma.hpp"
#include "../lininteg.hpp"

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
template<int DIM, int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemDLFAssembleSimplexMma_Batch(const int e0,
                                     const int NE,
                                     const real_t *p_,
                                     const real_t *d_,
                                     real_t *y_,
                                     const int vdim,
                                     const int vc,
                                     const int d1d,
                                     const int nq1)
{
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   constexpr int MAP = simplex_mma::MmaMapFor<DIM, T_D1D, T_Q1D>();
   constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
   constexpr int NB = simplex_mma::MassLikeNB<DIM, T_D1D, T_Q1D>();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = simplex_mma::SimplexNdofFromD1D(DIM, D1D);
   const int NQ1 = T_Q1D ? T_Q1D : nq1;

   const auto D = ConstDeviceMatrix(d_, NQ1, NE);

   struct alignas(16) Smem
   {
      real_t Us[U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   const int tid = simplex_mma::getThreadIdx();
   [[maybe_unused]]const int nthreads = simplex_mma::getBlockNthreads();

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    !defined(MFEM_USE_SINGLE)
   simplex_mma::SmemMatAcc<U_LD> Uacc {sm.Us};
   simplex_mma::YVdimAcc Yacc{y_, ndof, vdim, vc, e0};

   simplex_mma::LoadDToSmem(sm.Us, D, e0, NE, NQ1, U_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   simplex_mma::PAcc A{p_, NQ1, ndof};
   simplex_mma::BasisGemmT<MAP>(NQ1, ndof, NB, A, Uacc, Yacc, e0, NE);
#else
   if (tid == 0)
   {
      for (int b = 0; b < NB; ++b)
      {
         const int e = e0 + b;
         if (e >= NE) { continue; }
         for (int q = 0; q < NQ1; ++q)
         {
            sm.Us[q + U_LD * b] = D(q, e);
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < NQ1; ++q)
            {
               yi += p_[q + NQ1 * i] * sm.Us[q + U_LD * b];
            }
            y_[i + ndof * (vc + vdim * e)] += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM = 2, int T_D1D = 0, int T_Q1D = 0>
inline void SmemDLFAssembleSimplexMma(const int NE,
                                      const Array<real_t> &p_,
                                      const Vector &d_,
                                      real_t *y_,
                                      const int vdim,
                                      const int vc,
                                      const int d1d = 0,
                                      const int nq1 = 0)
{
   constexpr int NB = simplex_mma::MassLikeNB<DIM, T_D1D, T_Q1D>();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int ndof = simplex_mma::SimplexNdofFromD1D(DIM, D1D);
   const int max_d1d = T_D1D ? T_D1D
                       : ((DIM == 3) ? simplex_mma::FallbackMaxD1D3
                          : DeviceDofQuadLimits::Get().MAX_D1D);
   const int max_nq = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(NQ1 <= max_nq, "");
   MFEM_VERIFY(NQ1 > 0 && NE > 0 && d_.Size() == NQ1 * NE, "");
   MFEM_VERIFY(p_.Size() == NQ1 * ndof, "");
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   {
      constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
      // constexpr int BASIS = simplex_mma::SimplexNdof<DIM, T_D1D>();
      constexpr int MAP = simplex_mma::MmaMapFor<DIM, T_D1D, T_Q1D>();
      constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
      // DomainLF only stages Us[U_LD * NB] in shared memory.
      simplex_mma::VerifySharedMemBytes(int(sizeof(real_t)) * U_LD * NB);
   }

   const auto P = p_.Read();
   const auto D = d_.Read();

   const int nthreads = simplex_mma::LaunchNthreads(NQ1, ndof);
   const int nbatches = (NE + NB - 1) / NB;
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemDLFAssembleSimplexMma_Batch<DIM, T_D1D, T_Q1D>(
         batch * NB, NE, P, D, y_, vdim, vc, d1d, nq1);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
DomainLFIntegrator::AssembleSimplexMmaKernelType
DomainLFIntegrator::AssembleSimplexMmaKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemDLFAssembleSimplexMma<2, T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemDLFAssembleSimplexMma<3, T_D1D, T_Q1D>;
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
   if (dim == 3)
   {
      return internal::SmemDLFAssembleSimplexMma<3>;
   }
   return internal::SmemDLFAssembleSimplexMma<2>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
