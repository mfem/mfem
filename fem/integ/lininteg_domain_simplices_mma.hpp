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
                                 const real_t *p_,
                                 const real_t *d_,
                                 real_t *y_,
                                 const int vdim,
                                 const int vc)
{
   static_assert(D1D > 0 && QND > 0, "Simplex MMA DomainLF requires specialized D1D/QND");
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
   constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
   constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
   constexpr int NB = simplex_mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();

   const auto D = ConstDeviceMatrix(d_, QND, NE);

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

   simplex_mma::LoadDToSmem(sm.Us, D, e0, NE, QND, U_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   simplex_mma::PAcc A{p_, QND, ndof};
   simplex_mma::BasisGemmT<MAP>(QND, ndof, NB, A, Uacc, Yacc, e0, NE);
#else
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
               yi += p_[q + QND * i] * sm.Us[q + U_LD * b];
            }
            y_[i + ndof * (vc + vdim * e)] += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM, int D1D, int QND>
inline void MmaDLFAssembleSimplex(const int NE,
                                  const Array<real_t> &p_,
                                  const Vector &d_,
                                  real_t *y_,
                                  const int vdim,
                                  const int vc)
{
   static_assert(D1D > 0 && QND > 0, "Simplex MMA DomainLF requires specialized D1D/QND");
   constexpr int NB = simplex_mma::MassLikeNB<DIM, D1D, QND>();
   constexpr int ndof = simplex_mma::SimplexNdof<DIM, D1D>();
   MFEM_VERIFY(NE > 0 && d_.Size() == QND * NE, "");
   MFEM_VERIFY(p_.Size() == QND * ndof, "");
   MFEM_VERIFY(vdim >= 1 && vc >= 0 && vc < vdim, "");

   {
      constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, QND>();
      constexpr int MAP = simplex_mma::MmaMapFor<DIM, D1D, QND>();
      constexpr int U_LD = simplex_mma::PadLdBank<MAP>(MQ);
      // DomainLF only stages Us[U_LD * NB] in shared memory.
      simplex_mma::VerifySharedMemBytes(int(sizeof(real_t)) * U_LD * NB);
   }

   const auto P = p_.Read();
   const auto D = d_.Read();

   const int nthreads = simplex_mma::LaunchNthreads<QND>(QND, ndof);
   const int nbatches = (NE + NB - 1) / NB;
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      MmaDLFAssembleSimplex_Batch<DIM, D1D, QND>(
         batch * NB, NE, P, D, y_, vdim, vc);
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
   MFEM_ABORT("No fallback for Simplex MMA DomainLF");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
