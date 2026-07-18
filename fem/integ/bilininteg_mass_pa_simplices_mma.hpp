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
#include "../bilininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template<int DIM, int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplySimplexMma_Batch(const int e0,
                                     const int NE,
                                     const real_t *p_,
                                     const real_t *d_,
                                     const real_t *x_,
                                     real_t *y_,
                                     const int d1d,
                                     const int nq1)
{
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS_DIM = simplex_mma::SimplexNdof<DIM, T_D1D>();
   constexpr int MAGIC = simplex_mma::MagicFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAGIC>(BASIS_DIM);
   constexpr int U_LD = simplex_mma::PadLdBank<MAGIC>(MQ);
   constexpr int NB = simplex_mma::MassLikeNB<DIM, T_D1D, T_Q1D>();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = simplex_mma::SimplexNdofFromD1D(DIM, D1D);
   const int NQ1 = T_Q1D ? T_Q1D : nq1;

   const auto D = ConstDeviceMatrix(d_, NQ1, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   const int tid = simplex_mma::getThreadIdx();
   const int nthreads = simplex_mma::getBlockNthreads();

#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   simplex_mma::SmemMatAcc<X_LD> Xacc {sm.XY};
   simplex_mma::SmemMatAcc<U_LD> Uacc{sm.Us};
   simplex_mma::YBatchAcc Yacc{y_, ndof, e0};

   simplex_mma::LoadXToSmem(sm.XY, x, e0, NE, ndof, X_LD, NB, tid, nthreads);
   MFEM_SYNC_THREAD;

   simplex_mma::PAcc A{p_, NQ1, ndof};
   simplex_mma::BasisGemmForward<MAGIC, true>(NQ1, ndof, NB, A, Xacc, Uacc,
                                              D, e0, NE);
   MFEM_SYNC_THREAD;
   simplex_mma::BasisGemmT<MAGIC>(NQ1, ndof, NB, A, Uacc, Yacc, e0, NE);
#else
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
         for (int q = 0; q < NQ1; ++q)
         {
            real_t u = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               u += p_[q + NQ1 * i] * sm.XY[i + X_LD * b];
            }
            sm.Us[q + U_LD * b] = u * D(q, e);
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < NQ1; ++q)
            {
               yi += p_[q + NQ1 * i] * sm.Us[q + U_LD * b];
            }
            Y(i, e) += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM = 2, int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplySimplexMma(const int NE,
                                      const Array<real_t> &p_,
                                      const Vector &d_,
                                      const Vector &x_,
                                      Vector &y_,
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

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int nthreads = simplex_mma::LaunchNthreads(NQ1, ndof);
   const int nbatches = (NE + NB - 1) / NB;
   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemPAMassApplySimplexMma_Batch<DIM, T_D1D, T_Q1D>(
         batch * NB, NE, P, D, X, Y, d1d, nq1);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPAMassApplySimplexMma<2, T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAMassApplySimplexMma<3, T_D1D, T_Q1D>;
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
   if (dim == 3)
   {
      return internal::SmemPAMassApplySimplexMma<3>;
   }
   return internal::SmemPAMassApplySimplexMma<2>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
