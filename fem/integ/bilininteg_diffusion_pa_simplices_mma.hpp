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

/** Choose NB so X + DIM*U shared buffers stay within 48 KiB. */
template <int DIM, int T_D1D, int T_Q1D>
constexpr int DiffusionMmaNB()
{
   constexpr int MQ = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS = simplex_mma::SimplexNdof<DIM, T_D1D>();
   constexpr int MAGIC = simplex_mma::MagicFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = simplex_mma::PadLdBank<MAGIC>(BASIS);
   constexpr int U_LD = simplex_mma::PadLdBank<MAGIC>(MQ);
   constexpr int per_batch_col = X_LD + DIM * U_LD;
   constexpr int max_nb =
      (48 * 1024) / (int(sizeof(real_t)) * per_batch_col);
   // Prefer NBATCH when specialized and it fits; else mmaN; else shrink.
   if (T_D1D && T_Q1D && simplex_mma::NBATCH <= max_nb)
   {
      return simplex_mma::NBATCH;
   }
   if (simplex_mma::mmaN <= max_nb) { return simplex_mma::mmaN; }
   return max_nb > 0 ? max_nb : 1;
}

/** One-kernel diffusion PA data: J from (nodes_e, Gn), then metric into D. */
inline void PADiffusionSetupSimplexMma(const int dim,
                                       const int coeffDim,
                                       const int NE,
                                       const int NQ,
                                       const int ND,
                                       const Array<real_t> &w,
                                       const Array<real_t> &g,
                                       const Vector &nodes_e,
                                       const Vector &c,
                                       Vector &d)
{
   const bool symmetric = (coeffDim != dim * dim);
   const bool const_c = c.Size() == coeffDim;
   const int pa_size = symmetric ? (dim * (dim + 1)) / 2 : dim * dim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   auto D = Reshape(d.Write(), NQ, pa_size, NE);

   if (dim == 2)
   {
      const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1)
                     : Reshape(c.Read(), coeffDim, NQ, NE);
      auto get_coeff = [const_c] MFEM_HOST_DEVICE
                       (const decltype(C) &C, int i, int q, int e)
      {
         return const_c ? C(i, 0, 0) : C(i, q, e);
      };
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ;
         const int q = idx - NQ * e;
         real_t J11 = 0.0, J21 = 0.0, J12 = 0.0, J22 = 0.0;
         for (int i = 0; i < ND; i++)
         {
            const real_t x = E(i, 0, e), y = E(i, 1, e);
            const real_t gx = G(q, 0, i), gy = G(q, 1, i);
            J11 += x * gx; J21 += y * gx;
            J12 += x * gy; J22 += y * gy;
         }
         const real_t w_detJ = W(q) / ((J11 * J22) - (J21 * J12));
         if (coeffDim == 3 || coeffDim == 4)
         {
            const real_t M11 = get_coeff(C, 0, q, e);
            const real_t M12 = get_coeff(C, 1, q, e);
            const real_t M21 = symmetric ? M12 : get_coeff(C, 2, q, e);
            const real_t M22 = symmetric ? get_coeff(C, 2, q, e)
                               : get_coeff(C, 3, q, e);
            const real_t R11 = M11 * J22 - M12 * J12;
            const real_t R21 = M21 * J22 - M22 * J12;
            const real_t R12 = -M11 * J21 + M12 * J11;
            const real_t R22 = -M21 * J21 + M22 * J11;
            D(q, 0, e) = w_detJ * (J22 * R11 - J12 * R21);
            D(q, 1, e) = w_detJ * (-J21 * R11 + J11 * R21);
            D(q, 2, e) = w_detJ * (symmetric ? (-J21 * R12 + J11 * R22)
                                   : (J22 * R12 - J12 * R22));
            if (!symmetric)
            {
               D(q, 3, e) = w_detJ * (-J21 * R12 + J11 * R22);
            }
         }
         else
         {
            const real_t C1 = get_coeff(C, 0, q, e);
            const real_t C2 = get_coeff(C, coeffDim == 2 ? 1 : 0, q, e);
            D(q, 0, e) = w_detJ * (C2 * J12 * J12 + C1 * J22 * J22);
            D(q, 1, e) = -w_detJ * (C2 * J12 * J11 + C1 * J22 * J21);
            D(q, 2, e) = w_detJ * (C2 * J11 * J11 + C1 * J21 * J21);
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PADiffusionSetupSimplexMma only supports dim 2 or 3");
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1)
                  : Reshape(c.Read(), coeffDim, NQ, NE);
   auto get_coeff = [const_c] MFEM_HOST_DEVICE
                    (const decltype(C) &C, int i, int q, int e)
   {
      return const_c ? C(i, 0, 0) : C(i, q, e);
   };
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ;
      const int q = idx - NQ * e;
      real_t J11 = 0.0, J21 = 0.0, J31 = 0.0;
      real_t J12 = 0.0, J22 = 0.0, J32 = 0.0;
      real_t J13 = 0.0, J23 = 0.0, J33 = 0.0;
      for (int i = 0; i < ND; i++)
      {
         const real_t x = E(i, 0, e), y = E(i, 1, e), z = E(i, 2, e);
         const real_t gx = G(q, 0, i), gy = G(q, 1, i), gz = G(q, 2, i);
         J11 += x * gx; J21 += y * gx; J31 += z * gx;
         J12 += x * gy; J22 += y * gy; J32 += z * gy;
         J13 += x * gz; J23 += y * gz; J33 += z * gz;
      }
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      const real_t w_detJ = W(q) / detJ;
      const real_t A11 = (J22 * J33) - (J23 * J32);
      const real_t A12 = (J32 * J13) - (J12 * J33);
      const real_t A13 = (J12 * J23) - (J22 * J13);
      const real_t A21 = (J31 * J23) - (J21 * J33);
      const real_t A22 = (J11 * J33) - (J13 * J31);
      const real_t A23 = (J21 * J13) - (J11 * J23);
      const real_t A31 = (J21 * J32) - (J31 * J22);
      const real_t A32 = (J31 * J12) - (J11 * J32);
      const real_t A33 = (J11 * J22) - (J12 * J21);

      if (coeffDim == 6 || coeffDim == 9)
      {
         const real_t M11 = get_coeff(C, 0, q, e);
         const real_t M12 = get_coeff(C, 1, q, e);
         const real_t M13 = get_coeff(C, 2, q, e);
         const real_t M21 = (!symmetric) ? get_coeff(C, 3, q, e) : M12;
         const real_t M22 = (!symmetric) ? get_coeff(C, 4, q, e)
                            : get_coeff(C, 3, q, e);
         const real_t M23 = (!symmetric) ? get_coeff(C, 5, q, e)
                            : get_coeff(C, 4, q, e);
         const real_t M31 = (!symmetric) ? get_coeff(C, 6, q, e) : M13;
         const real_t M32 = (!symmetric) ? get_coeff(C, 7, q, e) : M23;
         const real_t M33 = (!symmetric) ? get_coeff(C, 8, q, e)
                            : get_coeff(C, 5, q, e);

         const real_t R11 = M11 * A11 + M12 * A12 + M13 * A13;
         const real_t R12 = M11 * A21 + M12 * A22 + M13 * A23;
         const real_t R13 = M11 * A31 + M12 * A32 + M13 * A33;
         const real_t R21 = M21 * A11 + M22 * A12 + M23 * A13;
         const real_t R22 = M21 * A21 + M22 * A22 + M23 * A23;
         const real_t R23 = M21 * A31 + M22 * A32 + M23 * A33;
         const real_t R31 = M31 * A11 + M32 * A12 + M33 * A13;
         const real_t R32 = M31 * A21 + M32 * A22 + M33 * A23;
         const real_t R33 = M31 * A31 + M32 * A32 + M33 * A33;

         D(q, 0, e) = w_detJ * (A11 * R11 + A12 * R21 + A13 * R31);
         const real_t D12 = w_detJ * (A11 * R12 + A12 * R22 + A13 * R32);
         D(q, 1, e) = D12;
         D(q, 2, e) = w_detJ * (A11 * R13 + A12 * R23 + A13 * R33);
         const real_t D22 = w_detJ * (A21 * R12 + A22 * R22 + A23 * R32);
         const real_t D23 = w_detJ * (A21 * R13 + A22 * R23 + A23 * R33);
         const real_t D33 = w_detJ * (A31 * R13 + A32 * R23 + A33 * R33);
         D(q, 4, e) = symmetric ? D23 : D22;
         D(q, 5, e) = symmetric ? D33 : D23;
         if (symmetric)
         {
            D(q, 3, e) = D22;
         }
         else
         {
            D(q, 3, e) = w_detJ * (A21 * R11 + A22 * R21 + A23 * R31);
            D(q, 6, e) = w_detJ * (A31 * R11 + A32 * R21 + A33 * R31);
            D(q, 7, e) = w_detJ * (A31 * R12 + A32 * R22 + A33 * R32);
            D(q, 8, e) = D33;
         }
      }
      else
      {
         const real_t C1 = get_coeff(C, 0, q, e);
         const real_t C2 = get_coeff(C, coeffDim == 3 ? 1 : 0, q, e);
         const real_t C3 = get_coeff(C, coeffDim == 3 ? 2 : 0, q, e);
         D(q, 0, e) = w_detJ * (C1 * A11 * A11 + C2 * A12 * A12 + C3 * A13 * A13);
         D(q, 1, e) = w_detJ * (C1 * A11 * A21 + C2 * A12 * A22 + C3 * A13 * A23);
         D(q, 2, e) = w_detJ * (C1 * A11 * A31 + C2 * A12 * A32 + C3 * A13 * A33);
         D(q, 3, e) = w_detJ * (C1 * A21 * A21 + C2 * A22 * A22 + C3 * A23 * A23);
         D(q, 4, e) = w_detJ * (C1 * A21 * A31 + C2 * A22 * A32 + C3 * A23 * A33);
         D(q, 5, e) = w_detJ * (C1 * A31 * A31 + C2 * A32 * A32 + C3 * A33 * A33);
      }
   });
}

template<int DIM, int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPADiffusionApplySimplexMma_Batch(const int e0,
                                          const int NE,
                                          const bool symmetric,
                                          const real_t *g_,
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
   constexpr int NB = DiffusionMmaNB<DIM, T_D1D, T_Q1D>();
   constexpr int PA_SIZE = (DIM * (DIM + 1)) /
                           2; // applied path uses symmetric layout
   static_assert(sizeof(real_t) * (X_LD + DIM * U_LD) * NB <= 48 * 1024,
                 "Diffusion simplex MMA shared memory exceeds 48 KiB");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                    : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int pa_size = symmetric ? PA_SIZE : (DIM * DIM);

   const auto D = Reshape(d_, NQ1, pa_size, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t UV[DIM * U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   struct GAcc
   {
      const real_t *g;
      int nq1_, ndof_, d_;
      MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
      {
         return g[row + nq1_ * (col + ndof_ * d_)];
      }
   };

   const int tid = simplex_mma::getThreadIdx();
#ifdef __CUDA_ARCH__
   const int nthreads = blockDim.x * blockDim.y * blockDim.z;
#else
   [[maybe_unused]] const int nthreads = 1;
#endif

#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   simplex_mma::SmemMatAcc<X_LD> Xacc {sm.XY};
   simplex_mma::YBatchAcc Yacc{y_, ndof, e0};
   simplex_mma::NullDAcc nullD;

   for (int i = tid; i < X_LD * NB; i += nthreads)
   {
      const int b = i / X_LD;
      const int r = i - b * X_LD;
      const int e = e0 + b;
      sm.XY[i] = (e < NE && r < ndof) ? x(r, e) : real_t(0);
   }
   MFEM_SYNC_THREAD;

   for (int d = 0; d < DIM; ++d)
   {
      GAcc A{g_, NQ1, ndof, d};
      simplex_mma::SmemMatAcc<U_LD> Uacc{sm.UV + d * U_LD * NB};
      simplex_mma::dmma_Gemm<MAGIC, false>(NQ1, ndof, NB, A, Xacc, Uacc,
                                           nullD, e0, NE);
   }
   MFEM_SYNC_THREAD;

   for (int i = tid; i < NQ1 * NB; i += nthreads)
   {
      const int b = i / NQ1;
      const int q = i - b * NQ1;
      const int e = e0 + b;
      if (e >= NE || q >= NQ1) { continue; }

      if constexpr (DIM == 2)
      {
         const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
         const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
         const real_t O11 = D(q, 0, e);
         const real_t O21 = D(q, 1, e);
         const real_t O12 = symmetric ? O21 : D(q, 2, e);
         const real_t O22 = symmetric ? D(q, 2, e) : D(q, 3, e);
         sm.UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2;
         sm.UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2;
      }
      else
      {
         const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
         const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
         const real_t u3 = sm.UV[2 * U_LD * NB + q + U_LD * b];
         const real_t O11 = D(q, 0, e);
         const real_t O12 = D(q, 1, e);
         const real_t O13 = D(q, 2, e);
         const real_t O21 = symmetric ? O12 : D(q, 3, e);
         const real_t O22 = symmetric ? D(q, 3, e) : D(q, 4, e);
         const real_t O23 = symmetric ? D(q, 4, e) : D(q, 5, e);
         const real_t O31 = symmetric ? O13 : D(q, 6, e);
         const real_t O32 = symmetric ? O23 : D(q, 7, e);
         const real_t O33 = symmetric ? D(q, 5, e) : D(q, 8, e);
         sm.UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
         sm.UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2 + O23 * u3;
         sm.UV[2 * U_LD * NB + q + U_LD * b] = O31 * u1 + O32 * u2 + O33 * u3;
      }
   }
   MFEM_SYNC_THREAD;

   for (int d = 0; d < DIM; ++d)
   {
      GAcc A{g_, NQ1, ndof, d};
      simplex_mma::SmemMatAcc<U_LD> Vacc{sm.UV + d * U_LD * NB};
      simplex_mma::dmma_GemmT<MAGIC>(NQ1, ndof, NB, A, Vacc, Yacc, e0, NE);
   }
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
         for (int d = 0; d < DIM; ++d)
         {
            for (int q = 0; q < NQ1; ++q)
            {
               real_t u = 0.0;
               for (int i = 0; i < ndof; ++i)
               {
                  u += g_[q + NQ1 * (i + ndof * d)] * sm.XY[i + X_LD * b];
               }
               sm.UV[d * U_LD * NB + q + U_LD * b] = u;
            }
         }
         for (int q = 0; q < NQ1; ++q)
         {
            if constexpr (DIM == 2)
            {
               const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
               const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
               const real_t O11 = D(q, 0, e);
               const real_t O21 = D(q, 1, e);
               const real_t O12 = symmetric ? O21 : D(q, 2, e);
               const real_t O22 = symmetric ? D(q, 2, e) : D(q, 3, e);
               sm.UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2;
               sm.UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2;
            }
            else
            {
               const real_t u1 = sm.UV[0 * U_LD * NB + q + U_LD * b];
               const real_t u2 = sm.UV[1 * U_LD * NB + q + U_LD * b];
               const real_t u3 = sm.UV[2 * U_LD * NB + q + U_LD * b];
               const real_t O11 = D(q, 0, e);
               const real_t O12 = D(q, 1, e);
               const real_t O13 = D(q, 2, e);
               const real_t O21 = symmetric ? O12 : D(q, 3, e);
               const real_t O22 = symmetric ? D(q, 3, e) : D(q, 4, e);
               const real_t O23 = symmetric ? D(q, 4, e) : D(q, 5, e);
               const real_t O31 = symmetric ? O13 : D(q, 6, e);
               const real_t O32 = symmetric ? O23 : D(q, 7, e);
               const real_t O33 = symmetric ? D(q, 5, e) : D(q, 8, e);
               sm.UV[0 * U_LD * NB + q + U_LD * b] = O11 * u1 + O12 * u2 + O13 * u3;
               sm.UV[1 * U_LD * NB + q + U_LD * b] = O21 * u1 + O22 * u2 + O23 * u3;
               sm.UV[2 * U_LD * NB + q + U_LD * b] = O31 * u1 + O32 * u2 + O33 * u3;
            }
         }
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int d = 0; d < DIM; ++d)
            {
               for (int q = 0; q < NQ1; ++q)
               {
                  yi += g_[q + NQ1 * (i + ndof * d)] *
                        sm.UV[d * U_LD * NB + q + U_LD * b];
               }
            }
            Y(i, e) += yi;
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM = 2, int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApplySimplexMma(const int NE,
                                           const bool symmetric,
                                           const Array<real_t> &g_,
                                           const Vector &d_,
                                           const Vector &x_,
                                           Vector &y_,
                                           const int d1d = 0,
                                           const int nq1 = 0)
{
   constexpr int NB = DiffusionMmaNB<DIM, T_D1D, T_Q1D>();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                    : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int max_d1d = T_D1D ? T_D1D
                       : ((DIM == 3) ? simplex_mma::FallbackMaxD1D3
                          : DeviceDofQuadLimits::Get().MAX_D1D);
   const int max_nq = simplex_mma::SimplexMaxNq<DIM, T_Q1D>();
   const int pa_size = symmetric ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(NQ1 <= max_nq, "");
   MFEM_VERIFY(NQ1 > 0 && NE > 0 && d_.Size() == pa_size * NQ1 * NE, "");
   MFEM_VERIFY(g_.Size() == NQ1 * ndof * DIM, "");

   const auto G = g_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int mPassQ = (NQ1 + simplex_mma::mmaM - 1) / simplex_mma::mmaM;
   const int mPassD = (ndof + simplex_mma::mmaM - 1) / simplex_mma::mmaM;
   const int nWarps = (mPassQ < mPassD) ? (mPassQ > 1 ? mPassQ : 1)
                      : (mPassD > 1 ? mPassD : 1);
   const int nthreads = nWarps * 32;
   const int nbatches = (NE + NB - 1) / NB;

   mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
   {
      SmemPADiffusionApplySimplexMma_Batch<DIM, T_D1D, T_Q1D>(
         batch * NB, NE, symmetric, G, D, X, Y, d1d, nq1);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPADiffusionApplySimplexMma<2, T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPADiffusionApplySimplexMma<3, T_D1D, T_Q1D>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA diffusion only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA diffusion PA is only implemented for triangles/tets");
   if (dim == 3)
   {
      return internal::SmemPADiffusionApplySimplexMma<3>;
   }
   return internal::SmemPADiffusionApplySimplexMma<2>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
