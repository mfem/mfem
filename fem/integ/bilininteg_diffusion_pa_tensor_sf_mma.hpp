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
#include "bilininteg_tensors_mma.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Sum-factored DMMA diffusion apply (3D hex). MFEM packed pa_data metric. */
template <int T_D1D, int T_Q1D, bool SYM>
inline void SmemPADiffusionApplyTensorSfMma3D(const int NE,
                                              const Array<real_t> &b_,
                                              const Array<real_t> &g_,
                                              const Vector &d_,
                                              const Vector &x_,
                                              Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   constexpr int PA_SIZE = SYM ? 6 : 9;
   constexpr int NB = tensor_sf_mma::SfMmaDiffNB3D<D1D, Q1D>();
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(d_.Size() == PA_SIZE * Q1D * Q1D * Q1D * NE, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto G = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, PA_SIZE, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);

   const int nthreads = tensor_sf_mma::SfMmaDiffThreads3D<D1D, Q1D>();
   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[3][MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[3][MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];

      // One global B/G load for all NB elements in this block.
      tensor_sf_mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, B, G, BG, BGt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         tensor_sf_mma::LoadX<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         tensor_sf_mma::GradX<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradY<MD1, MQ1>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradZ<MD1, MQ1>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;

         // Q-fn: grid-stride so thread count can be < Q^3.
         {
            const int tid = tensor_sf_mma::getThreadIdx();
            const int nq = Q1D * Q1D * Q1D;
#ifdef __CUDA_ARCH__
            const int stride = blockDim.x;
#else
            const int stride = nq;
#endif
            for (int thread = tid; thread < nq; thread += stride)
            {
               const int qx = thread % Q1D;
               const int div = thread / Q1D;
               const int qy = div % Q1D;
               const int qz = div / Q1D;
               // GradZ DeviceMatrix(Q*Q,Q) layout: (qx+Q*qy) + Q*Q*qz
               const int idx = qx + Q1D * qy + Q1D * Q1D * qz;
               const int q = qx + Q1D * (qy + Q1D * qz);
               const real_t gX = sm1[0][idx], gY = sm1[1][idx], gZ = sm1[2][idx];
               const real_t O11 = D(q, 0, e);
               const real_t O12 = D(q, 1, e);
               const real_t O13 = D(q, 2, e);
               real_t O21, O22, O23, O31, O32, O33;
               if constexpr (SYM)
               {
                  O21 = O12; O22 = D(q, 3, e); O23 = D(q, 4, e);
                  O31 = O13; O32 = O23; O33 = D(q, 5, e);
               }
               else
               {
                  O21 = D(q, 3, e); O22 = D(q, 4, e); O23 = D(q, 5, e);
                  O31 = D(q, 6, e); O32 = D(q, 7, e); O33 = D(q, 8, e);
               }
               sm0[0][idx] = O11 * gX + O12 * gY + O13 * gZ;
               sm0[1][idx] = O21 * gX + O22 * gY + O23 * gZ;
               sm0[2][idx] = O31 * gX + O32 * gY + O33 * gZ;
            }
         }
         // No mid-kernel global BtGt reload; use BGt from LoadBGBoth.
         MFEM_SYNC_THREAD;

         tensor_sf_mma::GradZt<MD1, MQ1>(D1D, Q1D, BGt, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradYt<MD1, MQ1>(D1D, Q1D, BGt, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradXt<MD1, MQ1>(D1D, Q1D, BGt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApplyTensorSfMma3DDispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Vector &d, const Vector &x, Vector &y,
   const int, const int)
{
   if (symmetric)
   {
      SmemPADiffusionApplyTensorSfMma3D<T_D1D, T_Q1D, true>(NE, b, g, d, x, y);
   }
   else
   {
      SmemPADiffusionApplyTensorSfMma3D<T_D1D, T_Q1D, false>(NE, b, g, d, x, y);
   }
}

template <int T_D1D, int T_Q1D, bool SYM>
inline void SmemPADiffusionApplyTensorSfMma2D(const int NE,
                                              const Array<real_t> &b_,
                                              const Array<real_t> &g_,
                                              const Vector &d_,
                                              const Vector &x_,
                                              Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   constexpr int PA_SIZE = SYM ? 3 : 4;
   constexpr int MDQ = (Q1D > D1D ? Q1D : D1D);
   constexpr int NB = tensor_sf_mma::SfMmaDiffNB2D<D1D, Q1D>();
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto G = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D, PA_SIZE, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   const int nthreads = tensor_sf_mma::SfMmaDiffThreads2D<D1D, Q1D>();
   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[2][MDQ * MDQ];
      MFEM_SHARED real_t sm1[2][MDQ * MDQ];
      MFEM_SHARED real_t BG[2][MD1 * MQ1];
      MFEM_SHARED real_t BGt[2][MD1 * MQ1];

      tensor_sf_mma::LoadBGBoth<MD1, MQ1>(D1D, Q1D, B, G, BG, BGt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         tensor_sf_mma::LoadX2D<MQ1>(e, D1D, X, sm0[0]);
         MFEM_SYNC_THREAD;

         tensor_sf_mma::GradX2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradY2D<MD1, MQ1, MDQ>(D1D, Q1D, BG, sm1, sm0);
         MFEM_SYNC_THREAD;

         {
            const int tid = tensor_sf_mma::getThreadIdx();
            const int nq = Q1D * Q1D;
#ifdef __CUDA_ARCH__
            const int stride = blockDim.x;
#else
            const int stride = nq;
#endif
            for (int t = tid; t < nq; t += stride)
            {
               const int qx = t % Q1D;
               const int qy = t / Q1D;
               const int idx = qx + Q1D * qy;
               const real_t gX = sm0[0][idx];
               const real_t gY = sm0[1][idx];
               const real_t O11 = D(idx, 0, e);
               const real_t O21 = D(idx, 1, e);
               const real_t O12 = SYM ? O21 : D(idx, 2, e);
               const real_t O22 = SYM ? D(idx, 2, e) : D(idx, 3, e);
               sm1[0][idx] = O11 * gX + O12 * gY;
               sm1[1][idx] = O21 * gX + O22 * gY;
            }
         }
         MFEM_SYNC_THREAD;

         tensor_sf_mma::GradYt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensor_sf_mma::GradXt2D<MD1, MQ1, MDQ>(D1D, Q1D, BGt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApplyTensorSfMma2DDispatch(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Vector &d, const Vector &x, Vector &y,
   const int, const int)
{
   if (symmetric)
   {
      SmemPADiffusionApplyTensorSfMma2D<T_D1D, T_Q1D, true>(NE, b, g, d, x, y);
   }
   else
   {
      SmemPADiffusionApplyTensorSfMma2D<T_D1D, T_Q1D, false>(NE, b, g, d, x, y);
   }
}

template <int DIM, int T_D1D, int T_Q1D>
inline void SmemPADiffusionApplyTensorSfMma(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &, const Array<real_t> &,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   if constexpr (DIM == 3)
   {
      SmemPADiffusionApplyTensorSfMma3DDispatch<T_D1D, T_Q1D>(
         NE, symmetric, b, g, d, x, y, d1d, q1d);
   }
   else
   {
      SmemPADiffusionApplyTensorSfMma2DDispatch<T_D1D, T_Q1D>(
         NE, symmetric, b, g, d, x, y, d1d, q1d);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::SmemPADiffusionApplyTensorSfMma<DIM, T_D1D, T_Q1D>;
}

inline DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_ABORT("Tensors MMA diffusion requires a specialized (D1D,Q1D) kernel"
              " (dim=" << dim << ")");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
