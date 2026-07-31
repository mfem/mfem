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

template <int T_D1D, int T_Q1D>
inline void MmaPAMassApplyTensors3D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   constexpr int NB = tensors_mma::MassNB3D<D1D, Q1D>();
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);

   const int nthreads = tensors_mma::MassThreads3D<D1D, Q1D>();
   const int nblocks = (NE + NB - 1) / NB;
   // Serial multi-element batch: shared B once; one element smem at a time.
   // Parallel z-batch (threadIdx.z) was correct but ~30% slower at (5,6).
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      tensors_mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         tensors_mma::LoadX<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         tensors_mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpZMass<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1, D, e);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D, int T_Q1D>
inline void MmaPAMassApplyTensors2D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   constexpr int MDQ = (Q1D > D1D ? Q1D : D1D);
   constexpr int NB = tensors_mma::NB2D<D1D, Q1D>();
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   const int nthreads = tensors_mma::Threads2D<D1D, Q1D>();
   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      tensors_mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         tensors_mma::LoadX2D<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         tensors_mma::InterpX2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpY2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
         MFEM_SYNC_THREAD;

         {
            const int tid = tensors_mma::getThreadIdx();
            const int nq = Q1D * Q1D;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            const int stride = blockDim.x;
#else
            const int stride = nq;
#endif
            for (int t = tid; t < nq; t += stride)
            {
               const int qx = t % Q1D;
               const int qy = t / Q1D;
               const int idx = qx + Q1D * qy;
               sm1[idx] = sm0[idx] * D(idx, e);
            }
         }
         MFEM_SYNC_THREAD;

         tensors_mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         tensors_mma::InterpXt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaPAMassApplyTensors(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &,
   const Vector &d, const Vector &x, Vector &y,
   const int, const int)
{
   if constexpr (DIM == 3)
   {
      MmaPAMassApplyTensors3D<T_D1D, T_Q1D>(NE, b, d, x, y);
   }
   else
   {
      MmaPAMassApplyTensors2D<T_D1D, T_Q1D>(NE, b, d, x, y);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaPAMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_ABORT("Tensors MMA mass requires a specialized (D1D,Q1D) kernel"
              " (dim=" << dim << ")");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
