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
#include "bilininteg_pa_tensor_sf_mma.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

template <int T_D1D, int T_Q1D>
inline void SmemPAMassApplyTensorSfMma3D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);

   const int nthreads = ((Q1D * Q1D * Q1D + 31) / 32) * 32;
   mfem::forall_3D(NE, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sB[MD1 * MQ1];

      tensor_sf_mma::LoadB<MD1, MQ1>(D1D, Q1D, B, sB);
      {
         const int tid = tensor_sf_mma::getThreadIdx();
         if (tid < D1D * D1D * D1D)
         {
            const int dx = tid % D1D;
            const int div = tid / D1D;
            const int dy = div % D1D;
            const int dz = div / D1D;
            sm0[dx + D1D * (dy + D1D * dz)] = X(dx, dy, dz, e);
         }
      }
      MFEM_SYNC_THREAD;

      tensor_sf_mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpZ<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
      MFEM_SYNC_THREAD;

      {
         const int tid = tensor_sf_mma::getThreadIdx();
         if (tid < Q1D * Q1D * Q1D)
         {
            const int qx = tid % Q1D;
            const int div = tid / Q1D;
            const int qy = div % Q1D;
            const int qz = div / Q1D;
            const int idx = qx + Q1D * qy + Q1D * Q1D * qz;
            const int q = qx + Q1D * (qy + Q1D * qz);
            sm0[idx] = sm1[idx] * D(q, e);
         }
      }
      MFEM_SYNC_THREAD;

      tensor_sf_mma::LoadBt<MD1, MQ1>(D1D, Q1D, B, sB);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpZt<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpYt<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpXt<MD1, MQ1>(D1D, Q1D, sB, sm0, Y, e);
   });
}

template <int T_D1D, int T_Q1D>
inline void SmemPAMassApplyTensorSfMma2D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_)
{
   constexpr int D1D = T_D1D, Q1D = T_Q1D;
   constexpr int MDQ = (Q1D > D1D ? Q1D : D1D);
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   const int nthreads = ((Q1D * Q1D + 31) / 32) * 32;
   mfem::forall_3D(NE, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D, MD1 = T_D1D;
      MFEM_SHARED real_t sm0[MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sB[MD1 * MQ1];

      tensor_sf_mma::LoadB<MD1, MQ1>(D1D, Q1D, B, sB);
      tensor_sf_mma::LoadX2D<MQ1>(e, D1D, X, sm0);
      MFEM_SYNC_THREAD;

      tensor_sf_mma::InterpX2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, sm1);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpY2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
      MFEM_SYNC_THREAD;

      {
         const int tid = tensor_sf_mma::getThreadIdx();
         if (tid < Q1D * Q1D)
         {
            const int qx = tid % Q1D;
            const int qy = tid / Q1D;
            const int idx = qx + Q1D * qy;
            sm1[idx] = sm0[idx] * D(idx, e);
         }
      }
      MFEM_SYNC_THREAD;

      tensor_sf_mma::LoadBt<MD1, MQ1>(D1D, Q1D, B, sB);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
      MFEM_SYNC_THREAD;
      tensor_sf_mma::InterpXt2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, Y, e);
   });
}

template <int DIM, int T_D1D, int T_Q1D>
inline void SmemPAMassApplyTensorSfMma(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &,
   const Vector &d, const Vector &x, Vector &y,
   const int, const int)
{
   if constexpr (DIM == 3)
   {
      SmemPAMassApplyTensorSfMma3D<T_D1D, T_Q1D>(NE, b, d, x, y);
   }
   else
   {
      SmemPAMassApplyTensorSfMma2D<T_D1D, T_Q1D>(NE, b, d, x, y);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTensorSfMmaKernelType
MassIntegrator::ApplyTensorSfMmaPAKernels::Kernel()
{
   return internal::SmemPAMassApplyTensorSfMma<DIM, T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTensorSfMmaKernelType
MassIntegrator::ApplyTensorSfMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_ABORT("Tensor SF-MMA mass requires a specialized (D1D,Q1D) kernel"
              " (dim=" << dim << ")");
   return nullptr;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
