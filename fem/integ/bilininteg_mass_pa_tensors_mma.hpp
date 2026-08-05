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

namespace mma
{
namespace blas
{

// ---- Host mass apply (dense sum-fact) -------------------------------------

/** Dense sum-fact host mass 2D with serial over element tiles. */
template <int D1D, int Q1D>
inline void MassApplyTensors2D(const int NE, const real_t *B,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   auto apply_e = [&](int e)
   {
      real_t sol_xy[Q1D][Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx) { sol_xy[qy][qx] = real_t(0); }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t sol_x[Q1D];
         for (int qx = 0; qx < Q1D; ++qx) { sol_x[qx] = real_t(0); }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X[dx + D1D * (dy + D1D * e)];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B[qx + Q1D * dx] * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t d2q = B[qy + Q1D * dy];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] *= Dv[qx + Q1D * (qy + Q1D * e)];
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t sol_x[D1D];
         for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = real_t(0); }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t s = sol_xy[qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += B[qx + Q1D * dx] * s; // Bt(dx,qx)
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t q2d = B[qy + Q1D * dy]; // Bt(dy,qy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y[dx + D1D * (dy + D1D * e)] += q2d * sol_x[dx];
            }
         }
      }
   };
   // Tile over elements for multi-RHS batching.
   const int NB = mma::TensorTileNB(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   }
}

/** blas_ sum-fact host mass 3D with serial over element tiles. */
template <int D1D, int Q1D>
inline void MassApplyTensors3D(const int NE, const real_t *B,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   auto apply_e = [&](int e)
   {
      real_t sol_xyz[Q1D][Q1D][Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xyz[qz][qy][qx] = real_t(0);

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t sol_xy[Q1D][Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xy[qy][qx] = real_t(0);
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t sol_x[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) { sol_x[qx] = real_t(0); }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X[dx + D1D * (dy + D1D * (dz + D1D * e))];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B[qx + Q1D * dx] * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = B[qy + Q1D * dy];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz = B[qz + Q1D * dz];
            for (int qy = 0; qy < Q1D; ++qy)
               for (int qx = 0; qx < Q1D; ++qx)
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
               sol_xyz[qz][qy][qx] *=
                  Dv[qx + Q1D * (qy + Q1D * (qz + Q1D * e))];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t sol_xy[D1D][D1D];
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
               sol_xy[dy][dx] = real_t(0);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t sol_x[D1D];
            for (int dx = 0; dx < D1D; ++dx) { sol_x[dx] = real_t(0); }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += B[qx + Q1D * dx] * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy = B[qy + Q1D * dy];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz = B[qz + Q1D * dz];
            for (int dy = 0; dy < D1D; ++dy)
               for (int dx = 0; dx < D1D; ++dx)
                  Y[dx + D1D * (dy + D1D * (dz + D1D * e))] +=
                     wz * sol_xy[dy][dx];
         }
      }
   };
   const int NB = mma::TensorTileNB3D(D1D, Q1D);
   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      for (int b = 0; b < nbe; ++b) { apply_e(e0 + b); }
   }
}

/** Host mass tensor apply: dense sum-fact (blas_). Always preferred for the
    registered D1D range (4..TensorsMmaMax) over the MMA Emulate shell. */
template <int D1D, int Q1D>
inline bool TryMassApplyTensors2D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> & /*bt*/,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y)
{
   if (!mma::host_PreferTensor(D1D, Q1D, NE)) { return false; }
   MassApplyTensors2D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                     x.Read(), y.ReadWrite());
   return true;
}

template <int D1D, int Q1D>
inline bool TryMassApplyTensors3D(const int NE,
                                       const Array<real_t> &b,
                                       const Array<real_t> & /*bt*/,
                                       const Vector &d,
                                       const Vector &x,
                                       Vector &y)
{
   if (!mma::host_PreferTensor(D1D, Q1D, NE)) { return false; }
   MassApplyTensors3D<D1D, Q1D>(NE, b.Read(), d.Read(),
                                     x.Read(), y.ReadWrite());
   return true;
}

} // namespace blas
} // namespace mma

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors3D(const int NE,
                                  const Array<real_t> &b,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : mma::TensorsMmaMaxQ1D;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 3D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::MassNB3D<T_D1D, T_Q1D>()
                        : mma::MassNB3DRuntime(D1D);
   // Host forall_3D workers all see getThreadIdxX()==0; keep one thread to avoid
   // races on MFEM_SHARED (device uses full thread count + Emulate/MMA).
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::MassThreads3D<T_D1D, T_Q1D>()
                                 : mma::MassThreads3DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D * Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   // Serial multi-element batch: shared B once; one element smem at a time.
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sm1[MQ1 * MQ1 * MQ1];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         mma::InterpX<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpY<MD1, MQ1>(D1D, Q1D, sB, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpZMass<MD1, MQ1>(D1D, Q1D, sB, sm0, sm1, D, e);
         MFEM_SYNC_THREAD;
         mma::InterpZt<MD1, MQ1>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpYt<MD1, MQ1>(D1D, Q1D, sBt, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpXt<MD1, MQ1>(D1D, Q1D, sBt, sm1, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors2D(const int NE,
                                  const Array<real_t> &b,
                                  const Vector &d,
                                  const Vector &x,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : mma::TensorsMmaMaxQ1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 2D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? mma::NB2D<T_D1D, T_Q1D>()
                        : mma::NB2DRuntime(D1D);
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? mma::Threads2D<T_D1D, T_Q1D>()
                                 : mma::Threads2DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D * Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
      MFEM_SHARED real_t sm0[MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ];
      MFEM_SHARED real_t sB[MD1 * MQ1];
      MFEM_SHARED real_t sBt[MD1 * MQ1];

      mma::LoadBBoth<MD1, MQ1>(D1D, Q1D, B, sB, sBt);
      MFEM_SYNC_THREAD;

      for (int i = 0; i < NB; i++)
      {
         const int e = b * NB + i;
         if (e >= NE) { break; }

         mma::LoadX2D<MQ1>(e, D1D, X, sm0);
         MFEM_SYNC_THREAD;

         mma::InterpX2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm0, sm1);
         MFEM_SYNC_THREAD;
         mma::InterpY2D<MD1, MQ1, MDQ>(D1D, Q1D, sB, sm1, sm0);
         MFEM_SYNC_THREAD;

         {
            const int tid = mma::getThreadIdxX();
            const int nq = Q1D * Q1D;
            const int stride = mma::getBlockNthreadsX();
            for (int t = tid; t < nq; t += stride)
            {
               const int qx = t % Q1D;
               const int qy = t / Q1D;
               const int idx = qx + Q1D * qy;
               sm1[idx] = sm0[idx] * D(idx, e);
            }
         }
         MFEM_SYNC_THREAD;

         mma::InterpYt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm1, sm0);
         MFEM_SYNC_THREAD;
         mma::InterpXt2D<MD1, MQ1, MDQ>(D1D, Q1D, sBt, sm0, Y, e);
         MFEM_SYNC_THREAD;
      }
   });
}

/** Runtime overloads for Fallback / unregistered (D1D,Q1D). */
inline void MmaMassApplyTensors2D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &bt,
                                  const Vector &d, const Vector &x,
                                  Vector &y,
                                  const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MmaMassApplyTensors2D<0, 0>(NE, b, d, x, y, d1d, q1d);
}

inline void MmaMassApplyTensors3D(const int NE,
                                  const Array<real_t> &b,
                                  const Array<real_t> &bt,
                                  const Vector &d, const Vector &x,
                                  Vector &y,
                                  const int d1d, const int q1d)
{
   MFEM_CONTRACT_VAR(bt);
   MmaMassApplyTensors3D<0, 0>(NE, b, d, x, y, d1d, q1d);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaMassApplyTensors(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   // Host: blas_ sum-fact when profitable, else MMA shell (Emulate).
   // Device: MMA shell (real MMA or fine-grained Emulate).
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      if constexpr (DIM == 3)
      {
         if (mma::blas::TryMassApplyTensors3D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
         { return; }
      }
      else
      {
         if (mma::blas::TryMassApplyTensors2D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
         { return; }
      }
   }
   if constexpr (DIM == 3)
   {
      MmaMassApplyTensors3D<T_D1D, T_Q1D>(NE, b, d, x, y, d1d, q1d);
   }
   else
   {
      MmaMassApplyTensors2D<T_D1D, T_Q1D>(NE, b, d, x, y, d1d, q1d);
   }
}

} // namespace internal

template <int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

// Fallback defined in bilininteg_mass_pa_tensors_mma.cpp (MMA shell runtime).

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
