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

#ifdef MFEM_USE_LAPACK
#include <vector>
#endif

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Prefer 1D multi-RHS GEMM for tensor SUM host apply. */
inline bool PreferTensorSumBlas(int D1D, int Q1D, int NE)
{
#ifdef MFEM_USE_LAPACK
   // Prefer when element tiling gives a wide enough 1D multi-RHS.
   const long long rhs = static_cast<long long>(D1D) * NE;
   return NE >= 4 && rhs >= 32 && (D1D * Q1D) >= 16;
#else
   (void)D1D; (void)Q1D; (void)NE;
   return false;
#endif
}


// ---- Mass SUM BLAS (1D multi-RHS GEMM) -------------------------------------

#ifdef MFEM_USE_LAPACK
/** 2D mass SUM via 1D HostGemm. Column-major; panel stages use tb='T'. */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumBlas2D(const int NE,
                                      const real_t *B, const real_t *Bt,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   const int NB = simplex_mma::HostBlasNB(Q1D, D1D);
   const int n_xy = D1D * NB;
   std::vector<real_t> xloc(static_cast<size_t>(D1D) * n_xy);
   std::vector<real_t> qq(static_cast<size_t>(Q1D) * n_xy);
   std::vector<real_t> U(static_cast<size_t>(Q1D) * Q1D * NB);
   std::vector<real_t> T(static_cast<size_t>(D1D) * Q1D * NB);
   std::vector<real_t> ytmp(static_cast<size_t>(D1D) * n_xy);

   const int ntiles = (NE + NB - 1) / NB;
   for (int tile = 0; tile < ntiles; ++tile)
   {
      const int e0 = tile * NB;
      const int nbe = std::min(NB, NE - e0);
      std::fill(xloc.begin(), xloc.end(), real_t(0));

      for (int b = 0; b < nbe; ++b)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               xloc[dx + D1D * (dy + D1D * b)] =
                  X[dx + D1D * (dy + D1D * (e0 + b))];
            }
         }
      }

      // Fat multi-RHS: qq[qx + Q*(dy + D*b)]
      simplex_mma::HostGemm('N', 'N', Q1D, n_xy, D1D, real_t(1), B, Q1D,
                            xloc.data(), D1D, real_t(0), qq.data(), Q1D);

      // U_b = B * qq_b^T  (absorbs qx↔dy pack)
      for (int b = 0; b < nbe; ++b)
      {
         simplex_mma::HostGemm('N', 'T', Q1D, Q1D, D1D, real_t(1), B, Q1D,
                               qq.data() + Q1D * D1D * b, Q1D,
                               real_t(0), U.data() + Q1D * Q1D * b, Q1D);
      }

      for (int b = 0; b < nbe; ++b)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               U[qy + Q1D * (qx + Q1D * b)] *=
                  Dv[qx + Q1D * (qy + Q1D * (e0 + b))];
            }
         }
      }

      // T_b = Bt * U_b^T
      for (int b = 0; b < nbe; ++b)
      {
         simplex_mma::HostGemm('N', 'T', D1D, Q1D, Q1D, real_t(1), Bt, D1D,
                               U.data() + Q1D * Q1D * b, Q1D,
                               real_t(0), T.data() + D1D * Q1D * b, D1D);
      }

      // ytmp_b = Bt * T_b^T
      for (int b = 0; b < nbe; ++b)
      {
         simplex_mma::HostGemm('N', 'T', D1D, D1D, Q1D, real_t(1), Bt, D1D,
                               T.data() + D1D * Q1D * b, D1D,
                               real_t(0), ytmp.data() + D1D * D1D * b, D1D);
      }

      for (int b = 0; b < nbe; ++b)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y[dx + D1D * (dy + D1D * (e0 + b))] +=
                  ytmp[dy + D1D * (dx + D1D * b)];
            }
         }
      }
   }
}

/** 3D mass SUM via 1D HostGemm, one element at a time (RHS = D1D*D1D). */
template <int D1D, int Q1D>
inline void MassApplyTensorsSumBlas3D(const int NE,
                                      const real_t *B, const real_t *Bt,
                                      const real_t *Dv, const real_t *X,
                                      real_t *Y)
{
   const int nd2 = D1D * D1D;
   const int nq2 = Q1D * Q1D;
   std::vector<real_t> xloc(static_cast<size_t>(D1D) * nd2);
   std::vector<real_t> t0(static_cast<size_t>(Q1D) * nd2);
   std::vector<real_t> t1(static_cast<size_t>(Q1D) * Q1D * D1D);
   std::vector<real_t> Az(static_cast<size_t>(D1D) * nq2);
   std::vector<real_t> U(static_cast<size_t>(Q1D) * nq2);
   std::vector<real_t> Tz(static_cast<size_t>(D1D) * nq2);
   std::vector<real_t> Ay(static_cast<size_t>(Q1D) * Q1D * D1D);
   std::vector<real_t> Ty(static_cast<size_t>(D1D) * Q1D * D1D);
   std::vector<real_t> ytmp(static_cast<size_t>(D1D) * nd2);

   for (int e = 0; e < NE; ++e)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               xloc[dx + D1D * (dy + D1D * dz)] =
                  X[dx + D1D * (dy + D1D * (dz + D1D * e))];
            }
         }
      }

      simplex_mma::HostGemm('N', 'N', Q1D, nd2, D1D, real_t(1), B, Q1D,
                            xloc.data(), D1D, real_t(0), t0.data(), Q1D);

      // t1_dz = B * t0_dz^T  (absorbs qx↔dy pack; dz as batch)
      for (int dz = 0; dz < D1D; ++dz)
      {
         simplex_mma::HostGemm('N', 'T', Q1D, Q1D, D1D, real_t(1), B, Q1D,
                               t0.data() + Q1D * D1D * dz, Q1D,
                               real_t(0), t1.data() + Q1D * Q1D * dz, Q1D);
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               Az[dz + D1D * (qx + Q1D * qy)] = t1[qy + Q1D * (qx + Q1D * dz)];
            }
         }
      }
      simplex_mma::HostGemm('N', 'N', Q1D, nq2, D1D, real_t(1), B, Q1D,
                            Az.data(), D1D, real_t(0), U.data(), Q1D);

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qz = 0; qz < Q1D; ++qz)
            {
               U[qz + Q1D * (qx + Q1D * qy)] *=
                  Dv[qx + Q1D * (qy + Q1D * (qz + Q1D * e))];
            }
         }
      }

      simplex_mma::HostGemm('N', 'N', D1D, nq2, Q1D, real_t(1), Bt, D1D,
                            U.data(), Q1D, real_t(0), Tz.data(), D1D);

      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               Ay[qy + Q1D * (qx + Q1D * dz)] = Tz[dz + D1D * (qx + Q1D * qy)];
            }
         }
      }
      simplex_mma::HostGemm('N', 'N', D1D, Q1D * D1D, Q1D, real_t(1), Bt, D1D,
                            Ay.data(), Q1D, real_t(0), Ty.data(), D1D);

      // ytmp_dz = Bt * Ty_dz^T  (absorbs dy↔qx pack)
      for (int dz = 0; dz < D1D; ++dz)
      {
         simplex_mma::HostGemm('N', 'T', D1D, D1D, Q1D, real_t(1), Bt, D1D,
                               Ty.data() + D1D * Q1D * dz, D1D,
                               real_t(0), ytmp.data() + D1D * D1D * dz, D1D);
      }

      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y[dx + D1D * (dy + D1D * (dz + D1D * e))] +=
                  ytmp[dx + D1D * (dy + D1D * dz)];
            }
         }
      }
   }
}
#endif // MFEM_USE_LAPACK

template <int D1D, int Q1D>
inline bool TryMassApplyTensorsSumBlas2D(const int NE,
                                         const Array<real_t> &b,
                                         const Array<real_t> &bt,
                                         const Vector &d,
                                         const Vector &x,
                                         Vector &y)
{
#ifdef MFEM_USE_LAPACK
   if (!PreferTensorSumBlas(D1D, Q1D, NE)) { return false; }
   MassApplyTensorsSumBlas2D<D1D, Q1D>(NE, b.Read(), bt.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#else
   (void)NE; (void)b; (void)bt; (void)d; (void)x; (void)y;
   return false;
#endif
}

template <int D1D, int Q1D>
inline bool TryMassApplyTensorsSumBlas3D(const int NE,
                                         const Array<real_t> &b,
                                         const Array<real_t> &bt,
                                         const Vector &d,
                                         const Vector &x,
                                         Vector &y)
{
#ifdef MFEM_USE_LAPACK
   if (!PreferTensorSumBlas(D1D, Q1D, NE)) { return false; }
   MassApplyTensorsSumBlas3D<D1D, Q1D>(NE, b.Read(), bt.Read(), d.Read(),
                                       x.Read(), y.ReadWrite());
   return true;
#else
   (void)NE; (void)b; (void)bt; (void)d; (void)x; (void)y;
   return false;
#endif
}

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors3D(const int NE,
                                  const Array<real_t> &b_,
                                  const Vector &d_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : tensors_mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : tensors_mma::TensorsMmaMaxQ1D;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 3D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? tensors_mma::MassNB3D<T_D1D, T_Q1D>()
                        : tensors_mma::MassNB3DRuntime(D1D);
   // Host forall_3D workers all see getThreadIdx()==0; keep one thread to avoid
   // races on MFEM_SHARED (device uses full thread count + Emulate/MMA).
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? tensors_mma::MassThreads3D<T_D1D, T_Q1D>()
                                 : tensors_mma::MassThreads3DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   // Serial multi-element batch: shared B once; one element smem at a time.
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
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

template <int T_D1D = 0, int T_Q1D = 0>
inline void MmaMassApplyTensors2D(const int NE,
                                  const Array<real_t> &b_,
                                  const Vector &d_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : tensors_mma::TensorsMmaMaxD1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : tensors_mma::TensorsMmaMaxQ1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   MFEM_VERIFY(D1D > 0 && Q1D > 0 && NE > 0, "");
   MFEM_VERIFY(D1D <= MD1 && Q1D <= MQ1, "Tensors MMA mass 2D D1D/Q1D exceeds shell cap");

   const int NB = T_D1D ? tensors_mma::NB2D<T_D1D, T_Q1D>()
                        : tensors_mma::NB2DRuntime(D1D);
   const int nthreads = Device::Allows(Backend::DEVICE_MASK)
                        ? (T_D1D ? tensors_mma::Threads2D<T_D1D, T_Q1D>()
                                 : tensors_mma::Threads2DRuntime(D1D, Q1D))
                        : 1;

   const auto B = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D * Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   const int nblocks = (NE + NB - 1) / NB;
   mfem::forall_3D(nblocks, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int b)
   {
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
            const int stride = tensors_mma::getBlockNthreads();
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
   // Host: 1D BLAS when profitable, else MMA shell (Interp/Grad Emulate).
   // Device: MMA shell (real MMA or fine-grained Emulate).
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      if constexpr (DIM == 3)
      {
         if (TryMassApplyTensorsSumBlas3D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
         { return; }
      }
      else
      {
         if (TryMassApplyTensorsSumBlas2D<T_D1D, T_Q1D>(NE, b, bt, d, x, y))
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
