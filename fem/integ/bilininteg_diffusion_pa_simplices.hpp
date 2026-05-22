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

#include <cuda/atomic>

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/* This function computes the action of the diffusion integrator for the Bernstein basis on triangles.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C2 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F2 and roughly corresponding to Algorithm 3 of [1]).

   [1] Bernstein–Bézier finite elements of arbitrary order and optimal assembly procedures.
       Ainsworth, M., Andriamaro, G., & Davydov, O. (2011).
       SIAM Journal on Scientific Computing, 33(6), 3087-3109.
   */
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApplyTriangle(const int NE,
                                     const bool symmetric,
                                     const Array<int> &lex_map_,
                                     const Array<int> &/*forward_map2d_*/,
                                     const Array<int> &/*inverse_map2d_*/,
                                     const Array<int> &/*forward_map3d_*/,
                                     const Array<int> &/*inverse_map3d_*/,
                                     const Array<real_t> &ga1_,
                                     const Array<real_t> &ga2_,
                                     const Array<real_t> &/*ga3_*/,
                                     const Array<real_t> &ga1t_,
                                     const Array<real_t> &ga2t_,
                                     const Array<real_t> &/*ga3t_*/,
                                     const Vector &d_,
                                     const Vector &x_,
                                     Vector &y_,
                                     const int d1d = 0,
                                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM = D1D * (D1D+1) / 2;
   const int p2 = (D1D-1) * (D1D-1);

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto lex_map = lex_map_.Read();
   const auto Ga1 = ConstDeviceMatrix(ga1_.Read(), D1D-1, Q1D);
   const auto Ga2 = ConstDeviceCube(ga2_.Read(), D1D-1, D1D-1, Q1D);
   const auto Ga1t = ConstDeviceMatrix(ga1t_.Read(), Q1D, D1D-1);
   const auto Ga2t = ConstDeviceCube(ga2t_.Read(), Q1D, D1D-1, D1D-1);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   const auto X = Reshape(x_.Read(), BASIS_DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), BASIS_DIM, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;

      real_t cin[2 * (max_D1D-1) * (max_D1D-1)];
      real_t C1[2 * (max_D1D-1) * max_Q1D];
      real_t C2[2 * max_Q1D * max_Q1D];
      real_t fin[2 * max_Q1D * max_Q1D];
      real_t F1[2 * (max_D1D-1) * max_Q1D];
      real_t F2[2 * (max_D1D-1) * (max_D1D-1)];

      for (int a1 = 0; a1 < D1D-1; ++a1)
      {
         for (int a2 = 0; a2 < D1D-1-a1; ++a2)
         {
            const int q = 2*(a2 + (D1D-1)*a1);
            cin[q] = 0.0;
            cin[1+q] = 0.0;
            F2[q] = 0.0;
            F2[1+q] = 0.0;
         }
         for (int i2 = 0; i2 < Q1D; ++i2)
         {
            const int q = 2*(i2 + Q1D*a1);
            C1[q] = 0.0;
            C1[1+q] = 0.0;
            F1[q] = 0.0;
            F1[1+q] = 0.0;
         }
      }
      for (int i1 = 0; i1 < Q1D; ++i1)
      {
         for (int i2 = 0; i2 < Q1D; ++i2)
         {
            const int q = 2*(i2 + Q1D*i1);
            C2[q] = 0.0;
            C2[1+q] = 0.0;
         }
      }

      // cin contains the vector coefficient
      //    cin_{\beta} = \sum_{k=1}^{3} \nabla\lambda_{k} * X_{\beta + e_{k}},
      // where \lambda_{k} are the standard barycentric coordinates and e_{k}
      // is the unit vector with nonzero value in entry k. C2 will contain the
      // value of the Bernstein polynomial
      //    \sum_{\beta} cin_{\beta} * B_{\beta}^{p-1}(\Phi(t1,t2)),
      // where \Phi is the Duffy transform and (t1,t2) is a Stroud quadrature node
      // in the unit square.
      for (int a1 = 0; a1 < D1D-1; ++a1)
      {
         for (int a2 = 0; a2 < D1D-a1-1; ++a2)
         {
            // k=0, component 0
            int idx = lex_map[a2 + D1D*(a1+1)];
            const int a1a2 = 2*(a1 + (D1D-1)*a2);
            cin[a1a2] += X(idx, e);

            // // k=1, component 0
            // idx = lex_map[(a2+1) + D1D*a1];
            // cin[a1a2] += X(idx, e) * 0.0;

            // k=2, component 0
            idx = lex_map[a2 + D1D*a1];
            cin[a1a2] -= X(idx, e);

            // // k=0, component 1
            // idx = lex_map[a2 + D1D*(a1+1)];
            // cin[1 + a1a2] += X(idx, e) * 0.0;

            // k=1, component 1
            idx = lex_map[(a2+1) + D1D*a1];
            cin[1 + a1a2] += X(idx, e);

            // k=2, component 1
            idx = lex_map[a2 + D1D*a1];
            cin[1 + a1a2] -= X(idx, e);
         }
      }

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const int a1i2 = 2*(i2 + Q1D*a1);
            for (int a2 = 0; a2 < D1D-a1-1; a2++)
            {
               const int a1a2 = 2*(a1 + (D1D-1)*a2);
               const real_t Gai = Ga2t(i2, a1, a2);
               C1[a1i2] += cin[a1a2] * Gai;
               C1[1 + a1i2] += cin[1 + a1a2] * Gai;
            }
         }
      }

      // C2 contains the Bernstein polynomial on a triangle with coefficients cin evaluated at
      // all of the Stroud quadrature nodes. E.g. if (t1,t2) is a Stroud node, then
      //    C2[i,j] = \sum_{\alpha} cin_{\alpha} * B_{\alpha}^{p-1}(\Phi(t1,t2)),
      // where \Phi is the Duffy transform.
      for (int i1 = 0; i1 < Q1D; i1++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const real_t Gai = Ga1t(i1, a1);
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int i1i2 = 2*(i2 + Q1D*i1);
               const int a1i2 = 2*(i2 + Q1D*a1);
               C2[i1i2] += C1[a1i2] * Gai;
               C2[1 + i1i2] += C1[1 + a1i2] * Gai;
            }
         }
      }

      // now evaluate the Bernstein moments
      // fin contains (B_{K})^{-1} * D(x) * (B_{K})^{-T} * C2(x). the result stored in F2
      // will be all Bernstein moments
      //    \int_{K} B_{\alpha}^{p-1}(x) * fin(x) dx.
      for (int i1 = 0; i1 < Q1D; ++i1)
      {
         for (int i2 = 0; i2 < Q1D; ++i2)
         {
            const real_t O11 = D(i1, i2, 0, e);
            const real_t O21 = D(i1, i2, 1, e);
            const real_t O12 = symmetric ? O21 : D(i1, i2, 2, e);
            const real_t O22 = symmetric ? D(i1, i2, 2, e) : D(i1, i2, 3, e);

            const int i1i2 = 2*(i2 + Q1D*i1);
            fin[i1i2] = O11 * C2[i1i2] + O12 * C2[1 + i1i2];
            fin[1 + i1i2] = O21 * C2[i1i2] + O22 * C2[1 + i1i2];
         }
      }

      // F1 computes the Bernstein moment over the first ragged tensor dimension.
      for (int i1 = 0; i1 < Q1D; i1++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const real_t Gai = Ga1(a1, i1);
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int i1i2 = 2*(i2 + Q1D*i1);
               const int a1i2 = 2*(i2 + Q1D*a1);
               F1[a1i2] += fin[i1i2] * Gai;
               F1[1 + a1i2] += fin[1 + i1i2] * Gai;
            }
         }
      }

      // F2 computes the Bernstein moment over the second/last ragged tensor dimension.
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const int a1i2 = 2*(i2 + Q1D*a1);
            for (int a2 = 0; a2 < D1D-a1-1; a2++)
            {
               const int a1a2 = 2*(a2 + (D1D-1)*a1);
               const real_t Gai = Ga2(a2, a1, i2);
               F2[a1a2] += F1[a1i2] * Gai;
               F2[1 + a1a2] += F1[1 + a1i2] * Gai;
            }
         }
      }

      // compute contributions to local RHS. we have
      //       Y_{\alpha + e_{k}} = p^{2} * \nabla\lambda_{k} * F2_{\alpha}
      // where \lambda_{k} is the kth barycentric coordinate and
      // e_{k} is the unit vector with nonzero entry k, for k=1,2,3.
      for (int a1 = 0; a1 < D1D-1; ++a1)
      {
         for (int a2 = 0; a2 < D1D-a1-1; ++a2)
         {
            // k=0
            int idx = lex_map[a2 + D1D*(a1+1)];
            const int a2a1 = 2*(a2 + (D1D-1)*a1);
            Y(idx,e) += p2 * F2[a2a1];

            // k=1
            idx = lex_map[(a2+1) + D1D*a1];
            Y(idx,e) += p2 * F2[1 + a2a1];

            // k=2
            idx = lex_map[a2 + D1D*a1];
            Y(idx,e) -= p2 * (F2[a2a1] + F2[1 + a2a1]);
         }
      }
   });
}

#if (defined(MFEM_USE_CUDA) && defined(__CUDACC__))
template<int T_D1D, int T_Q1D>
__constant__ real_t Ga1[T_Q1D][std::max(T_D1D-1, 1)];

template<int T_D1D, int T_Q1D>
int Ga1_initialized = false;
#endif

MFEM_HOST_DEVICE inline int ij_to_index(const int p, const int i, const int j) {
   return i * (2 * p - i + 1) / 2 + j;
}

MFEM_HOST_DEVICE inline int ijk_to_index(const int p, const int i, const int j, const int k) {
   const int n = p - k;

   const int layer_sum = (p * (p + 1) * (p + 2) - n * (n + 1) * (n + 2)) / 6;

   const int offset_in_layer = j * (2 * n - j + 1) / 2 + i;

   return layer_sum + offset_in_layer;
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApplyTriangle(const int NE,
                                         const bool symmetric,
                                         const Array<int> &lex_map_,
                                         const Array<int> &/*forward_map2d_*/,
                                         const Array<int> &/*inverse_map2d_*/,
                                         const Array<int> &/*forward_map3d_*/,
                                         const Array<int> &/*inverse_map3d_*/,
                                         const Array<real_t> &ga1_,
                                         const Array<real_t> &ga2_,
                                         const Array<real_t> &/*ga3_*/,
                                         const Array<real_t> &ga1t_,
                                         const Array<real_t> &ga2t_,
                                         const Array<real_t> &/*ga3t_*/,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_,
                                         const int d1d = 0,
                                         const int q1d = 0)
{
   const int D1D = T_D1D;
   const int Q1D = T_Q1D;
   const int BASIS_DIM = D1D * (D1D+1) / 2;
   const int p2 = (D1D-1) * (D1D-1);

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");
   const auto ga2_ptr = ga2_.Read();
   const auto D = Reshape(d_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), BASIS_DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), BASIS_DIM, NE);

   if (!Ga1_initialized<T_D1D, T_Q1D>)
   {
      MFEM_GPU_CHECK(cudaMemcpyToSymbol(Ga1<D1D, Q1D>, ga1_.Read(), sizeof(Ga1<D1D, Q1D>), 0, cudaMemcpyDeviceToDevice));
      Ga1_initialized<T_D1D, T_Q1D> = true;
   }

   static const int BLK = std::max(Q1D, D1D-1);
   static const int BZ = std::min(64, 128 / BLK);

   mfem::forall_2D_batch(NE, Q1D, 1, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
#if !defined(__CUDA_ARCH__)
MFEM_ABORT("Non CUDA diffusion partial assemble is not implemented");
#endif

      const int D1D = T_D1D;
      const int Q1D = T_Q1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      constexpr int BASIS_DIM = MD1 * (MD1+1) / 2;

      MFEM_SHARED real_t Ga2[MQ1][std::max(1, D1D-1)][std::max(1, D1D-1)];
      MFEM_SHARED real_t GD[BZ][2][MDQ][MDQ];

      MFEM_SHARED union
      {
         real_t Xz[BZ][BASIS_DIM];
         real_t GQ[BZ][2][MDQ][MDQ];
      } Xz_GQ;
      auto Xz = Xz_GQ.Xz;
      auto GQ = Xz_GQ.GQ;

      auto X = Xz[MFEM_THREAD_ID(z)];
      auto DQ0 = (real_t (*)[MQ1])(GD[MFEM_THREAD_ID(z)][0]);
      auto DQ1 = (real_t (*)[MQ1])(GD[MFEM_THREAD_ID(z)][1]);
      auto QQ0 = (real_t (*)[MQ1])(GQ[MFEM_THREAD_ID(z)][0]);
      auto QQ1 = (real_t (*)[MQ1])(GQ[MFEM_THREAD_ID(z)][1]);

      // load in input vector and basis data
      const int local_3d_id = MFEM_THREAD_ID(x) + MFEM_THREAD_ID(z) * BLK;
      const int local_2d_id = MFEM_THREAD_ID(x);

      const int alive_thread_count = BLK * min(BZ, NE - MFEM_BLOCK_ID(x) * BZ);

      for (int i = local_3d_id; i < Q1D * (D1D-1) * (D1D-1); i += alive_thread_count)
      {
         ((real_t *)Ga2)[i] = ga2_ptr[i];
      }

      for (int i = local_2d_id; i < BASIS_DIM; i += BLK)
      {
         X[i] = x(i,e);
      }
      MFEM_SYNC_THREAD;
      // DQ corresponds to C1 in AAD algorithm
      if (int a1 = MFEM_THREAD_ID(x); a1 < D1D-1)
      {
         real_t us[std::max(1, D1D - 1)];
         real_t vs[std::max(1, D1D - 1)];
         MFEM_UNROLL(D1D-1)
         for (int a2 = 0; a2 < D1D-1; a2++)
         {
            if (a1 + a2 >= D1D-1)
            {
               break;
            }
            const real_t x = X[ij_to_index(D1D, a2, a1)];

            us[a2] = X[ij_to_index(D1D, a2, a1+1)] - x;
            vs[a2] = X[ij_to_index(D1D, a2+1, a1)] - x;
         }

         MFEM_UNROLL(Q1D)
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            real_t uu = 0.0, vv = 0.0;
            MFEM_UNROLL(D1D-1)
            for (int a2 = 0; a2 < D1D-1; ++a2)
            {
               if (a2 >= D1D-a1-1)
               {
                  break;
               }
               const real_t Gai = Ga2[i2][a1][a2];
               uu += us[a2] * Gai;
               vv += vs[a2] * Gai;
            }
            DQ0[a1][i2] = uu;
            DQ1[a1][i2] = vv;
         }
      }
      MFEM_SYNC_THREAD;
      // QQ corresponds to C2 in AAD algorithm
      if (int i2 = MFEM_THREAD_ID(x); i2 < Q1D)
      {
         real_t us[std::max(1, D1D - 1)], vs[std::max(1, D1D - 1)];
         MFEM_UNROLL(D1D-1)
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            us[a1] = DQ0[a1][i2];
            vs[a1] = DQ1[a1][i2];
         }

         MFEM_UNROLL(Q1D)
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            real_t u = 0.0, v = 0.0;
            MFEM_UNROLL(D1D-1)
            for (int a1 = 0; a1 < D1D-1; a1++)
            {
               const real_t Gai = Ga1<D1D, Q1D>[i1][a1];
               u += us[a1] * Gai;
               v += vs[a1] * Gai;
            }

            const real_t O11 = D(i1, i2, 0, e);
            const real_t O21 = D(i1, i2, 1, e);
            const real_t O12 = symmetric ? O21 : D(i1, i2, 2, e);
            const real_t O22 = symmetric ? D(i1, i2, 2, e) : D(i1, i2, 3, e);
            const real_t gX = u;
            const real_t gY = v;

            QQ0[i1][i2] = (O11 * gX) + (O12 * gY);
            QQ1[i1][i2] = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      // DQ corresponds to F1 in AAD algorithm
      if (int i2 = MFEM_THREAD_ID(x); i2 < Q1D)
      {
         real_t us[Q1D], vs[Q1D];

         MFEM_UNROLL(Q1D)
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            us[i1] = QQ0[i1][i2];
            vs[i1] = QQ1[i1][i2];
         }

         MFEM_UNROLL(D1D-1)
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            real_t u = 0.0, v = 0.0;
            MFEM_UNROLL(Q1D)
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               u += us[i1] * Ga1<D1D, Q1D>[i1][a1];
               v += vs[i1] * Ga1<D1D, Q1D>[i1][a1];
            }
            DQ0[a1][i2] = u;
            DQ1[a1][i2] = v;
         }
      }
      MFEM_SYNC_THREAD;

      for (int i = local_2d_id; i < BASIS_DIM; i += BLK)
      {
         X[i] = Y(i,e);
      }
      MFEM_SYNC_THREAD;
      // compute F2 from AAD algorithm and add contributions to RHS
      if (int a1 = MFEM_THREAD_ID(x); a1 < D1D-1)
      {

         real_t us[Q1D], vs[Q1D];
         MFEM_UNROLL(Q1D)
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            us[i2] = DQ0[a1][i2];
            vs[i2] = DQ1[a1][i2];
         }

         MFEM_UNROLL(D1D-1)
         for (int a2 = 0; a2 < D1D-1; ++a2)
         {
            if (a2 >= D1D-a1-1)
            {
               break;
            }
            real_t u = 0.0, v = 0.0;
            MFEM_UNROLL(Q1D)
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               u += us[i2] * Ga2[i2][a1][a2];
               v += vs[i2] * Ga2[i2][a1][a2];
            }


            using Atomic = cuda::atomic<real_t, cuda::thread_scope_block>;
            // k=0
            reinterpret_cast<Atomic*>(&X[ij_to_index(D1D, a2, a1+1)])->fetch_add(p2 * u, cuda::memory_order_relaxed);
            // k=1
            reinterpret_cast<Atomic*>(&X[ij_to_index(D1D, a2+1, a1)])->fetch_add(p2 * v, cuda::memory_order_relaxed);
            // k=2
            reinterpret_cast<Atomic*>(&X[ij_to_index(D1D, a2, a1)])->fetch_add(-p2 * (u + v), cuda::memory_order_relaxed);
         }
      }
      MFEM_SYNC_THREAD;
      for (int i = local_2d_id; i < BASIS_DIM; i += BLK)
      {
         Y(i,e) = X[i];
      }
   });
}

/* This function computes the action of the diffusion integrator for the Bernstein basis on tetrahedrons.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C3 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F3 and roughly corresponding to Algorithm 3 of [1]).

   [1] Bernstein–Bézier finite elements of arbitrary order and optimal assembly procedures.
       Ainsworth, M., Andriamaro, G., & Davydov, O. (2011).
       SIAM Journal on Scientific Computing, 33(6), 3087-3109.
   */
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApplyTetrahedron(const int NE,
                                        const bool symmetric,
                                        const Array<int> &lex_map_,
                                        const Array<int> &forward_map2d_,
                                        const Array<int> &inverse_map2d_,
                                        const Array<int> &/*forward_map3d_*/,
                                        const Array<int> &inverse_map3d_,
                                        const Array<real_t> &ga1_,
                                        const Array<real_t> &ga2_,
                                        const Array<real_t> &/*ga3_*/,
                                        const Array<real_t> &ga1t_,
                                        const Array<real_t> &ga2t_,
                                        const Array<real_t> &ga3t_,
                                        const Vector &d_,
                                        const Vector &x_,
                                        Vector &y_,
                                        const int d1d = 0,
                                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM3D = D1D * (D1D+1) * (D1D+2) / 6;
   const int BASIS_DIM2D_DIFF = (D1D-1) * D1D / 2;
   const int BASIS_DIM3D_DIFF = (D1D-1) * D1D * (D1D+1) / 6;
   const int p2 = (D1D-1) * (D1D-1);

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto lex_map = lex_map_.Read();
   const auto forward_map2d = forward_map2d_.Read();
   const auto inverse_map2d = inverse_map2d_.Read();
   const auto inverse_map3d = inverse_map3d_.Read();
   const auto Ga1 = ConstDeviceMatrix(ga1_.Read(), D1D-1, Q1D);
   const auto Ga2 = ConstDeviceMatrix(ga2_.Read(), BASIS_DIM2D_DIFF, Q1D);
   const auto Ga1t = ConstDeviceMatrix(ga1t_.Read(), Q1D, D1D-1);
   const auto Ga2t = ConstDeviceMatrix(ga2t_.Read(), Q1D, BASIS_DIM2D_DIFF);
   const auto Ga3t = ConstDeviceMatrix(ga3t_.Read(), Q1D, BASIS_DIM3D_DIFF);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   const auto X = Reshape(x_.Read(), BASIS_DIM3D, NE);
   auto Y = Reshape(y_.ReadWrite(), BASIS_DIM3D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;

      constexpr int basis_dim2d = (int) 3 * (max_D1D-1) * (max_D1D) / 2;
      real_t C1[(int) 3 * basis_dim2d * max_Q1D];
      real_t C2[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t C3[3 * max_Q1D * max_Q1D * max_Q1D];
      real_t F1[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t F2[(int) 3 * basis_dim2d * max_Q1D];

      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const int q = 3*(i1 + Q1D*(i2 + Q1D*i3));
               C3[q] = 0.0;
               C3[1+q] = 0.0;
               C3[2+q] = 0.0;
            }
            for (int a1 = 0; a1 < D1D-1; a1++)
            {
               const int q = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               C2[q] = 0.0;
               C2[1+q] = 0.0;
               C2[2+q] = 0.0;
               F1[q] = 0.0;
               F1[1+q] = 0.0;
               F1[2+q] = 0.0;
            }
         }

         for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
         {
            const int q = 3*(a + BASIS_DIM2D_DIFF*i3);
            C1[q] = 0.0;
            C1[1+q] = 0.0;
            C1[2+q] = 0.0;
            F2[q] = 0.0;
            F2[1+q] = 0.0;
            F2[2+q] = 0.0;
         }
      }

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int a = 0; a < BASIS_DIM3D_DIFF; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];

         // aggregate input vector
         real_t u = 0.0, v = 0.0, w = 0.0;
         // k=3, component 0
         int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
         u -= X(idx, e);

         // k=3, component 1
         v -= X(idx, e);

         // k=3, component 2
         w -= X(idx, e);

         // k=2, component 2
         idx = lex_map[a3+1 + D1D*(a2 + D1D*a1)];
         w += X(idx, e);

         // k=1, component 1 (not computed because \nabla\lambda_{k}
         // component is 0)
         idx = lex_map[a3 + D1D*(a2+1 + D1D*a1)];
         v += X(idx, e);

         // k=0, component 0
         idx = lex_map[a3 + D1D*(a2 + D1D*(a1 + 1))];
         u += X(idx, e);

         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a_2d);
            const real_t Gai = Ga3t(i3,a);

            C1[a1a2i3] += u * Gai;
            C1[1 + a1a2i3] += v * Gai;
            C1[2 + a1a2i3] += w * Gai;
         }
      }

      // C2 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the second spatial dimension
      for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
      {
         const int a1 = inverse_map2d[2*a];
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a);
            const real_t C1x = C1[a1a2i3];
            const real_t C1y = C1[1 + a1a2i3];
            const real_t C1z = C1[2 + a1a2i3];
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const real_t Gai = Ga2t(i2,a);

               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               C2[a1i2i3] += C1x * Gai;
               C2[1 + a1i2i3] += C1y * Gai;
               C2[2 + a1i2i3] += C1z * Gai;
            }
         }
      }

      for (int a1 = 0; a1 < D1D-1; a1++)
      {
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               const real_t C2x = C2[a1i2i3];
               const real_t C2y = C2[1 + a1i2i3];
               const real_t C2z = C2[2 + a1i2i3];
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  const real_t Gai = Ga1t(i1,a1);
                  const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
                  C3[i1i2i3] += C2x * Gai;
                  C3[1 + i1i2i3] += C2y * Gai;
                  C3[2 + i1i2i3] += C2z * Gai;
               }
            }
         }
      }

      // F1 computes the Bernstein moment over the first ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const real_t O11 = D(i1,i2,i3,0,e);
               const real_t O12 = D(i1,i2,i3,1,e);
               const real_t O13 = D(i1,i2,i3,2,e);
               const real_t O21 = symmetric ? O12 : D(i1,i2,i3,3,e);
               const real_t O22 = symmetric ? D(i1,i2,i3,3,e) : D(i1,i2,i3,4,e);
               const real_t O23 = symmetric ? D(i1,i2,i3,4,e) : D(i1,i2,i3,5,e);
               const real_t O31 = symmetric ? O13 : D(i1,i2,i3,6,e);
               const real_t O32 = symmetric ? O23 : D(i1,i2,i3,7,e);
               const real_t O33 = symmetric ? D(i1,i2,i3,5,e) : D(i1,i2,i3,8,e);

               const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
               real_t gX = C3[i1i2i3];
               real_t gY = C3[1 + i1i2i3];
               real_t gZ = C3[2 + i1i2i3];

               const real_t fin1 = O11 * gX + O12 * gY + O13 * gZ;
               const real_t fin2 = O21 * gX + O22 * gY + O23 * gZ;
               const real_t fin3 = O31 * gX + O32 * gY + O33 * gZ;
               for (int a1 = 0; a1 < D1D-1; a1++)
               {
                  const real_t Gai = Ga1(a1,i1);
                  const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
                  F1[a1i2i3] += fin1 * Gai;
                  F1[1 + a1i2i3] += fin2 * Gai;
                  F1[2 + a1i2i3] += fin3 * Gai;
               }
            }
         }
      }

      // F2 computes the Bernstein moment over the second ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
            {
               const int a1 = inverse_map2d[2*a];
               const real_t Gai = Ga2(a,i2);

               const int a1a2i3 = 3*(a + BASIS_DIM2D_DIFF*i3);
               const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               F2[a1a2i3] += F1[a1i2i3] * Gai;
               F2[1 + a1a2i3] += F1[1 + a1i2i3] * Gai;
               F2[2 + a1a2i3] += F1[2 + a1i2i3] * Gai;
            }
         }
      }

      for (int a = 0; a < BASIS_DIM3D_DIFF; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];

         real_t u = 0.0, v = 0.0, w = 0.0;
         // const int a1a2a3 = 3*(a3 + a1a2);
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            // const int idx = i3 + Q1D*a;
            const real_t Gai = Ga3t(i3,a);
            const int a1a2i3 = 3*(a_2d + BASIS_DIM2D_DIFF*i3);
            u += F2[a1a2i3] * Gai;
            v += F2[1 + a1a2i3] * Gai;
            w += F2[2 + a1a2i3] * Gai;
         }

         // k=3
         int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
         Y(idx,e) -= p2 * (u + v + w);

         // k=0
         idx = lex_map[a3 + D1D*(a2 + D1D*(a1+1))];
         Y(idx,e) += p2 * u;

         // k=1
         idx = lex_map[a3 + D1D*(a2+1 + D1D*a1)];
         Y(idx,e) += p2 * v;

         // k=2
         idx = lex_map[a3+1 + D1D*(a2 + D1D*a1)];
         Y(idx,e) += p2 * w;
      }
   });
}

// collapsed algorithm with bulk loading basis
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApplyTetrahedron(const int NE,
                                            const bool symmetric,
                                            const Array<int> &lex_map_,
                                            const Array<int> &forward_map2d_,
                                            const Array<int> &inverse_map2d_,
                                            const Array<int> &forward_map3d_,
                                            const Array<int> &/*inverse_map3d_*/,
                                            const Array<real_t> &ga1_,
                                            const Array<real_t> &ga2_,
                                            const Array<real_t> &ga3_,
                                            const Array<real_t> &ga1t_,
                                            const Array<real_t> &ga2t_,
                                            const Array<real_t> &ga3t_,
                                            const Vector &d_,
                                            const Vector &x_,
                                            Vector &y_,
                                            const int d1d = 0,
                                            const int q1d = 0)
{
   static const int D1D = T_D1D;
   static const int Q1D = T_Q1D;
   const int BASIS_DIM3D = D1D * (D1D+1) * (D1D+2) / 6;
   const int BASIS_DIM2D_DIFF = (D1D-1) * D1D / 2;
   const int BASIS_DIM3D_DIFF = (D1D-1) * D1D * (D1D+1) / 6;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   const int p2 = (D1D-1) * (D1D-1);

   const int MQ1 = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int MD1 = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   const auto ga2_ptr = ga2_.Read();
   const auto ga3_ptr = ga3_.Read();
   auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), BASIS_DIM3D, NE);
   auto y = Reshape(y_.ReadWrite(), BASIS_DIM3D, NE);

   if (!Ga1_initialized<T_D1D, T_Q1D>)
   {
      MFEM_GPU_CHECK(cudaMemcpyToSymbol(Ga1<D1D, Q1D>, ga1_.Read(), sizeof(Ga1<D1D, Q1D>), 0, cudaMemcpyDeviceToDevice));
      Ga1_initialized<T_D1D, T_Q1D> = true;
   }

   static constexpr int BLK = std::max(Q1D, D1D-1);
   static constexpr int BZ = std::min(64, 128 / (BLK * BLK));

   mfem::forall_2D_batch(NE, BLK, BLK, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
#if !defined(__CUDA_ARCH__)
      MFEM_ABORT("Non CUDA diffusion partial assemble is not implemented");
#endif

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int BASIS_DIM2D_DIFF = (MD1-1) * MD1 / 2;
      constexpr int BASIS_DIM3D_DIFF = (MD1-1) * MD1 * (MD1 + 1) / 6;

      static constexpr int lds = (BLK == 2 || BLK == 4 || BLK == 8) ? BLK + 1 : BLK;

      MFEM_SHARED union
      {
         real_t contraction[BZ][3][BLK][BLK][lds];
         real_t X[BZ][BASIS_DIM3D];
      } sm0;

      MFEM_SHARED real_t Ga3[MQ1][std::max(1, BASIS_DIM3D_DIFF)];
      MFEM_SHARED real_t Ga2[MQ1][std::max(1, BASIS_DIM2D_DIFF)];
      auto X = sm0.X[MFEM_THREAD_ID(z)];
      auto DDQ0 = sm0.contraction[MFEM_THREAD_ID(z)][0];
      auto DDQ1 = sm0.contraction[MFEM_THREAD_ID(z)][1];
      auto DDQ2 = sm0.contraction[MFEM_THREAD_ID(z)][2];
      auto DQQ0 = sm0.contraction[MFEM_THREAD_ID(z)][0];
      auto DQQ1 = sm0.contraction[MFEM_THREAD_ID(z)][1];
      auto DQQ2 = sm0.contraction[MFEM_THREAD_ID(z)][2];
      auto QQQ0 = sm0.contraction[MFEM_THREAD_ID(z)][0];
      auto QQQ1 = sm0.contraction[MFEM_THREAD_ID(z)][1];
      auto QQQ2 = sm0.contraction[MFEM_THREAD_ID(z)][2];
      auto QQD0 = sm0.contraction[MFEM_THREAD_ID(z)][0];
      auto QQD1 = sm0.contraction[MFEM_THREAD_ID(z)][1];
      auto QQD2 = sm0.contraction[MFEM_THREAD_ID(z)][2];
      auto QDD0 = sm0.contraction[MFEM_THREAD_ID(z)][0];
      auto QDD1 = sm0.contraction[MFEM_THREAD_ID(z)][1];
      auto QDD2 = sm0.contraction[MFEM_THREAD_ID(z)][2];

      const int local_3d_id = MFEM_THREAD_ID(x) + MFEM_THREAD_ID(y) * BLK + MFEM_THREAD_ID(z) * BLK * BLK;
      const int local_2d_id = MFEM_THREAD_ID(x) + MFEM_THREAD_ID(y) * BLK;

      const int alive_thread_count = BLK * BLK * min(BZ, NE - MFEM_BLOCK_ID(x) * BZ);

      for (int i = local_3d_id; i < Q1D * BASIS_DIM2D_DIFF; i += alive_thread_count)
      {
         ((real_t *)Ga2)[i] = ga2_ptr[i];
      }

      for (int i = local_3d_id; i < Q1D * BASIS_DIM3D_DIFF; i += alive_thread_count)
      {
         ((real_t *)Ga3)[i] = ga3_ptr[i];
      }

      for (int i = local_2d_id; i < BASIS_DIM3D; i += BLK * BLK)
      {
         X[i] = x(i,e);
      }
      MFEM_SYNC_THREAD;
      {
         real_t us[std::max(1, D1D - 1)];
         real_t vs[std::max(1, D1D - 1)];
         real_t ws[std::max(1, D1D - 1)];
         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a2 = MFEM_THREAD_ID(x); a2 < D1D-a1-1)
            {
               MFEM_UNROLL(D1D-1)
               for (int a3 = 0; a3 < D1D-1; a3++)
               {
                  if (a1 + a2 + a3 >= D1D-1)
                  {
                     break;
                  }
                  const real_t x = X[ijk_to_index(D1D, a1, a2, a3)];

                  us[a3] = X[ijk_to_index(D1D, a1 + 1, a2, a3)] - x;
                  vs[a3] = X[ijk_to_index(D1D, a1, a2 + 1, a3)] - x;
                  ws[a3] = X[ijk_to_index(D1D, a1, a2, a3 + 1)] - x;
               }
            }
         }

         MFEM_SYNC_THREAD;

         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a2 = MFEM_THREAD_ID(x); a2 < D1D-a1-1)
            {

              MFEM_UNROLL(Q1D)
              for (int i3 = 0; i3 < Q1D; i3++)
              {
                 real_t uu = 0.0, vv = 0.0, ww = 0.0;
                 MFEM_UNROLL(D1D-1)
                 for (int a3 = 0; a3 < D1D-1; a3++)
                 {
                    if (a1 + a2 + a3 >= D1D-1)
                    {
                       break;
                    }
                    const int a = ijk_to_index(D1D - 1, a1, a2, a3);

                    const auto u = us[a3];
                    const auto v = vs[a3];
                    const auto w = ws[a3];

                    const real_t Gai = Ga3[i3][a];
                    uu += u * Gai;
                    vv += v * Gai;
                    ww += w * Gai;
                 }
                 DDQ0[a1][a2][i3] = uu;
                 DDQ1[a1][a2][i3] = vv;
                 DDQ2[a1][a2][i3] = ww;
              }
            }
         }
      }
      MFEM_SYNC_THREAD;
      {
         real_t us[std::max(1, D1D - 1)];
         real_t vs[std::max(1, D1D - 1)];
         real_t ws[std::max(1, D1D - 1)];
         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(D1D-1)
               for (int a2 = 0; a2 < D1D-1; a2++)
               {
                  if (a1 + a2 >= D1D-1)
                  {
                     break;
                  }

                  us[a2] = DDQ0[a1][a2][a3];
                  vs[a2] = DDQ1[a1][a2][a3];
                  ws[a2] = DDQ2[a1][a2][a3];
               }
            }
         }

         MFEM_SYNC_THREAD;

         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(Q1D)
               for (int i2 = 0; i2 < Q1D; i2++)
               {
                  real_t u = 0.0, v = 0.0, w = 0.0;
                  MFEM_UNROLL(D1D-1)
                  for (int a2 = 0; a2 < D1D-1; a2++)
                  {
                     if (a1 + a2 >= D1D-1)
                     {
                        break;
                     }
                     const int a_2d = ij_to_index(D1D - 1, a1, a2);
                     u += us[a2] * Ga2[i2][a_2d];
                     v += vs[a2] * Ga2[i2][a_2d];
                     w += ws[a2] * Ga2[i2][a_2d];
                  }
                  DQQ0[a1][i2][a3] = u;
                  DQQ1[a1][i2][a3] = v;
                  DQQ2[a1][i2][a3] = w;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      {
         real_t us[std::max(1, D1D - 1)];
         real_t vs[std::max(1, D1D - 1)];
         real_t ws[std::max(1, D1D - 1)];
         if (int a2 = MFEM_THREAD_ID(y); a2 < Q1D)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(D1D-1)
               for (int a1 = 0; a1 < D1D-1; a1++)
               {
                  us[a1] = DQQ0[a1][a2][a3];
                  vs[a1] = DQQ1[a1][a2][a3];
                  ws[a1] = DQQ2[a1][a2][a3];
               }
            }
         }

         MFEM_SYNC_THREAD;

         if (int a2 = MFEM_THREAD_ID(y); a2 < Q1D)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(Q1D)
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  real_t u = 0.0, v = 0.0, w = 0.0;
                  MFEM_UNROLL(D1D-1)
                  for (int a1 = 0; a1 < D1D-1; a1++)
                  {
                     u += us[a1] * Ga1<D1D, Q1D>[i1][a1];
                     v += vs[a1] * Ga1<D1D, Q1D>[i1][a1];
                     w += ws[a1] * Ga1<D1D, Q1D>[i1][a1];
                  }
                  const real_t O11 = d(i1,a2,a3,0,e);
                  const real_t O12 = d(i1,a2,a3,1,e);
                  const real_t O13 = d(i1,a2,a3,2,e);
                  const real_t O21 = symmetric ? O12 : d(i1,a2,a3,3,e);
                  const real_t O22 = symmetric ? d(i1,a2,a3,3,e) : d(i1,a2,a3,4,e);
                  const real_t O23 = symmetric ? d(i1,a2,a3,4,e) : d(i1,a2,a3,5,e);
                  const real_t O31 = symmetric ? O13 : d(i1,a2,a3,6,e);
                  const real_t O32 = symmetric ? O23 : d(i1,a2,a3,7,e);
                  const real_t O33 = symmetric ? d(i1,a2,a3,5,e) : d(i1,a2,a3,8,e);
                  const real_t gX = u;
                  const real_t gY = v;
                  const real_t gZ = w;
                  QQQ0[i1][a2][a3] = O11 * gX + O12 * gY + O13 * gZ;
                  QQQ1[i1][a2][a3] = O21 * gX + O22 * gY + O23 * gZ;
                  QQQ2[i1][a2][a3] = O31 * gX + O32 * gY + O33 * gZ;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      {
         real_t us[Q1D];
         real_t vs[Q1D];
         real_t ws[Q1D];
         if (int a2 = MFEM_THREAD_ID(y); a2 < Q1D)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(Q1D)
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  us[i1] = QQQ0[i1][a2][a3];
                  vs[i1] = QQQ1[i1][a2][a3];
                  ws[i1] = QQQ2[i1][a2][a3];
               }
            }
         }

         MFEM_SYNC_THREAD;

         if (int a2 = MFEM_THREAD_ID(y); a2 < Q1D)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(D1D-1)
               for (int a1 = 0; a1 < D1D-1; a1++)
               {
                  real_t u = 0.0, v = 0.0, w = 0.0;
                  MFEM_UNROLL(Q1D)
                  for (int i1 = 0; i1 < Q1D; i1++)
                  {
                     u += us[i1] * Ga1<D1D, Q1D>[i1][a1];
                     v += vs[i1] * Ga1<D1D, Q1D>[i1][a1];
                     w += ws[i1] * Ga1<D1D, Q1D>[i1][a1];
                  }
                  QQD0[a1][a2][a3] = u;
                  QQD1[a1][a2][a3] = v;
                  QQD2[a1][a2][a3] = w;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      {
         real_t us[Q1D];
         real_t vs[Q1D];
         real_t ws[Q1D];
         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(Q1D)
               for (int i2 = 0; i2 < Q1D; i2++)
               {
                  us[i2] = QQD0[a1][i2][a3];
                  vs[i2] = QQD1[a1][i2][a3];
                  ws[i2] = QQD2[a1][i2][a3];
               }
            }
         }

         MFEM_SYNC_THREAD;

         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a3 = MFEM_THREAD_ID(x); a3 < Q1D)
            {
               MFEM_UNROLL(D1D-1)
               for (int a2 = 0; a2 < D1D-1; a2++)
               {
                  if (a1 + a2 >= D1D-1)
                  {
                     break;
                  }
                  const int a_2d = ij_to_index(D1D - 1, a1, a2);
                  real_t u = 0.0, v = 0.0, w = 0.0;
                  MFEM_UNROLL(Q1D)
                  for (int i2 = 0; i2 < Q1D; i2++) {
                     u += us[i2] * Ga2[i2][a_2d];
                     v += vs[i2] * Ga2[i2][a_2d];
                     w += ws[i2] * Ga2[i2][a_2d];
                  }
                  QDD0[a1][a2][a3] = u;
                  QDD1[a1][a2][a3] = v;
                  QDD2[a1][a2][a3] = w;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      {

         real_t us[Q1D];
         real_t vs[Q1D];
         real_t ws[Q1D];
         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a2 = MFEM_THREAD_ID(x); a2 < D1D-a1-1)
            {
               MFEM_UNROLL(Q1D)
               for (int i3 = 0; i3 < Q1D; i3++)
               {
                  us[i3] = QDD0[a1][a2][i3];
                  vs[i3] = QDD1[a1][a2][i3];
                  ws[i3] = QDD2[a1][a2][i3];
               }
            }
         }

         MFEM_SYNC_THREAD;

         for (int i = local_2d_id; i < BASIS_DIM3D; i += BLK * BLK)
         {
            X[i] = y(i,e);
         }

         MFEM_SYNC_THREAD;

         if (int a1 = MFEM_THREAD_ID(y); a1 < D1D-1)
         {
            if (int a2 = MFEM_THREAD_ID(x); a2 < D1D-a1-1)
            {
              MFEM_UNROLL(D1D-1)
              for (int a3 = 0; a3 < D1D-1; a3++)
              {
                 if (a1 + a2 + a3 >= D1D-1)
                 {
                    break;
                 }
                 const int a = ijk_to_index(D1D - 1, a1, a2, a3);
                 real_t u = 0.0, v = 0.0, w = 0.0;
                 MFEM_UNROLL(Q1D)
                 for (int i3 = 0; i3 < Q1D; i3++) {
                    u += us[i3] * Ga3[i3][a];
                    v += vs[i3] * Ga3[i3][a];
                    w += ws[i3] * Ga3[i3][a];
                 }

                 using Atomic = cuda::atomic<real_t, cuda::thread_scope_block>;
                 reinterpret_cast<Atomic*>(&X[ijk_to_index(D1D, a1, a2, a3)])->fetch_add(-p2 * (u + v + w), cuda::memory_order_relaxed);
                 reinterpret_cast<Atomic*>(&X[ijk_to_index(D1D, a1+1, a2, a3)])->fetch_add(p2 * u, cuda::memory_order_relaxed);
                 reinterpret_cast<Atomic*>(&X[ijk_to_index(D1D, a1, a2+1, a3)])->fetch_add(p2 * v, cuda::memory_order_relaxed);
                 reinterpret_cast<Atomic*>(&X[ijk_to_index(D1D, a1, a2, a3+1)])->fetch_add(p2 * w, cuda::memory_order_relaxed);
              }
            }
         }

         MFEM_SYNC_THREAD;

         for (int i = local_2d_id; i < BASIS_DIM3D; i += BLK * BLK)
         {
            y(i,e) = X[i];
         }
      }
   });
}
} // namespace internal

template<int DIM, int D1D, int Q1D>
DiffusionIntegrator::ApplySimplexKernelType
DiffusionIntegrator::ApplySimplexPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPADiffusionApplyTriangle<D1D, Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPADiffusionApplyTetrahedron<D1D, Q1D>;
   }
   else { MFEM_ABORT(""); }
}

inline DiffusionIntegrator::ApplySimplexKernelType
DiffusionIntegrator::ApplySimplexPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2)
   {
      return internal::PADiffusionApplyTriangle;
   }
   else if (dim == 3)
   {
      return internal::PADiffusionApplyTetrahedron;
   }
   else { MFEM_ABORT(""); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

