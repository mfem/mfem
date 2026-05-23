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

// On CUDA/HIP, store the small 1D "Ba1" basis (used by the
// SmemPAMassApply{Triangle,Tetrahedron} kernels) in device __constant__
// memory: broadcast-read by every thread, the constant cache is faster
// than shared memory for these accesses. The two kernels use distinct
// symbols because triangle and tetrahedron use different first-dimension
// 1D quadrature rules in Stroud's construction. On non-GPU builds the
// symbol is a regular global; the host-side kernel body falls through
// to the shared-memory copy loaded cooperatively.
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#define MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST 1
#define MFEM_PA_SIMPLEX_MASS_DEVICE_CONST __constant__
#define MFEM_PA_SIMPLEX_MASS_MEMCPY_TO_SYMBOL cudaMemcpyToSymbol
#define MFEM_PA_SIMPLEX_MASS_MEMCPY_D2D       cudaMemcpyDeviceToDevice
#define MFEM_PA_SIMPLEX_MASS_SYMBOL(...) (__VA_ARGS__)
#elif defined(MFEM_USE_HIP) && defined(__HIPCC__)
#define MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST 1
#define MFEM_PA_SIMPLEX_MASS_DEVICE_CONST __constant__
#define MFEM_PA_SIMPLEX_MASS_MEMCPY_TO_SYMBOL hipMemcpyToSymbol
#define MFEM_PA_SIMPLEX_MASS_MEMCPY_D2D       hipMemcpyDeviceToDevice
// Per the HIP docs, hipMemcpyToSymbol must wrap the symbol in HIP_SYMBOL().
#define MFEM_PA_SIMPLEX_MASS_SYMBOL(...) HIP_SYMBOL((__VA_ARGS__))
#else
#define MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST 0
#define MFEM_PA_SIMPLEX_MASS_DEVICE_CONST
#endif

#if MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST
template<int T_D1D, int T_Q1D>
MFEM_PA_SIMPLEX_MASS_DEVICE_CONST real_t SmemPAMassTriBa1[T_Q1D][T_D1D];
template<int T_D1D, int T_Q1D>
bool SmemPAMassTriBa1Init = false;

template<int T_D1D, int T_Q1D>
MFEM_PA_SIMPLEX_MASS_DEVICE_CONST real_t SmemPAMassTetBa1[T_Q1D][T_D1D];
template<int T_D1D, int T_Q1D>
bool SmemPAMassTetBa1Init = false;
#endif

/* This function computes the action of the mass integrator for the Bernstein basis on triangles.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C2 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F2 and roughly corresponding to Algorithm 3 of [1]).

   [1] Bernstein–Bézier finite elements of arbitrary order and optimal assembly procedures.
       Ainsworth, M., Andriamaro, G., & Davydov, O. (2011).
       SIAM Journal on Scientific Computing, 33(6), 3087-3109.
   */
template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApplyTriangle_Element(const int e,
                                 const int NE,
                                 const int BASIS_DIM,
                                 const int *lex_map_,
                                 const real_t *ba1_,
                                 const real_t *ba2_,
                                 const real_t *ba1t_,
                                 const real_t *ba2t_,
                                 const real_t *d_,
                                 const real_t *x_,
                                 real_t *y_,
                                 const int d1d = 0,
                                 const int q1d = 0)
{
   const int D1D = d1d, Q1D = q1d;
   constexpr int max_D1D = DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int max_Q1D = DofQuadLimits::MAX_Q1D_SIMPLEX;

   const auto lex_map = DeviceTensor<2,const int>(lex_map_, D1D, D1D);
   const auto Ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto Ba2 = ConstDeviceCube(ba2_, D1D, D1D, Q1D);
   const auto Ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto Ba2t = ConstDeviceCube(ba2t_, Q1D, D1D, D1D);

   const auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   const auto X = ConstDeviceMatrix(x_, BASIS_DIM, NE);
   auto Y = DeviceMatrix(y_, BASIS_DIM, NE);

   if (!ACCUMULATE)
   {
      for (int idx = 0; idx < BASIS_DIM; idx++)
      {
         Y(idx, e) = 0.0;
      }
   }

   // C2 will contain the Bernstein polynomial with coefficients X
   // evaluated at all of the quadrature nodes in O(p^{d+1}). we have
   //    C2[t1,t2] = \sum_{\alpha} X_{\alpha} * B_{\alpha}^{p}(\Phi(t1,t2)),
   // where \Phi is the Duffy transformation and (t1,t2) is a Stroud node.
   real_t C2[max_Q1D * max_Q1D];
   real_t C1[max_D1D * max_Q1D];

   for (int i1 = 0; i1 < Q1D; i1++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         const int q = i2 + Q1D*i1;
         C2[q] = 0.0;
      }
      for (int a1 = 0; a1 < D1D; a1++)
      {
         const int q = a1 + D1D*i1;
         C1[q] = 0.0;
      }
   }

   // quad to dofs operation (i.e. evaluating Bernstein polynomial at all quad nodes),
   // step 1: convert first quadrature index to first multiindex.
   for (int i2 = 0; i2 < Q1D; i2++)
   {
      for (int a1 = 0; a1 < D1D; a1++)
      {
         const int a1i2 = a1 + D1D*i2;
         for (int a2 = 0; a2 < D1D-a1; a2++)
         {
            const int idx = lex_map(a2, a1);
            C1[a1i2] += X(idx, e) * Ba2(a2, a1, i2);
         }
      }
   }
   // quad to dofs operation, step 2: convert second quadrature index to second
   // multiindex. C2 contains the Bernstein polynomial on a triangle with
   // coefficients X evaluated at all of the Stroud quadrature nodes. E.g. if
   // (t1,t2) is a Stroud node, then
   //    C2[i,j] = \sum_{\alpha} X_{\alpha} * B_{\alpha}^{p-1}(\Phi(t1,t2)),
   // where \Phi is the Duffy transform.
   for (int i1 = 0; i1 < Q1D; i1++)
   {
      for (int a1 = 0; a1 < D1D; a1++)
      {
         const real_t Bai = Ba1(a1, i1);
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            C2[i2 + Q1D*i1] += C1[a1 + D1D*i2] * Bai;
         }
      }
   }
   for (int i1 = 0; i1 < Q1D; i1++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         C2[i2 + Q1D*i1] *= D(i1, i2, e);
      }
   }
   // dofs to quad operation (i.e. evaluating all Bernstein moments of the form
   // \int_{K} B_{\alpha}^{p}(x) * C2(x) dx), step 1: convert first multiindex to
   // first quadrature index. Note: here, C1 corresponds to F1 in the AAD
   // algorithm.
   for (int i2 = 0; i2 < Q1D; i2++)
   {
      for (int a1 = 0; a1 < D1D; a1++)
      {
         C1[a1 + D1D*i2] = 0.0;
      }
   }
   for (int i1 = 0; i1 < Q1D; i1++)
   {
      for (int a1 = 0; a1 < D1D; a1++)
      {
         const real_t Bai = Ba1t(i1, a1);
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            C1[i2 + Q1D*a1] += C2[i2 + Q1D*i1] * Bai;
         }
      }
   }
   // dofs to quad operation, step 2: convert second multiindex to second
   // quadrature index. The contribution to the local RHS is
   //    Y_{\alpha} = F2_{\alpha}.
   for (int a1 = 0; a1 < D1D; a1++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         const int a1i2 = i2 + Q1D*a1;
         for (int a2 = 0; a2 < D1D-a1; a2++)
         {
            const int idx = lex_map(a2, a1);
            Y(idx,e) += C1[a1i2] * Ba2t(i2, a1, a2);
         }
      }
   }
}

// PA Mass Apply 2D kernel on triangles (Bernstein only)
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApplyTriangle(const int NE,
                                const Array<int> &lex_map_,
                                const Array<int> &/*forward_map2d_*/,
                                const Array<int> &/*inverse_map2d_*/,
                                const Array<int> &/*forward_map3d_*/,
                                const Array<int> &/*inverse_map3d_*/,
                                const Array<real_t> &ba1_,
                                const Array<real_t> &ba2_,
                                const Array<real_t> &/*ba3_*/,
                                const Array<real_t> &ba1t_,
                                const Array<real_t> &ba2t_,
                                const Array<real_t> &/*ba3t_*/,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM = D1D * (D1D + 1) / 2;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto lex_map = lex_map_.Read();
   const auto Ba1 = ba1_.Read();
   const auto Ba2 = ba2_.Read();
   const auto Ba1t = ba1t_.Read();
   const auto Ba2t = ba2t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApplyTriangle_Element(e, NE, BASIS_DIM,
                                            lex_map, Ba1, Ba2, Ba1t, Ba2t, D,
                                            X, Y,
                                            d1d, q1d);
   });
}

MFEM_HOST_DEVICE inline int ij_to_index(const int p, const int i, const int j)
{
   return i * (2 * p - i + 1) / 2 + j;
}

MFEM_HOST_DEVICE inline int ijk_to_index(const int p, const int i, const int j,
                                         const int k)
{
   const int n = p - k;
   const int layer_sum = (p * (p + 1) * (p + 2) - n * (n + 1) * (n + 2)) / 6;
   const int offset_in_layer = j * (2 * n - j + 1) / 2 + i;
   return layer_sum + offset_in_layer;
}

// PA Mass Apply 2D kernel on triangles with shared memory
template<int T_D1D, int T_Q1D>
inline void SmemPAMassApplyTriangle(const int NE,
                                    const Array<int> &lex_map_,
                                    const Array<int> &forward_map2d_,
                                    const Array<int> &inverse_map2d_,
                                    const Array<int> &forward_map3d_,
                                    const Array<int> &inverse_map3d_,
                                    const Array<real_t> &ba1_,
                                    const Array<real_t> &ba2_,
                                    const Array<real_t> &ba3_,
                                    const Array<real_t> &ba1t_,
                                    const Array<real_t> &ba2t_,
                                    const Array<real_t> &ba3t_,
                                    const Vector &d_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int = 0,
                                    const int = 0)
{
   static_assert(T_D1D != 0 && T_Q1D != 0,
                 "SmemPAMassApplyTriangle requires compile-time D1D and Q1D");

   // Host backend: dispatch to the non-Smem fallback. The Smem kernel
   // reads Ba1 from __constant__ memory on GPU.
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      PAMassApplyTriangle<T_D1D, T_Q1D>(
         NE, lex_map_, forward_map2d_, inverse_map2d_, forward_map3d_,
         inverse_map3d_, ba1_, ba2_, ba3_, ba1t_, ba2t_, ba3t_,
         d_, x_, y_, T_D1D, T_Q1D);
      return;
   }

   constexpr int D1D = T_D1D;
   constexpr int Q1D = T_Q1D;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto Ba1_ptr = ba1_.Read();
   const auto Ba2_ptr = ba2_.Read();
   const auto D_ptr = d_.Read();
   const auto X_ptr = x_.Read();
   auto Y_ptr = y_.ReadWrite();

   constexpr int BLK = (D1D > Q1D) ? D1D : Q1D;
   constexpr int BZ = (128 / BLK < 64) ? 128 / BLK : 64;

#if MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST
   // Copy Ba1 into __constant__ memory once per (T_D1D, T_Q1D) instantiation.
   if (!SmemPAMassTriBa1Init<T_D1D, T_Q1D>)
   {
      MFEM_GPU_CHECK(MFEM_PA_SIMPLEX_MASS_MEMCPY_TO_SYMBOL(
                        MFEM_PA_SIMPLEX_MASS_SYMBOL(SmemPAMassTriBa1<T_D1D, T_Q1D>),
                        Ba1_ptr,
                        sizeof(SmemPAMassTriBa1<T_D1D, T_Q1D>), 0,
                        MFEM_PA_SIMPLEX_MASS_MEMCPY_D2D));
      SmemPAMassTriBa1Init<T_D1D, T_Q1D> = true;
   }

   mfem::forall_2D_batch(NE, BLK, 1, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int BASIS_DIM = D1D * (D1D + 1) / 2;

      const auto D = ConstDeviceCube(D_ptr, Q1D, Q1D, NE);
      const auto x = ConstDeviceMatrix(X_ptr, BASIS_DIM, NE);
      auto Y = DeviceMatrix(Y_ptr, BASIS_DIM, NE);

      // Ba1 lives in device __constant__ memory (initialized above): the
      // kernel reads `SmemPAMassTriBa1<T_D1D, T_Q1D>` directly and does
      // not allocate a shared-memory copy.
      MFEM_SHARED real_t Ba2[Q1D][D1D][D1D];
      MFEM_SHARED real_t sm1[BZ][BLK * BLK];
      auto DQ = (real_t (*)[BLK]) (sm1[MFEM_THREAD_ID(z)]);

      MFEM_SHARED union
      {
         real_t Xz[BZ][BASIS_DIM];
         real_t QQ[BZ][BLK][BLK];
      } Xz_QQ;
      auto X = Xz_QQ.Xz[MFEM_THREAD_ID(z)];
      auto QQ = Xz_QQ.QQ[MFEM_THREAD_ID(z)];

      // Cooperatively load Ba1, Ba2 into shared memory, and X for this batch.
      // Cooperative-load helpers. MFEM_THREAD_SIZE returns 1 on host so a
      // single host iteration covers the entire range; on GPU the x-threads
      // of this batch share the work.
      const int tid_x = MFEM_THREAD_ID(x);
      const int stride_x = MFEM_THREAD_SIZE(x);

      if (MFEM_THREAD_ID(z) == 0)
      {
         for (int i = tid_x; i < Q1D * D1D * D1D; i += stride_x)
         {
            ((real_t *)Ba2)[i] = Ba2_ptr[i];
         }
      }
      for (int i = tid_x; i < BASIS_DIM; i += stride_x)
      {
         X[i] = x(i,e);
      }
      MFEM_SYNC_THREAD;

      // Ba1 lives in __constant__ memory: read it directly.
      const auto &Ba1_view = SmemPAMassTriBa1<T_D1D, T_Q1D>;

      // quad to dofs operation, step 1: convert first quadrature index to first
      // multiindex. DQ corresponds to C1 in the AAD algorithm.
      MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
      {
         real_t us[D1D];
         MFEM_UNROLL(D1D)
         for (int a2 = 0; a2 < D1D; a2++)
         {
            if (a1 + a2 >= D1D) { break; }
            us[a2] = X[ij_to_index(D1D, a2, a1)];
         }

         MFEM_UNROLL(Q1D)
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            real_t uu = 0.0;
            MFEM_UNROLL(D1D)
            for (int a2 = 0; a2 < D1D; ++a2)
            {
               if (a2 >= D1D - a1) { break; }
               uu += us[a2] * Ba2[i2][a1][a2];
            }
            DQ[a1][i2] = uu;
         }
      }
      MFEM_SYNC_THREAD;
      // quad to dofs operation, step 2: convert second quadrature index to
      // second multiindex. QQ corresponds to C2 in the AAD algorithm.
      MFEM_FOREACH_THREAD_DIRECT(i2, x, Q1D)
      {
         real_t us[D1D];
         MFEM_UNROLL(D1D)
         for (int a1 = 0; a1 < D1D; a1++)
         {
            us[a1] = DQ[a1][i2];
         }

         MFEM_UNROLL(Q1D)
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            real_t u = 0.0;
            MFEM_UNROLL(D1D)
            for (int a1 = 0; a1 < D1D; a1++)
            {
               u += us[a1] * Ba1_view[i1][a1];
            }
            QQ[i1][i2] = u * D(i1, i2, e);
         }
      }
      MFEM_SYNC_THREAD;
      // dofs to quad operation, step 1: convert first multiindex to first
      // quadrature index. DQ corresponds to F1 in the AAD algorithm.
      MFEM_FOREACH_THREAD_DIRECT(i2, x, Q1D)
      {
         real_t us[Q1D];
         MFEM_UNROLL(Q1D)
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            us[i1] = QQ[i1][i2];
         }

         MFEM_UNROLL(D1D)
         for (int a1 = 0; a1 < D1D; a1++)
         {
            real_t u = 0.0;
            MFEM_UNROLL(Q1D)
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               u += us[i1] * Ba1_view[i1][a1];
            }
            DQ[a1][i2] = u;
         }
      }
      MFEM_SYNC_THREAD;
      // dofs to quad operation, step 2: convert second multiindex to second
      // quadrature index. The contribution to the local RHS is
      //       Y_{\alpha} = F2_{\alpha}.
      MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
      {
         real_t us[Q1D];
         MFEM_UNROLL(Q1D)
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            us[i2] = DQ[a1][i2];
         }

         MFEM_UNROLL(D1D)
         for (int a2 = 0; a2 < D1D; ++a2)
         {
            if (a2 >= D1D - a1) { break; }
            real_t u = 0.0;
            MFEM_UNROLL(Q1D)
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               u += us[i2] * Ba2[i2][a1][a2];
            }
            Y(ij_to_index(D1D, a2, a1), e) += u;
         }
      }
   });
#endif
}

/* This function computes the action of the mass integrator for the Bernstein basis on tetrahedrons.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C3 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F3 and roughly corresponding to Algorithm 3 of [1]).

   [1] Bernstein–Bézier finite elements of arbitrary order and optimal assembly procedures.
       Ainsworth, M., Andriamaro, G., & Davydov, O. (2011).
       SIAM Journal on Scientific Computing, 33(6), 3087-3109.
   */
template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApplyTetrahedron_Element(const int e,
                                    const int NE,
                                    const int BASIS_DIM,
                                    const int BASIS_DIM2D,
                                    const int *forward_map2d,
                                    const int */*inverse_map2d*/,
                                    const int *forward_map3d,
                                    const int */*inverse_map3d*/,
                                    const real_t *ba1_,
                                    const real_t *ba2_,
                                    const real_t *ba3_,
                                    const real_t *ba1t_,
                                    const real_t *ba2t_,
                                    const real_t *ba3t_,
                                    const real_t *d_,
                                    const real_t *x_,
                                    real_t *y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   const int D1D = d1d, Q1D = q1d;

   const auto Ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto Ba2 = ConstDeviceMatrix(ba2_, BASIS_DIM2D, Q1D);
   const auto Ba3 = ConstDeviceMatrix(ba3_, BASIS_DIM, Q1D);
   const auto Ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto Ba2t = ConstDeviceMatrix(ba2t_, Q1D, BASIS_DIM2D);
   const auto Ba3t = ConstDeviceMatrix(ba3t_, Q1D, BASIS_DIM);
   const auto D = DeviceTensor<4,const real_t>(d_, Q1D, Q1D, Q1D, NE);
   const auto X = ConstDeviceMatrix(x_, BASIS_DIM, NE);
   auto Y = DeviceMatrix(y_, BASIS_DIM, NE);

   if (!ACCUMULATE)
   {
      for (int idx = 0; idx < BASIS_DIM; idx++)
      {
         Y(idx, e) = 0.0;
      }
   }

   // C3 will contain the Bernstein polynomial with coefficients X
   // evaluated at all of the quadrature nodes in O(p^{d+1}). we have
   //    C3[t1,t2] = \sum_{\alpha} X_{\alpha} * B_{\alpha}^{p}(\Phi(t1,t2)),
   // where \Phi is the Duffy transformation and (t1,t2) is a Stroud node.

   // evaluate Bernstein polynomial over the first ragged tensor dimension
   constexpr int max_D1D = DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int max_Q1D = DofQuadLimits::MAX_Q1D_SIMPLEX;
   constexpr int BASIS_DIM2D_ = max_D1D * (max_D1D) / 2;

   real_t C3[max_Q1D][max_Q1D][max_Q1D];
   for (int i3 = 0; i3 < Q1D; i3++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            C3[i3][i2][i1] = 0.0;
         }
      }
   }

   for (int a1 = 0; a1 < D1D; a1++)
   {
      real_t C2[max_Q1D][max_Q1D];
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            C2[i3][i2] = 0.0;
         }
      }

      for (int a2 = 0; a2 < D1D-a1; a2++)
      {
         real_t C1[max_Q1D];
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            C1[i3] = 0.0;
         }

         for (int a3 = 0; a3 < D1D-a1-a2; a3++)
         {
            const int a = forward_map3d[a3 + D1D*(a2 + D1D*a1)];
            const real_t s = X(a,e);
            for (int i3 = 0; i3 < Q1D; i3++)
            {
               C1[i3] += s * Ba3t(i3,a);
            }
         }

         const int a_2d = forward_map2d[a2 + D1D*a1];
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const real_t s = C1[i3];
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               C2[i3][i2] += Ba2t(i2,a_2d) * s;
            }
         }
      }

      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            const real_t s = C2[i3][i2];
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               C3[i3][i2][i1] += Ba1t(i1,a1) * s;
            }
         }
      }
   }

   for (int i3 = 0; i3 < Q1D; i3++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            C3[i3][i2][i1] *= D(i1,i2,i3,e);
         }
      }
   }

   for (int i3 = 0; i3 < Q1D; i3++)
   {
      real_t F2[BASIS_DIM2D_];
      for (int a = 0; a < BASIS_DIM2D; a++)
      {
         F2[a] = 0.0;
      }

      for (int i2 = 0; i2 < Q1D; i2++)
      {
         real_t F1[max_D1D];
         for (int a1 = 0; a1 < D1D; a1++)
         {
            F1[a1] = 0.0;
         }

         for (int i1 = 0; i1 < Q1D; i1++)
         {
            const real_t s = C3[i3][i2][i1];
            for (int a1 = 0; a1 < D1D; a1++)
            {
               F1[a1] += Ba1(a1,i1) * s;
            }
         }

         for (int a1 = 0; a1 < D1D; a1++)
         {
            const real_t s = F1[a1];
            for (int a2 = 0; a2 < D1D-a1; a2++)
            {
               const int a_2d = forward_map2d[a2 + D1D*a1];
               F2[a_2d] += Ba2(a_2d,i2) * s;
            }
         }
      }

      for (int a1 = 0; a1 < D1D; a1++)
      {
         for (int a2 = 0; a2 < D1D-a1; a2++)
         {
            const int a_2d = forward_map2d[a2 + D1D*a1];
            const real_t s = F2[a_2d];
            for (int a3 = 0; a3 < D1D-a1-a2; a3++)
            {
               const int a = forward_map3d[a3 + D1D*(a2 + D1D*a1)];
               Y(a,e) += Ba3(a,i3) * s;
            }
         }
      }
   }
}

// PA Mass Apply 3D kernel on tetrahedrons (Bernstein only)
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApplyTetrahedron(const int NE,
                                   const Array<int> &/*lex_map_*/,
                                   const Array<int> &forward_map2d_,
                                   const Array<int> &inverse_map2d_,
                                   const Array<int> &forward_map3d_,
                                   const Array<int> &inverse_map3d_,
                                   const Array<real_t> &ba1_,
                                   const Array<real_t> &ba2_,
                                   const Array<real_t> &ba3_,
                                   const Array<real_t> &ba1t_,
                                   const Array<real_t> &ba2t_,
                                   const Array<real_t> &ba3t_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM = D1D * (D1D + 1) * (D1D + 2) / 6;
   const int BASIS_DIM2D = D1D * (D1D + 1) / 2;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto forward_map2d = forward_map2d_.Read();
   const auto inverse_map2d = inverse_map2d_.Read();
   const auto forward_map3d = forward_map3d_.Read();
   const auto inverse_map3d = inverse_map3d_.Read();
   const auto Ba1 = ba1_.Read();
   const auto Ba2 = ba2_.Read();
   const auto Ba3 = ba3_.Read();
   const auto Ba1t = ba1t_.Read();
   const auto Ba2t = ba2t_.Read();
   const auto Ba3t = ba3t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApplyTetrahedron_Element(e, NE, BASIS_DIM, BASIS_DIM2D,
                                               forward_map2d, inverse_map2d,
                                               forward_map3d, inverse_map3d,
                                               Ba1, Ba2, Ba3, Ba1t, Ba2t, Ba3t,
                                               D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 3D Kernel on tetrahedrons (Bernstein only)
template<int T_D1D, int T_Q1D>
inline void SmemPAMassApplyTetrahedron(const int NE,
                                       const Array<int> &lex_map_,
                                       const Array<int> &forward_map2d_,
                                       const Array<int> &inverse_map2d_,
                                       const Array<int> &forward_map3d_,
                                       const Array<int> &inverse_map3d_,
                                       const Array<real_t> &ba1_,
                                       const Array<real_t> &ba2_,
                                       const Array<real_t> &ba3_,
                                       const Array<real_t> &ba1t_,
                                       const Array<real_t> &ba2t_,
                                       const Array<real_t> &ba3t_,
                                       const Vector &d_,
                                       const Vector &x_,
                                       Vector &y_,
                                       const int = 0,
                                       const int = 0)
{
   static_assert(T_D1D != 0 && T_Q1D != 0,
                 "SmemPAMassApplyTetrahedron requires compile-time D1D and Q1D");

   // Host backend: dispatch to the non-Smem fallback. The Smem kernel
   // reads Ba1 from __constant__ memory on GPU.
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      PAMassApplyTetrahedron<T_D1D, T_Q1D>(
         NE, lex_map_, forward_map2d_, inverse_map2d_, forward_map3d_,
         inverse_map3d_, ba1_, ba2_, ba3_, ba1t_, ba2t_, ba3t_,
         d_, x_, y_, T_D1D, T_Q1D);
      return;
   }

   constexpr int D1D = T_D1D;
   constexpr int Q1D = T_Q1D;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto Ba1_ptr = ba1_.Read();
   const auto Ba2_ptr = ba2_.Read();
   const auto Ba3_ptr = ba3_.Read();
   const auto D_ptr = d_.Read();
   const auto X_ptr = x_.Read();
   auto Y_ptr = y_.ReadWrite();

   constexpr int BLK = (D1D > Q1D) ? D1D : Q1D;
   constexpr int BZ_RAW = 128 / (BLK * BLK);
   constexpr int BZ = (BZ_RAW < 64 ? (BZ_RAW < 1 ? 1 : BZ_RAW) : 64);

#if MFEM_PA_SIMPLEX_MASS_HAS_DEVICE_CONST
   // Copy Ba1 into __constant__ memory once per (T_D1D, T_Q1D) instantiation.
   if (!SmemPAMassTetBa1Init<T_D1D, T_Q1D>)
   {
      MFEM_GPU_CHECK(MFEM_PA_SIMPLEX_MASS_MEMCPY_TO_SYMBOL(
                        MFEM_PA_SIMPLEX_MASS_SYMBOL(SmemPAMassTetBa1<T_D1D, T_Q1D>),
                        Ba1_ptr,
                        sizeof(SmemPAMassTetBa1<T_D1D, T_Q1D>), 0,
                        MFEM_PA_SIMPLEX_MASS_MEMCPY_D2D));
      SmemPAMassTetBa1Init<T_D1D, T_Q1D> = true;
   }

   mfem::forall_2D_batch(NE, BLK, BLK, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int BASIS_DIM2D_MASS = D1D * (D1D + 1) / 2;
      constexpr int BASIS_DIM3D_MASS = D1D * (D1D + 1) * (D1D + 2) / 6;

      const auto d = DeviceTensor<4,const real_t>(D_ptr, Q1D, Q1D, Q1D, NE);
      const auto x = ConstDeviceMatrix(X_ptr, BASIS_DIM3D_MASS, NE);
      auto y = DeviceMatrix(Y_ptr, BASIS_DIM3D_MASS, NE);

      // Ba1 lives in device __constant__ memory (initialized above): the
      // kernel reads `SmemPAMassTetBa1<T_D1D, T_Q1D>` directly and does
      // not allocate a shared-memory copy.
      MFEM_SHARED real_t Ba2[Q1D][BASIS_DIM2D_MASS];
      MFEM_SHARED real_t Ba3[Q1D][BASIS_DIM3D_MASS];
      MFEM_SHARED union
      {
         real_t points[BZ][D1D][D1D][Q1D];
         real_t X[BZ][BASIS_DIM3D_MASS];
      } sm[2];
      auto X = (real_t (*)) sm[1].X[MFEM_THREAD_ID(z)];

      auto C1 = sm[0].points[MFEM_THREAD_ID(z)];
      auto C2 = sm[1].points[MFEM_THREAD_ID(z)];
      auto C3 = sm[0].points[MFEM_THREAD_ID(z)];
      auto F1 = sm[1].points[MFEM_THREAD_ID(z)];
      auto F2 = sm[0].points[MFEM_THREAD_ID(z)];

      // Cooperatively load Ba1, Ba2, Ba3 into shared memory, and X for this batch.
      // Cooperative-load helpers. MFEM_THREAD_SIZE returns 1 on host so a
      // single host iteration covers the entire range; on GPU the (x, y)
      // threads of this batch share the work.
      const int tid_xy = MFEM_THREAD_ID(x) +
                         MFEM_THREAD_ID(y) * MFEM_THREAD_SIZE(x);
      const int stride_xy = MFEM_THREAD_SIZE(x) * MFEM_THREAD_SIZE(y);

      if (MFEM_THREAD_ID(z) == 0)
      {
         for (int i = tid_xy; i < Q1D * BASIS_DIM2D_MASS; i += stride_xy)
         {
            ((real_t *)Ba2)[i] = Ba2_ptr[i];
         }
         for (int i = tid_xy; i < Q1D * BASIS_DIM3D_MASS; i += stride_xy)
         {
            ((real_t *)Ba3)[i] = Ba3_ptr[i];
         }
      }
      for (int i = tid_xy; i < BASIS_DIM3D_MASS; i += stride_xy)
      {
         X[i] = x(i,e);
      }
      MFEM_SYNC_THREAD;

      // Ba1 lives in __constant__ memory: read it directly.
      const auto &Ba1_view = SmemPAMassTetBa1<T_D1D, T_Q1D>;

      // Bernstein polynomial evaluation, contraction in third index.
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
         {
            real_t us[D1D];
            MFEM_UNROLL(D1D)
            for (int a3 = 0; a3 < D1D; a3++)
            {
               if (a1 + a2 + a3 >= D1D) { break; }
               us[a3] = X[ijk_to_index(D1D, a1, a2, a3)];
            }

            MFEM_UNROLL(Q1D)
            for (int i3 = 0; i3 < Q1D; i3++)
            {
               real_t u = 0.0;
               MFEM_UNROLL(D1D)
               for (int a3 = 0; a3 < D1D; a3++)
               {
                  if (a1 + a2 + a3 >= D1D) { break; }
                  const int a = ijk_to_index(D1D, a1, a2, a3);
                  u += us[a3] * Ba3[i3][a];
               }
               C1[a1][a2][i3] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Contraction in second index.
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a3, x, Q1D)
         {
            real_t us[D1D];
            MFEM_UNROLL(D1D)
            for (int a2 = 0; a2 < D1D; a2++)
            {
               if (a1 + a2 >= D1D) { break; }
               us[a2] = C1[a1][a2][a3];
            }

            MFEM_UNROLL(Q1D)
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               real_t u = 0.0;
               MFEM_UNROLL(D1D)
               for (int a2 = 0; a2 < D1D; a2++)
               {
                  if (a1 + a2 >= D1D) { break; }
                  const int a_2d = ij_to_index(D1D, a2, a1);
                  u += us[a2] * Ba2[i2][a_2d];
               }
               C2[a1][i2][a3] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Contraction in first index, then apply D.
      MFEM_FOREACH_THREAD_DIRECT(a2, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a3, x, Q1D)
         {
            real_t us[D1D];
            MFEM_UNROLL(D1D)
            for (int a1 = 0; a1 < D1D; a1++)
            {
               us[a1] = C2[a1][a2][a3];
            }

            MFEM_UNROLL(Q1D)
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               real_t u = 0.0;
               MFEM_UNROLL(D1D)
               for (int a1 = 0; a1 < D1D; a1++)
               {
                  u += us[a1] * Ba1_view[i1][a1];
               }
               C3[i1][a2][a3] = u * d(i1,a2,a3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Bernstein moment, contraction in first index.
      MFEM_FOREACH_THREAD_DIRECT(a2, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a3, x, Q1D)
         {
            real_t us[Q1D];
            MFEM_UNROLL(Q1D)
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               us[i1] = C3[i1][a2][a3];
            }

            MFEM_UNROLL(D1D)
            for (int a1 = 0; a1 < D1D; a1++)
            {
               real_t u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  u += us[i1] * Ba1_view[i1][a1];
               }
               F1[a1][a2][a3] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Contraction in second index.
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a3, x, Q1D)
         {
            real_t us[Q1D];
            MFEM_UNROLL(Q1D)
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               us[i2] = F1[a1][i2][a3];
            }

            MFEM_UNROLL(D1D)
            for (int a2 = 0; a2 < D1D; a2++)
            {
               if (a1 + a2 >= D1D) { break; }
               const int a_2d = ij_to_index(D1D, a2, a1);
               real_t u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i2 = 0; i2 < Q1D; i2++)
               {
                  u += us[i2] * Ba2[i2][a_2d];
               }
               F2[a1][a2][a3] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Contraction in third index, accumulate into Y.
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
         {
            real_t us[Q1D];
            MFEM_UNROLL(Q1D)
            for (int i3 = 0; i3 < Q1D; i3++)
            {
               us[i3] = F2[a1][a2][i3];
            }

            MFEM_UNROLL(D1D)
            for (int a3 = 0; a3 < D1D; a3++)
            {
               if (a1 + a2 + a3 >= D1D) { break; }
               const int a = ijk_to_index(D1D, a1, a2, a3);
               real_t u = 0.0;
               MFEM_UNROLL(Q1D)
               for (int i3 = 0; i3 < Q1D; i3++)
               {
                  u += us[i3] * Ba3[i3][a];
               }
               y(a,e) += u;
            }
         }
      }
   });
#endif
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplySimplexKernelType
MassIntegrator::ApplySimplexPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPAMassApplyTriangle<T_D1D,T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAMassApplyTetrahedron<T_D1D, T_Q1D>;
   }
   else { MFEM_ABORT(""); }
}

inline MassIntegrator::ApplySimplexKernelType
MassIntegrator::ApplySimplexPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2)
   {
      return internal::PAMassApplyTriangle;
   }
   else if (dim == 3)
   {
      return internal::PAMassApplyTetrahedron;
   }
   else { MFEM_ABORT(""); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
