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

/* This function computes the action of the mass integrator for the Bernstein basis on triangles.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C2 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F2 and roughly corresponding to Algorithm 3 of [1]).

   [1] Ainsworth, M., Andriamaro, G., & Davydov, O. (2011). Bernstein–Bézier finite elements
       of arbitrary order and optimal assembly procedures. SIAM Journal on Scientific Computing,
       33(6), 3087-3109.
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
   const int D1D = d1d;
   const int Q1D = q1d;
   constexpr int max_D1D = DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int max_Q1D = DofQuadLimits::MAX_Q1D_SIMPLEX;

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
            const int idx = lex_map_[a2 + D1D*a1];
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
   //       Y_{\alpha} = F2_{\alpha}.
   for (int a1 = 0; a1 < D1D; a1++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         const int a1i2 = i2 + Q1D*a1;
         for (int a2 = 0; a2 < D1D-a1; a2++)
         {
            const int idx = lex_map_[a2 + D1D*a1];
            Y(idx,e) += C1[a1i2] * Ba2t(i2, a1, a2);
         }
      }
   }
}

template<int T_D1D, int T_Q1D, int T_NBZ, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTriangle_Element(const int e,
                                     const int NE,
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
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int BASIS_DIM = MD1 * (MD1+1) / 2;

   const auto lex_map__ = DeviceTensor<2,const int>(lex_map_, D1D, D1D);
   const auto ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto ba2 = ConstDeviceCube(ba2_, D1D, D1D, Q1D);
   const auto ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto ba2t = ConstDeviceCube(ba2t_, Q1D, D1D, D1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto x = ConstDeviceMatrix(x_, BASIS_DIM, NE);
   auto Y = DeviceMatrix(y_, BASIS_DIM, NE);

   MFEM_SHARED real_t B[2][MQ1*MD1*MD1];
   auto Ba1 = (real_t (*)[MD1]) (B+0);
   auto Ba2 = (real_t (*)[MD1][MD1]) (B+1);
   auto Ba1t = (real_t (*)[MQ1]) (B+0);
   auto Ba2t = (real_t (*)[MD1][MQ1]) (B+1);
   MFEM_SHARED real_t Xz[BASIS_DIM];
   MFEM_SHARED real_t sm0[MDQ*MDQ];
   MFEM_SHARED real_t sm1[MDQ*MDQ];
   auto X = (real_t (*)) (Xz);
   auto DQ = (real_t (*)[MD1]) (sm1);
   auto QQ = (real_t (*)[MQ1]) (sm0);
   auto QD = (real_t (*)[MQ1]) (sm1);
   MFEM_SHARED int s_lex[MD1*MD1];
   auto lex_map = (int (*)[MD1])(s_lex);

   // load in input vector and basis data
   MFEM_FOREACH_THREAD(a1,y,D1D)
   {
      MFEM_FOREACH_THREAD(a2,x,D1D-a1)
      {
         const int idx = lex_map__(a2,a1);
         lex_map[a1][a2] = idx;
         X[idx] = x(idx,e);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a1,y,D1D)
   {
      MFEM_FOREACH_THREAD(i1,x,Q1D)
      {
         Ba1[i1][a1] = ba1(a1,i1);
         for (int a2 = 0; a2 < D1D-a1; ++a2)
         {
            Ba2[i1][a1][a2] = ba2(a2,a1,i1);
         }
      }
   }
   MFEM_SYNC_THREAD;
   // quad to dofs operation, step 1: convert first quadrature index to first
   // multiindex. DQ corresponds to C1 in the AAD algorithm.
   MFEM_FOREACH_THREAD(i2,y,Q1D)
   {
      MFEM_FOREACH_THREAD(a1,x,D1D)
      {
         real_t u = 0.0;
         for (int a2 = 0; a2 < D1D-a1; ++a2)
         {
            int idx = lex_map[a1][a2];
            u += X[idx] * Ba2[i2][a1][a2];
         }
         DQ[i2][a1] = u;
      }
   }
   MFEM_SYNC_THREAD;
   // quad to dofs operation, step 2: convert second quadrature index to second
   // multiindex. QQ corresponds to C2 in the AAD algorithm, which contains the Bernstein
   // polynomial on a triangle with coefficients X evaluated at
   // all of the Stroud quadrature nodes. E.g. if (t1,t2) is a Stroud node, then
   //    C2[i,j] = \sum_{\alpha} X_{\alpha} * B_{\alpha}^{p-1}(\Phi(t1,t2)),
   // where \Phi is the Duffy transform.
   MFEM_FOREACH_THREAD(i1,y,Q1D)
   {
      MFEM_FOREACH_THREAD(i2,x,Q1D)
      {
         real_t u = 0.0;
         for (int a1 = 0; a1 < D1D; ++a1)
         {
            u += DQ[i2][a1] * Ba1[i1][a1];
         }
         QQ[i1][i2] = u * D(i1, i2, e);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a1,y,D1D)
   {
      MFEM_FOREACH_THREAD(i1,x,Q1D)
      {
         Ba1t[a1][i1] = ba1t(i1,a1);
         for (int a2 = 0; a2 < D1D-a1; ++a2)
         {
            Ba2t[a2][a1][i1] = ba2t(i1,a1,a2);
         }
      }
   }
   // dofs to quad operation, step 1: convert first multiindex to first quadrature
   // index. DQ corresponds to F1 in the AAD algorithm, with F0 corresponding to
   // C2 * D.
   MFEM_FOREACH_THREAD(i2,y,Q1D)
   {
      MFEM_FOREACH_THREAD(a1,x,D1D)
      {
         real_t u = 0.0;
         for (int i1 = 0; i1 < Q1D; ++i1)
         {
            u += QQ[i1][i2] * Ba1t[a1][i1];
         }
         QD[a1][i2] = u;
      }
   }
   MFEM_SYNC_THREAD;
   // dofs to quad operation, step 2: convert second multiindex to second
   // quadrature index. u corresponds to F2 in the AAD algorithm. The contribution
   // to the local RHS is
   //       Y_{\alpha} = F2_{\alpha}.
   MFEM_FOREACH_THREAD(a1,y,D1D)
   {
      MFEM_FOREACH_THREAD(a2,x,D1D-a1)
      {
         real_t u = 0.0;
         for (int i2 = 0; i2 < Q1D; ++i2)
         {
            u += QD[a1][i2] * Ba2t[a2][a1][i2];
         }
         int idx = lex_map[a1][a2];
         if (ACCUMULATE)
         {
            Y(idx,e) += u;
         }
         else
         {
            Y(idx,e) = u;
         }
      }
   }
}

// current optimal version with 2D collapsed loops...
template<int T_D1D, int T_Q1D, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTetrahedron_Element(const int e,
                                        const int NE,
                                        const int BASIS_DIM,
                                        const int BASIS_DIM2D,
                                        // const int *lex_map,
                                        const int *forward_map2d_,
                                        const int *inverse_map2d_,
                                        const int *forward_map3d_,
                                        //  const int *inverse_map3d_,
                                        const real_t *ba1_,
                                        const real_t *ba2_,
                                        const real_t *ba3_,
                                        const real_t *t_,
                                        const real_t *ba1t_,
                                        const real_t *ba2t_,
                                        const real_t *ba3t_,
                                        const real_t *d_,
                                        const real_t *x_,
                                        real_t *y_,
                                        const int d1d = 0,
                                        const int q1d = 0)
{
   constexpr int D1D = T_D1D ? T_D1D : d1d;
   constexpr int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int BASIS_DIM2D_ = MD1 * (MD1 + 1) / 2;
   constexpr int BASIS_DIM_ = MD1 * (MD1 + 1) * (MD1 + 2) / 6;

   const auto ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto ba2 = ConstDeviceMatrix(ba2_, BASIS_DIM2D, Q1D);
   const auto ba3 = ConstDeviceMatrix(ba3_, BASIS_DIM, Q1D);
   const auto ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto ba2t = ConstDeviceMatrix(ba2t_, Q1D, BASIS_DIM2D);
   const auto ba3t = ConstDeviceMatrix(ba3t_, Q1D, BASIS_DIM);
   const auto d = DeviceTensor<4,const real_t>(d_, Q1D, Q1D, Q1D, NE);
   const auto x = ConstDeviceMatrix(x_, BASIS_DIM, NE);
   auto y = DeviceMatrix(y_, BASIS_DIM, NE);
   const auto forward_map3d__ = DeviceTensor<3,const int>(forward_map3d_, D1D, D1D,
                                                          D1D);
   const auto forward_map2d__ = DeviceTensor<2,const int>(forward_map2d_, D1D,
                                                          D1D);
   const auto inverse_map2d__ = DeviceTensor<2,const int>(inverse_map2d_, 2,
                                                          BASIS_DIM2D);

   MFEM_SHARED real_t sDQ[BASIS_DIM_*MQ1];
   auto Ba1 = (real_t (*)[MD1]) sDQ;
   auto Ba1t = (real_t (*)[MQ1]) sDQ;
   auto Ba2 = (real_t (*)[BASIS_DIM2D_]) sDQ;
   auto Ba2t = (real_t (*)[MQ1]) sDQ;
   auto Ba3 = (real_t (*)[BASIS_DIM_]) sDQ;
   auto Ba3t = (real_t (*)[MQ1]) sDQ;
   MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
   MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];
   auto X = (real_t (*)) sm0;
   auto C1 = (real_t (*)[MQ1]) sm1;
   auto C2 = (real_t (*)[MQ1][MQ1]) sm0;
   auto C3 = (real_t (*)[MQ1][MQ1]) sm1;
   auto F1 = (real_t (*)[MQ1][MD1]) sm0;
   auto F2 = (real_t (*)[MQ1]) sm1;
   MFEM_SHARED int s3D[MD1*MD1*MD1];
   MFEM_SHARED int s2D[MD1*MD1];
   auto forward_map3d = (int (*)[MD1][MD1]) s3D;
   auto forward_map2d = (int (*)[MD1]) s2D;
   MFEM_SHARED int s2D_inv[BASIS_DIM2D_*2];
   auto inverse_map2d = (int (*)[2]) s2D_inv;

   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D)
   {
      inverse_map2d[a_2d][0] = inverse_map2d__(0,a_2d);
      inverse_map2d[a_2d][1] = inverse_map2d__(1,a_2d);
      const int a1 = inverse_map2d[a_2d][0];
      const int a2 = inverse_map2d[a_2d][1];
      const int a_2d_ = forward_map2d__(a2, a1);
      forward_map2d[a1][a2] = a_2d_;
      MFEM_FOREACH_THREAD(i3,x,Q1D)
      {
         MFEM_UNROLL(MD1)
         for (int a3 = 0; a3 < D1D-a1-a2; ++a3)
         {
            const int a = forward_map3d__(a3, a2, a1);
            forward_map3d[a1][a2][a3] = a;
            X[a] = x(a,e);
            Ba3[i3][a] = ba3(a,i3);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D)
   {
      const int a1 = inverse_map2d[a_2d][0];
      const int a2 = inverse_map2d[a_2d][1];
      MFEM_FOREACH_THREAD(i3,x,Q1D)
      {
         real_t u = 0.0;
         MFEM_UNROLL(MD1)
         for (int a3 = 0; a3 < D1D-a1-a2; ++a3)
         {
            const int a = forward_map3d[a1][a2][a3];
            u += X[a] * Ba3[i3][a];
         }
         C1[a_2d][i3] = u;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D) // load in Ba2
   {
      MFEM_FOREACH_THREAD(i2,x,Q1D)
      {
         Ba2[i2][a_2d] = ba2(a_2d,i2);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a1i2,y,Q1D*D1D)
   {
      const int i2 = a1i2 % Q1D;
      const int a1 = (int) a1i2 / Q1D;
      MFEM_FOREACH_THREAD(i3,x,Q1D)
      {
         real_t u = 0.0;
         MFEM_UNROLL(MD1)
         for (int a2 = 0; a2 < D1D-a1; a2++)
         {
            const int a_2d = forward_map2d[a1][a2];
            u += C1[a_2d][i3] * Ba2[i2][a_2d];
         }
         C2[a1][i2][i3] = u;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a1i1,y,Q1D*D1D) // load in Ba1
   {
      const int i1 = a1i1 % Q1D;
      const int a1 = (int) a1i1 / Q1D;
      Ba1[i1][a1] = ba1(a1,i1);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(i2i3,y,Q1D*Q1D)
   {
      const int i3 = i2i3 % Q1D;
      const int i2 = (int) i2i3 / Q1D;
      MFEM_FOREACH_THREAD(i1,x,Q1D)
      {
         real_t u = 0.0;
         MFEM_UNROLL(MD1)
         for (int a1 = 0; a1 < D1D; a1++)
         {
            u += C2[a1][i2][i3] * Ba1[i1][a1];
         }
         C3[i2][i3][i1] = u * d(i1,i2,i3,e);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a1i1,y,Q1D*D1D) // load in Ba1
   {
      const int i1 = a1i1 % Q1D;
      const int a1 = (int) a1i1 / Q1D;
      Ba1t[a1][i1] = ba1t(i1,a1);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(i2i3,y,Q1D*Q1D)
   {
      const int i3 = i2i3 % Q1D;
      const int i2 = (int) i2i3 / Q1D;
      MFEM_FOREACH_THREAD(a1,x,D1D)
      {
         real_t u = 0.0;
         MFEM_UNROLL(MQ1)
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            u += C3[i2][i3][i1] * Ba1t[a1][i1];
         }
         F1[i2][i3][a1] = u;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D) // load in Ba2
   {
      MFEM_FOREACH_THREAD(i2,x,Q1D)
      {
         Ba2t[a_2d][i2] = ba2t(i2,a_2d);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D)
   {
      const int a1 = inverse_map2d[a_2d][0];
      MFEM_FOREACH_THREAD(i3,x,Q1D)
      {
         real_t u = 0.0;
         MFEM_UNROLL(MQ1)
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            u += F1[i2][i3][a1] * Ba2t[a_2d][i2];
         }
         F2[a_2d][i3] = u;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D) // load in Ba3t
   {
      const int a1 = inverse_map2d[a_2d][0];
      const int a2 = inverse_map2d[a_2d][1];
      MFEM_FOREACH_THREAD(i3,x,Q1D)
      {
         // MFEM_UNROLL(MD1)
         for (int a3 = 0; a3 < D1D-a1-a2; ++a3)
         {
            const int a = forward_map3d[a1][a2][a3];
            Ba3t[a][i3] = ba3t(i3,a);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(a_2d,y,BASIS_DIM2D)
   {
      const int a1 = inverse_map2d[a_2d][0];
      const int a2 = inverse_map2d[a_2d][1];
      MFEM_FOREACH_THREAD(a3,x,D1D-a1-a2)
      {
         real_t u = 0.0;
         const int a = forward_map3d[a1][a2][a3];
         MFEM_UNROLL(MQ1)
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            u += F2[a_2d][i3] * Ba3t[a][i3];
         }
         if (ACCUMULATE)
         {
            y(a,e) += u;
         }
         else
         {
            y(a,e) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/* This function computes the action of the mass integrator for the Bernstein basis on tetrahedrons.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C3 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F3 and roughly corresponding to Algorithm 3 of [1]).

   [1] Ainsworth, M., Andriamaro, G., & Davydov, O. (2011). Bernstein–Bézier finite elements
       of arbitrary order and optimal assembly procedures. SIAM Journal on Scientific Computing,
       33(6), 3087-3109.
   */
template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApplyTetrahedron_Element(const int e,
                                    const int NE,
                                    const int BASIS_DIM,
                                    const int BASIS_DIM2D,
                                    const int *forward_map2d,
                                    const int *inverse_map2d,
                                    const int *forward_map3d,
                                    const int *inverse_map3d,
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
   const int D1D = d1d;
   const int Q1D = q1d;
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

// PA Mass Apply 2D kernel on triangles (Bernstein only)
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApplyTriangle(const int NE,
                                const Array<int> &lex_map_,
                                const Array<int> &forward_map2d_,
                                const Array<int> &inverse_map2d_,
                                const Array<int> &forward_map3d_,
                                const Array<int> &inverse_map3d_,
                                const Array<real_t> &ba1_,
                                const Array<real_t> &ba2_,
                                const Array<real_t> &ba3_, // unused in 2D...
                                const Array<real_t> &ba1t_,
                                const Array<real_t> &ba2t_,
                                const Array<real_t> &ba3t_, // unused in 2D...
                                const Array<real_t> &t_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX,
               "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX,
               "");

   const int BASIS_DIM = d1d * (d1d + 1) / 2;
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
      internal::PAMassApplyTriangle_Element(e, NE, BASIS_DIM, lex_map, Ba1, Ba2, Ba1t,
                                            Ba2t, D, X,
                                            Y, d1d, q1d);
   });
}

// PA Mass Apply 2D kernel on triangles with shared memory
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangle(const int NE,
                                    const Array<int> &lex_map_,
                                    const Array<int> &forward_map2d_,
                                    const Array<int> &inverse_map2d_,
                                    const Array<int> &forward_map3d_,
                                    const Array<int> &inverse_map3d_,
                                    const Array<real_t> &ba1_,
                                    const Array<real_t> &ba2_,
                                    const Array<real_t> &ba3_, // unused in 2D...
                                    const Array<real_t> &ba1t_,
                                    const Array<real_t> &ba2t_,
                                    const Array<real_t> &ba3t_, // unused in 2D...
                                    const Array<real_t> &t_,
                                    const Vector &d_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   static constexpr int T_NBZ = mass::NBZ(T_D1D);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");

   const auto lex_map = lex_map_.Read();
   const auto Ba1 = ba1_.Read();
   const auto Ba2 = ba2_.Read();
   const auto Ba1t = ba1t_.Read();
   const auto Ba2t = ba2t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApplyTriangle_Element<T_D1D,T_Q1D,T_NBZ>(e, NE, lex_map,
                                                                   Ba1, Ba2, Ba1t, Ba2t, D, X,
                                                                   Y, d1d, q1d);
   });
}
// PA Mass Apply 3D kernel on tetrahedrons (Bernstein only)
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApplyTetrahedron(const int NE,
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
                                   const Array<real_t> &t_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const int BASIS_DIM = D1D * (D1D + 1) * (D1D + 2) / 6;
   const int BASIS_DIM2D = D1D * (D1D + 1) / 2;
   // const auto lex_map = lex_map_.Read();
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
                                               forward_map2d,
                                               inverse_map2d, forward_map3d, inverse_map3d, Ba1, Ba2,
                                               Ba3, Ba1t, Ba2t, Ba3t, D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 3D Kernel on tetrahedrons (Bernstein only)
template<int T_D1D = 0, int T_Q1D = 0>
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
                                       const Array<real_t> &t_,
                                       const Vector &d_,
                                       const Vector &x_,
                                       Vector &y_,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");

   const int BASIS_DIM = D1D * (D1D + 1) * (D1D + 2) / 6;
   const int BASIS_DIM2D = D1D * (D1D + 1) / 2;
   const auto forward_map2d = forward_map2d_.Read();
   const auto inverse_map2d = inverse_map2d_.Read();
   const auto forward_map3d = forward_map3d_.Read();
   const auto Ba1 = ba1_.Read();
   const auto Ba2 = ba2_.Read();
   const auto Ba3 = ba3_.Read();
   const auto Ba1t = ba1t_.Read();
   const auto Ba2t = ba2t_.Read();
   const auto Ba3t = ba3t_.Read();
   const auto T = t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall_2D(NE, Q1D, Q1D*Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApplyTetrahedron_Element<T_D1D, T_Q1D>(e, NE, BASIS_DIM,
                                                                 BASIS_DIM2D, forward_map2d,
                                                                 inverse_map2d, forward_map3d, Ba1, Ba2, Ba3, T,
                                                                 Ba1t, Ba2t, Ba3t, D, X, Y, d1d, q1d);
   });
}

} // namespace internal

namespace
{
using ApplySimplexKernelType = MassIntegrator::ApplySimplexKernelType;
}

template<int DIM, int T_D1D, int T_Q1D>
ApplySimplexKernelType MassIntegrator::ApplySimplexPAKernels::Kernel()
{
   if constexpr (DIM == 2) { return internal::SmemPAMassApplyTriangle<T_D1D,T_Q1D>; }
   else if constexpr (DIM == 3) { return internal::SmemPAMassApplyTetrahedron<T_D1D, T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline ApplySimplexKernelType MassIntegrator::ApplySimplexPAKernels::Fallback(
   int dim, int, int)
{
   if (dim == 2) { return internal::PAMassApplyTriangle; }
   else if (dim == 3) { return internal::PAMassApplyTetrahedron; }
   else { MFEM_ABORT(""); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
