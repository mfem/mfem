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

/** Tensor Mirror Overlap (TMO) helpers and apply kernels.
    - Duffy: vertex-permuted Stroud + Bernstein Ba1/Ba2 sum-fac
    - Composite: Duffy sum-fac on each even-fold half × mirrors
    - Tensor / BernsteinDense: dense P at chart quadrature pts
    - Bernstein: even-prolong to tensor Bernstein + B/Bt */
namespace tmo
{

constexpr int NMIRRORS = 3;
constexpr int NHALVES = 2; // COMPOSITE: lower + upper even-fold halves

/** Duffy helper: map Stroud (s,t) on T through vertex permutation k. */
MFEM_HOST_DEVICE inline void MapSquareToTriangle(const int k,
                                                 const real_t s, const real_t t,
                                                 real_t &xi, real_t &eta)
{
   if (k == 0) { xi = s; eta = t; }
   else if (k == 1) { xi = real_t(1) - s - t; eta = s; }
   else { xi = t; eta = real_t(1) - s - t; }
}

MFEM_HOST_DEVICE inline int NatIdx(const int i, const int j, const int p)
{
   return j * (2 * p - j + 3) / 2 + i;
}

MFEM_HOST_DEVICE inline int ProlongSrc(const int k, const int i, const int j,
                                       const int p)
{
   if (k == 0) { return NatIdx(i, j, p); }
   if (k == 1) { return NatIdx(p - i - j, i, p); }
   return NatIdx(j, p - i - j, p);
}

/** Tensor path: map unit-square (s,t) into parallelogram Q_k in ref coords.
    k=0 from A=(0,0); k=1 from B=(1,0); k=2 from C=(0,1). */
MFEM_HOST_DEVICE inline void MapSquareToParallelogram(const int k,
                                                      const real_t s,
                                                      const real_t t,
                                                      real_t &x, real_t &y)
{
   if (k == 0) { x = s; y = t; }
   else if (k == 1) { x = real_t(1) - s - t; y = t; }
   else { x = t; y = real_t(1) - s - t; }
}

/** Even fold on the reference square (s+t=1), then φ_k → point in T. */
MFEM_HOST_DEVICE inline void EvenEvalPoint(const int k,
                                           const real_t s, const real_t t,
                                           real_t &x, real_t &y)
{
   real_t sf = s, tf = t;
   if (s + t > real_t(1))
   {
      sf = real_t(1) - t;
      tf = real_t(1) - s;
   }
   MapSquareToParallelogram(k, sf, tf, x, y);
}

} // namespace tmo

// ---------------------------------------------------------------------------
// Duffy / Bernstein sum-factorized TMO apply
// ---------------------------------------------------------------------------

template<int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTriangleTmoDuffy_Element(const int e,
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
   const int p = D1D - 1;
   const int ndof = D1D * (D1D + 1) / 2;

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int BASIS_DIM = MD1 * (MD1 + 1) / 2;

   const auto map = DeviceTensor<2, const int>(lex_map_, D1D, D1D);
   const auto ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto ba2 = ConstDeviceCube(ba2_, D1D, D1D, Q1D);
   const auto ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto ba2t = ConstDeviceCube(ba2t_, Q1D, D1D, D1D);
   const auto D = DeviceTensor<4, const real_t>(d_, Q1D, Q1D, tmo::NMIRRORS, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   auto Y = DeviceMatrix(y_, ndof, NE);

   MFEM_SHARED real_t B[2][MQ1 * MD1 * MD1];
   auto Ba1 = (real_t (*)[MD1]) (B + 0);
   auto Ba2 = (real_t (*)[MD1][MD1]) (B + 1);
   auto Ba1t = (real_t (*)[MQ1]) (B + 0);
   auto Ba2t = (real_t (*)[MD1][MQ1]) (B + 1);
   MFEM_SHARED real_t X0[BASIS_DIM];
   MFEM_SHARED real_t Xz[BASIS_DIM];
   MFEM_SHARED real_t sm0[MDQ * MDQ], sm1[MDQ * MDQ];
   auto X = (real_t (*)) (Xz);
   auto DQ = (real_t (*)[MD1]) (sm1);
   auto QQ = (real_t (*)[MQ1]) (sm0);
   auto QD = (real_t (*)[MQ1]) (sm1);
   MFEM_SHARED int s_lex[MD1 * MD1];
   auto lex_map = (int (*)[MD1])(s_lex);

   MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
   {
      MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
      {
         const int idx = map(a2, a1);
         lex_map[a1][a2] = idx;
         X0[idx] = x(idx, e);
      }
   }
   MFEM_SYNC_THREAD;

   MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
   {
      MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
      {
         Ba1[i1][a1] = ba1(a1, i1);
         for (int a2 = 0; a2 < D1D - a1; ++a2)
         {
            Ba2[i1][a1][a2] = ba2(a2, a1, i1);
         }
      }
   }
   MFEM_SYNC_THREAD;

   for (int k = 0; k < tmo::NMIRRORS; ++k)
   {
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
         {
            const int dst = lex_map[a1][a2];
            X[dst] = X0[tmo::ProlongSrc(k, a1, a2, p)];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i2, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
         {
            real_t u = 0.0;
            for (int a2 = 0; a2 < D1D - a1; ++a2)
            {
               u += X[lex_map[a1][a2]] * Ba2[i2][a1][a2];
            }
            DQ[i2][a1] = u;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i1, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(i2, x, Q1D)
         {
            real_t u = 0.0;
            for (int a1 = 0; a1 < D1D; ++a1)
            {
               u += DQ[i2][a1] * Ba1[i1][a1];
            }
            QQ[i1][i2] = u * D(i1, i2, k, e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
         {
            Ba1t[a1][i1] = ba1t(i1, a1);
            for (int a2 = 0; a2 < D1D - a1; ++a2)
            {
               Ba2t[a2][a1][i1] = ba2t(i1, a1, a2);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i2, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
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

      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
         {
            real_t u = 0.0;
            for (int i2 = 0; i2 < Q1D; ++i2)
            {
               u += QD[a1][i2] * Ba2t[a2][a1][i2];
            }
            Y(tmo::ProlongSrc(k, a1, a2, p), e) += u;
         }
      }
      MFEM_SYNC_THREAD;

      if (k + 1 < tmo::NMIRRORS)
      {
         MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
            {
               Ba1[i1][a1] = ba1(a1, i1);
               for (int a2 = 0; a2 < D1D - a1; ++a2)
               {
                  Ba2[i1][a1][a2] = ba2(a2, a1, i1);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangleTmoDuffy(
   const int NE,
   const Array<int> &lex_map_,
   const Array<int> &, const Array<int> &,
   const Array<int> &, const Array<int> &,
   const Array<real_t> &ba1_, const Array<real_t> &ba2_,
   const Array<real_t> &,
   const Array<real_t> &ba1t_, const Array<real_t> &ba2t_,
   const Array<real_t> &,
   const Vector &d_, const Vector &x_, Vector &y_,
   const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");

   const auto lex_map = lex_map_.Read();
   const auto Ba1 = ba1_.Read(), Ba2 = ba2_.Read();
   const auto Ba1t = ba1t_.Read(), Ba2t = ba2t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int T1D = (Q1D > D1D) ? Q1D : D1D;
   constexpr int T_T1D = (T_Q1D > T_D1D) ? T_Q1D : T_D1D;

   mfem::forall_2D<T_T1D * T_T1D>(NE, T1D, T1D, [=] MFEM_HOST_DEVICE (int e)
   {
      SmemPAMassApplyTriangleTmoDuffy_Element<T_D1D, T_Q1D>(
         e, NE, lex_map, Ba1, Ba2, Ba1t, Ba2t, D, X, Y, d1d, q1d);
   });
}

// ---------------------------------------------------------------------------
// COMPOSITE: Duffy Ba1/Ba2 sum-fac on each even-fold half × 3 mirrors.
// D layout: (Q1D, Q1D, NHALVES, NMIRRORS, NE); weight 1/(2m) in assemble.
// ---------------------------------------------------------------------------

template<int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTriangleTmoComposite_Element(const int e,
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
   const int p = D1D - 1;
   const int ndof = D1D * (D1D + 1) / 2;

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int BASIS_DIM = MD1 * (MD1 + 1) / 2;

   const auto map = DeviceTensor<2, const int>(lex_map_, D1D, D1D);
   const auto ba1 = ConstDeviceMatrix(ba1_, D1D, Q1D);
   const auto ba2 = ConstDeviceCube(ba2_, D1D, D1D, Q1D);
   const auto ba1t = ConstDeviceMatrix(ba1t_, Q1D, D1D);
   const auto ba2t = ConstDeviceCube(ba2t_, Q1D, D1D, D1D);
   const auto D = DeviceTensor<5, const real_t>(d_, Q1D, Q1D, tmo::NHALVES,
                                                tmo::NMIRRORS, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   auto Y = DeviceMatrix(y_, ndof, NE);

   MFEM_SHARED real_t B[2][MQ1 * MD1 * MD1];
   auto Ba1 = (real_t (*)[MD1]) (B + 0);
   auto Ba2 = (real_t (*)[MD1][MD1]) (B + 1);
   auto Ba1t = (real_t (*)[MQ1]) (B + 0);
   auto Ba2t = (real_t (*)[MD1][MQ1]) (B + 1);
   MFEM_SHARED real_t X0[BASIS_DIM];
   MFEM_SHARED real_t Xz[BASIS_DIM];
   MFEM_SHARED real_t sm0[MDQ * MDQ], sm1[MDQ * MDQ];
   auto X = (real_t (*)) (Xz);
   auto DQ = (real_t (*)[MD1]) (sm1);
   auto QQ = (real_t (*)[MQ1]) (sm0);
   auto QD = (real_t (*)[MQ1]) (sm1);
   MFEM_SHARED int s_lex[MD1 * MD1];
   auto lex_map = (int (*)[MD1])(s_lex);

   MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
   {
      MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
      {
         const int idx = map(a2, a1);
         lex_map[a1][a2] = idx;
         X0[idx] = x(idx, e);
      }
   }
   MFEM_SYNC_THREAD;

   MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
   {
      MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
      {
         Ba1[i1][a1] = ba1(a1, i1);
         for (int a2 = 0; a2 < D1D - a1; ++a2)
         {
            Ba2[i1][a1][a2] = ba2(a2, a1, i1);
         }
      }
   }
   MFEM_SYNC_THREAD;

   for (int k = 0; k < tmo::NMIRRORS; ++k)
   {
      MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
         {
            const int dst = lex_map[a1][a2];
            X[dst] = X0[tmo::ProlongSrc(k, a1, a2, p)];
         }
      }
      MFEM_SYNC_THREAD;

      for (int h = 0; h < tmo::NHALVES; ++h)
      {
         MFEM_FOREACH_THREAD_DIRECT(i2, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
            {
               real_t u = 0.0;
               for (int a2 = 0; a2 < D1D - a1; ++a2)
               {
                  u += X[lex_map[a1][a2]] * Ba2[i2][a1][a2];
               }
               DQ[i2][a1] = u;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(i1, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(i2, x, Q1D)
            {
               real_t u = 0.0;
               for (int a1 = 0; a1 < D1D; ++a1)
               {
                  u += DQ[i2][a1] * Ba1[i1][a1];
               }
               QQ[i1][i2] = u * D(i1, i2, h, k, e);
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
            {
               Ba1t[a1][i1] = ba1t(i1, a1);
               for (int a2 = 0; a2 < D1D - a1; ++a2)
               {
                  Ba2t[a2][a1][i1] = ba2t(i1, a1, a2);
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(i2, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(a1, x, D1D)
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

         MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(a2, x, D1D - a1)
            {
               real_t u = 0.0;
               for (int i2 = 0; i2 < Q1D; ++i2)
               {
                  u += QD[a1][i2] * Ba2t[a2][a1][i2];
               }
               Y(tmo::ProlongSrc(k, a1, a2, p), e) += u;
            }
         }
         MFEM_SYNC_THREAD;

         // Reload Ba after Ba1t/Ba2t alias the same shared buffers.
         MFEM_FOREACH_THREAD_DIRECT(a1, y, D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(i1, x, Q1D)
            {
               Ba1[i1][a1] = ba1(a1, i1);
               for (int a2 = 0; a2 < D1D - a1; ++a2)
               {
                  Ba2[i1][a1][a2] = ba2(a2, a1, i1);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangleTmoComposite(
   const int NE,
   const Array<int> &lex_map_,
   const Array<int> &, const Array<int> &,
   const Array<int> &, const Array<int> &,
   const Array<real_t> &ba1_, const Array<real_t> &ba2_,
   const Array<real_t> &,
   const Array<real_t> &ba1t_, const Array<real_t> &ba2t_,
   const Array<real_t> &,
   const Vector &d_, const Vector &x_, Vector &y_,
   const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");

   const auto lex_map = lex_map_.Read();
   const auto Ba1 = ba1_.Read(), Ba2 = ba2_.Read();
   const auto Ba1t = ba1t_.Read(), Ba2t = ba2t_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int T1D = (Q1D > D1D) ? Q1D : D1D;
   constexpr int T_T1D = (T_Q1D > T_D1D) ? T_Q1D : T_D1D;

   mfem::forall_2D<T_T1D * T_T1D>(NE, T1D, T1D, [=] MFEM_HOST_DEVICE (int e)
   {
      SmemPAMassApplyTriangleTmoComposite_Element<T_D1D, T_Q1D>(
         e, NE, lex_map, Ba1, Ba2, Ba1t, Ba2t, D, X, Y, d1d, q1d);
   });
}

// ---------------------------------------------------------------------------
// Tensor TMO apply: even extension evaluated at tensor quad pts on each Q_k
// (GLL nodal prolong into Q_p⊗Q_p is not exact for the even map, so we
// evaluate φ_T∘Fold at (s,t) quadrature nodes and apply D there.)
// ---------------------------------------------------------------------------

template<int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTriangleTmoTensor_Element(const int e,
                                              const int NE,
                                              const real_t *p_,
                                              const real_t *d_,
                                              const real_t *x_,
                                              real_t *y_,
                                              const int d1d,
                                              const int nq1)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = D1D * (D1D + 1) / 2;
   // nq1 = quadrature points per mirror (triangle rule count)
   const int NQ1 = T_Q1D ? T_Q1D : nq1;

   constexpr int MQ = T_Q1D ? T_Q1D : (DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D);
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   constexpr int BASIS_DIM = MD1 * (MD1 + 1) / 2;

   const auto P = DeviceTensor<3, const real_t>(p_, NQ1, ndof, tmo::NMIRRORS);
   const auto D = DeviceTensor<3, const real_t>(d_, NQ1, tmo::NMIRRORS, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   auto Y = DeviceMatrix(y_, ndof, NE);

   MFEM_SHARED real_t Xtri[BASIS_DIM];
   MFEM_SHARED real_t Uq[MQ];

   // y-thread 0 owns dof loads/stores; x-thread 0 owns Uq writes
   if (MFEM_THREAD_ID(y) == 0)
   {
      MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
      {
         Xtri[i] = x(i, e);
      }
   }
   MFEM_SYNC_THREAD;

   for (int k = 0; k < tmo::NMIRRORS; ++k)
   {
      if (MFEM_THREAD_ID(x) == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, y, NQ1)
         {
            real_t u = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               u += P(q, i, k) * Xtri[i];
            }
            Uq[q] = u * D(q, k, e);
         }
      }
      MFEM_SYNC_THREAD;

      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
         {
            real_t yi = 0.0;
            for (int q = 0; q < NQ1; ++q)
            {
               yi += P(q, i, k) * Uq[q];
            }
            Y(i, e) += yi;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangleTmoTensor(const int NE,
                                             const Array<real_t> &p_,
                                             const Vector &d_,
                                             const Vector &x_,
                                             Vector &y_,
                                             const int d1d = 0,
                                             const int nq1 = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(NQ1 <= DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D, "");

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int BX = D1D * (D1D + 1) / 2;
   const int BY = NQ1;

   mfem::forall_2D(NE, BX, BY, [=] MFEM_HOST_DEVICE (int e)
   {
      SmemPAMassApplyTriangleTmoTensor_Element<T_D1D, T_Q1D>(
         e, NE, P, D, X, Y, d1d, nq1);
   });
}

// ---------------------------------------------------------------------------
// MMA Tensor TMO apply: batched (NBATCH=mmaN) shared-memory DMMA GEMM
// Same P/D as Tensor / BernsteinDense; used by TMO_MMA, TMO_MMA_1, BernsteinMMA.
// nmirrors inferred from pa_data size (1 for MMA_1, 3 for full TMO).
// Smem: Xs/Us/Ys with odd leading dims (bank-conflict padding); no Ps staging.
// ---------------------------------------------------------------------------

namespace tmo
{
namespace mma
{

MFEM_HOST_DEVICE inline int getThreadIdx()
{
#ifdef __CUDA_ARCH__
   return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
   return 0;
#endif
}

MFEM_HOST_DEVICE inline int getWarpId(int thread) { return thread / 32; }
MFEM_HOST_DEVICE inline int getLaneId(int thread) { return thread % 32; }
MFEM_HOST_DEVICE inline int getGroupId(int laneId) { return laneId / 4; }
MFEM_HOST_DEVICE inline int getThreadIdInGroup(int laneId) { return laneId % 4; }

constexpr int mmaM = 8;
constexpr int mmaN = 8;
constexpr int mmaK = 4;

// Default packed column map for m8n8k4.row.col: [0,5,1,6,2,7,3,4].
// Shape-specific maps (paper §III-C f_n) are selected by MagicForDims and
// packed the same way: map[slot] occupies bits 3*slot .. 3*slot+2.
constexpr int MagicDefault = 0b100011111010110001101000; // 0x8fac68

/** Effective column map for known (ndof,nq1) BP1tri GLL shapes.
    Falls back to MagicDefault when untuned. */
constexpr int MagicForDims(int ndof, int nq1)
{
   // order 2: ndof=6,  nq1=15
   if (ndof == 6 && nq1 == 15) { return 0xaf9ca0; } // [0,4,2,6,1,7,3,5]
   // order 3: ndof=10, nq1=19
   if (ndof == 10 && nq1 == 19) { return 0xceae60; } // [0,4,1,7,2,5,3,6]
   // order 4: ndof=15, nq1=28
   if (ndof == 15 && nq1 == 28) { return 0xcd7328; } // [0,5,4,1,7,2,3,6]
   // order 5: ndof=21, nq1=37
   if (ndof == 21 && nq1 == 37) { return 0xcfa868; } // [0,5,1,4,2,7,3,6]
   // order 6: ndof=28, nq1=49
   if (ndof == 28 && nq1 == 49) { return 0xcd7328; } // [0,5,4,1,7,2,3,6]
   return MagicDefault;
}

template <int DIM, int D1D, int Q1D>
constexpr int MagicFor()
{
   if (D1D == 0 || Q1D == 0) { return MagicDefault; }
   constexpr int ndof = (DIM == 2)
                        ? (D1D * (D1D + 1) / 2)
                        : (D1D * (D1D + 1) * (D1D + 2) / 6);
   return MagicForDims(ndof, Q1D);
}

// Fallback (T_D1D==0) caps: keep static smem under ~48KB with NB=8.
// 2D: MAX_D1D triangle ndof; 3D: D1D<=8 (p<=7) covers BP1tet / unit tests.
constexpr int FallbackMaxD1D2 = DofQuadLimits::MAX_D1D;
constexpr int FallbackMaxD1D3 = 8;
constexpr int FallbackMaxNq2 = DofQuadLimits::MAX_Q1D * DofQuadLimits::MAX_Q1D;
constexpr int FallbackMaxNq3 = 256; // covers tet IntRules through order ~16

template <int DIM, int D1D>
constexpr int SimplexNdof()
{
   if constexpr (DIM == 2)
   {
      return D1D ? (D1D * (D1D + 1) / 2)
                 : (FallbackMaxD1D2 * (FallbackMaxD1D2 + 1) / 2);
   }
   else
   {
      const int d = D1D ? D1D : FallbackMaxD1D3;
      return d * (d + 1) * (d + 2) / 6;
   }
}

template <int DIM, int Q1D>
constexpr int SimplexMaxNq()
{
   if (Q1D) { return Q1D; }
   return (DIM == 2) ? FallbackMaxNq2 : FallbackMaxNq3;
}

template <int MAGIC>
constexpr int MagicCol(int slot)
{
   return (MAGIC >> (3 * slot)) & 0b111;
}

/** True if leading dim `ld` is bank-conflict free for MAGIC + blocked f_m. */
template <int MAGIC>
constexpr bool LdBankOkM8(int ld)
{
   constexpr int cog[8] = {
      MagicCol<MAGIC>(0), MagicCol<MAGIC>(1), MagicCol<MAGIC>(2), MagicCol<MAGIC>(3),
      MagicCol<MAGIC>(4), MagicCol<MAGIC>(5), MagicCol<MAGIC>(6), MagicCol<MAGIC>(7)
   };
   for (int phase = 0; phase < 2; ++phase)
   {
      unsigned used = 0u;
      for (int gi = 0; gi < 4; ++gi)
      {
         const int col = cog[phase * 4 + gi];
         for (int r = 0; r < 4; ++r)
         {
            const unsigned b = (unsigned)((r + ld * col) & 31);
            if (used & (1u << b)) { return false; }
            used |= (1u << b);
         }
      }
   }
   // C stores (c0/c1) with blocked rows = groupId.
   for (int phase = 0; phase < 2; ++phase)
   {
      for (int i = 0; i < 2; ++i)
      {
         unsigned used = 0u;
         for (int g = 0; g < 4; ++g)
         {
            const int row = phase * 4 + g;
            for (int tinG = 0; tinG < 4; ++tinG)
            {
               const int col = MagicCol<MAGIC>(tinG * 2 + i);
               const unsigned b = (unsigned)((row + ld * col) & 31);
               if (used & (1u << b)) { return false; }
               used |= (1u << b);
            }
         }
      }
   }
   return true;
}

template <int MAGIC>
constexpr int PadLdBank(int n)
{
   for (int ld = n; ld < n + 48; ++ld)
   {
      if (LdBankOkM8<MAGIC>(ld)) { return ld; }
   }
   return n;
}

MFEM_HOST_DEVICE inline void dmmaSync([[maybe_unused]] double aReg[1],
                                      [[maybe_unused]] double bReg[1],
                                      [[maybe_unused]] double cReg[2])
{
#ifdef __CUDA_ARCH__
   asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+d"(cReg[0]), "+d"(cReg[1]) : "d"(aReg[0]), "d"(bReg[0]));
#endif
}

/** Column-major smem matrix accessor with compile-time leading dimension. */
template<int LD>
struct SmemMatAcc
{
   real_t *p;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int c) const
   {
      return p[r + LD * c];
   }
};

/** Max N-tiles / elements per specialized CTA (A fragment shared across tiles). */
constexpr int MAX_N_TILES = 2;
constexpr int NBATCH = MAX_N_TILES * mmaN; // 16

MFEM_HOST_DEVICE inline int getNumWarps()
{
#ifdef __CUDA_ARCH__
   return (blockDim.x * blockDim.y * blockDim.z) / 32;
#else
   return 1;
#endif
}

/** C = A * B with fused D-scale on the C store (U *= D from registers). */
template <int MAGIC, bool SCALE, typename AAcc, typename BAcc, typename CAcc,
          typename DAcc>
MFEM_HOST_DEVICE inline void dmma_Gemm8(const int M, const int K, const int N,
                                        AAcc A, BAcc B, CAcc C,
                                        DAcc D, const int e0, const int NE,
                                        const int k_mir)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (M + mmaM - 1) / mmaM;
   // One A fragment feeds all N-tiles (NBATCH may be 2×mmaN).
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {}; // [nTile][frag]

      for (int mK = 0; mK < (K + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aRow = row0 + groupId;
         const int aColumn = threadIdInGroup + mK * mmaK;
         aReg[0] = (aRow < M && aColumn < K)
                   ? static_cast<double>(A(aRow, aColumn)) : 0.0;

         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAGIC>(groupId);
            bReg[0] = (bRow < K && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAGIC>(threadIdInGroup * 2 + i);
            if (cRow < M && cColumn < nTile)
            {
               real_t v = static_cast<real_t>(cReg[nt][i]);
               if constexpr (SCALE)
               {
                  const int e = e0 + n0 + cColumn;
                  v = (e < NE) ? v * D(cRow, k_mir, e) : real_t(0);
               }
               C(cRow, n0 + cColumn) = v;
            }
         }
      }
   }
}

/** C += A^T * B into global Y (via C accessor); skip if e0+b >= NE. */
template <int MAGIC, typename AAcc, typename BAcc, typename CAcc>
MFEM_HOST_DEVICE inline void dmma_GemmT8(const int M, const int K, const int N,
                                         AAcc A, BAcc B, CAcc C,
                                         const int e0, const int NE)
{
   const int thread = getThreadIdx();
   const int warpId = getWarpId(thread);
   const int nWarps = getNumWarps();
   const int laneId = getLaneId(thread);
   const int groupId = getGroupId(laneId);
   const int threadIdInGroup = getThreadIdInGroup(laneId);
   const int mPass = (K + mmaM - 1) / mmaM;
   const int nTiles = (N + mmaN - 1) / mmaN;

   for (int tile = warpId; tile < mPass; tile += nWarps)
   {
      const int row0 = tile * mmaM;
      double cReg[MAX_N_TILES][2] = {}; // [nTile][frag]

      for (int mK = 0; mK < (M + mmaK - 1) / mmaK; mK++)
      {
         double aReg[1];
         const int aT_row = row0 + groupId;
         const int aT_col = threadIdInGroup + mK * mmaK;
         aReg[0] = (aT_row < K && aT_col < M)
                   ? static_cast<double>(A(aT_col, aT_row)) : 0.0;

         for (int nt = 0; nt < nTiles; ++nt)
         {
            const int n0 = nt * mmaN;
            const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
            double bReg[1];
            const int bRow = threadIdInGroup + mK * mmaK;
            const int bColumn = MagicCol<MAGIC>(groupId);
            bReg[0] = (bRow < M && bColumn < nTile)
                      ? static_cast<double>(B(bRow, n0 + bColumn)) : 0.0;
            dmmaSync(aReg, bReg, cReg[nt]);
         }
      }
      for (int nt = 0; nt < nTiles; ++nt)
      {
         const int n0 = nt * mmaN;
         const int nTile = (N - n0 < mmaN) ? (N - n0) : mmaN;
         for (int i = 0; i < 2; i++)
         {
            const int cRow = row0 + groupId;
            const int cColumn = MagicCol<MAGIC>(threadIdInGroup * 2 + i);
            const int e = e0 + n0 + cColumn;
            if (cRow < K && cColumn < nTile && e < NE)
            {
               C(cRow, n0 + cColumn) += static_cast<real_t>(cReg[nt][i]);
            }
         }
      }
   }
}

template <int MAGIC, bool SCALE, typename AAcc, typename BAcc, typename CAcc,
          typename DAcc>
MFEM_HOST_DEVICE inline void dmma_Gemm(const int M, const int K, const int N,
                                       AAcc A, BAcc B, CAcc C,
                                       DAcc D, const int e0, const int NE,
                                       const int k_mir)
{
   dmma_Gemm8<MAGIC, SCALE>(M, K, N, A, B, C, D, e0, NE, k_mir);
}

template <int MAGIC, typename AAcc, typename BAcc, typename CAcc>
MFEM_HOST_DEVICE inline void dmma_GemmT(const int M, const int K, const int N,
                                        AAcc A, BAcc B, CAcc C,
                                        const int e0, const int NE)
{
   dmma_GemmT8<MAGIC>(M, K, N, A, B, C, e0, NE);
}

/** Y(r, e0+b) for in-register GemmT store (skips XY scratch / final smem store). */
struct YBatchAcc
{
   real_t *y;
   int ndof, e0;
   MFEM_HOST_DEVICE inline real_t &operator()(int r, int b) const
   {
      return y[r + ndof * (e0 + b)];
   }
};

} // namespace mma
} // namespace tmo

template<int DIM, int T_D1D, int T_Q1D, bool SINGLE_CHART>
MFEM_HOST_DEVICE inline
void SmemPAMassApplySimplexTmoMma_Batch(const int e0,
                                        const int NE,
                                        const real_t *p_,
                                        const real_t *d_,
                                        const real_t *x_,
                                        real_t *y_,
                                        const int d1d,
                                        const int nq1,
                                        const int nmirrors)
{
   constexpr int MQ = tmo::mma::SimplexMaxNq<DIM, T_Q1D>();
   constexpr int BASIS_DIM = tmo::mma::SimplexNdof<DIM, T_D1D>();
   constexpr int MAGIC = tmo::mma::MagicFor<DIM, T_D1D, T_Q1D>();
   constexpr int X_LD = tmo::mma::PadLdBank<MAGIC>(BASIS_DIM);
   constexpr int U_LD = tmo::mma::PadLdBank<MAGIC>(MQ);
   // Specialized: NBATCH=16. Fallback: NB=8 to stay under 48KB smem.
   // Large 3D tet rules (nq>~200): NB=8 even when specialized.
   constexpr int NB = (T_D1D && T_Q1D && !(DIM == 3 && T_Q1D > 160))
                      ? tmo::mma::NBATCH : tmo::mma::mmaN;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                               : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int NQ1 = T_Q1D ? T_Q1D : nq1;

   const auto D = DeviceTensor<3, const real_t>(d_, NQ1, nmirrors, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);

   // XY = X only; Us = U; Y accumulates in global memory from GemmT.
   // P stays in global/L2 (staging was measured slower — mesh-wide P is hot).
   struct alignas(16) Smem
   {
      real_t XY[X_LD * NB];
      real_t Us[U_LD * NB];
   };
   MFEM_SHARED Smem sm;

   struct PSlice
   {
      const real_t *p;
      int nq1_, ndof_, k_;
      MFEM_HOST_DEVICE inline real_t operator()(int row, int col) const
      {
         return p[row + nq1_ * (col + ndof_ * k_)];
      }
   };

   const int tid = tmo::mma::getThreadIdx();
#ifdef __CUDA_ARCH__
   const int nthreads = blockDim.x * blockDim.y * blockDim.z;
#else
   const int nthreads = 1;
#endif

#if defined(__CUDA_ARCH__) && !defined(MFEM_USE_SINGLE)
   tmo::mma::SmemMatAcc<X_LD> Xacc{sm.XY};
   tmo::mma::SmemMatAcc<U_LD> Uacc{sm.Us};
   tmo::mma::YBatchAcc Yacc{y_, ndof, e0};

   // X is chart-independent: load once. D-scale fused into Gemm. GemmT
   // accumulates straight into global Y (no XY clear / smem Y store).
   for (int i = tid; i < X_LD * NB; i += nthreads)
   {
      const int b = i / X_LD;
      const int r = i - b * X_LD;
      const int e = e0 + b;
      sm.XY[i] = (e < NE && r < ndof) ? x(r, e) : real_t(0);
   }
   MFEM_SYNC_THREAD;

   const int k_end = SINGLE_CHART ? 1 : nmirrors;
   for (int k = 0; k < k_end; ++k)
   {
      PSlice A{p_, NQ1, ndof, k};
      tmo::mma::dmma_Gemm<MAGIC, true>(NQ1, ndof, NB, A, Xacc, Uacc,
                                       D, e0, NE, k);
      MFEM_SYNC_THREAD;
      tmo::mma::dmma_GemmT<MAGIC>(NQ1, ndof, NB, A, Uacc, Yacc, e0, NE);
      if (k + 1 < k_end) { MFEM_SYNC_THREAD; }
   }
#else
   auto Y = DeviceMatrix(y_, ndof, NE);
   if (tid == 0)
   {
      for (int k = 0; k < nmirrors; ++k)
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
                  u += p_[q + NQ1 * (i + ndof * k)] * sm.XY[i + X_LD * b];
               }
               sm.Us[q + U_LD * b] = u * D(q, k, e);
            }
            for (int i = 0; i < ndof; ++i)
            {
               real_t yi = 0.0;
               for (int q = 0; q < NQ1; ++q)
               {
                  yi += p_[q + NQ1 * (i + ndof * k)] * sm.Us[q + U_LD * b];
               }
               Y(i, e) += yi;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
#endif
}

template<int DIM = 2, int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplySimplexTmoMma(const int NE,
                                         const Array<real_t> &p_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_,
                                         const int d1d = 0,
                                         const int nq1 = 0)
{
   constexpr int NB = (T_D1D && T_Q1D && !(DIM == 3 && T_Q1D > 160))
                      ? tmo::mma::NBATCH : tmo::mma::mmaN;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int NQ1 = T_Q1D ? T_Q1D : nq1;
   const int ndof = (DIM == 2) ? (D1D * (D1D + 1) / 2)
                               : (D1D * (D1D + 1) * (D1D + 2) / 6);
   const int max_d1d = T_D1D ? T_D1D
                       : ((DIM == 3) ? tmo::mma::FallbackMaxD1D3
                          : DeviceDofQuadLimits::Get().MAX_D1D);
   const int max_nq = tmo::mma::SimplexMaxNq<DIM, T_Q1D>();
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(NQ1 <= max_nq, "");
   MFEM_VERIFY(NQ1 > 0 && NE > 0 && d_.Size() % (NQ1 * NE) == 0, "");
   const int nmirrors = d_.Size() / (NQ1 * NE);
   MFEM_VERIFY(nmirrors >= 1 && nmirrors <= tmo::NMIRRORS, "");
   MFEM_VERIFY(p_.Size() == NQ1 * ndof * nmirrors, "");
   if constexpr (DIM == 3)
   {
      MFEM_VERIFY(nmirrors == 1, "3D TMO MMA currently supports nmirrors=1 only");
   }

   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   // Keep all warps busy in both Gemm phases (strided M-tiles).
   const int mPassQ = (NQ1 + tmo::mma::mmaM - 1) / tmo::mma::mmaM;
   const int mPassD = (ndof + tmo::mma::mmaM - 1) / tmo::mma::mmaM;
   const int nWarps = (mPassQ < mPassD) ? (mPassQ > 1 ? mPassQ : 1)
                                        : (mPassD > 1 ? mPassD : 1);
   const int nthreads = nWarps * 32;
   const int nbatches = (NE + NB - 1) / NB;

   if (nmirrors == 1)
   {
      mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
      {
         SmemPAMassApplySimplexTmoMma_Batch<DIM, T_D1D, T_Q1D, true>(
            batch * NB, NE, P, D, X, Y, d1d, nq1, 1);
      });
   }
   else
   {
      mfem::forall_3D(nbatches, nthreads, 1, 1, [=] MFEM_HOST_DEVICE (int batch)
      {
         SmemPAMassApplySimplexTmoMma_Batch<DIM, T_D1D, T_Q1D, false>(
            batch * NB, NE, P, D, X, Y, d1d, nq1, nmirrors);
      });
   }
}

// Backward-compatible alias used by existing 2D call sites / specializations.
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangleTmoMma(const int NE,
                                          const Array<real_t> &p_,
                                          const Vector &d_,
                                          const Vector &x_,
                                          Vector &y_,
                                          const int d1d = 0,
                                          const int nq1 = 0)
{
   SmemPAMassApplySimplexTmoMma<2, T_D1D, T_Q1D>(NE, p_, d_, x_, y_, d1d, nq1);
}

// Bernstein parallelogram TMO: even-prolong to tensor Bernstein coeffs, then
// stock-like B/Bt sum-fac (no Duffy / Stroud).
// ---------------------------------------------------------------------------

template<int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApplyTriangleTmoBernstein_Element(const int e,
                                                 const int NE,
                                                 const real_t *b_,
                                                 const real_t *p_,
                                                 const real_t *d_,
                                                 const real_t *x_,
                                                 real_t *y_,
                                                 const int d1d,
                                                 const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int ndof = D1D * (D1D + 1) / 2;
   const int nqdof = D1D * D1D;

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   constexpr int BASIS_DIM = MD1 * (MD1 + 1) / 2;

   const auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   const auto P = DeviceTensor<3, const real_t>(p_, nqdof, ndof, tmo::NMIRRORS);
   const auto D = DeviceTensor<4, const real_t>(d_, Q1D, Q1D, tmo::NMIRRORS, NE);
   const auto x = ConstDeviceMatrix(x_, ndof, NE);
   auto Y = DeviceMatrix(y_, ndof, NE);

   MFEM_SHARED real_t BBt[MQ1 * MD1];
   real_t (*B)[MD1] = (real_t (*)[MD1]) BBt;
   real_t (*Bt)[MQ1] = (real_t (*)[MQ1]) BBt;
   MFEM_SHARED real_t Xtri[BASIS_DIM];
   MFEM_SHARED real_t sm0[MDQ * MDQ], sm1[MDQ * MDQ];
   real_t (*Xq)[MD1] = (real_t (*)[MD1]) sm0;
   real_t (*DQ)[MQ1] = (real_t (*)[MQ1]) sm1;
   real_t (*QQ)[MQ1] = (real_t (*)[MQ1]) sm0;
   real_t (*QD)[MD1] = (real_t (*)[MD1]) sm1;

   MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
   {
      Xtri[i] = x(i, e);
   }
   MFEM_SYNC_THREAD;

   for (int k = 0; k < tmo::NMIRRORS; ++k)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
         {
            const int iq = dx + D1D * dy;
            real_t u = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               u += P(iq, i, k) * Xtri[i];
            }
            Xq[dy][dx] = u;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(q, x, Q1D)
         {
            B[q][dy] = b(q, dy);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            real_t dq = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               dq += Xq[dy][dx] * B[qx][dx];
            }
            DQ[dy][qx] = dq;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            real_t qq = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               qq += DQ[dy][qx] * B[qy][dy];
            }
            QQ[qy][qx] = qq * D(qx, qy, k, e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
         {
            Bt[dx][qy] = b(qy, dx);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
         {
            real_t qd = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               qd += QQ[qy][qx] * Bt[dx][qx];
            }
            QD[qy][dx] = qd;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
         {
            real_t u = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += QD[qy][dx] * Bt[dy][qy];
            }
            Xq[dy][dx] = u;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
      {
         real_t yi = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               yi += P(dx + D1D * dy, i, k) * Xq[dy][dx];
            }
         }
         Y(i, e) += yi;
      }
      MFEM_SYNC_THREAD;
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApplyTriangleTmoBernstein(const int NE,
                                                const Array<real_t> &b_,
                                                const Array<real_t> &p_,
                                                const Vector &d_,
                                                const Vector &x_,
                                                Vector &y_,
                                                const int d1d = 0,
                                                const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");

   const auto B = b_.Read();
   const auto P = p_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   const int T1D = (Q1D > D1D) ? Q1D : D1D;
   constexpr int T_T1D = (T_Q1D > T_D1D) ? T_Q1D : T_D1D;

   mfem::forall_2D<T_T1D * T_T1D>(NE, T1D, T1D, [=] MFEM_HOST_DEVICE (int e)
   {
      SmemPAMassApplyTriangleTmoBernstein_Element<T_D1D, T_Q1D>(
         e, NE, B, P, D, X, Y, d1d, q1d);
   });
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTmoKernelType
MassIntegrator::ApplyTmoPAKernels::Kernel()
{
   MFEM_CONTRACT_VAR(DIM);
   return internal::SmemPAMassApplyTriangleTmoDuffy<T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTmoKernelType
MassIntegrator::ApplyTmoPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2, "TMO Duffy mass PA is only implemented for triangles");
   return internal::SmemPAMassApplyTriangleTmoDuffy;
}

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTmoKernelType
MassIntegrator::ApplyTmoCompositePAKernels::Kernel()
{
   MFEM_CONTRACT_VAR(DIM);
   return internal::SmemPAMassApplyTriangleTmoComposite<T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTmoKernelType
MassIntegrator::ApplyTmoCompositePAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2,
               "TMO Composite mass PA is only implemented for triangles");
   return internal::SmemPAMassApplyTriangleTmoComposite;
}

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTmoTensorKernelType
MassIntegrator::ApplyTmoTensorPAKernels::Kernel()
{
   MFEM_CONTRACT_VAR(DIM);
   return internal::SmemPAMassApplyTriangleTmoTensor<T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTmoTensorKernelType
MassIntegrator::ApplyTmoTensorPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2, "TMO Tensor mass PA is only implemented for triangles");
   return internal::SmemPAMassApplyTriangleTmoTensor;
}

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTmoTensorKernelType
MassIntegrator::ApplyTmoMmaPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::SmemPAMassApplySimplexTmoMma<2, T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::SmemPAMassApplySimplexTmoMma<3, T_D1D, T_Q1D>;
   }
   else
   {
      MFEM_ABORT("TMO MMA only supports DIM 2 or 3");
      return nullptr;
   }
}

inline MassIntegrator::ApplyTmoTensorKernelType
MassIntegrator::ApplyTmoMmaPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "TMO MMA mass PA is only implemented for triangles/tets");
   if (dim == 3)
   {
      return internal::SmemPAMassApplySimplexTmoMma<3>;
   }
   return internal::SmemPAMassApplySimplexTmoMma<2>;
}

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTmoBernsteinKernelType
MassIntegrator::ApplyTmoBernsteinPAKernels::Kernel()
{
   MFEM_CONTRACT_VAR(DIM);
   return internal::SmemPAMassApplyTriangleTmoBernstein<T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplyTmoBernsteinKernelType
MassIntegrator::ApplyTmoBernsteinPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 2,
               "TMO Bernstein mass PA is only implemented for triangles");
   return internal::SmemPAMassApplyTriangleTmoBernstein;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
