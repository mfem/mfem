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
    - Duffy path: vertex-permuted Stroud + Bernstein Ba1/Ba2 sum-fac
    - Tensor path: even prolong onto 3 parallelograms + fused GLL quad tensor */
namespace tmo
{

constexpr int NMIRRORS = 3;

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

   MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
   {
      Xtri[i] = x(i, e);
   }
   MFEM_SYNC_THREAD;

   for (int k = 0; k < tmo::NMIRRORS; ++k)
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
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD_DIRECT(i, x, ndof)
      {
         real_t yi = 0.0;
         for (int q = 0; q < NQ1; ++q)
         {
            yi += P(q, i, k) * Uq[q];
         }
         Y(i, e) += yi;
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
