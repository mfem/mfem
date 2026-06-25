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

// Internal header, included only by .cpp files.
// Template function implementations.

#ifndef MFEM_QUADINTERP_EVAL
#define MFEM_QUADINTERP_EVAL

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/kernels.hpp"
#include "../kernels.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

template<QVectorLayout Q_LAYOUT>
static void Values1D(const int NE,
                     const real_t *b_,
                     const real_t *x_,
                     real_t *y_,
                     const int vdim,
                     const int d1d,
                     const int q1d)
{
   const auto b = Reshape(b_, q1d, d1d);
   const auto x = Reshape(x_, d1d, vdim, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES ? Reshape(y_, q1d, vdim, NE)
               : Reshape(y_, vdim, q1d, NE);
      for (int c = 0; c < vdim; c++)
      {
         for (int q = 0; q < q1d; q++)
         {
            real_t u = 0.0;
            for (int d = 0; d < d1d; d++)
            {
               u += b(q, d) * x(d, c, e);
            }
            if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
            {
               y(c, q, e) = u;
            }
            if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
            {
               y(q, c, e) = u;
            }
         }
      }
   });
}

template <QVectorLayout Q_LAYOUT>
static void IntValues1D(const int NE, const real_t *b_, const real_t *detJ_,
                        const real_t *x_, real_t *y_, const int vdim,
                        const int d1d, const int q1d)
{
   const auto b = Reshape(b_, q1d, d1d);
   const auto x = Reshape(x_, d1d, vdim, NE);
   const auto detJ = Reshape(detJ_, d1d, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES ? Reshape(y_, q1d, vdim, NE)
               : Reshape(y_, vdim, q1d, NE);
      for (int c = 0; c < vdim; c++)
      {
         for (int q = 0; q < q1d; q++)
         {
            real_t u = 0.0;
            for (int d = 0; d < d1d; d++)
            {
               u += b(q, d) * x(d, c, e) * detJ(d, e);
            }
            if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
            {
               y(c, q, e) = u;
            }
            if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
            {
               y(q, c, e) = u;
            }
         }
      }
   });
}

// Template compute kernel for Values in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1>
static void Values2D(const int NE,
                     const real_t *b_,
                     const real_t *x_,
                     real_t *y_,
                     const int vdim = 0,
                     const int d1d = 0,
                     const int q1d = 0)
{
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES
               ? Reshape(y_, Q1D, Q1D, VDIM, NE)
               : Reshape(y_, VDIM, Q1D, Q1D, NE);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_SHARED real_t sB[MQ1*MD1];
      MFEM_SHARED real_t sm0[NBZ][MDQ*MDQ];
      MFEM_SHARED real_t sm1[NBZ][MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceMatrix DD(sm0[tidz], MD1, MD1);
      DeviceMatrix DQ(sm1[tidz], MD1, MQ1);
      DeviceMatrix QQ(sm0[tidz], MQ1, MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX(e,D1D,c,x,DD);
         kernels::internal::EvalX(D1D,Q1D,B,DD,DQ);
         kernels::internal::EvalY(D1D,Q1D,B,DQ,QQ);
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = QQ(qx,qy);
               if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
               {
                  y(c, qx, qy, e) = u;
               }
               if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx, qy, c, e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for Values in 2D: tensor product version.
template <QVectorLayout Q_LAYOUT, int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
          int T_NBZ = 1>
static void IntValues2D(const int NE, const real_t *b_, const real_t *detJ_,
                        const real_t *x_, real_t *y_, const int vdim = 0,
                        const int d1d = 0, const int q1d = 0)
{
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   const auto detJ = Reshape(detJ_, d1d, d1d, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES
               ? Reshape(y_, Q1D, Q1D, VDIM, NE)
               : Reshape(y_, VDIM, Q1D, Q1D, NE);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_SHARED real_t sB[MQ1*MD1];
      MFEM_SHARED real_t sm0[NBZ][MDQ*MDQ];
      MFEM_SHARED real_t sm1[NBZ][MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceMatrix DD(sm0[tidz], MD1, MD1);
      DeviceMatrix DQ(sm1[tidz], MD1, MQ1);
      DeviceMatrix QQ(sm0[tidz], MQ1, MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               DD(dx, dy) = x(dx, dy, c, e) * detJ(dx, dy, e);
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::EvalX(D1D,Q1D,B,DD,DQ);
         kernels::internal::EvalY(D1D,Q1D,B,DQ,QQ);
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = QQ(qx, qy);
               if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
               {
                  y(c, qx, qy, e) = u;
               }
               if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx, qy, c, e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for Values in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void Values3D(const int NE,
                     const real_t *b_,
                     const real_t *x_,
                     real_t *y_,
                     const int vdim = 0,
                     const int d1d = 0,
                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES
               ? Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE)
               : Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t sB[MQ1*MD1];
      MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX(e,D1D,c,x,DDD);
         kernels::internal::EvalX(D1D,Q1D,B,DDD,DDQ);
         kernels::internal::EvalY(D1D,Q1D,B,DDQ,DQQ);
         kernels::internal::EvalZ(D1D,Q1D,B,DQQ,QQQ);
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const real_t u = QQQ(qz,qy,qx);
                  if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     y(c, qx, qy, qz, e) = u;
                  }
                  if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx, qy, qz, c, e) = u;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for Values in 3D: tensor product version.
template <QVectorLayout Q_LAYOUT, int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void IntValues3D(const int NE, const real_t *b_, const real_t *detJ_,
                        const real_t *x_, real_t *y_, const int vdim = 0,
                        const int d1d = 0, const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   const auto detJ = Reshape(detJ_, d1d, d1d, d1d, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      // nvcc limitation: can't capture y inside a constexpr first
      auto y = Q_LAYOUT == QVectorLayout::byNODES
               ? Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE)
               : Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t sB[MQ1*MD1];
      MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  DDD(dx, dy, dz) = x(dx, dy, dz, c, e) * detJ(dx, dy, dz, e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::EvalX(D1D,Q1D,B,DDD,DDQ);
         kernels::internal::EvalY(D1D,Q1D,B,DDQ,DQQ);
         kernels::internal::EvalZ(D1D,Q1D,B,DQQ,QQQ);
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const real_t u = QQQ(qz,qy,qx);
                  if constexpr (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     y(c, qx, qy, qz, e) = u;
                  }
                  if constexpr (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx, qy, qz, c, e) = u;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void Eval1D(const int NE, const int vdim, const QVectorLayout q_layout,
            const GeometricFactors *geom, const DofToQuad &maps,
            const Vector &e_vec, Vector &q_val, Vector &q_der, Vector &q_det,
            const int eval_flags);

void IntEval1D(const int NE, const int vdim, const QVectorLayout q_layout,
               const GeometricFactors *detJgeom, const GeometricFactors *geom,
               const DofToQuad &maps, const Vector &e_vec, Vector &q_val,
               Vector &q_der, Vector &q_det, const int eval_flags);

// Template compute kernel for 2D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
template<const int T_VDIM, const int T_ND, const int T_NQ>
static void Eval2D(const int NE,
                   const int vdim,
                   const QVectorLayout q_layout,
                   const GeometricFactors *geom,
                   const DofToQuad &maps,
                   const Vector &e_vec,
                   Vector &q_val,
                   Vector &q_der,
                   Vector &q_det,
                   const int eval_flags)
{
   using QI = QuadratureInterpolator;

   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int NMAX = NQ > ND ? NQ : ND;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_ASSERT(maps.mode == DofToQuad::FULL, "internal error");
   MFEM_ASSERT(!geom || geom->mesh->SpaceDimension() == 2, "");
   MFEM_VERIFY(ND <= QI::MAX_ND2D, "");
   MFEM_VERIFY(NQ <= QI::MAX_NQ2D, "");
   MFEM_VERIFY(bool(geom) == bool(eval_flags & QI::PHYSICAL_DERIVATIVES),
               "'geom' must be given (non-null) only when evaluating physical"
               " derivatives");
   const auto B = Reshape(maps.B.Read(), NQ, ND);
   const auto G = Reshape(maps.G.Read(), NQ, 2, ND);
   const auto J = Reshape(geom ? geom->J.Read() : nullptr, NQ, 2, 2, NE);
   const auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), NQ, VDIM, NE):
              Reshape(q_val.Write(), VDIM, NQ, NE);
   auto der = q_layout == QVectorLayout::byNODES ?
              Reshape(q_der.Write(), NQ, VDIM, 2, NE):
              Reshape(q_der.Write(), VDIM, 2, NQ, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   mfem::forall_2D(NE, NMAX, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : QI::MAX_ND2D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : QI::MAX_VDIM2D;
      MFEM_SHARED real_t s_E[max_VDIM*max_ND];
      MFEM_FOREACH_THREAD(d, x, ND)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(q, x, NQ)
      {
         if (eval_flags & (QI::VALUES | QI::PHYSICAL_VALUES))
         {
            real_t ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const real_t b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++)
            {
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,e) = ed[c]; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,e) = ed[c]; }
            }
         }
         if ((eval_flags & QI::DERIVATIVES) ||
             (eval_flags & QI::PHYSICAL_DERIVATIVES) ||
             (eval_flags & QI::DETERMINANTS))
         {
            // use MAX_VDIM2D to avoid "subscript out of range" warnings
            real_t D[QI::MAX_VDIM2D*2];
            for (int i = 0; i < 2*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const real_t wx = G(q,0,d);
               const real_t wy = G(q,1,d);
               for (int c = 0; c < VDIM; c++)
               {
                  real_t s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
               }
            }
            if (eval_flags & QI::DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = D[c+VDIM*0];
                     der(c,1,q,e) = D[c+VDIM*1];
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = D[c+VDIM*0];
                     der(q,c,1,e) = D[c+VDIM*1];
                  }
               }
            }
            if (eval_flags & QI::PHYSICAL_DERIVATIVES)
            {
               real_t Jloc[4], Jinv[4];
               Jloc[0] = J(q,0,0,e);
               Jloc[1] = J(q,1,0,e);
               Jloc[2] = J(q,0,1,e);
               Jloc[3] = J(q,1,1,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               for (int c = 0; c < VDIM; c++)
               {
                  const real_t u = D[c+VDIM*0];
                  const real_t v = D[c+VDIM*1];
                  const real_t JiU = Jinv[0]*u + Jinv[1]*v;
                  const real_t JiV = Jinv[2]*u + Jinv[3]*v;
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = JiU;
                     der(c,1,q,e) = JiV;
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = JiU;
                     der(q,c,1,e) = JiV;
                  }
               }
            }
            if (eval_flags & QI::DETERMINANTS)
            {
               if (VDIM == 2) { det(q,e) = kernels::Det<2>(D); }
               else
               {
                  DeviceTensor<2> j(D, 3, 2);
                  const double E = j(0,0)*j(0,0) + j(1,0)*j(1,0) + j(2,0)*j(2,0);
                  const double F = j(0,0)*j(0,1) + j(1,0)*j(1,1) + j(2,0)*j(2,1);
                  const double G = j(0,1)*j(0,1) + j(1,1)*j(1,1) + j(2,1)*j(2,1);
                  det(q,e) = std::sqrt(E*G - F*F);
               }
            }
         }
      }
   });
}

// Template compute kernel for 2D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
template <const int T_VDIM, const int T_ND, const int T_NQ>
static void
IntEval2D(const int NE, const int vdim, const QVectorLayout q_layout,
          const GeometricFactors *detJgeom, const GeometricFactors *geom,
          const DofToQuad &maps, const Vector &e_vec, Vector &q_val,
          Vector &q_der, Vector &q_det, const int eval_flags)
{
   // TODO
   MFEM_ABORT("Not implemented yet");
}

// Template compute kernel for 3D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
template<const int T_VDIM, const int T_ND, const int T_NQ>
static void Eval3D(const int NE,
                   const int vdim,
                   const QVectorLayout q_layout,
                   const GeometricFactors *geom,
                   const DofToQuad &maps,
                   const Vector &e_vec,
                   Vector &q_val,
                   Vector &q_der,
                   Vector &q_det,
                   const int eval_flags)
{
   using QI = QuadratureInterpolator;

   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int NMAX = NQ > ND ? NQ : ND;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_ASSERT(maps.mode == DofToQuad::FULL, "internal error");
   MFEM_ASSERT(!geom || geom->mesh->SpaceDimension() == 3, "");
   MFEM_VERIFY(ND <= QI::MAX_ND3D, "");
   MFEM_VERIFY(NQ <= QI::MAX_NQ3D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & QI::DETERMINANTS), "");
   MFEM_VERIFY(bool(geom) == bool(eval_flags & QI::PHYSICAL_DERIVATIVES),
               "'geom' must be given (non-null) only when evaluating physical"
               " derivatives");
   const auto B = Reshape(maps.B.Read(), NQ, ND);
   const auto G = Reshape(maps.G.Read(), NQ, 3, ND);
   const auto J = Reshape(geom ? geom->J.Read() : nullptr, NQ, 3, 3, NE);
   const auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), NQ, VDIM, NE):
              Reshape(q_val.Write(), VDIM, NQ, NE);
   auto der = q_layout == QVectorLayout::byNODES ?
              Reshape(q_der.Write(), NQ, VDIM, 3, NE):
              Reshape(q_der.Write(), VDIM, 3, NQ, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   mfem::forall_2D(NE, NMAX, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : QI::MAX_ND3D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : QI::MAX_VDIM3D;
      MFEM_SHARED real_t s_E[max_VDIM*max_ND];
      MFEM_FOREACH_THREAD(d, x, ND)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(q, x, NQ)
      {
         if (eval_flags & (QI::VALUES | QI::PHYSICAL_VALUES))
         {
            real_t ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const real_t b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++)
            {
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,e) = ed[c]; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,e) = ed[c]; }
            }
         }
         if ((eval_flags & QI::DERIVATIVES) ||
             (eval_flags & QI::PHYSICAL_DERIVATIVES) ||
             (eval_flags & QI::DETERMINANTS))
         {
            // use MAX_VDIM3D to avoid "subscript out of range" warnings
            real_t D[QI::MAX_VDIM3D*3];
            for (int i = 0; i < 3*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const real_t wx = G(q,0,d);
               const real_t wy = G(q,1,d);
               const real_t wz = G(q,2,d);
               for (int c = 0; c < VDIM; c++)
               {
                  real_t s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
                  D[c+VDIM*2] += s_e * wz;
               }
            }
            if (eval_flags & QI::DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = D[c+VDIM*0];
                     der(c,1,q,e) = D[c+VDIM*1];
                     der(c,2,q,e) = D[c+VDIM*2];
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = D[c+VDIM*0];
                     der(q,c,1,e) = D[c+VDIM*1];
                     der(q,c,2,e) = D[c+VDIM*2];
                  }
               }
            }
            if (eval_flags & QI::PHYSICAL_DERIVATIVES)
            {
               real_t Jloc[9], Jinv[9];
               for (int col = 0; col < 3; col++)
               {
                  for (int row = 0; row < 3; row++)
                  {
                     Jloc[row+3*col] = J(q,row,col,e);
                  }
               }
               kernels::CalcInverse<3>(Jloc, Jinv);
               for (int c = 0; c < VDIM; c++)
               {
                  const real_t u = D[c+VDIM*0];
                  const real_t v = D[c+VDIM*1];
                  const real_t w = D[c+VDIM*2];
                  const real_t JiU = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                  const real_t JiV = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                  const real_t JiW = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = JiU;
                     der(c,1,q,e) = JiV;
                     der(c,2,q,e) = JiW;
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = JiU;
                     der(q,c,1,e) = JiV;
                     der(q,c,2,e) = JiW;
                  }
               }
            }
            if (VDIM == 3 && (eval_flags & QI::DETERMINANTS))
            {
               // The check (VDIM == 3) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 3).
               det(q,e) = kernels::Det<3>(D);
            }
         }
      }
   });
}

// Template compute kernel for 3D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
template <const int T_VDIM, const int T_ND, const int T_NQ>
static void
IntEval3D(const int NE, const int vdim, const QVectorLayout q_layout,
          const GeometricFactors *detJgeom, const GeometricFactors *geom,
          const DofToQuad &maps, const Vector &e_vec, Vector &q_val,
          Vector &q_der, Vector &q_det, const int eval_flags)
{
   // TODO
   MFEM_ABORT("Not implemented yet");
}

} // namespace quadrature_interpolator

} // namespace internal

/// @cond Suppress_Doxygen_warnings

template<int DIM, QVectorLayout Q_LAYOUT,
         int VDIM, int D1D, int Q1D, int NBZ>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel()
{
   if constexpr (DIM == 1) { return internal::quadrature_interpolator::Values1D<Q_LAYOUT>; }
   else if constexpr (DIM == 2) { return internal::quadrature_interpolator::Values2D<Q_LAYOUT, VDIM, D1D, Q1D, NBZ>; }
   else if constexpr (DIM == 3) { return internal::quadrature_interpolator::Values3D<Q_LAYOUT, VDIM, D1D, Q1D>; }
   MFEM_ABORT("");
}

template<int DIM, QVectorLayout Q_LAYOUT,
         int VDIM, int D1D, int Q1D, int NBZ>
QuadratureInterpolator::IntTensorEvalKernelType
QuadratureInterpolator::IntTensorEvalKernels::Kernel()
{
   if constexpr (DIM == 1) { return internal::quadrature_interpolator::IntValues1D<Q_LAYOUT>; }
   else if constexpr (DIM == 2) { return internal::quadrature_interpolator::IntValues2D<Q_LAYOUT, VDIM, D1D, Q1D, NBZ>; }
   else if constexpr (DIM == 3) { return internal::quadrature_interpolator::IntValues3D<Q_LAYOUT, VDIM, D1D, Q1D>; }
   MFEM_ABORT("");
}

template <int DIM, int VDIM, int ND, int NQ>
QuadratureInterpolator::EvalKernelType
QuadratureInterpolator::EvalKernels::Kernel()
{
   using namespace internal::quadrature_interpolator;
   if constexpr (DIM == 1) { return Eval1D; }
   else if constexpr (DIM == 2) { return Eval2D<VDIM,ND,NQ>; }
   else if constexpr (DIM == 3) { return Eval3D<VDIM,ND,NQ>; }
   MFEM_ABORT("");
}

template <int DIM, int VDIM, int ND, int NQ>
QuadratureInterpolator::IntEvalKernelType
QuadratureInterpolator::IntEvalKernels::Kernel()
{
   using namespace internal::quadrature_interpolator;
   if constexpr (DIM == 1) { return IntEval1D; }
   else if constexpr (DIM == 2) { return IntEval2D<VDIM,ND,NQ>; }
   else if constexpr (DIM == 3) { return IntEval3D<VDIM,ND,NQ>; }
   MFEM_ABORT("");
}

/// @endcond

} // namespace mfem

#endif
