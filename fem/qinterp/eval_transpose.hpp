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
static void ValuesTranspose1D(const int NE,
                              const real_t *b_,
                              const real_t *q_,
                              real_t *e_,
                              const int vdim,
                              const int d1d,
                              const int q1d)
{
   const auto b = Reshape(b_, q1d, d1d);
   const auto qd = Q_LAYOUT == QVectorLayout::byNODES ?
                   Reshape(q_, q1d, vdim, NE) :
                   Reshape(q_, vdim, q1d, NE);
   auto e = Reshape(e_, d1d, vdim, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int el)
   {
      for (int c = 0; c < vdim; c++)
      {
         for (int d = 0; d < d1d; d++)
         {
            real_t u = 0.0;
            for (int q = 0; q < q1d; q++)
            {
               const real_t qval = Q_LAYOUT == QVectorLayout::byVDIM ?
                                   qd(c, q, el) : qd(q, c, el);
               u += b(q, d) * qval;
            }
            e(d, c, el) = u;
         }
      }
   });
}

template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1>
static void ValuesTranspose2D(const int NE,
                              const real_t *b_,
                              const real_t *q_,
                              real_t *e_,
                              const int vdim = 0,
                              const int d1d = 0,
                              const int q1d = 0)
{
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto q = Q_LAYOUT == QVectorLayout::byNODES ?
                  Reshape(q_, Q1D, Q1D, VDIM, NE) :
                  Reshape(q_, VDIM, Q1D, Q1D, NE);
   auto e = Reshape(e_, D1D, D1D, VDIM, NE);

   mfem::forall_2D_batch(NE, D1D, D1D, NBZ, [=] MFEM_HOST_DEVICE (int el)
   {
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

      ConstDeviceMatrix B(sB, D1D, Q1D);
      DeviceMatrix QQ(sm0[tidz], MQ1, MQ1);
      DeviceMatrix DQ(sm1[tidz], MD1, MQ1);
      DeviceMatrix DD(sm0[tidz], MD1, MD1);

      for (int c = 0; c < VDIM; c++)
      {
         // Load Q data
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               QQ(qx,qy) = Q_LAYOUT == QVectorLayout::byVDIM ?
                           q(c,qx,qy,el) : q(qx,qy,c,el);
            }
         }
         MFEM_SYNC_THREAD;

         // Transpose in y: QQ -> DQ (apply B^T in y-direction)
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += B(dy,qy) * QQ(qx,qy);
               }
               DQ(dy,qx) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Transpose in x: DQ -> DD (apply B^T in x-direction)
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += B(dx,qx) * DQ(dy,qx);
               }
               DD(dx,dy) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Store result
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               e(dx,dy,c,el) = DD(dx,dy);
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void ValuesTranspose3D(const int NE,
                              const real_t *b_,
                              const real_t *q_,
                              real_t *e_,
                              const int vdim = 0,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto q = Q_LAYOUT == QVectorLayout::byNODES ?
                  Reshape(q_, Q1D, Q1D, Q1D, VDIM, NE) :
                  Reshape(q_, VDIM, Q1D, Q1D, Q1D, NE);
   auto e = Reshape(e_, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int el)
   {
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

      ConstDeviceMatrix B(sB, D1D, Q1D);
      DeviceCube QQQ(sm0, MQ1, MQ1, MQ1);
      DeviceCube DQQ(sm1, MD1, MQ1, MQ1);
      DeviceCube DDQ(sm0, MD1, MD1, MQ1);
      DeviceCube DDD(sm1, MD1, MD1, MD1);

      for (int c = 0; c < VDIM; c++)
      {
         // Load Q data
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  QQQ(qx,qy,qz) = Q_LAYOUT == QVectorLayout::byVDIM ?
                                  q(c,qx,qy,qz,el) : q(qx,qy,qz,c,el);
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Transpose in z
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += B(dz,qz) * QQQ(qx,qy,qz);
                  }
                  DQQ(dz,qx,qy) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Transpose in y
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += B(dy,qy) * DQQ(dz,qx,qy);
                  }
                  DDQ(dz,dy,qx) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Transpose in x
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += B(dx,qx) * DDQ(dz,dy,qx);
                  }
                  DDD(dx,dy,dz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  e(dx,dy,dz,c,el) = DDD(dx,dy,dz);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace quadrature_interpolator
} // namespace internal

template<int DIM, QVectorLayout Q_LAYOUT,
         int VDIM, int D1D, int Q1D, int NBZ>
QuadratureInterpolator::TensorEvalTransposeKernelType
QuadratureInterpolator::TensorEvalTransposeKernels::Kernel()
{
   if (DIM == 1) { return internal::quadrature_interpolator::ValuesTranspose1D<Q_LAYOUT>; }
   else if (DIM == 2) { return internal::quadrature_interpolator::ValuesTranspose2D<Q_LAYOUT, VDIM, D1D, Q1D, NBZ>; }
   else if (DIM == 3) { return internal::quadrature_interpolator::ValuesTranspose3D<Q_LAYOUT, VDIM, D1D, Q1D>; }
   else { MFEM_ABORT(""); }
}

} // namespace mfem
