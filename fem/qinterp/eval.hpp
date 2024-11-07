// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, q1d, vdim, NE):
            Reshape(y_, vdim, q1d, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int c = 0; c < vdim; c++)
      {
         for (int q = 0; q < q1d; q++)
         {
            real_t u = 0.0;
            for (int d = 0; d < d1d; d++)
            {
               u += b(q, d) * x(d, c, e);
            }
            if (Q_LAYOUT == QVectorLayout::byVDIM)  { y(c, q, e) = u; }
            if (Q_LAYOUT == QVectorLayout::byNODES) { y(q, c, e) = u; }
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
   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
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
               if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,e) = u; }
               if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,c,e) = u; }
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
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
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
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,qz,e) = u; }
                  if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,qz,c,e) = u; }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace quadrature_interpolator

} // namespace internal

/// @cond Suppress_Doxygen_warnings

template<int DIM, QVectorLayout Q_LAYOUT,
         int VDIM, int D1D, int Q1D, int NBZ>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel()
{
   if (DIM == 1) { return internal::quadrature_interpolator::Values1D<Q_LAYOUT>; }
   else if (DIM == 2) { return internal::quadrature_interpolator::Values2D<Q_LAYOUT, VDIM, D1D, Q1D, NBZ>; }
   else if (DIM == 3) { return internal::quadrature_interpolator::Values3D<Q_LAYOUT, VDIM, D1D, Q1D>; }
   else { MFEM_ABORT(""); }
}

/// @endcond

} // namespace mfem

#endif
