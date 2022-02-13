// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEM_KERNELS_EVAL_2D_HPP
#define MFEM_FEM_KERNELS_EVAL_2D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace eval
{

/// 2D Scalar Evaluation, 1/2
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               ConstDeviceMatrix &B,
                               DeviceMatrix &DD,
                               DeviceMatrix &DQ)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            u += B(dx,qx) * DD(dx,dy);
         }
         DQ(dy,qx) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               double (&sDD)[NBZ][MD1*MD1],
                               double (&sDQ)[NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix DD(sDD[tidz], D1D, D1D);
   DeviceMatrix DQ(sDQ[tidz], D1D, Q1D);
   X(D1D,Q1D,B,DD,DQ);
}

/// 2D Scalar Evaluation, 2/2
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               ConstDeviceMatrix &B,
                               DeviceMatrix &DQ,
                               DeviceMatrix &QQ)
{
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            u += DQ(dy,qx) * B(dy,qy);
         }
         QQ(qx,qy) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               double (&sDQ)[NBZ][MD1*MQ1],
                               double (&sQQ)[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix DQ(sDQ[tidz], D1D, Q1D);
   DeviceMatrix QQ(sQQ[tidz], Q1D, Q1D);
   Y(D1D,Q1D,B,DQ,QQ);
}

/// 2D Evaluation, 1/2 (only B)
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sX)[2][NBZ][MD1*MD1],
                               double (&sDQ)[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceMatrix X0(sX[0][tidz], D1D, D1D);
   ConstDeviceMatrix X1(sX[1][tidz], D1D, D1D);
   DeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   DeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double xx = X0(dx,dy);
            const double xy = X1(dx,dy);
            u[0] += B(dx,qx) * xx;
            u[1] += B(dx,qx) * xy;
         }
         DQ0(qx,dy) = u[0];
         DQ1(qx,dy) = u[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Evaluation, 2/2 (only B)
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDQ)[2][NBZ][MD1*MQ1],
                               double (&sQQ)[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);
   DeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         for (int dy = 0; dy < D1D; ++dy)
         {
            u[0] += DQ0(qx,dy) * B(dy,qy);
            u[1] += DQ1(qx,dy) * B(dy,qy);
         }
         QQ0(qx,qy) = u[0];
         QQ1(qx,qy) = u[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Transposed evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const double (&sB)[MQ1*MD1],
                                const double (&sQQ)[2][NBZ][MQ1*MQ1],
                                double (&sDQ)[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);
   DeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   DeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u[0] += QQ0(qx,qy) * Bt(qx,dx);
            u[1] += QQ1(qx,qy) * Bt(qx,dx);
         }
         DQ0(qy,dx) = u[0];
         DQ1(qy,dx) = u[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Transposed evaluation, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const double (&sB)[MQ1*MD1],
                                const double (&sDQ)[2][NBZ][MD1*MQ1],
                                const DeviceTensor<4> &Y, // output
                                const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u[0] += Bt(qy,dy) * DQ0(qy,dx);
            u[1] += Bt(qy,dy) * DQ1(qy,dx);
         }
         Y(dx,dy,0,e) += u[0];
         Y(dx,dy,1,e) += u[1];
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace kernels::internal::eval

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_EVAL_2D_HPP
