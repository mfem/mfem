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

#ifndef MFEM_FEM_KERNELS_GRAD_2D_HPP
#define MFEM_FEM_KERNELS_GRAD_2D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace grad
{

/// 2D Gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (&sBG)[2][MQ1*MD1],
                               const double (&sX)[2][NBZ][MD1*MD1],
                               double (&sDQ)[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   ConstDeviceMatrix X0(sX[0][tidz], D1D, D1D);
   ConstDeviceMatrix X1(sX[1][tidz], D1D, D1D);
   DeviceMatrix X0B(sDQ[0][tidz], Q1D, D1D);
   DeviceMatrix X0G(sDQ[1][tidz], Q1D, D1D);
   DeviceMatrix X1B(sDQ[2][tidz], Q1D, D1D);
   DeviceMatrix X1G(sDQ[3][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double Bx = B(dx,qx);
            const double Gx = G(dx,qx);
            const double x0 = X0(dx,dy);
            const double x1 = X1(dx,dy);
            u[0] += Bx * x0;
            v[0] += Gx * x0;
            u[1] += Bx * x1;
            v[1] += Gx * x1;
         }
         X0B(qx,dy) = u[0];
         X0G(qx,dy) = v[0];
         X1B(qx,dy) = u[1];
         X1G(qx,dy) = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Gradient, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (&sBG)[2][MQ1*MD1],
                               const double (&sDQ)[4][NBZ][MD1*MQ1],
                               double (&sQQ)[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   ConstDeviceMatrix X0B(sDQ[0][tidz], Q1D, D1D);
   ConstDeviceMatrix X0G(sDQ[1][tidz], Q1D, D1D);
   ConstDeviceMatrix X1B(sDQ[2][tidz], Q1D, D1D);
   ConstDeviceMatrix X1G(sDQ[3][tidz], Q1D, D1D);
   DeviceMatrix X0GB(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix X0BG(sQQ[1][tidz], Q1D, Q1D);
   DeviceMatrix X1GB(sQQ[2][tidz], Q1D, Q1D);
   DeviceMatrix X1BG(sQQ[3][tidz], Q1D, Q1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double By = B(dy,qy);
            const double Gy = G(dy,qy);
            u[0] += X0G(qx,dy) * By;
            v[0] += X0B(qx,dy) * Gy;
            u[1] += X1G(qx,dy) * By;
            v[1] += X1B(qx,dy) * Gy;
         }
         X0GB(qx,qy) = u[0];
         X0BG(qx,qy) = v[0];
         X1GB(qx,qy) = u[1];
         X1BG(qx,qy) = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Transposed gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const double (&sBG)[2][MQ1*MD1],
                                const double (&GQ)[4][NBZ][MQ1*MQ1],
                                double (&GD)[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceMatrix QQx0(GQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQx1(GQ[1][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQy0(GQ[2][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQy1(GQ[3][tidz], Q1D, Q1D);
   DeviceMatrix DQxB(GD[0][tidz], Q1D, D1D);
   DeviceMatrix DQxG(GD[1][tidz], Q1D, D1D);
   DeviceMatrix DQyB(GD[2][tidz], Q1D, D1D);
   DeviceMatrix DQyG(GD[3][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u[0] += Gt(qx,dx) * QQx0(qx,qy);
            u[1] += Gt(qx,dx) * QQy0(qx,qy);
            v[0] += Bt(qx,dx) * QQx1(qx,qy);
            v[1] += Bt(qx,dx) * QQy1(qx,qy);
         }
         DQxB(qy,dx) = u[0];
         DQyB(qy,dx) = u[1];
         DQxG(qy,dx) = v[0];
         DQyG(qy,dx) = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Transposed gradient, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const double (&sBG)[2][MQ1*MD1],
                                const double (&GD)[4][NBZ][MD1*MQ1],
                                const DeviceTensor<4> &Y, // output
                                const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceMatrix DQxB(GD[0][tidz], Q1D, D1D);
   ConstDeviceMatrix DQxG(GD[1][tidz], Q1D, D1D);
   ConstDeviceMatrix DQyB(GD[2][tidz], Q1D, D1D);
   ConstDeviceMatrix DQyG(GD[3][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u[0] += DQxB(qy,dx) * Bt(qy,dy);
            u[1] += DQyB(qy,dx) * Bt(qy,dy);
            v[0] += DQxG(qy,dx) * Gt(qy,dy);
            v[1] += DQyG(qy,dx) * Gt(qy,dy);
         }
         Y(dx,dy,0,e) += u[0] + v[0];
         Y(dx,dy,1,e) += u[1] + v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace kernels::internal::grad

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_GRAD_2D_HPP
