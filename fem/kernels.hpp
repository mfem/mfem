// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEM_KERNELS_HPP
#define MFEM_FEM_KERNELS_HPP

#include "../config/config.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

namespace kernels
{

/// Load B1d & G1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBG(const int D1D, const int Q1D,
                                    const ConstDeviceMatrix b,
                                    const ConstDeviceMatrix g,
                                    double sBG[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sBG[0], MD1, MQ1);
   DeviceMatrix G(sBG[1], MD1, MQ1);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B(d,q) = b(q,d);
            G(d,q) = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into shared memory, with comp
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<4, const double> x,
                                   double sm[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X(sm[tidz], MD1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X(dx,dy) = x(dx,dy,c,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input vector into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const double> X,
                                   double sX[2][NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0(sX[0][tidz], MD1, MD1);
   DeviceMatrix X1(sX[1][tidz], MD1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X0(dx,dy) = X(dx,dy,0,e);
         X1(dx,dy) = X(dx,dy,1,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sX[2][NBZ][MD1*MD1],
                                   double sDQ[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   ConstDeviceMatrix X0(sX[0][tidz], MD1, MD1);
   ConstDeviceMatrix X1(sX[1][tidz], MD1, MD1);
   DeviceMatrix X0B(sDQ[0][tidz], MQ1, MD1);
   DeviceMatrix X0G(sDQ[1][tidz], MQ1, MD1);
   DeviceMatrix X1B(sDQ[2][tidz], MQ1, MD1);
   DeviceMatrix X1G(sDQ[3][tidz], MQ1, MD1);

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
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQ[4][NBZ][MD1*MQ1],
                                   double sQQ[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   ConstDeviceMatrix X0B(sDQ[0][tidz], MQ1, MD1);
   ConstDeviceMatrix X0G(sDQ[1][tidz], MQ1, MD1);
   ConstDeviceMatrix X1B(sDQ[2][tidz], MQ1, MD1);
   ConstDeviceMatrix X1G(sDQ[3][tidz], MQ1, MD1);
   DeviceMatrix X0GB(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix X0BG(sQQ[1][tidz], MQ1, MQ1);
   DeviceMatrix X1GB(sQQ[2][tidz], MQ1, MQ1);
   DeviceMatrix X1BG(sQQ[3][tidz], MQ1, MQ1);

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

/// Pull 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullGrad(const int qx, const int qy,
                                      const double sQQ[4][NBZ][MQ1*MQ1],
                                      double *Jpr)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix X0GB(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix X0BG(sQQ[1][tidz], MQ1, MQ1);
   ConstDeviceMatrix X1GB(sQQ[2][tidz], MQ1, MQ1);
   ConstDeviceMatrix X1BG(sQQ[3][tidz], MQ1, MQ1);

   Jpr[0] = X0GB(qx,qy);
   Jpr[1] = X1GB(qx,qy);
   Jpr[2] = X0BG(qx,qy);
   Jpr[3] = X1BG(qx,qy);
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const double> X,
                                   double sm[3][MD1*MD1*MD1])
{
   DeviceCube Xx(sm[0], MD1, MD1, MD1);
   DeviceCube Xy(sm[1], MD1, MD1, MD1);
   DeviceCube Xz(sm[2], MD1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx(dx,dy,dz) = X(dx,dy,dz,0,e);
            Xy(dx,dy,dz) = X(dx,dy,dz,1,e);
            Xz(dx,dy,dz) = X(dx,dy,dz,2,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDD[3][MD1*MD1*MD1],
                                   double sDDQ[9][MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   ConstDeviceCube Xx(sDDD[0], MD1, MD1, MD1);
   ConstDeviceCube Xy(sDDD[1], MD1, MD1, MD1);
   ConstDeviceCube Xz(sDDD[2], MD1, MD1, MD1);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   DeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   DeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   DeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   DeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double xx = Xx(dx,dy,dz);
               const double xy = Xy(dx,dy,dz);
               const double xz = Xz(dx,dy,dz);
               const double Bx = B(dx,qx);
               const double Gx = G(dx,qx);
               u[0] += Bx * xx;
               u[1] += Bx * xy;
               u[2] += Bx * xz;

               v[0] += Gx * xx;
               v[1] += Gx * xy;
               v[2] += Gx * xz;
            }
            XxB(qx,dy,dz) = u[0];
            XyB(qx,dy,dz) = u[1];
            XzB(qx,dy,dz) = u[2];

            XxG(qx,dy,dz) = v[0];
            XyG(qx,dy,dz) = v[1];
            XzG(qx,dy,dz) = v[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Gradient, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDQ[9][MD1*MD1*MQ1],
                                   double sDQQ[9][MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   ConstDeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   ConstDeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   ConstDeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   ConstDeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   ConstDeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   DeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   DeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   DeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   DeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   DeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   DeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            double w[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B(dy,qy);
               const double Gy = G(dy,qy);

               u[0] += XxB(qx,dy,dz) * By;
               u[1] += XyB(qx,dy,dz) * By;
               u[2] += XzB(qx,dy,dz) * By;

               v[0] += XxG(qx,dy,dz) * By;
               v[1] += XyG(qx,dy,dz) * By;
               v[2] += XzG(qx,dy,dz) * By;

               w[0] += XxB(qx,dy,dz) * Gy;
               w[1] += XyB(qx,dy,dz) * Gy;
               w[2] += XzB(qx,dy,dz) * Gy;
            }
            XxBB(qx,qy,dz) = u[0];
            XyBB(qx,qy,dz) = u[1];
            XzBB(qx,qy,dz) = u[2];

            XxBG(qx,qy,dz) = v[0];
            XyBG(qx,qy,dz) = v[1];
            XzBG(qx,qy,dz) = v[2];

            XxGB(qx,qy,dz) = w[0];
            XyGB(qx,qy,dz) = w[1];
            XzGB(qx,qy,dz) = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Gradient, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZ(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQQ[9][MD1*MQ1*MQ1],
                                   double sQQQ[9][MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   ConstDeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   ConstDeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   ConstDeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   ConstDeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   ConstDeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   ConstDeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   ConstDeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   ConstDeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);
   DeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   DeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   DeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   DeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   DeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   DeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   DeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   DeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            double w[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B(dz,qz);
               const double Gz = G(dz,qz);

               u[0] += XxBG(qx,qy,dz) * Bz;
               u[1] += XyBG(qx,qy,dz) * Bz;
               u[2] += XzBG(qx,qy,dz) * Bz;

               v[0] += XxGB(qx,qy,dz) * Bz;
               v[1] += XyGB(qx,qy,dz) * Bz;
               v[2] += XzGB(qx,qy,dz) * Bz;

               w[0] += XxBB(qx,qy,dz) * Gz;
               w[1] += XyBB(qx,qy,dz) * Gz;
               w[2] += XzBB(qx,qy,dz) * Gz;
            }
            XxBBG(qx,qy,qz) = u[0];
            XyBBG(qx,qy,qz) = u[1];
            XzBBG(qx,qy,qz) = u[2];

            XxBGB(qx,qy,qz) = v[0];
            XyBGB(qx,qy,qz) = v[1];
            XzBGB(qx,qy,qz) = v[2];

            XxGBB(qx,qy,qz)= w[0];
            XyGBB(qx,qy,qz) = w[1];
            XzGBB(qx,qy,qz) = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PullGrad(const int x, const int y, const int z,
                                      const double sQQQ[9][MQ1*MQ1*MQ1],
                                      double *Jpr)
{
   ConstDeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   ConstDeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   ConstDeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   ConstDeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   ConstDeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

   Jpr[0] = XxBBG(x,y,z);
   Jpr[3] = XxBGB(x,y,z);
   Jpr[6] = XxGBB(x,y,z);
   Jpr[1] = XyBBG(x,y,z);
   Jpr[4] = XyBGB(x,y,z);
   Jpr[7] = XyGBB(x,y,z);
   Jpr[2] = XzBBG(x,y,z);
   Jpr[5] = XzBGB(x,y,z);
   Jpr[8] = XzGBB(x,y,z);
}

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
