// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

// Experimental helper functions for MFEM_FORALL FEM kernels
// For the 2D functions, NBZ should be tied to '1' for now
namespace internal
{

/// Load B1d & G1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBG(const int D1D, const int Q1D,
                                    const ConstDeviceMatrix &b,
                                    const ConstDeviceMatrix &g,
                                    double (&sBG)[2][MQ1*MD1])
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
                                   const DeviceTensor<4, const double> &x,
                                   double (&sm)[NBZ][MD1*MD1])
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
                                   const DeviceTensor<4, const double> &X,
                                   double (&sX)[2][NBZ][MD1*MD1])
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
                                   const double (&sBG)[2][MQ1*MD1],
                                   const double (&sX)[2][NBZ][MD1*MD1],
                                   double (&sDQ)[4][NBZ][MD1*MQ1])
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
                                   const double (&sBG)[2][MQ1*MD1],
                                   const double (&sDQ)[4][NBZ][MD1*MQ1],
                                   double (&sQQ)[4][NBZ][MQ1*MQ1])
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
                                      const double (&sQQ)[4][NBZ][MQ1*MQ1],
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

/// Load 3D scalar input vector into shared memory, with comp
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<5, const double> &x,
                                   DeviceTensor<3> &X)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X(dx,dy,dz) = x(dx,dy,dz,c,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const double> &X,
                                   double (*sm)[MD1*MD1*MD1])
{
   DeviceTensor<3,double> Xx(sm[0], MD1, MD1, MD1);
   DeviceTensor<3,double> Xy(sm[1], MD1, MD1, MD1);
   DeviceTensor<3,double> Xz(sm[2], MD1, MD1, MD1);

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
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDDD)[MD1*MD1*MD1],
                                   double (*sDDQ)[MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   DeviceTensor<3,const double> Xx(sDDD[0], MD1, MD1, MD1);
   DeviceTensor<3,const double> Xy(sDDD[1], MD1, MD1, MD1);
   DeviceTensor<3,const double> Xz(sDDD[2], MD1, MD1, MD1);
   DeviceTensor<3,double> XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,double> XxG(sDDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,double> XyB(sDDQ[2], MQ1, MD1, MD1);
   DeviceTensor<3,double> XyG(sDDQ[3], MQ1, MD1, MD1);
   DeviceTensor<3,double> XzB(sDDQ[4], MQ1, MD1, MD1);
   DeviceTensor<3,double> XzG(sDDQ[5], MQ1, MD1, MD1);

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
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDDQ)[MD1*MD1*MQ1],
                                   double (*sDQQ)[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   DeviceTensor<3,const double> XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,const double> XxG(sDDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,const double> XyB(sDDQ[2], MQ1, MD1, MD1);
   DeviceTensor<3,const double> XyG(sDDQ[3], MQ1, MD1, MD1);
   DeviceTensor<3,const double> XzB(sDDQ[4], MQ1, MD1, MD1);
   DeviceTensor<3,const double> XzG(sDDQ[5], MQ1, MD1, MD1);
   DeviceTensor<3,double> XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XxBG(sDQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XxGB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XyBB(sDQQ[3], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XyBG(sDQQ[4], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XyGB(sDQQ[5], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XzBB(sDQQ[6], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XzBG(sDQQ[7], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XzGB(sDQQ[8], MQ1, MQ1, MD1);

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
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDQQ)[MD1*MQ1*MQ1],
                                   double (*sQQQ)[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);
   DeviceTensor<3,const double> XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XxBG(sDQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XxGB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XyBB(sDQQ[3], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XyBG(sDQQ[4], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XyGB(sDQQ[5], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XzBB(sDQQ[6], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XzBG(sDQQ[7], MQ1, MQ1, MD1);
   DeviceTensor<3,const double> XzGB(sDQQ[8], MQ1, MQ1, MD1);
   DeviceTensor<3,double> XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   DeviceTensor<3,double> XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

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
                                      const double (*sQQQ)[MQ1*MQ1*MQ1],
                                      double *Jpr)
{
   DeviceTensor<3,const double> XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   DeviceTensor<3,const double> XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

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

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
