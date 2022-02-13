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

#ifndef MFEM_FEM_KERNELS_EVAL_3D_HPP
#define MFEM_FEM_KERNELS_EVAL_3D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace eval
{

/// 3D Scalar Evaluation, 1/3
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               ConstDeviceMatrix &B,
                               const DeviceCube &DDD,
                               DeviceCube &DDQ)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B(dx,qx);
               u += Bx * DDD(dx,dy,dz);
            }
            DDQ(dz,dy,qx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 1/3 - bis
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDDD)[MD1*MD1*MD1],
                               double (&sDDQ)[MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DDD(sDDD, D1D, D1D, D1D);
   DeviceCube DDQ(sDDQ, Q1D, D1D, D1D);
   X(D1D,Q1D,B,DDD,DDQ);
}

/// 3D Scalar Evaluation, 2/3
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               ConstDeviceMatrix &B,
                               const DeviceCube &DDQ,
                               DeviceCube &DQQ)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B(dy,qy);
               u += DDQ(dz,dy,qx) * By;
            }
            DQQ(dz,qy,qx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 2/3 - bis
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDDQ)[MD1*MD1*MQ1],
                               double (&sDQQ)[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DDQ(sDDQ, Q1D, D1D, D1D);
   DeviceCube DQQ(sDQQ, Q1D, Q1D, D1D);
   Y(D1D,Q1D,B,DDQ,DQQ);
}

/// 3D Scalar Evaluation, 3/3
MFEM_HOST_DEVICE inline void Z(const int D1D, const int Q1D,
                               ConstDeviceMatrix &B,
                               const DeviceCube &DQQ,
                               DeviceCube &QQQ)
{
   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B(dz,qz);
               u += DQQ(dz,qy,qx) * Bz;
            }
            QQQ(qz,qy,qx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE inline void Mult(const int D1D, const int Q1D,
                                  ConstDeviceMatrix &B,
                                  DeviceCube &DDD,
                                  DeviceCube &DDQ,
                                  DeviceCube &DQQ,
                                  DeviceCube &QQQ)
{
   eval::X(D1D,Q1D,B,DDD,DDQ);
   eval::Y(D1D,Q1D,B,DDQ,DQQ);
   eval::Z(D1D,Q1D,B,DQQ,QQQ);
}

/// 3D Scalar Evaluation, 3/3 - bis
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Z(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDQQ)[MD1*MQ1*MQ1],
                               double (&sQQQ)[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DQQ(sDQQ, Q1D, Q1D, D1D);
   DeviceCube QQQ(sQQQ, Q1D, Q1D, Q1D);
   Z(D1D,Q1D,B,DQQ,QQQ);
}

/// 3D Vector Evaluation, 1/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDDD)[3][MD1*MD1*MD1],
                               double (&sDDQ)[3][MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceCube Xx(sDDD[0], D1D, D1D, D1D);
   ConstDeviceCube Xy(sDDD[1], D1D, D1D, D1D);
   ConstDeviceCube Xz(sDDD[2], D1D, D1D, D1D);
   DeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   DeviceCube XyB(sDDQ[1], Q1D, D1D, D1D);
   DeviceCube XzB(sDDQ[2], Q1D, D1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B(dx,qx);
               u[0] += Bx * Xx(dx,dy,dz);
               u[1] += Bx * Xy(dx,dy,dz);
               u[2] += Bx * Xz(dx,dy,dz);
            }
            XxB(qx,dy,dz) = u[0];
            XyB(qx,dy,dz) = u[1];
            XzB(qx,dy,dz) = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Vector Evaluation, 2/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDDQ)[3][MD1*MD1*MQ1],
                               double (&sDQQ)[3][MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   ConstDeviceCube XyB(sDDQ[1], Q1D, D1D, D1D);
   ConstDeviceCube XzB(sDDQ[2], Q1D, D1D, D1D);
   DeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   DeviceCube XyBB(sDQQ[1], Q1D, Q1D, D1D);
   DeviceCube XzBB(sDQQ[2], Q1D, Q1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B(dy,qy);
               u[0] += XxB(qx,dy,dz) * By;
               u[1] += XyB(qx,dy,dz) * By;
               u[2] += XzB(qx,dy,dz) * By;
            }
            XxBB(qx,qy,dz) = u[0];
            XyBB(qx,qy,dz) = u[1];
            XzBB(qx,qy,dz) = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Vector Evaluation, 3/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Z(const int D1D, const int Q1D,
                               const double (&sB)[MQ1*MD1],
                               const double (&sDQQ)[3][MD1*MQ1*MQ1],
                               double (&sQQQ)[3][MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   ConstDeviceCube XyBB(sDQQ[1], Q1D, Q1D, D1D);
   ConstDeviceCube XzBB(sDQQ[2], Q1D, Q1D, D1D);
   DeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B(dz,qz);
               u[0] += XxBB(qx,qy,dz) * Bz;
               u[1] += XyBB(qx,qy,dz) * Bz;
               u[2] += XzBB(qx,qy,dz) * Bz;
            }
            XxBBB(qx,qy,qz) = u[0];
            XyBBB(qx,qy,qz) = u[1];
            XzBBB(qx,qy,qz) = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Vector Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const double (&sB)[MQ1*MD1],
                                const double (&sQQQ)[3][MQ1*MQ1*MQ1],
                                double (&sDQQ)[3][MD1*MQ1*MQ1])
{
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);
   DeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   DeviceCube XyBB(sDQQ[1], Q1D, Q1D, D1D);
   DeviceCube XzBB(sDQQ[2], Q1D, Q1D, D1D);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(qx,dx);
               u[0] += XxBBB(qx,qy,qz) * Btx;
               u[1] += XyBBB(qx,qy,qz) * Btx;
               u[2] += XzBBB(qx,qy,qz) * Btx;
            }
            XxBB(qz,qy,dx) = u[0];
            XyBB(qz,qy,dx) = u[1];
            XzBB(qz,qy,dx) = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Vector Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const double (&sB)[MQ1*MD1],
                                const double (&sDQQ)[3][MD1*MQ1*MQ1],
                                double (&sDDQ)[3][MD1*MD1*MQ1])
{
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   ConstDeviceCube XyBB(sDQQ[1], Q1D, Q1D, D1D);
   ConstDeviceCube XzBB(sDQQ[2], Q1D, Q1D, D1D);
   DeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   DeviceCube XyB(sDDQ[1], Q1D, D1D, D1D);
   DeviceCube XzB(sDDQ[2], Q1D, D1D, D1D);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt(qy,dy);
               u[0] += XxBB(qz,qy,dx) * Bty;
               u[1] += XyBB(qz,qy,dx) * Bty;
               u[2] += XzBB(qz,qy,dx) * Bty;

            }
            XxB(qz,dy,dx) = u[0];
            XyB(qz,dy,dx) = u[1];
            XzB(qz,dy,dx)= u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Vector Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Zt(const int D1D, const int Q1D,
                                const double (&sB)[MQ1*MD1],
                                const double (&sDDQ)[3][MD1*MD1*MQ1],
                                const DeviceTensor<5> &Y, // output
                                const int e)
{
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   ConstDeviceCube XyB(sDDQ[1], Q1D, D1D, D1D);
   ConstDeviceCube XzB(sDDQ[2], Q1D, D1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz,dz);
               u[0] += XxB(qz,dy,dx) * Btz;
               u[1] += XyB(qz,dy,dx) * Btz;
               u[2] += XzB(qz,dy,dx) * Btz;
            }
            Y(dx,dy,dz,0,e) += u[0];
            Y(dx,dy,dz,1,e) += u[1];
            Y(dx,dy,dz,2,e) += u[2];
         }
      }
   }
}

} // namespace kernels::internal::eval

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_EVAL_3D_HPP
