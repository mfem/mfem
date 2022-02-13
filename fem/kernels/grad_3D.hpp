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

#ifndef MFEM_FEM_KERNELS_GRAD_3D_HPP
#define MFEM_FEM_KERNELS_GRAD_3D_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace grad
{

/// 3D Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void X(const int D1D, const int Q1D,
                               const double (*sBG)[MQ1*MD1],
                               const double (*sDDD)[MD1*MD1*MD1],
                               double (*sDDQ)[MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   ConstDeviceCube Xx(sDDD[0], D1D, D1D, D1D);
   ConstDeviceCube Xy(sDDD[1], D1D, D1D, D1D);
   ConstDeviceCube Xz(sDDD[2], D1D, D1D, D1D);
   DeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   DeviceCube XxG(sDDQ[1], Q1D, D1D, D1D);
   DeviceCube XyB(sDDQ[2], Q1D, D1D, D1D);
   DeviceCube XyG(sDDQ[3], Q1D, D1D, D1D);
   DeviceCube XzB(sDDQ[4], Q1D, D1D, D1D);
   DeviceCube XzG(sDDQ[5], Q1D, D1D, D1D);

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
MFEM_HOST_DEVICE inline void Y(const int D1D, const int Q1D,
                               const double (*sBG)[MQ1*MD1],
                               const double (*sDDQ)[MD1*MD1*MQ1],
                               double (*sDQQ)[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   ConstDeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   ConstDeviceCube XxG(sDDQ[1], Q1D, D1D, D1D);
   ConstDeviceCube XyB(sDDQ[2], Q1D, D1D, D1D);
   ConstDeviceCube XyG(sDDQ[3], Q1D, D1D, D1D);
   ConstDeviceCube XzB(sDDQ[4], Q1D, D1D, D1D);
   ConstDeviceCube XzG(sDDQ[5], Q1D, D1D, D1D);
   DeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   DeviceCube XxBG(sDQQ[1], Q1D, Q1D, D1D);
   DeviceCube XxGB(sDQQ[2], Q1D, Q1D, D1D);
   DeviceCube XyBB(sDQQ[3], Q1D, Q1D, D1D);
   DeviceCube XyBG(sDQQ[4], Q1D, Q1D, D1D);
   DeviceCube XyGB(sDQQ[5], Q1D, Q1D, D1D);
   DeviceCube XzBB(sDQQ[6], Q1D, Q1D, D1D);
   DeviceCube XzBG(sDQQ[7], Q1D, Q1D, D1D);
   DeviceCube XzGB(sDQQ[8], Q1D, Q1D, D1D);

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
MFEM_HOST_DEVICE inline void Z(const int D1D, const int Q1D,
                               const double (*sBG)[MQ1*MD1],
                               const double (*sDQQ)[MD1*MQ1*MQ1],
                               double (*sQQQ)[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sBG[0], D1D, Q1D);
   ConstDeviceMatrix G(sBG[1], D1D, Q1D);
   ConstDeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   ConstDeviceCube XxBG(sDQQ[1], Q1D, Q1D, D1D);
   ConstDeviceCube XxGB(sDQQ[2], Q1D, Q1D, D1D);
   ConstDeviceCube XyBB(sDQQ[3], Q1D, Q1D, D1D);
   ConstDeviceCube XyBG(sDQQ[4], Q1D, Q1D, D1D);
   ConstDeviceCube XyGB(sDQQ[5], Q1D, Q1D, D1D);
   ConstDeviceCube XzBB(sDQQ[6], Q1D, Q1D, D1D);
   ConstDeviceCube XzBG(sDQQ[7], Q1D, Q1D, D1D);
   ConstDeviceCube XzGB(sDQQ[8], Q1D, Q1D, D1D);
   DeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   DeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   DeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   DeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   DeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   DeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   DeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);

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

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Zt(const int D1D, const int Q1D,
                                const double (&sBG)[2][MQ1*MD1],
                                const double (&sQQQ)[9][MQ1*MQ1*MQ1],
                                double (&sDQQ)[9][MD1*MQ1*MQ1])
{

   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   ConstDeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   ConstDeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);
   DeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   DeviceCube XxBG(sDQQ[1], Q1D, Q1D, D1D);
   DeviceCube XxGB(sDQQ[2], Q1D, Q1D, D1D);
   DeviceCube XyBB(sDQQ[3], Q1D, Q1D, D1D);
   DeviceCube XyBG(sDQQ[4], Q1D, Q1D, D1D);
   DeviceCube XyGB(sDQQ[5], Q1D, Q1D, D1D);
   DeviceCube XzBB(sDQQ[6], Q1D, Q1D, D1D);
   DeviceCube XzBG(sDQQ[7], Q1D, Q1D, D1D);
   DeviceCube XzGB(sDQQ[8], Q1D, Q1D, D1D);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            double w[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(qx,dx);
               const double Gtx = Gt(qx,dx);

               u[0] += XxBBG(qx,qy,qz) * Gtx;
               v[0] += XxBGB(qx,qy,qz) * Btx;
               w[0] += XxGBB(qx,qy,qz) * Btx;

               u[1] += XyBBG(qx,qy,qz) * Gtx;
               v[1] += XyBGB(qx,qy,qz) * Btx;
               w[1] += XyGBB(qx,qy,qz) * Btx;

               u[2] += XzBBG(qx,qy,qz) * Gtx;
               v[2] += XzBGB(qx,qy,qz) * Btx;
               w[2] += XzGBB(qx,qy,qz) * Btx;
            }
            XxBB(qz,qy,dx) = u[0];
            XxBG(qz,qy,dx) = v[0];
            XxGB(qz,qy,dx) = w[0];

            XyBB(qz,qy,dx) = u[1];
            XyBG(qz,qy,dx) = v[1];
            XyGB(qz,qy,dx) = w[1];

            XzBB(qz,qy,dx) = u[2];
            XzBG(qz,qy,dx) = v[2];
            XzGB(qz,qy,dx) = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Gradient, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Yt(const int D1D, const int Q1D,
                                const double (&sBG)[2][MQ1*MD1],
                                const double (&sDQQ)[9][MD1*MQ1*MQ1],
                                double (&sDDQ)[9][MD1*MD1*MQ1])
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceCube XxBB(sDQQ[0], Q1D, Q1D, D1D);
   ConstDeviceCube XxBG(sDQQ[1], Q1D, Q1D, D1D);
   ConstDeviceCube XxGB(sDQQ[2], Q1D, Q1D, D1D);
   ConstDeviceCube XyBB(sDQQ[3], Q1D, Q1D, D1D);
   ConstDeviceCube XyBG(sDQQ[4], Q1D, Q1D, D1D);
   ConstDeviceCube XyGB(sDQQ[5], Q1D, Q1D, D1D);
   ConstDeviceCube XzBB(sDQQ[6], Q1D, Q1D, D1D);
   ConstDeviceCube XzBG(sDQQ[7], Q1D, Q1D, D1D);
   ConstDeviceCube XzGB(sDQQ[8], Q1D, Q1D, D1D);
   DeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   DeviceCube XxG(sDDQ[1], Q1D, D1D, D1D);
   DeviceCube XyB(sDDQ[2], Q1D, D1D, D1D);
   DeviceCube XyG(sDDQ[3], Q1D, D1D, D1D);
   DeviceCube XzB(sDDQ[4], Q1D, D1D, D1D);
   DeviceCube XzG(sDDQ[5], Q1D, D1D, D1D);
   DeviceCube XxC(sDDQ[6], Q1D, D1D, D1D);
   DeviceCube XyC(sDDQ[7], Q1D, D1D, D1D);
   DeviceCube XzC(sDDQ[8], Q1D, D1D, D1D);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            double w[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt(qy,dy);
               const double Gty = Gt(qy,dy);

               u[0] += XxBB(qz,qy,dx) * Bty;
               v[0] += XxBG(qz,qy,dx) * Gty;
               w[0] += XxGB(qz,qy,dx) * Bty;

               u[1] += XyBB(qz,qy,dx) * Bty;
               v[1] += XyBG(qz,qy,dx) * Gty;
               w[1] += XyGB(qz,qy,dx) * Bty;

               u[2] += XzBB(qz,qy,dx) * Bty;
               v[2] += XzBG(qz,qy,dx) * Gty;
               w[2] += XzGB(qz,qy,dx) * Bty;

            }
            XxB(qz,dy,dx) = u[0];
            XxC(qz,dy,dx) = v[0];
            XxG(qz,dy,dx) = w[0];

            XyB(qz,dy,dx) = u[1];
            XyC(qz,dy,dx) = v[1];
            XyG(qz,dy,dx) = w[1];

            XzB(qz,dy,dx) = u[2];
            XzC(qz,dy,dx) = v[2];
            XzG(qz,dy,dx) = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Gradient, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Xt(const int D1D, const int Q1D,
                                const double (&sBG)[2][MQ1*MD1],
                                const double (&sDDQ)[9][MD1*MD1*MQ1],
                                const DeviceTensor<5> &Y, // output
                                const int e)
{
   ConstDeviceMatrix Bt(sBG[0], Q1D, D1D);
   ConstDeviceMatrix Gt(sBG[1], Q1D, D1D);
   ConstDeviceCube XxB(sDDQ[0], Q1D, D1D, D1D);
   ConstDeviceCube XxG(sDDQ[1], Q1D, D1D, D1D);
   ConstDeviceCube XyB(sDDQ[2], Q1D, D1D, D1D);
   ConstDeviceCube XyG(sDDQ[3], Q1D, D1D, D1D);
   ConstDeviceCube XzB(sDDQ[4], Q1D, D1D, D1D);
   ConstDeviceCube XzG(sDDQ[5], Q1D, D1D, D1D);
   ConstDeviceCube XxC(sDDQ[6], Q1D, D1D, D1D);
   ConstDeviceCube XyC(sDDQ[7], Q1D, D1D, D1D);
   ConstDeviceCube XzC(sDDQ[8], Q1D, D1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            double v[3] = {0.0, 0.0, 0.0};
            double w[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz,dz);
               const double Gtz = Gt(qz,dz);

               u[0] += XxB(qz,dy,dx) * Btz;
               v[0] += XxC(qz,dy,dx) * Btz;
               w[0] += XxG(qz,dy,dx) * Gtz;

               u[1] += XyB(qz,dy,dx) * Btz;
               v[1] += XyC(qz,dy,dx)* Btz;
               w[1] += XyG(qz,dy,dx) * Gtz;

               u[2] += XzB(qz,dy,dx) * Btz;
               v[2] += XzC(qz,dy,dx) * Btz;
               w[2] += XzG(qz,dy,dx) * Gtz;
            }
            Y(dx,dy,dz,0,e) += u[0] + v[0] + w[0];
            Y(dx,dy,dz,1,e) += u[1] + v[1] + w[1];
            Y(dx,dy,dz,2,e) += u[2] + v[2] + w[2];
         }
      }
   }
}

} // namespace kernels::internal::grad

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_GRAD_3D_HPP
