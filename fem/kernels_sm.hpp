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

#ifndef MFEM_FEM_KERNELS_SM_HPP
#define MFEM_FEM_KERNELS_SM_HPP

#include "kernels.hpp"
#include "../config/config.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace sm
{

///////////////////////////////////////////////////////////////////////////////
/// Load 3D scalar tensor into shared memory
template<int MDQ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &X,
                                   real_t (&sm)[MDQ*MDQ*MDQ])
{
   DeviceCube Xx(sm, D1D, D1D, D1D);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx(dx,dy,dz) = X(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D scalar input vector into shared memory, with comp & pointer
template<int MDQ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<5, const real_t> &x,
                                   real_t (&sm)[MDQ*MDQ*MDQ])
{
   DeviceCube X(sm, D1D, D1D, D1D);
   mfem::kernels::internal::LoadX(e, D1D, c, x, X);
}

/// Load 3D vector tensor into shared memory
template<int MDQ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const real_t> &X,
                                   real_t (*sm)[MDQ*MDQ*MDQ])
{
   DeviceCube Xx(sm[0], D1D, D1D, D1D);
   DeviceCube Xy(sm[1], D1D, D1D, D1D);
   DeviceCube Xz(sm[2], D1D, D1D, D1D);

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

///////////////////////////////////////////////////////////////////////////////
/// 3D Scalar Evaluation, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   real_t (&sm0)[MDQ*MDQ*MDQ],
                                   real_t (&sm1)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceCube DDD(sm0, D1D, D1D, D1D);
   DeviceCube DDQ(sm1, D1D, D1D, Q1D);
   mfem::kernels::internal::EvalX(D1D, Q1D, B, DDD, DDQ);
}

/// 3D Scalar Evaluation, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   real_t (&sm1)[MDQ*MDQ*MDQ],
                                   real_t (&sm0)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceCube DDQ(sm1, D1D, D1D, Q1D);
   DeviceCube DQQ(sm0, D1D, Q1D, Q1D);
   mfem::kernels::internal::EvalY(D1D, Q1D, B, DDQ, DQQ);
}

/// 3D Scalar Evaluation, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   real_t (&sm0)[MDQ*MDQ*MDQ],
                                   real_t (&sm1)[MDQ*MDQ*MDQ])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceCube DQQ(sm0, D1D, Q1D, Q1D);
   DeviceCube QQQ(sm1, Q1D, Q1D, Q1D);
   mfem::kernels::internal::EvalZ(D1D, Q1D, B, DQQ, QQQ);
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Evaluation, 1/3 (only B)
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDD)[3][MDQ*MDQ*MDQ],
                                   real_t (&sDDQ)[3][MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t Bx = B(dx,qx);
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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDQ)[3][MDQ*MDQ*MDQ],
                                   real_t (&sDQQ)[3][MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t By = B(dy,qy);
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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDQQ)[3][MDQ*MDQ*MDQ],
                                   real_t (&sQQQ)[3][MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const real_t Bz = B(dz,qz);
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

///////////////////////////////////////////////////////////////////////////////
/// Pull 3D Scalar Evaluation
template<int MDQ>
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int qx, const int qy, const int qz,
                                      const real_t (&sm)[MDQ*MDQ*MDQ],
                                      real_t &X)
{
   ConstDeviceCube QQQ(sm, Q1D, Q1D, Q1D);
   X = QQQ(qz, qy, qx);
}

/// Pull 3D Vector Evaluation
template<int MDQ>
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (&sm)[3][MDQ*MDQ*MDQ],
                                      real_t (&X)[3])
{
   ConstDeviceCube Xx(sm[0], Q1D, Q1D, Q1D);
   ConstDeviceCube Xy(sm[1], Q1D, Q1D, Q1D);
   ConstDeviceCube Xz(sm[2], Q1D, Q1D, Q1D);

   X[0] = Xx(x,y,z);
   X[1] = Xy(x,y,z);
   X[2] = Xz(x,y,z);
}

/// Push 3D Vector Evaluation
template<int MDQ>
MFEM_HOST_DEVICE inline void PushEval(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (&A)[3],
                                      real_t (&sm)[3][MDQ*MDQ*MDQ])
{
   DeviceCube Xx(sm[0], Q1D, Q1D, Q1D);
   DeviceCube Xy(sm[1], Q1D, Q1D, Q1D);
   DeviceCube Xz(sm[2], Q1D, Q1D, Q1D);

   Xx(x,y,z) = A[0];
   Xy(x,y,z) = A[1];
   Xz(x,y,z) = A[2];
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Transposed Vector Evaluation, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sQQQ)[3][MDQ*MDQ*MDQ],
                                    real_t (&sDQQ)[3][MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t Btx = Bt(qx,dx);
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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sDQQ)[3][MDQ*MDQ*MDQ],
                                    real_t (&sDDQ)[3][MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t Bty = Bt(qy,dy);
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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void EvalZt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sDDQ)[3][MDQ*MDQ*MDQ],
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
            real_t u[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t Btz = Bt(qz,dz);
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

///////////////////////////////////////////////////////////////////////////////
/// 3D Vector Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDD)[MDQ*MDQ*MDQ],
                                   real_t (*sDDQ)[MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t xx = Xx(dx,dy,dz);
               const real_t xy = Xy(dx,dy,dz);
               const real_t xz = Xz(dx,dy,dz);
               const real_t Bx = B(dx,qx);
               const real_t Gx = G(dx,qx);
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

/// 3D Vector Gradient, 2/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDDQ)[MDQ*MDQ*MDQ],
                                   real_t (*sDQQ)[MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            real_t w[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t By = B(dy,qy);
               const real_t Gy = G(dy,qy);

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

/// 3D Vector Gradient, 3/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZ(const int D1D, const int Q1D,
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                   real_t (*sQQQ)[MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            real_t w[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const real_t Bz = B(dz,qz);
               const real_t Gz = G(dz,qz);

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

///////////////////////////////////////////////////////////////////////////////
/// Pull 3D Gradient
template<int MDQ>
MFEM_HOST_DEVICE inline void PullGrad(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (*sQQQ)[MDQ*MDQ*MDQ],
                                      real_t *Jpr)
{
   ConstDeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   ConstDeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   ConstDeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);

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

/// Push 3D Gradient
template<int MDQ>
MFEM_HOST_DEVICE inline void PushGrad(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t *A,
                                      real_t (&sQQQ)[9][MDQ*MDQ*MDQ])
{
   DeviceCube XxBBG(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XxBGB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XxGBB(sQQQ[2], Q1D, Q1D, Q1D);
   DeviceCube XyBBG(sQQQ[3], Q1D, Q1D, Q1D);
   DeviceCube XyBGB(sQQQ[4], Q1D, Q1D, Q1D);
   DeviceCube XyGBB(sQQQ[5], Q1D, Q1D, Q1D);
   DeviceCube XzBBG(sQQQ[6], Q1D, Q1D, Q1D);
   DeviceCube XzBGB(sQQQ[7], Q1D, Q1D, Q1D);
   DeviceCube XzGBB(sQQQ[8], Q1D, Q1D, Q1D);

   XxBBG(x,y,z) = A[0];
   XxBGB(x,y,z) = A[1];
   XxGBB(x,y,z) = A[2];
   XyBBG(x,y,z) = A[3];
   XyBGB(x,y,z) = A[4];
   XyGBB(x,y,z) = A[5];
   XzBBG(x,y,z) = A[6];
   XzBGB(x,y,z) = A[7];
   XzGBB(x,y,z) = A[8];
}

///////////////////////////////////////////////////////////////////////////////
/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sQQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDQQ)[MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            real_t w[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t Btx = Bt(qx,dx);
               const real_t Gtx = Gt(qx,dx);

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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDQQ)[MDQ*MDQ*MDQ],
                                    real_t (*sDDQ)[MDQ*MDQ*MDQ])
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            real_t w[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t Bty = Bt(qy,dy);
               const real_t Gty = Gt(qy,dy);

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
template<int MD1, int MQ1, int MDQ = (MQ1 > MD1 ? MQ1 : MD1)>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (*sDDQ)[MDQ*MDQ*MDQ],
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
            real_t u[3] = {0.0, 0.0, 0.0};
            real_t v[3] = {0.0, 0.0, 0.0};
            real_t w[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t Btz = Bt(qz,dz);
               const real_t Gtz = Gt(qz,dz);

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

} // namespace sm

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_SM_HPP
