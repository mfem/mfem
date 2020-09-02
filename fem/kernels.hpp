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

/// Load B1d matrice into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadB(const int D1D, const int Q1D,
                                   const ConstDeviceMatrix b,
                                   double sB[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sB, MD1, MQ1);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B(d,q) = b(q,d);
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

/// 2D Scalar Evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sX[NBZ][MD1*MD1],
                                   double sDQ[NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceMatrix X(sX[tidz], MD1, MD1);
   DeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            u += B(dx,qx) * X(dx,dy);
         }
         DQ(qx,dy) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Evaluation, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDQ[NBZ][MD1*MQ1],
                                   double sQQ[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceMatrix DQ(sDQ[tidz], MQ1, MD1);
   DeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            u += DQ(qx,dy) * B(dy,qy);
         }
         QQ(qx,qy) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 2D Scalar Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline double PullEval(const int qx, const int qy,
                                        const double sQQ[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   return QQ(qx,qy);
}

/// Load 3D scalar input vector into shared memory, with comp
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<5, const double> x,
                                   double sm[MD1*MD1*MD1])
{
   DeviceCube X(sm, MD1, MD1, MD1);

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

/// 3D Scalar Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDDD[MD1*MD1*MD1],
                                   double sDDQ[MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube Xx(sDDD, MD1, MD1, MD1);
   DeviceCube XxB(sDDQ, MQ1, MD1, MD1);

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
               u += Bx * Xx(dx,dy,dz);
            }
            XxB(qx,dy,dz) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDDQ[MD1*MD1*MQ1],
                                   double sDQQ[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube XxB(sDDQ, MQ1, MD1, MD1);
   DeviceCube XxBB(sDQQ, MQ1, MQ1, MD1);

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
               u += XxB(qx,dy,dz) * By;
            }
            XxBB(qx,qy,dz) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDQQ[MD1*MQ1*MQ1],
                                   double sQQQ[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube XxBB(sDQQ, MQ1, MQ1, MD1);
   DeviceCube XxBBB(sQQQ, MQ1, MQ1, MQ1);

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
               u += XxBB(qx,qy,dz) * Bz;
            }
            XxBBB(qx,qy,qz) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 3D Scalar Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline double PullEval(const int x, const int y, const int z,
                                        const double sQQQ[MQ1*MQ1*MQ1])
{
   ConstDeviceCube XxBBB(sQQQ, MQ1, MQ1, MQ1);
   return XxBBB(x,y,z);
}

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
