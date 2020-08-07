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

/// Load Bt1d & Gt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBGt(const int D1D, const int Q1D,
                                     const ConstDeviceMatrix b,
                                     const ConstDeviceMatrix g,
                                     double sBG[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sBG[0], MQ1, MD1);
   DeviceMatrix Gt(sBG[1], MQ1, MD1);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt(q,d) = b(q,d);
            Gt(q,d) = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadS(const int e, const int D1D,
                                   const DeviceTensor<3, const double> x,
                                   double sX[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X(sX[tidz], MD1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X(dx,dy) = x(dx,dy,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input scalar into shared memory, with comp
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadXS(const int e, const int D1D, const int c,
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
MFEM_HOST_DEVICE inline void EvalXS(const int D1D, const int Q1D,
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
MFEM_HOST_DEVICE inline void EvalYS(const int D1D, const int Q1D,
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
MFEM_HOST_DEVICE inline void PullEvalS(const int qx, const int qy,
                                       const double sQQ[NBZ][MQ1*MQ1],
                                       double *P)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   P[0] = QQ(qx,qy);
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

/// 2D Evaluation, 1/2 (only B)
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sX[2][NBZ][MD1*MD1],
                                   double sDQ[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceMatrix X0(sX[0][tidz], MD1, MD1);
   ConstDeviceMatrix X1(sX[1][tidz], MD1, MD1);
   DeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   DeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);

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
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDQ[2][NBZ][MD1*MQ1],
                                   double sQQ[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);
   DeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

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

/// 2D Evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sX[2][NBZ][MD1*MD1],
                                   double sDQ[2][NBZ][MD1*MQ1])
{
   EvalX<MD1,MQ1,NBZ>(D1D,Q1D,sBG[0],sX,sDQ);
   /*const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix X0(sX[0][tidz], MD1, MD1);
   ConstDeviceMatrix X1(sX[1][tidz], MD1, MD1);
   DeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   DeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double xx = X0(dx,dy);
            const double xy = X1(dx,dy);
            u[0] += B(dx,qx)* xx;
            u[1] += B(dx,qx) * xy;
         }
         DQ0(qx,dy) = u[0];
         DQ1(qx,dy) = u[1];
      }
   }
   MFEM_SYNC_THREAD;*/
}

/// 2D Evaluation, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQ[2][NBZ][MD1*MQ1],
                                   double sQQ[2][NBZ][MQ1*MQ1])
{
   EvalY<MD1,MQ1,NBZ>(D1D,Q1D,sBG,sDQ,sQQ);
   /*const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);
   DeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

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
   MFEM_SYNC_THREAD;*/
}

/// Pull 2D Scalar Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEvalXYS(const int qx, const int qy,
                                         const double sQQ[NBZ][MQ1*MQ1],
                                         double &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ0(sQQ[tidz], MQ1, MQ1);

   P = QQ0(qx,qy);
}

/// Pull 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEvalXY(const int qx, const int qy,
                                        const double sQQ[2][NBZ][MQ1*MQ1],
                                        double *P)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

   P[0] = QQ0(qx,qy);
   P[1] = QQ1(qx,qy);
}

/// Push 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushEvalXY(const int qx, const int qy,
                                        const double *P,
                                        double sQQ[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

   QQ0(qx,qy) = P[0];
   QQ1(qx,qy) = P[1];
}

/// 2D Transposed evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sQQ[2][NBZ][MQ1*MQ1],
                                    double sDQ[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);
   DeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   DeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);

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
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sDQ[2][NBZ][MD1*MQ1],
                                    DeviceTensor<4, double> Y, const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], MQ1, MD1);

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
MFEM_HOST_DEVICE inline void PullGradXY(const int qx, const int qy,
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

/// Push 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushGradXY(const int qx, const int qy,
                                        const double *A,
                                        double sQQ[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0GB(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix X0BG(sQQ[1][tidz], MQ1, MQ1);
   DeviceMatrix X1GB(sQQ[2][tidz], MQ1, MQ1);
   DeviceMatrix X1BG(sQQ[3][tidz], MQ1, MQ1);

   X0GB(qx,qy) = A[0];
   X1GB(qx,qy) = A[2];
   X0BG(qx,qy) = A[1];
   X1BG(qx,qy) = A[3];
}

/// 2D Transposed gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double GQ[4][NBZ][MQ1*MQ1],
                                    double GD[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);

   ConstDeviceMatrix QQx0(GQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQx1(GQ[1][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQy0(GQ[2][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQy1(GQ[3][tidz], MQ1, MQ1);

   DeviceMatrix DQxB(GD[0][tidz], MQ1, MD1);
   DeviceMatrix DQxG(GD[1][tidz], MQ1, MD1);
   DeviceMatrix DQyB(GD[2][tidz], MQ1, MD1);
   DeviceMatrix DQyG(GD[3][tidz], MQ1, MD1);

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
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double GD[4][NBZ][MD1*MQ1],
                                    mfem::DeviceTensor<4, double> Y,
                                    const int e)
{
   const int tidz = MFEM_THREAD_ID(z);

   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);

   ConstDeviceMatrix DQxB(GD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQxG(GD[1][tidz], MQ1, MD1);
   ConstDeviceMatrix DQyB(GD[2][tidz], MQ1, MD1);
   ConstDeviceMatrix DQyG(GD[3][tidz], MQ1, MD1);

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

/// Load 3D scalar input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadXS(const int e, const int D1D, const int c,
                                    const DeviceTensor<5, const double> x,
                                    double sm[MD1*MD1*MD1])
{
   //double (*X)[MD1][MD1] = (double (*)[MD1][MD1])(sm);
   DeviceCube X(sm, MD1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X(dx,dy,dz)/*[dz][dy][dx]*/ = x(dx,dy,dz,c,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXS(const int D1D, const int Q1D,
                                    const double sB[MQ1*MD1],
                                    const double sDDD[MD1*MD1*MD1],
                                    double sDDQ[MD1*MD1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   //double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD);
   ConstDeviceCube Xx(sDDD, MD1, MD1, MD1);
   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ);
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
               u += Bx * Xx(dx,dy,dz);//[dz][dy][dx];
            }
            XxB(qx,dy,dz)/*[dz][dy][qx]*/ = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalYS(const int D1D, const int Q1D,
                                    const double sB[MQ1*MD1],
                                    const double sDDQ[MD1*MD1*MQ1],
                                    double sDQQ[MD1*MQ1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ);
   ConstDeviceCube XxB(sDDQ, MQ1, MD1, MD1);
   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ);
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
               const double By = B(dy,qy);//[qy][dy];
               u += XxB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
            }
            XxBB(qx,qy,dz)/*[dz][qy][qx]*/ = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZS(const int D1D, const int Q1D,
                                    const double sB[MQ1*MD1],
                                    const double sDQQ[MD1*MQ1*MQ1],
                                    double sQQQ[MQ1*MQ1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ);
   ConstDeviceCube XxBB(sDQQ, MQ1, MQ1, MD1);
   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ);
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
               const double Bz = B(dz,qz);//[qz][dz];
               u += XxBB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
            }
            XxBBB(qx,qy,qz)/*[qz][qy][qx]*/ = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 3D Scalar Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PullEvalXYZS(const int x, const int y, const int z,
                                          const double sQQQ[MQ1*MQ1*MQ1],
                                          double &X)
{
   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ);
   ConstDeviceCube XxBBB(sQQQ, MQ1, MQ1, MQ1);
   X = XxBBB(x,y,z);//[z][y][x];
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const double> X,
                                   double sm[3][MD1*MD1*MD1])
{
   //double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sm+0);
   DeviceCube Xx(sm[0], MD1, MD1, MD1);
   //double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sm+1);
   DeviceCube Xy(sm[1], MD1, MD1, MD1);
   //double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sm+2);
   DeviceCube Xz(sm[2], MD1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx(dx,dy,dz)/*[dz][dy][dx]*/ = X(dx,dy,dz,0,e);
            Xy(dx,dy,dz)/*[dz][dy][dx]*/ = X(dx,dy,dz,1,e);
            Xz(dx,dy,dz)/*[dz][dy][dx]*/ = X(dx,dy,dz,2,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 1/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDDD[3][MD1*MD1*MD1],
                                   double sDDQ[3][MD1*MD1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);

   //double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+0);
   ConstDeviceCube Xx(sDDD[0], MD1, MD1, MD1);
   //double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+1);
   ConstDeviceCube Xy(sDDD[1], MD1, MD1, MD1);
   //double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+2);
   ConstDeviceCube Xz(sDDD[2], MD1, MD1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   DeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   DeviceCube XzB(sDDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B(dx,qx);//[qx][dx];
               u[0] += Bx * Xx(dx,dy,dz);//[dz][dy][dx];
               u[1] += Bx * Xy(dx,dy,dz);//[dz][dy][dx];
               u[2] += Bx * Xz(dx,dy,dz);//[dz][dy][dx];
            }
            XxB(qx,dy,dz)/*[dz][dy][qx]*/ = u[0];
            XyB(qx,dy,dz)/*[dz][dy][qx]*/ = u[1];
            XzB(qx,dy,dz)/*[dz][dy][qx]*/ = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 2/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDDQ[3][MD1*MD1*MQ1],
                                   double sDQQ[3][MD1*MQ1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   ConstDeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   ConstDeviceCube XzB(sDDQ[2], MQ1, MD1, MD1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   DeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   DeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B(dy,qy);//[qy][dy];
               u[0] += XxB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               u[1] += XyB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               u[2] += XzB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
            }
            XxBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[0];
            XyBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[1];
            XzBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 3/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDQQ[3][MD1*MQ1*MQ1],
                                   double sQQQ[3][MQ1*MQ1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   ConstDeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   ConstDeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);

   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   DeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   DeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   DeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B(dz,qz);//[qz][dz];
               u[0] += XxBB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               u[1] += XyBB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               u[2] += XzBB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
            }
            XxBBB(qx,qy,qz)/*[qz][qy][qx]*/ = u[0];
            XyBBB(qx,qy,qz)/*[qz][qy][qx]*/ = u[1];
            XzBBB(qx,qy,qz)/*[qz][qy][qx]*/ = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDD[3][MD1*MD1*MD1],
                                   double sDDQ[3][MD1*MD1*MQ1])
{
   EvalX<MD1,MQ1>(D1D,Q1D,sBG[0],sDDD,sDDQ);
   /*double (*B)[MD1] = (double (*)[MD1])(sBG+0);

   double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+0);
   double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+1);
   double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+2);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B[qx][dx];
               u[0] += Bx * Xx[dz][dy][dx];
               u[1] += Bx * Xy[dz][dy][dx];
               u[2] += Bx * Xz[dz][dy][dx];
            }
            XxB[dz][dy][qx] = u[0];
            XyB[dz][dy][qx] = u[1];
            XzB[dz][dy][qx] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;*/
}

/// 3D Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDQ[3][MD1*MD1*MQ1],
                                   double sDQQ[3][MD1*MQ1*MQ1])
{
   EvalY<MD1,MQ1>(D1D,Q1D,sBG[0],sDDQ,sDQQ);
   /*double (*B)[MD1] = (double (*)[MD1])(sBG+0);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B[qy][dy];
               u[0] += XxB[dz][dy][qx] * By;
               u[1] += XyB[dz][dy][qx] * By;
               u[2] += XzB[dz][dy][qx] * By;
            }
            XxBB[dz][qy][qx] = u[0];
            XyBB[dz][qy][qx] = u[1];
            XzBB[dz][qy][qx] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;*/
}

/// 3D Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQQ[3][MD1*MQ1*MQ1],
                                   double sQQQ[3][MQ1*MQ1*MQ1])
{
   EvalZ<MD1,MQ1>(D1D,Q1D,sBG[0],sDQQ,sQQQ);
   /*double (*B)[MD1] = (double (*)[MD1])(sBG+0);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);

   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B[qz][dz];
               u[0] += XxBB[dz][qy][qx] * Bz;
               u[1] += XyBB[dz][qy][qx] * Bz;
               u[2] += XzBB[dz][qy][qx] * Bz;
            }
            XxBBB[qz][qy][qx] = u[0];
            XyBBB[qz][qy][qx] = u[1];
            XzBBB[qz][qy][qx] = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;*/
}

/// Pull 3D Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PullEvalXYZ(const int x, const int y, const int z,
                                         const double sQQQ[3][MQ1*MQ1*MQ1],
                                         double X[3])
{
   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   ConstDeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   ConstDeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   ConstDeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   X[0] = XxBBB(x,y,z);//[z][y][x];
   X[1] = XyBBB(x,y,z);//[z][y][x];
   X[2] = XzBBB(x,y,z);//[z][y][x];
}

/// Push 3D Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PushEvalXYZ(const int x, const int y, const int z,
                                         const double A[3],
                                         double sQQQ[3][MQ1*MQ1*MQ1])
{
   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   DeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   DeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   DeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   XxBBB(x,y,z)/*[z][y][x]*/ = A[0];
   XyBBB(x,y,z)/*[z][y][x]*/ = A[1];
   XzBBB(x,y,z)/*[z][y][x]*/ = A[2];
}

/// 3D Transposed Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sQQQ[3][MQ1*MQ1*MQ1],
                                    double sDQQ[3][MD1*MQ1*MQ1])
{

   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);

   //double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   ConstDeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   ConstDeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   ConstDeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[0]);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[1]);
   DeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[2]);
   DeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(qx,dx);//[dx][qx];
               u[0] += XxBBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
               u[1] += XyBBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
               u[2] += XzBBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
            }
            XxBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[0];
            XyBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[1];
            XzBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sDQQ[3][MD1*MQ1*MQ1],
                                    double sDDQ[3][MD1*MD1*MQ1])
{
   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   ConstDeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   ConstDeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   DeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   DeviceCube XzB(sDDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt(qy,dy);//[dy][qy];
               u[0] += XxBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;
               u[1] += XyBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;
               u[2] += XzBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;

            }
            XxB(qz,dy,dx)/*[dx][dy][qz]*/ = u[0];
            XyB(qz,dy,dx)/*[dx][dy][qz]*/ = u[1];
            XzB(qz,dy,dx)/*[dx][dy][qz]*/ = u[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sDDQ[3][MD1*MD1*MQ1],
                                    DeviceTensor<5, double> Y, const int e)
{
   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   ConstDeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   ConstDeviceCube XzB(sDDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz,dz);//[dz][qz];
               u[0] += XxB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               u[1] += XyB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               u[2] += XzB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
            }
            Y(dx,dy,dz,0,e) += u[0];
            Y(dx,dy,dz,1,e) += u[1];
            Y(dx,dy,dz,2,e) += u[2];
         }
      }
   }
}

/// 3D Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDD[3][MD1*MD1*MD1],
                                   double sDDQ[9][MD1*MD1*MQ1])
{
   //double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   //double (*G)[MD1] = (double (*)[MD1])(sBG+1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);

   //double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+0);
   ConstDeviceCube Xx(sDDD[0], MD1, MD1, MD1);
   //double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+1);
   ConstDeviceCube Xy(sDDD[1], MD1, MD1, MD1);
   //double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+2);
   ConstDeviceCube Xz(sDDD[2], MD1, MD1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   DeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   DeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   //double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   DeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   DeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   //double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
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
               const double xx = Xx(dx,dy,dz);//[dz][dy][dx];
               const double xy = Xy(dx,dy,dz);//[dz][dy][dx];
               const double xz = Xz(dx,dy,dz);//[dz][dy][dx];
               const double Bx = B(dx,qx);//[qx][dx];
               const double Gx = G(dx,qx);//[qx][dx];
               u[0] += Bx * xx;
               u[1] += Bx * xy;
               u[2] += Bx * xz;

               v[0] += Gx * xx;
               v[1] += Gx * xy;
               v[2] += Gx * xz;
            }
            XxB(qx,dy,dz)/*[dz][dy][qx]*/ = u[0];
            XyB(qx,dy,dz)/*[dz][dy][qx]*/ = u[1];
            XzB(qx,dy,dz)/*[dz][dy][qx]*/ = u[2];

            XxG(qx,dy,dz)/*[dz][dy][qx]*/ = v[0];
            XyG(qx,dy,dz)/*[dz][dy][qx]*/ = v[1];
            XzG(qx,dy,dz)/*[dz][dy][qx]*/ = v[2];
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
   //double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   //double (*G)[MD1] = (double (*)[MD1])(sBG+1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   ConstDeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   ConstDeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   //double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   ConstDeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   ConstDeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   //double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
   ConstDeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   DeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   DeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   DeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   //double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   DeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   //double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   DeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   DeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   //double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   DeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   //double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);
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
               const double By = B(dy,qy);//[qy][dy];
               const double Gy = G(dy,qy);//[qy][dy];

               u[0] += XxB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               u[1] += XyB(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               u[2] += XzB(qx,dy,dz)/*[dz][dy][qx]*/ * By;

               v[0] += XxG(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               v[1] += XyG(qx,dy,dz)/*[dz][dy][qx]*/ * By;
               v[2] += XzG(qx,dy,dz)/*[dz][dy][qx]*/ * By;

               w[0] += XxB(qx,dy,dz)/*[dz][dy][qx]*/ * Gy;
               w[1] += XyB(qx,dy,dz)/*[dz][dy][qx]*/ * Gy;
               w[2] += XzB(qx,dy,dz)/*[dz][dy][qx]*/ * Gy;
            }
            XxBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[0];
            XyBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[1];
            XzBB(qx,qy,dz)/*[dz][qy][qx]*/ = u[2];

            XxBG(qx,qy,dz)/*[dz][qy][qx]*/ = v[0];
            XyBG(qx,qy,dz)/*[dz][qy][qx]*/ = v[1];
            XzBG(qx,qy,dz)/*[dz][qy][qx]*/ = v[2];

            XxGB(qx,qy,dz)/*[dz][qy][qx]*/ = w[0];
            XyGB(qx,qy,dz)/*[dz][qy][qx]*/ = w[1];
            XzGB(qx,qy,dz)/*[dz][qy][qx]*/ = w[2];
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
   //double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   ConstDeviceMatrix B(sBG[0], MD1, MQ1);
   //double (*G)[MD1] = (double (*)[MD1])(sBG+1);
   ConstDeviceMatrix G(sBG[1], MD1, MQ1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   ConstDeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   ConstDeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   ConstDeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   //double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   ConstDeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   //double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   ConstDeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   ConstDeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   //double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   ConstDeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   //double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);
   ConstDeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);

   //double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   DeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   DeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   DeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   //double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   DeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   //double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   DeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   //double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   DeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   //double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   DeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   //double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   DeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   //double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);
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
               const double Bz = B(dz,qz);//[qz][dz];
               const double Gz = G(dz,qz);//[[qz][dz];

               u[0] += XxBG(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               u[1] += XyBG(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               u[2] += XzBG(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;

               v[0] += XxGB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               v[1] += XyGB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;
               v[2] += XzGB(qx,qy,dz)/*[dz][qy][qx]*/ * Bz;

               w[0] += XxBB(qx,qy,dz)/*[dz][qy][qx]*/ * Gz;
               w[1] += XyBB(qx,qy,dz)/*[dz][qy][qx]*/ * Gz;
               w[2] += XzBB(qx,qy,dz)/*[dz][qy][qx]*/ * Gz;
            }
            XxBBG(qx,qy,qz)/*[qz][qy][qx]*/ = u[0];
            XyBBG(qx,qy,qz)/*[qz][qy][qx]*/ = u[1];
            XzBBG(qx,qy,qz)/*[qz][qy][qx]*/ = u[2];

            XxBGB(qx,qy,qz)/*[qz][qy][qx]*/ = v[0];
            XyBGB(qx,qy,qz)/*[qz][qy][qx]*/ = v[1];
            XzBGB(qx,qy,qz)/*[qz][qy][qx]*/ = v[2];

            XxGBB(qx,qy,qz)/*[qz][qy][qx]*/ = w[0];
            XyGBB(qx,qy,qz)/*[qz][qy][qx]*/ = w[1];
            XzGBB(qx,qy,qz)/*[qz][qy][qx]*/ = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PullGradXYZ(const int x, const int y, const int z,
                                         const double sQQQ[9][MQ1*MQ1*MQ1],
                                         double *Jpr)
{
   //double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   ConstDeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   ConstDeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   ConstDeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   //double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   ConstDeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   //double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   ConstDeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   //double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   ConstDeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   //double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   ConstDeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   //double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   ConstDeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   //double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);
   ConstDeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

   Jpr[0] = XxBBG(x,y,z);//[z][y][x];
   Jpr[3] = XxBGB(x,y,z);//[z][y][x];
   Jpr[6] = XxGBB(x,y,z);//[z][y][x];
   Jpr[1] = XyBBG(x,y,z);//[z][y][x];
   Jpr[4] = XyBGB(x,y,z);//[z][y][x];
   Jpr[7] = XyGBB(x,y,z);//[z][y][x];
   Jpr[2] = XzBBG(x,y,z);//[z][y][x];
   Jpr[5] = XzBGB(x,y,z);//[z][y][x];
   Jpr[8] = XzGBB(x,y,z);//[z][y][x];
}

/// Push 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PushGradXYZ(const int x, const int y, const int z,
                                         const double *A,
                                         double sQQQ[9][MQ1*MQ1*MQ1])
{
   //double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+0);
   DeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+1);
   DeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+2);
   DeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   //double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+3);
   DeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   //double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+4);
   DeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   //double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+5);
   DeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   //double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+6);
   DeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   //double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+7);
   DeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   //double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+8);
   DeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

   XxBBG(x,y,z)/*[z][y][x]*/ = A[0];
   XxBGB(x,y,z)/*[z][y][x]*/ = A[1];
   XxGBB(x,y,z)/*[z][y][x]*/ = A[2];

   XyBBG(x,y,z)/*[z][y][x]*/ = A[3];
   XyBGB(x,y,z)/*[z][y][x]*/ = A[4];
   XyGBB(x,y,z)/*[z][y][x]*/ = A[5];

   XzBBG(x,y,z)/*[z][y][x]*/ = A[6];
   XzBGB(x,y,z)/*[z][y][x]*/ = A[7];
   XzGBB(x,y,z)/*[z][y][x]*/ = A[8];
}

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sQQQ[9][MQ1*MQ1*MQ1],
                                    double sDQQ[9][MD1*MQ1*MQ1])
{

   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   //double (*Gt)[MQ1] = (double (*)[MQ1])(sBG[1]);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);

   //double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   ConstDeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   //double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   ConstDeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   //double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   ConstDeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   //double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   ConstDeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   //double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   ConstDeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   //double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   ConstDeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   //double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   ConstDeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   //double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   ConstDeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   //double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);
   ConstDeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   DeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   DeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   DeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   //double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   DeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   //double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   DeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   DeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   //double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   DeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   //double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);
   DeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);

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
               const double Btx = Bt(qx,dx);//[dx][qx];
               const double Gtx = Gt(qx,dx);//[dx][qx];

               u[0] += XxBBG(qx,qy,qz)/*[qz][qy][qx]*/ * Gtx;
               v[0] += XxBGB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
               w[0] += XxGBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;

               u[1] += XyBBG(qx,qy,qz)/*[qz][qy][qx]*/ * Gtx;
               v[1] += XyBGB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
               w[1] += XyGBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;

               u[2] += XzBBG(qx,qy,qz)/*[qz][qy][qx]*/ * Gtx;
               v[2] += XzBGB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
               w[2] += XzGBB(qx,qy,qz)/*[qz][qy][qx]*/ * Btx;
            }
            XxBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[0];
            XxBG(qz,qy,dx)/*[dx][qy][qz]*/ = v[0];
            XxGB(qz,qy,dx)/*[dx][qy][qz]*/ = w[0];

            XyBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[1];
            XyBG(qz,qy,dx)/*[dx][qy][qz]*/ = v[1];
            XyGB(qz,qy,dx)/*[dx][qy][qz]*/ = w[1];

            XzBB(qz,qy,dx)/*[dx][qy][qz]*/ = u[2];
            XzBG(qz,qy,dx)/*[dx][qy][qz]*/ = v[2];
            XzGB(qz,qy,dx)/*[dx][qy][qz]*/ = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Gradient, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sDQQ[9][MD1*MQ1*MQ1],
                                    double sDDQ[9][MD1*MD1*MQ1])
{
   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   //double (*Gt)[MQ1] = (double (*)[MQ1])(sBG[1]);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);

   //double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   //double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   ConstDeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   //double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   ConstDeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   //double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   ConstDeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   //double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   ConstDeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   //double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   ConstDeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   //double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   ConstDeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   //double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   ConstDeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   //double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);
   ConstDeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   DeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   DeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   //double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   DeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   DeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   //double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
   DeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);
   //double (*XxC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+6);
   DeviceCube XxC(sDDQ[6], MQ1, MD1, MD1);
   //double (*XyC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+7);
   DeviceCube XyC(sDDQ[7], MQ1, MD1, MD1);
   //double (*XzC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+8);
   DeviceCube XzC(sDDQ[8], MQ1, MD1, MD1);

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
               const double Bty = Bt(qy,dy);//[dy][qy];
               const double Gty = Gt(qy,dy);//[dy][qy];

               u[0] += XxBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;
               v[0] += XxBG(qz,qy,dx)/*[dx][qy][qz]*/ * Gty;
               w[0] += XxGB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;

               u[1] += XyBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;
               v[1] += XyBG(qz,qy,dx)/*[dx][qy][qz]*/ * Gty;
               w[1] += XyGB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;

               u[2] += XzBB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;
               v[2] += XzBG(qz,qy,dx)/*[dx][qy][qz]*/ * Gty;
               w[2] += XzGB(qz,qy,dx)/*[dx][qy][qz]*/ * Bty;

            }
            XxB(qz,dy,dx)/*[dx][dy][qz]*/ = u[0];
            XxC(qz,dy,dx)/*[dx][dy][qz]*/ = v[0];
            XxG(qz,dy,dx)/*[dx][dy][qz]*/ = w[0];

            XyB(qz,dy,dx)/*[dx][dy][qz]*/ = u[1];
            XyC(qz,dy,dx)/*[dx][dy][qz]*/ = v[1];
            XyG(qz,dy,dx)/*[dx][dy][qz]*/ = w[1];

            XzB(qz,dy,dx)/*[dx][dy][qz]*/ = u[2];
            XzC(qz,dy,dx)/*[dx][dy][qz]*/ = v[2];
            XzG(qz,dy,dx)/*[dx][dy][qz]*/ = w[2];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Gradient, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sDDQ[9][MD1*MD1*MQ1],
                                    DeviceTensor<5, double> Y, const int e)
{
   //double (*Bt)[MQ1] = (double (*)[MQ1])(sBG+0);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   //double (*Gt)[MQ1] = (double (*)[MQ1])(sBG+1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);

   //double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   //double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   ConstDeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   //double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   ConstDeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   //double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   ConstDeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   //double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   ConstDeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   //double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
   ConstDeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);
   //double (*XxC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+6);
   ConstDeviceCube XxC(sDDQ[6], MQ1, MD1, MD1);
   //double (*XyC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+7);
   ConstDeviceCube XyC(sDDQ[7], MQ1, MD1, MD1);
   //double (*XzC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+8);
   ConstDeviceCube XzC(sDDQ[8], MQ1, MD1, MD1);

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
               const double Btz = Bt(qz,dz);//[dz][qz];
               const double Gtz = Gt(qz,dz);//[dz][qz];

               u[0] += XxB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               v[0] += XxC(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               w[0] += XxG(qz,dy,dx)/*[dx][dy][qz]*/ * Gtz;

               u[1] += XyB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               v[1] += XyC(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               w[1] += XyG(qz,dy,dx)/*[dx][dy][qz]*/ * Gtz;

               u[2] += XzB(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               v[2] += XzC(qz,dy,dx)/*[dx][dy][qz]*/ * Btz;
               w[2] += XzG(qz,dy,dx)/*[dx][dy][qz]*/ * Gtz;
            }
            Y(dx,dy,dz,0,e) += u[0] + v[0] + w[0];
            Y(dx,dy,dz,1,e) += u[1] + v[1] + w[1];
            Y(dx,dy,dz,2,e) += u[2] + v[2] + w[2];
         }
      }
   }
}

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
