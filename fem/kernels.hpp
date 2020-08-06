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
                                    const DeviceTensor<2, const double> b,
                                    const DeviceTensor<2, const double> g,
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
                                     const DeviceTensor<2, const double> b,
                                     const DeviceTensor<2, const double> g,
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
   MFEM_ABORT("");
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
   MFEM_ABORT("");
   const int tidz = MFEM_THREAD_ID(z);
   //double (*X)[MD1] = (double (*)[MD1])(sm + tidz);
   DeviceMatrix X(sm[tidz], MD1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         //X[dy][dx] = x(dx,dy,c,e);
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
   MFEM_ABORT("");
   const int tidz = MFEM_THREAD_ID(z);
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*X)[MD1]  = (double (*)[MD1])(sX + tidz);
   double (*DQ)[MQ1] = (double (*)[MQ1])(sDQ + tidz);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            u += B[qx][dx] * X[dy][dx];
         }
         DQ[dy][qx] = u;
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
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*DQ)[MQ1] = (double (*)[MQ1])(sDQ + tidz);
   double (*QQ)[MQ1] = (double (*)[MQ1])(sQQ + tidz);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            u += DQ[dy][qx] * B[qy][dy];
         }
         QQ[qy][qx] = u;
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
   double (*QQ)[MQ1] = (double (*)[MQ1])(sQQ + tidz);

   P[0] = QQ[qy][qx];
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
   //#warning MFEM_ABORT
   MFEM_ABORT("");
   const int tidz = MFEM_THREAD_ID(z);
   //double (*B)[MD1] = (double (*)[MD1])(sB);
   ConstDeviceMatrix B(sB, MD1, MQ1);
   //double (*X0)[MD1]  = (double (*)[MD1])(sX[0] + tidz);
   ConstDeviceMatrix X0(sX[0][tidz], MD1, MD1);
   //double (*X1)[MD1]  = (double (*)[MD1])(sX[1] + tidz);
   ConstDeviceMatrix X1(sX[1][tidz], MD1, MD1);
   double (*DQ0)[MQ1] = (double (*)[MQ1])(sDQ[0] + tidz);
   double (*DQ1)[MQ1] = (double (*)[MQ1])(sDQ[1] + tidz);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double xx = X0(dx,dy);//[dy][dx];
            const double xy = X1(dx,dy);//[dy][dx];
            u[0] += B(dx,qx) * xx;
            u[1] += B(dx,qx) * xy;
         }
         DQ0[dy][qx] = u[0];
         DQ1[dy][qx] = u[1];
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
   //#warning MFEM_ABORT
   MFEM_ABORT("");
   const int tidz = MFEM_THREAD_ID(z);
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*DQ0)[MQ1] = (double (*)[MQ1])(sDQ[0] + tidz);
   double (*DQ1)[MQ1] = (double (*)[MQ1])(sDQ[1] + tidz);
   double (*QQ0)[MQ1] = (double (*)[MQ1])(sQQ[0] + tidz);
   double (*QQ1)[MQ1] = (double (*)[MQ1])(sQQ[1] + tidz);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         for (int dy = 0; dy < D1D; ++dy)
         {
            u[0] += DQ0[dy][qx] * B[qy][dy];
            u[1] += DQ1[dy][qx] * B[qy][dy];
         }
         QQ0[qy][qx] = u[0];
         QQ1[qy][qx] = u[1];
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
   const int tidz = MFEM_THREAD_ID(z);
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
   MFEM_SYNC_THREAD;
}

/// 2D Evaluation, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQ[2][NBZ][MD1*MQ1],
                                   double sQQ[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
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
   MFEM_SYNC_THREAD;
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
                                    const DeviceTensor<5, const double> X,
                                    double sm[MD1*MD1*MD1])
{
   double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sm);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dz][dy][dx] = X(dx,dy,dz,c,e);
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
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD);
   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bx = B[qx][dx];
               u += Bx * Xx[dz][dy][dx];
            }
            XxB[dz][dy][qx] = u;
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
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ);
   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B[qy][dy];
               u += XxB[dz][dy][qx] * By;
            }
            XxBB[dz][qy][qx] = u;
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
   double (*B)[MD1] = (double (*)[MD1])(sB);
   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ);
   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B[qz][dz];
               u += XxBB[dz][qy][qx] * Bz;
            }
            XxBBB[qz][qy][qx] = u;
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
   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ);
   X = XxBBB[z][y][x];
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const double> X,
                                   double sm[3][MD1*MD1*MD1])
{
   double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sm+0);
   double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sm+1);
   double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sm+2);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dz][dy][dx] = X(dx,dy,dz,0,e);
            Xy[dz][dy][dx] = X(dx,dy,dz,1,e);
            Xz[dz][dy][dx] = X(dx,dy,dz,2,e);
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
   double (*B)[MD1] = (double (*)[MD1])(sB);

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
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 2/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDDQ[3][MD1*MD1*MQ1],
                                   double sDQQ[3][MD1*MQ1*MQ1])
{
   double (*B)[MD1] = (double (*)[MD1])(sB);

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
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 3/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double sB[MQ1*MD1],
                                   const double sDQQ[3][MD1*MQ1*MQ1],
                                   double sQQQ[3][MQ1*MQ1*MQ1])
{
   double (*B)[MD1] = (double (*)[MD1])(sB);

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
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDD[3][MD1*MD1*MD1],
                                   double sDDQ[3][MD1*MD1*MQ1])
{
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);

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
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDDQ[3][MD1*MD1*MQ1],
                                   double sDQQ[3][MD1*MQ1*MQ1])
{
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);

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
   MFEM_SYNC_THREAD;
}

/// 3D Evaluation, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double sBG[2][MQ1*MD1],
                                   const double sDQQ[3][MD1*MQ1*MQ1],
                                   double sQQQ[3][MQ1*MQ1*MQ1])
{
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);

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
   MFEM_SYNC_THREAD;
}

/// Pull 3D Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PullEvalXYZ(const int x, const int y, const int z,
                                         const double sQQQ[3][MQ1*MQ1*MQ1],
                                         double X[3])
{
   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);

   X[0] = XxBBB[z][y][x];
   X[1] = XyBBB[z][y][x];
   X[2] = XzBBB[z][y][x];
}

/// Push 3D Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PushEvalXYZ(const int x, const int y, const int z,
                                         const double A[3],
                                         double sQQQ[3][MQ1*MQ1*MQ1])
{
   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);

   XxBBB[z][y][x] = A[0];
   XyBBB[z][y][x] = A[1];
   XzBBB[z][y][x] = A[2];
}

/// 3D Transposed Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sQQQ[3][MQ1*MQ1*MQ1],
                                    double sDQQ[3][MD1*MQ1*MQ1])
{

   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);

   double (*XxBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XyBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XzBBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[0]);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[1]);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ[2]);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt[dx][qx];
               u[0] += XxBBB[qz][qy][qx] * Btx;
               u[1] += XyBBB[qz][qy][qx] * Btx;
               u[2] += XzBBB[qz][qy][qx] * Btx;
            }
            XxBB[dx][qy][qz] = u[0];
            XyBB[dx][qy][qz] = u[1];
            XzBB[dx][qy][qz] = u[2];
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
   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);

   MFEM_FOREACH_THREAD(qz,z,Q1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt[dy][qy];
               u[0] += XxBB[dx][qy][qz] * Bty;
               u[1] += XyBB[dx][qy][qz] * Bty;
               u[2] += XzBB[dx][qy][qz] * Bty;

            }
            XxB[dx][dy][qz] = u[0];
            XyB[dx][dy][qz] = u[1];
            XzB[dx][dy][qz] = u[2];
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
   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);

   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[3] = {0.0, 0.0, 0.0};
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt[dz][qz];
               u[0] += XxB[dx][dy][qz] * Btz;
               u[1] += XyB[dx][dy][qz] * Btz;
               u[2] += XzB[dx][dy][qz] * Btz;
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
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   double (*G)[MD1] = (double (*)[MD1])(sBG+1);

   double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+0);
   double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+1);
   double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(sDDD+2);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);

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
               const double xx = Xx[dz][dy][dx];
               const double xy = Xy[dz][dy][dx];
               const double xz = Xz[dz][dy][dx];
               const double Bx = B[qx][dx];
               const double Gx = G[qx][dx];
               u[0] += Bx * xx;
               u[1] += Bx * xy;
               u[2] += Bx * xz;

               v[0] += Gx * xx;
               v[1] += Gx * xy;
               v[2] += Gx * xz;
            }
            XxB[dz][dy][qx] = u[0];
            XyB[dz][dy][qx] = u[1];
            XzB[dz][dy][qx] = u[2];

            XxG[dz][dy][qx] = v[0];
            XyG[dz][dy][qx] = v[1];
            XzG[dz][dy][qx] = v[2];
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
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   double (*G)[MD1] = (double (*)[MD1])(sBG+1);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);

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
               const double By = B[qy][dy];
               const double Gy = G[qy][dy];

               u[0] += XxB[dz][dy][qx] * By;
               u[1] += XyB[dz][dy][qx] * By;
               u[2] += XzB[dz][dy][qx] * By;

               v[0] += XxG[dz][dy][qx] * By;
               v[1] += XyG[dz][dy][qx] * By;
               v[2] += XzG[dz][dy][qx] * By;

               w[0] += XxB[dz][dy][qx] * Gy;
               w[1] += XyB[dz][dy][qx] * Gy;
               w[2] += XzB[dz][dy][qx] * Gy;
            }
            XxBB[dz][qy][qx] = u[0];
            XyBB[dz][qy][qx] = u[1];
            XzBB[dz][qy][qx] = u[2];

            XxBG[dz][qy][qx] = v[0];
            XyBG[dz][qy][qx] = v[1];
            XzBG[dz][qy][qx] = v[2];

            XxGB[dz][qy][qx] = w[0];
            XyGB[dz][qy][qx] = w[1];
            XzGB[dz][qy][qx] = w[2];
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
   double (*B)[MD1] = (double (*)[MD1])(sBG+0);
   double (*G)[MD1] = (double (*)[MD1])(sBG+1);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);

   double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);

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
               const double Bz = B[qz][dz];
               const double Gz = G[qz][dz];

               u[0] += XxBG[dz][qy][qx] * Bz;
               u[1] += XyBG[dz][qy][qx] * Bz;
               u[2] += XzBG[dz][qy][qx] * Bz;

               v[0] += XxGB[dz][qy][qx] * Bz;
               v[1] += XyGB[dz][qy][qx] * Bz;
               v[2] += XzGB[dz][qy][qx] * Bz;

               w[0] += XxBB[dz][qy][qx] * Gz;
               w[1] += XyBB[dz][qy][qx] * Gz;
               w[2] += XzBB[dz][qy][qx] * Gz;
            }
            XxBBG[qz][qy][qx] = u[0];
            XyBBG[qz][qy][qx] = u[1];
            XzBBG[qz][qy][qx] = u[2];

            XxBGB[qz][qy][qx] = v[0];
            XyBGB[qz][qy][qx] = v[1];
            XzBGB[qz][qy][qx] = v[2];

            XxGBB[qz][qy][qx] = w[0];
            XyGBB[qz][qy][qx] = w[1];
            XzGBB[qz][qy][qx] = w[2];
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
   double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);

   Jpr[0] = XxBBG[z][y][x];
   Jpr[3] = XxBGB[z][y][x];
   Jpr[6] = XxGBB[z][y][x];
   Jpr[1] = XyBBG[z][y][x];
   Jpr[4] = XyBGB[z][y][x];
   Jpr[7] = XyGBB[z][y][x];
   Jpr[2] = XzBBG[z][y][x];
   Jpr[5] = XzBGB[z][y][x];
   Jpr[8] = XzGBB[z][y][x];
}

/// Push 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PushGradXYZ(const int x, const int y, const int z,
                                         const double *A,
                                         double s_QQQ[9][MQ1*MQ1*MQ1])
{
   double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+0);
   double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+1);
   double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+2);
   double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+3);
   double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+4);
   double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+5);
   double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+6);
   double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+7);
   double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+8);

   XxBBG[z][y][x] = A[0];
   XxBGB[z][y][x] = A[1];
   XxGBB[z][y][x] = A[2];

   XyBBG[z][y][x] = A[3];
   XyBGB[z][y][x] = A[4];
   XyGBB[z][y][x] = A[5];

   XzBBG[z][y][x] = A[6];
   XzBGB[z][y][x] = A[7];
   XzGBB[z][y][x] = A[8];
}

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double sQQQ[9][MQ1*MQ1*MQ1],
                                    double sDQQ[9][MD1*MQ1*MQ1])
{

   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   double (*Gt)[MQ1] = (double (*)[MQ1])(sBG[1]);

   double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+0);
   double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+1);
   double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+2);
   double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+3);
   double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+4);
   double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+5);
   double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+6);
   double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+7);
   double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sQQQ+8);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);

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
               const double Btx = Bt[dx][qx];
               const double Gtx = Gt[dx][qx];

               u[0] += XxBBG[qz][qy][qx] * Gtx;
               v[0] += XxBGB[qz][qy][qx] * Btx;
               w[0] += XxGBB[qz][qy][qx] * Btx;

               u[1] += XyBBG[qz][qy][qx] * Gtx;
               v[1] += XyBGB[qz][qy][qx] * Btx;
               w[1] += XyGBB[qz][qy][qx] * Btx;

               u[2] += XzBBG[qz][qy][qx] * Gtx;
               v[2] += XzBGB[qz][qy][qx] * Btx;
               w[2] += XzGBB[qz][qy][qx] * Btx;
            }
            XxBB[dx][qy][qz] = u[0];
            XxBG[dx][qy][qz] = v[0];
            XxGB[dx][qy][qz] = w[0];

            XyBB[dx][qy][qz] = u[1];
            XyBG[dx][qy][qz] = v[1];
            XyGB[dx][qy][qz] = w[1];

            XzBB[dx][qy][qz] = u[2];
            XzBG[dx][qy][qz] = v[2];
            XzGB[dx][qy][qz] = w[2];
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
   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG[0]);
   double (*Gt)[MQ1] = (double (*)[MQ1])(sBG[1]);

   double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+0);
   double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+1);
   double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+2);
   double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+3);
   double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+4);
   double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+5);
   double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+6);
   double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+7);
   double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(sDQQ+8);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
   double (*XxC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+6);
   double (*XyC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+7);
   double (*XzC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+8);

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
               const double Bty = Bt[dy][qy];
               const double Gty = Gt[dy][qy];

               u[0] += XxBB[dx][qy][qz] * Bty;
               v[0] += XxBG[dx][qy][qz] * Gty;
               w[0] += XxGB[dx][qy][qz] * Bty;

               u[1] += XyBB[dx][qy][qz] * Bty;
               v[1] += XyBG[dx][qy][qz] * Gty;
               w[1] += XyGB[dx][qy][qz] * Bty;

               u[2] += XzBB[dx][qy][qz] * Bty;
               v[2] += XzBG[dx][qy][qz] * Gty;
               w[2] += XzGB[dx][qy][qz] * Bty;

            }
            XxB[dx][dy][qz] = u[0];
            XxC[dx][dy][qz] = v[0];
            XxG[dx][dy][qz] = w[0];

            XyB[dx][dy][qz] = u[1];
            XyC[dx][dy][qz] = v[1];
            XyG[dx][dy][qz] = w[1];

            XzB[dx][dy][qz] = u[2];
            XzC[dx][dy][qz] = v[2];
            XzG[dx][dy][qz] = w[2];
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
   double (*Bt)[MQ1] = (double (*)[MQ1])(sBG+0);
   double (*Gt)[MQ1] = (double (*)[MQ1])(sBG+1);

   double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+0);
   double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+1);
   double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+2);
   double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+3);
   double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+4);
   double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+5);
   double (*XxC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+6);
   double (*XyC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+7);
   double (*XzC)[MD1][MQ1] = (double (*)[MD1][MQ1])(sDDQ+8);

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
               const double Btz = Bt[dz][qz];
               const double Gtz = Gt[dz][qz];

               u[0] += XxB[dx][dy][qz] * Btz;
               v[0] += XxC[dx][dy][qz] * Btz;
               w[0] += XxG[dx][dy][qz] * Gtz;

               u[1] += XyB[dx][dy][qz] * Btz;
               v[1] += XyC[dx][dy][qz] * Btz;
               w[1] += XyG[dx][dy][qz] * Gtz;

               u[2] += XzB[dx][dy][qz] * Btz;
               v[2] += XzC[dx][dy][qz] * Btz;
               w[2] += XzG[dx][dy][qz] * Gtz;
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
