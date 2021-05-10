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

/// Load B1d matrice into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadB(const int D1D, const int Q1D,
                                   const ConstDeviceMatrix &b,
                                   double (&sB)[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sB, D1D, Q1D);

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

/// Load Bt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBt(const int D1D, const int Q1D,
                                    const ConstDeviceMatrix &b,
                                    double (&sB)[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sB, MQ1, MD1);

   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt(q,d) = b(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

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

/// Load Bt1d & Gt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBGt(const int D1D, const int Q1D,
                                     const ConstDeviceMatrix &b,
                                     const ConstDeviceMatrix &g,
                                     double (&sBG)[2][MQ1*MD1])
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
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<3, const double> &x,
                                   double (&sX)[NBZ][MD1*MD1])
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
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<4, const double> &x,
                                   DeviceMatrix &DD)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         DD(dx,dy) = x(dx,dy,c,e);
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<4, const double> &x,
                                   double (&sm)[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix DD(sm[tidz], MD1, MD1);
   LoadX(e,D1D,c,x,DD);
}

/// 2D Scalar Evaluation, 1/2
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   ConstDeviceMatrix &B,
                                   const DeviceMatrix &DD,
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
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sX)[NBZ][MD1*MD1],
                                   double (&sDQ)[NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceMatrix DD((double*)sX[tidz], D1D, D1D);
   DeviceMatrix DQ(sDQ[tidz], Q1D, D1D);
   EvalX(D1D,Q1D,B,DD,DQ);
}

/// 2D Scalar Evaluation, 2/2
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   ConstDeviceMatrix &B,
                                   const DeviceMatrix &DQ,
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
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDQ)[NBZ][MD1*MQ1],
                                   double (&sQQ)[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceMatrix DQ((double*)sDQ[tidz], Q1D, D1D);
   DeviceMatrix QQ(sQQ[tidz], Q1D, Q1D);
   EvalY(D1D,Q1D,B,DQ,QQ);
}

/// Pull 2D Scalar Evaluation
MFEM_HOST_DEVICE inline void PullEval(const int qx, const int qy,
                                      const DeviceMatrix &QQ,
                                      double &P)
{
   P = QQ(qx,qy);
}

template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEval(const int qx, const int qy,
                                      const double (&sQQ)[NBZ][MQ1*MQ1],
                                      double &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   const DeviceMatrix QQ((double*)sQQ[tidz], MQ1, MQ1);
   PullEval(qx,qy,QQ,P);
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

/// 2D Evaluation, 1/2 (only B)
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sX)[2][NBZ][MD1*MD1],
                                   double (&sDQ)[2][NBZ][MD1*MQ1])
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
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDQ)[2][NBZ][MD1*MQ1],
                                   double (&sQQ)[2][NBZ][MQ1*MQ1])
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

/// Pull 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEval(const int qx, const int qy,
                                      const double (&sQQ)[2][NBZ][MQ1*MQ1],
                                      double (&P)[2])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

   P[0] = QQ0(qx,qy);
   P[1] = QQ1(qx,qy);
}

/// Push 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushEval(const int qx, const int qy,
                                      const double *P,
                                      double (&sQQ)[2][NBZ][MQ1*MQ1])
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
                                    const double (&sB)[MQ1*MD1],
                                    const double (&sQQ)[2][NBZ][MQ1*MQ1],
                                    double (&sDQ)[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
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
                                    const double (&sB)[MQ1*MD1],
                                    const double (&sDQ)[2][NBZ][MD1*MQ1],
                                    const DeviceTensor<4> &Y, // output
                                    const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
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

/// Push 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushGrad(const int qx, const int qy,
                                      const double *A,
                                      double (&sQQ)[4][NBZ][MQ1*MQ1])
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
                                    const double (&sBG)[2][MQ1*MD1],
                                    const double (&GQ)[4][NBZ][MQ1*MQ1],
                                    double (&GD)[4][NBZ][MD1*MQ1])
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
                                    const double (&sBG)[2][MQ1*MD1],
                                    const double (&GD)[4][NBZ][MD1*MQ1],
                                    const DeviceTensor<4> &Y, // output
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
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const double> &x,
                                   DeviceCube &X)
{
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X(dx,dy,dz) = x(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const double> &x,
                                   double (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, MD1,MD1,MD1);
   LoadX(e,D1D,x,X);
}

/// Load 3D scalar input vector into shared memory, with comp & DeviceTensor
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

/// Load 3D scalar input vector into shared memory, with comp & pointer
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<5, const double> &x,
                                   double (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, MD1, MD1, MD1);
   return LoadX<MD1>(e,D1D,c,x,X);
}

/// 3D Scalar Evaluation, 1/3
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
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

template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDDD)[MD1*MD1*MD1],
                                   double (&sDDQ)[MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DDD(sDDD, D1D, D1D, D1D);
   DeviceCube DDQ(sDDQ, Q1D, D1D, D1D);
   EvalX(D1D,Q1D,B,DDD,DDQ);
}

/// 3D Scalar Evaluation, 2/3
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
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

template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDDQ)[MD1*MD1*MQ1],
                                   double (&sDQQ)[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   const DeviceCube DDQ(sDDQ, MQ1, MD1, MD1);
   DeviceCube DQQ(sDQQ, MQ1, MQ1, MD1);
   EvalY(D1D,Q1D,B,DDQ,DQQ);
}

/// 3D Scalar Evaluation, 3/3
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
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
            QQQ(qx,qy,qz) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDQQ)[MD1*MQ1*MQ1],
                                   double (&sQQQ)[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   const DeviceCube DQQ(sDQQ, MQ1, MQ1, MD1);
   DeviceCube QQQ(sQQQ, MQ1, MQ1, MQ1);
   EvalZ(D1D,Q1D,B,DQQ,QQQ);
}

/// Pull 3D Scalar Evaluation
MFEM_HOST_DEVICE inline void PullEval(const int x, const int y, const int z,
                                      const DeviceCube &QQQ,
                                      double &X)
{
   X = QQQ(x,y,z);
}

template<int MQ1>
MFEM_HOST_DEVICE inline void PullEval(const int x, const int y, const int z,
                                      const double (&sQQQ)[MQ1*MQ1*MQ1],
                                      double &X)
{
   const DeviceCube QQQ(sQQQ, MQ1, MQ1, MQ1);
   PullEval(x,y,z,QQQ,X);
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const double> &X,
                                   double (*sm)[MD1*MD1*MD1])
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

/// 3D Vector Evaluation, 1/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDDD)[3][MD1*MD1*MD1],
                                   double (&sDDQ)[3][MD1*MD1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube Xx(sDDD[0], MD1, MD1, MD1);
   ConstDeviceCube Xy(sDDD[1], MD1, MD1, MD1);
   ConstDeviceCube Xz(sDDD[2], MD1, MD1, MD1);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
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
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDDQ)[3][MD1*MD1*MQ1],
                                   double (&sDQQ)[3][MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   ConstDeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
   ConstDeviceCube XzB(sDDQ[2], MQ1, MD1, MD1);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
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
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const double (&sB)[MQ1*MD1],
                                   const double (&sDQQ)[3][MD1*MQ1*MQ1],
                                   double (&sQQQ)[3][MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, MD1, MQ1);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   ConstDeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   ConstDeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
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

/// Pull 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PullEval(const int x, const int y, const int z,
                                      const double (&sQQQ)[3][MQ1*MQ1*MQ1],
                                      double (&X)[3])
{
   ConstDeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   X[0] = XxBBB(x,y,z);
   X[1] = XyBBB(x,y,z);
   X[2] = XzBBB(x,y,z);
}

/// Push 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PushEval(const int x, const int y, const int z,
                                      const double (&A)[3],
                                      double (&sQQQ)[3][MQ1*MQ1*MQ1])
{
   DeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   DeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);

   XxBBB(x,y,z) = A[0];
   XyBBB(x,y,z) = A[1];
   XzBBB(x,y,z) = A[2];
}

/// 3D Transposed Vector Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double (&sB)[MQ1*MD1],
                                    const double (&sQQQ)[3][MQ1*MQ1*MQ1],
                                    double (&sDQQ)[3][MD1*MQ1*MQ1])
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxBBB(sQQQ[0], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBBB(sQQQ[1], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBBB(sQQQ[2], MQ1, MQ1, MQ1);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
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
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double (&sB)[MQ1*MD1],
                                    const double (&sDQQ)[3][MD1*MQ1*MQ1],
                                    double (&sDDQ)[3][MD1*MD1*MQ1])
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   ConstDeviceCube XyBB(sDQQ[1], MQ1, MQ1, MD1);
   ConstDeviceCube XzBB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
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
MFEM_HOST_DEVICE inline void EvalZt(const int D1D, const int Q1D,
                                    const double (&sB)[MQ1*MD1],
                                    const double (&sDDQ)[3][MD1*MD1*MQ1],
                                    const DeviceTensor<5> &Y, // output
                                    const int e)
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   ConstDeviceCube XyB(sDDQ[1], MQ1, MD1, MD1);
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

/// 3D Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDDD)[MD1*MD1*MD1],
                                   double (*sDDQ)[MD1*MD1*MQ1])
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
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDDQ)[MD1*MD1*MQ1],
                                   double (*sDQQ)[MD1*MQ1*MQ1])
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
                                   const double (*sBG)[MQ1*MD1],
                                   const double (*sDQQ)[MD1*MQ1*MQ1],
                                   double (*sQQQ)[MQ1*MQ1*MQ1])
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
                                      const double (*sQQQ)[MQ1*MQ1*MQ1],
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

/// Push 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PushGrad(const int x, const int y, const int z,
                                      const double *A,
                                      double (&sQQQ)[9][MQ1*MQ1*MQ1])
{
   DeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   DeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   DeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   DeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   DeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   DeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   DeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   DeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   DeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);

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

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const double (&sBG)[2][MQ1*MD1],
                                    const double (&sQQQ)[9][MQ1*MQ1*MQ1],
                                    double (&sDQQ)[9][MD1*MQ1*MQ1])
{

   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceCube XxBBG(sQQQ[0], MQ1, MQ1, MQ1);
   ConstDeviceCube XxBGB(sQQQ[1], MQ1, MQ1, MQ1);
   ConstDeviceCube XxGBB(sQQQ[2], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBBG(sQQQ[3], MQ1, MQ1, MQ1);
   ConstDeviceCube XyBGB(sQQQ[4], MQ1, MQ1, MQ1);
   ConstDeviceCube XyGBB(sQQQ[5], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBBG(sQQQ[6], MQ1, MQ1, MQ1);
   ConstDeviceCube XzBGB(sQQQ[7], MQ1, MQ1, MQ1);
   ConstDeviceCube XzGBB(sQQQ[8], MQ1, MQ1, MQ1);
   DeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   DeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   DeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   DeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   DeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   DeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   DeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   DeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
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
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const double (&sBG)[2][MQ1*MD1],
                                    const double (&sDQQ)[9][MD1*MQ1*MQ1],
                                    double (&sDDQ)[9][MD1*MD1*MQ1])
{
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceCube XxBB(sDQQ[0], MQ1, MQ1, MD1);
   ConstDeviceCube XxBG(sDQQ[1], MQ1, MQ1, MD1);
   ConstDeviceCube XxGB(sDQQ[2], MQ1, MQ1, MD1);
   ConstDeviceCube XyBB(sDQQ[3], MQ1, MQ1, MD1);
   ConstDeviceCube XyBG(sDQQ[4], MQ1, MQ1, MD1);
   ConstDeviceCube XyGB(sDQQ[5], MQ1, MQ1, MD1);
   ConstDeviceCube XzBB(sDQQ[6], MQ1, MQ1, MD1);
   ConstDeviceCube XzBG(sDQQ[7], MQ1, MQ1, MD1);
   ConstDeviceCube XzGB(sDQQ[8], MQ1, MQ1, MD1);
   DeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   DeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   DeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   DeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   DeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   DeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);
   DeviceCube XxC(sDDQ[6], MQ1, MD1, MD1);
   DeviceCube XyC(sDDQ[7], MQ1, MD1, MD1);
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
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const double (&sBG)[2][MQ1*MD1],
                                    const double (&sDDQ)[9][MD1*MD1*MQ1],
                                    const DeviceTensor<5> &Y, // output
                                    const int e)
{
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceCube XxB(sDDQ[0], MQ1, MD1, MD1);
   ConstDeviceCube XxG(sDDQ[1], MQ1, MD1, MD1);
   ConstDeviceCube XyB(sDDQ[2], MQ1, MD1, MD1);
   ConstDeviceCube XyG(sDDQ[3], MQ1, MD1, MD1);
   ConstDeviceCube XzB(sDDQ[4], MQ1, MD1, MD1);
   ConstDeviceCube XzG(sDDQ[5], MQ1, MD1, MD1);
   ConstDeviceCube XxC(sDDQ[6], MQ1, MD1, MD1);
   ConstDeviceCube XyC(sDDQ[7], MQ1, MD1, MD1);
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

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
