// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

// Experimental helper functions for mfem::forall FEM kernels
// For the 2D functions, NBZ should be tied to '1' for now
namespace internal
{

/// Load B1d matrice into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadB(const int D1D, const int Q1D,
                                   const ConstDeviceMatrix &b,
                                   real_t (&sB)[MQ1*MD1])
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
                                    real_t (&sB)[MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sB, Q1D, D1D);

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
                                    real_t (&sBG)[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix B(sBG[0], D1D, Q1D);
   DeviceMatrix G(sBG[1], D1D, Q1D);

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
                                     real_t (&sBG)[2][MQ1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix Bt(sBG[0], Q1D, D1D);
   DeviceMatrix Gt(sBG[1], Q1D, D1D);

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

/// Load 2D input scalar into given DeviceMatrix
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<3, const real_t> &x,
                                   DeviceMatrix &DD)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         DD(dx,dy) = x(dx,dy,e);
      }
   }
   MFEM_SYNC_THREAD;
}


/// Load 2D input scalar into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<3, const real_t> &x,
                                   real_t (&sX)[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X(sX[tidz], D1D, D1D);
   LoadX(e, D1D, x, X);
}

/// Load 2D input scalar into shared memory, with comp
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<4, const real_t> &x,
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
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t (&sm)[NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix DD(sm[tidz], D1D, D1D);
   LoadX(e,D1D,c,x,DD);
}

/// 2D Scalar Evaluation, 1/2
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   ConstDeviceMatrix &B,
                                   DeviceMatrix &DD,
                                   DeviceMatrix &DQ)
{
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u = 0.0;
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
                                   const real_t (&sB)[MQ1*MD1],
                                   real_t (&sDD)[NBZ][MD1*MD1],
                                   real_t (&sDQ)[NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix DD(sDD[tidz], D1D, D1D);
   DeviceMatrix DQ(sDQ[tidz], D1D, Q1D);
   EvalX(D1D,Q1D,B,DD,DQ);
}

/// 2D Scalar Evaluation, 2/2
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   ConstDeviceMatrix &B,
                                   DeviceMatrix &DQ,
                                   DeviceMatrix &QQ)
{
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u = 0.0;
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
                                   const real_t (&sB)[MQ1*MD1],
                                   real_t (&sDQ)[NBZ][MD1*MQ1],
                                   real_t (&sQQ)[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   DeviceMatrix DQ(sDQ[tidz], D1D, Q1D);
   DeviceMatrix QQ(sQQ[tidz], Q1D, Q1D);
   EvalY(D1D,Q1D,B,DQ,QQ);
}

/// Pull 2D Scalar Evaluation
MFEM_HOST_DEVICE inline void PullEval(const int qx, const int qy,
                                      DeviceMatrix &QQ,
                                      real_t &P)
{
   P = QQ(qx,qy);
}

template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int qx, const int qy,
                                      real_t (&sQQ)[NBZ][MQ1*MQ1],
                                      real_t &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], Q1D, Q1D);
   PullEval(qx,qy,QQ,P);
}

/// Load 2D input vector into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<4, const real_t> &X,
                                   real_t (&sX)[2][NBZ][MD1*MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0(sX[0][tidz], D1D, D1D);
   DeviceMatrix X1(sX[1][tidz], D1D, D1D);

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
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sX)[2][NBZ][MD1*MD1],
                                   real_t (&sDQ)[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceMatrix X0(sX[0][tidz], D1D, D1D);
   ConstDeviceMatrix X1(sX[1][tidz], D1D, D1D);
   DeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   DeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t xx = X0(dx,dy);
            const real_t xy = X1(dx,dy);
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
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDQ)[2][NBZ][MD1*MQ1],
                                   real_t (&sQQ)[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sB, D1D, Q1D);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);
   DeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u[2] = {0.0, 0.0};
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
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int qx, const int qy,
                                      const real_t (&sQQ)[2][NBZ][MQ1*MQ1],
                                      real_t (&P)[2])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   P[0] = QQ0(qx,qy);
   P[1] = QQ1(qx,qy);
}

/// Push 2D Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushEval(const int Q1D,
                                      const int qx, const int qy,
                                      const real_t *P,
                                      real_t (&sQQ)[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);

   QQ0(qx,qy) = P[0];
   QQ1(qx,qy) = P[1];
}

/// 2D Transposed evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sQQ)[2][NBZ][MQ1*MQ1],
                                    real_t (&sDQ)[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceMatrix QQ0(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix QQ1(sQQ[1][tidz], Q1D, Q1D);
   DeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   DeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t u[2] = {0.0, 0.0};
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
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sDQ)[2][NBZ][MD1*MQ1],
                                    const DeviceTensor<4> &Y, // output
                                    const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, Q1D, D1D);
   ConstDeviceMatrix DQ0(sDQ[0][tidz], Q1D, D1D);
   ConstDeviceMatrix DQ1(sDQ[1][tidz], Q1D, D1D);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t u[2] = {0.0, 0.0};
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
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (&sX)[2][NBZ][MD1*MD1],
                                   real_t (&sDQ)[4][NBZ][MD1*MQ1])
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
         real_t u[2] = {0.0, 0.0};
         real_t v[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t Bx = B(dx,qx);
            const real_t Gx = G(dx,qx);
            const real_t x0 = X0(dx,dy);
            const real_t x1 = X1(dx,dy);
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
                                   const real_t (&sBG)[2][MQ1*MD1],
                                   const real_t (&sDQ)[4][NBZ][MD1*MQ1],
                                   real_t (&sQQ)[4][NBZ][MQ1*MQ1])
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
         real_t u[2] = {0.0, 0.0};
         real_t v[2] = {0.0, 0.0};
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t By = B(dy,qy);
            const real_t Gy = G(dy,qy);
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
MFEM_HOST_DEVICE inline void PullGrad(const int Q1D,
                                      const int qx, const int qy,
                                      const real_t (&sQQ)[4][NBZ][MQ1*MQ1],
                                      real_t *Jpr)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix X0GB(sQQ[0][tidz], Q1D, Q1D);
   ConstDeviceMatrix X0BG(sQQ[1][tidz], Q1D, Q1D);
   ConstDeviceMatrix X1GB(sQQ[2][tidz], Q1D, Q1D);
   ConstDeviceMatrix X1BG(sQQ[3][tidz], Q1D, Q1D);

   Jpr[0] = X0GB(qx,qy);
   Jpr[1] = X1GB(qx,qy);
   Jpr[2] = X0BG(qx,qy);
   Jpr[3] = X1BG(qx,qy);
}

/// Push 2D Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushGrad(const int Q1D,
                                      const int qx, const int qy,
                                      const real_t *A,
                                      real_t (&sQQ)[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0GB(sQQ[0][tidz], Q1D, Q1D);
   DeviceMatrix X0BG(sQQ[1][tidz], Q1D, Q1D);
   DeviceMatrix X1GB(sQQ[2][tidz], Q1D, Q1D);
   DeviceMatrix X1BG(sQQ[3][tidz], Q1D, Q1D);

   X0GB(qx,qy) = A[0];
   X1GB(qx,qy) = A[2];
   X0BG(qx,qy) = A[1];
   X1BG(qx,qy) = A[3];
}

/// 2D Transposed gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&GQ)[4][NBZ][MQ1*MQ1],
                                    real_t (&GD)[4][NBZ][MD1*MQ1])
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
         real_t u[2] = {0.0, 0.0};
         real_t v[2] = {0.0, 0.0};
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
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&GD)[4][NBZ][MD1*MQ1],
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
         real_t u[2] = {0.0, 0.0};
         real_t v[2] = {0.0, 0.0};
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
                                   const DeviceTensor<4, const real_t> &x,
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
                                   const DeviceTensor<4, const real_t> &x,
                                   real_t (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, D1D,D1D,D1D);
   LoadX(e,D1D,x,X);
}

/// Load 3D scalar input vector into shared memory, with comp & DeviceTensor
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D, const int c,
                                   const DeviceTensor<5, const real_t> &x,
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
                                   const DeviceTensor<5, const real_t> &x,
                                   real_t (&sm)[MD1*MD1*MD1])
{
   DeviceCube X(sm, D1D, D1D, D1D);
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
            real_t u = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t Bx = B(dx,qx);
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
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDD)[MD1*MD1*MD1],
                                   real_t (&sDDQ)[MD1*MD1*MQ1])
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
            real_t u = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t By = B(dy,qy);
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
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDQ)[MD1*MD1*MQ1],
                                   real_t (&sDQQ)[MD1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DDQ(sDDQ, Q1D, D1D, D1D);
   DeviceCube DQQ(sDQQ, Q1D, Q1D, D1D);
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
            real_t u = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
            {
               const real_t Bz = B(dz,qz);
               u += DQQ(dz,qy,qx) * Bz;
            }
            QQQ(qz,qy,qx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDQQ)[MD1*MQ1*MQ1],
                                   real_t (&sQQQ)[MQ1*MQ1*MQ1])
{
   ConstDeviceMatrix B(sB, D1D, Q1D);
   const DeviceCube DQQ(sDQQ, Q1D, Q1D, D1D);
   DeviceCube QQQ(sQQQ, Q1D, Q1D, Q1D);
   EvalZ(D1D,Q1D,B,DQQ,QQQ);
}

/// Pull 3D Scalar Evaluation
MFEM_HOST_DEVICE inline void PullEval(const int x, const int y, const int z,
                                      const DeviceCube &QQQ,
                                      real_t &X)
{
   X = QQQ(z,y,x);
}

template<int MQ1>
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (&sQQQ)[MQ1*MQ1*MQ1],
                                      real_t &X)
{
   const DeviceCube QQQ(sQQQ, Q1D, Q1D, Q1D);
   PullEval(x,y,z,QQQ,X);
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e, const int D1D,
                                   const DeviceTensor<5, const real_t> &X,
                                   real_t (*sm)[MD1*MD1*MD1])
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

/// 3D Vector Evaluation, 1/3 (only B)
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalX(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDD)[3][MD1*MD1*MD1],
                                   real_t (&sDDQ)[3][MD1*MD1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalY(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDDQ)[3][MD1*MD1*MQ1],
                                   real_t (&sDQQ)[3][MD1*MQ1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZ(const int D1D, const int Q1D,
                                   const real_t (&sB)[MQ1*MD1],
                                   const real_t (&sDQQ)[3][MD1*MQ1*MQ1],
                                   real_t (&sQQQ)[3][MQ1*MQ1*MQ1])
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

/// Pull 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PullEval(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (&sQQQ)[3][MQ1*MQ1*MQ1],
                                      real_t (&X)[3])
{
   ConstDeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   ConstDeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   ConstDeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);

   X[0] = XxBBB(x,y,z);
   X[1] = XyBBB(x,y,z);
   X[2] = XzBBB(x,y,z);
}

/// Push 3D Vector Evaluation
template<int MQ1>
MFEM_HOST_DEVICE inline void PushEval(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (&A)[3],
                                      real_t (&sQQQ)[3][MQ1*MQ1*MQ1])
{
   DeviceCube XxBBB(sQQQ[0], Q1D, Q1D, Q1D);
   DeviceCube XyBBB(sQQQ[1], Q1D, Q1D, Q1D);
   DeviceCube XzBBB(sQQQ[2], Q1D, Q1D, Q1D);

   XxBBB(x,y,z) = A[0];
   XyBBB(x,y,z) = A[1];
   XzBBB(x,y,z) = A[2];
}

/// 3D Transposed Vector Evaluation, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sQQQ)[3][MQ1*MQ1*MQ1],
                                    real_t (&sDQQ)[3][MD1*MQ1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sDQQ)[3][MD1*MQ1*MQ1],
                                    real_t (&sDDQ)[3][MD1*MD1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZt(const int D1D, const int Q1D,
                                    const real_t (&sB)[MQ1*MD1],
                                    const real_t (&sDDQ)[3][MD1*MD1*MQ1],
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

/// 3D Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const real_t (*sBG)[MQ1*MD1],
                                   const real_t (*sDDD)[MD1*MD1*MD1],
                                   real_t (*sDDQ)[MD1*MD1*MQ1])
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

/// 3D Gradient, 2/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const real_t (*sBG)[MQ1*MD1],
                                   const real_t (*sDDQ)[MD1*MD1*MQ1],
                                   real_t (*sDQQ)[MD1*MQ1*MQ1])
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

/// 3D Gradient, 3/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZ(const int D1D, const int Q1D,
                                   const real_t (*sBG)[MQ1*MD1],
                                   const real_t (*sDQQ)[MD1*MQ1*MQ1],
                                   real_t (*sQQQ)[MQ1*MQ1*MQ1])
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

/// Pull 3D Gradient
template<int MQ1>
MFEM_HOST_DEVICE inline void PullGrad(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t (*sQQQ)[MQ1*MQ1*MQ1],
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
template<int MQ1>
MFEM_HOST_DEVICE inline void PushGrad(const int Q1D,
                                      const int x, const int y, const int z,
                                      const real_t *A,
                                      real_t (&sQQQ)[9][MQ1*MQ1*MQ1])
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

/// 3D Transposed Gradient, 1/3
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradZt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sQQQ)[9][MQ1*MQ1*MQ1],
                                    real_t (&sDQQ)[9][MD1*MQ1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sDQQ)[9][MD1*MQ1*MQ1],
                                    real_t (&sDDQ)[9][MD1*MD1*MQ1])
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
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const real_t (&sBG)[2][MQ1*MD1],
                                    const real_t (&sDDQ)[9][MD1*MD1*MQ1],
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

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_HPP
