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

#ifndef TMOP_PA_HPP
#define TMOP_PA_HPP

#include "../config/config.hpp"

#include "../general/forall.hpp"
#include "../general/cuda.hpp"

#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

namespace kernels
{

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
#define MFEM_DEVICE_FORCEINLINE __device__ __forceinline__
#else
#define MFEM_DEVICE_FORCEINLINE inline
#endif

/// Load B1d & G1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBG(const int D1D, const int Q1D,
                                    double s_BG[2][MQ1*MD1],
                                    const DeviceTensor<2, const double> b,
                                    const DeviceTensor<2, const double> g)
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*B)[MD1] = (double (*)[MD1])(s_BG+0);
   double (*G)[MD1] = (double (*)[MD1])(s_BG+1);
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B[q][d] = b(q,d);
            G[q][d] = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load Bt1d & Gt1d matrices into shared memory
template<int MD1, int MQ1>
MFEM_HOST_DEVICE inline void LoadBGt(const int D1D, const int Q1D,
                                     double sBG[2][MQ1*MD1],
                                     const DeviceTensor<2, const double> b,
                                     const DeviceTensor<2, const double> g)
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
   double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt[d][q] = b(q,d);
            Gt[d][q] = g(q,d);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D input vector into shared memory
template<int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadX(const int e,
                                   const int D1D,
                                   double s_X[2][NBZ][MD1*MD1],
                                   const DeviceTensor<4, const double> X)
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*Xx)[MD1] = (double (*)[MD1])(s_X[0] + tidz);
   double (*Xy)[MD1] = (double (*)[MD1])(s_X[1] + tidz);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         Xx[dy][dx] = X(dx,dy,0,e);
         Xy[dy][dx] = X(dx,dy,1,e);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D input vector into shared memory
template<int MD1>
MFEM_HOST_DEVICE inline void LoadX(const int e,
                                   const int D1D,
                                   double sm[3][MD1*MD1*MD1],
                                   const DeviceTensor<5, const double> X)
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
}

/// 2D Grad, stage 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradX(const int D1D, const int Q1D,
                                   const double s_BG[2][MQ1*MD1],
                                   const double s_X[2][NBZ][MD1*MD1],
                                   double s_DQ[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*B)[MD1] = (double (*)[MD1])(s_BG+0);
   double (*G)[MD1] = (double (*)[MD1])(s_BG+1);

   double (*Xx)[MD1]  = (double (*)[MD1])(s_X[0] + tidz);
   double (*Xy)[MD1]  = (double (*)[MD1])(s_X[1] + tidz);

   double (*XxB)[MQ1] = (double (*)[MQ1])(s_DQ[0] + tidz);
   double (*XxG)[MQ1] = (double (*)[MQ1])(s_DQ[1] + tidz);
   double (*XyB)[MQ1] = (double (*)[MQ1])(s_DQ[2] + tidz);
   double (*XyG)[MQ1] = (double (*)[MQ1])(s_DQ[3] + tidz);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double xx = Xx[dy][dx];
            const double xy = Xy[dy][dx];
            u[0] += B[qx][dx] * xx;
            v[0] += G[qx][dx] * xx;
            u[1] += B[qx][dx] * xy;
            v[1] += G[qx][dx] * xy;
         }
         XxB[dy][qx] = u[0];
         XxG[dy][qx] = v[0];
         XyB[dy][qx] = u[1];
         XyG[dy][qx] = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Grad, stage 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradY(const int D1D, const int Q1D,
                                   const double s_BG[2][MQ1*MD1],
                                   const double s_DQ[4][NBZ][MD1*MQ1],
                                   double s_QQ[4][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*B)[MD1] = (double (*)[MD1])(s_BG+0);
   double (*G)[MD1] = (double (*)[MD1])(s_BG+1);

   double (*XxB)[MQ1] = (double (*)[MQ1])(s_DQ[0] + tidz);
   double (*XxG)[MQ1] = (double (*)[MQ1])(s_DQ[1] + tidz);
   double (*XyB)[MQ1] = (double (*)[MQ1])(s_DQ[2] + tidz);
   double (*XyG)[MQ1] = (double (*)[MQ1])(s_DQ[3] + tidz);

   double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[0] + tidz);
   double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1] + tidz);
   double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[2] + tidz);
   double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[3] + tidz);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int dy = 0; dy < D1D; ++dy)
         {
            u[0] += XxG[dy][qx] * B[qy][dy];
            v[0] += XxB[dy][qx] * G[qy][dy];
            u[1] += XyG[dy][qx] * B[qy][dy];
            v[1] += XyB[dy][qx] * G[qy][dy];
         }
         Xx0[qy][qx] = u[0];
         Xx1[qy][qx] = v[0];
         Xy0[qy][qx] = u[1];
         Xy1[qy][qx] = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D GradXY(X) to Jpr
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline
const double *PullGradXY(const int qx, const int qy,
                         const double s_QQ[4][NBZ][MQ1*MQ1],
                         double *Jpr)
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[0] + tidz);
   double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1] + tidz);
   double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[2] + tidz);
   double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[3] + tidz);
   Jpr[0] = Xx0[qy][qx];
   Jpr[1] = Xy0[qy][qx];
   Jpr[2] = Xx1[qy][qx];
   Jpr[3] = Xy1[qy][qx];
   return Jpr;
}

/// Push 2D GradXY(X) to A
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline
void PushGradXY(const int qx, const int qy,
                double s_QQ[4][NBZ][MQ1*MQ1],
                const double *A)
{
   const int tidz = MFEM_THREAD_ID(z);
   double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[0] + tidz);
   double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1] + tidz);
   double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[2] + tidz);
   double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[3] + tidz);

   Xx0[qy][qx] = A[0];
   Xy0[qy][qx] = A[2];
   Xx1[qy][qx] = A[1];
   Xy1[qy][qx] = A[3];
}

/// 2D GradT, stage 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradYt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double GQ[4][NBZ][MQ1*MQ1],
                                    double GD[4][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);

   double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
   double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);

   double (*DQxB)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
   double (*DQxG)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);
   double (*DQyB)[MQ1] = (double (*)[MQ1])(GD[2] + tidz);
   double (*DQyG)[MQ1] = (double (*)[MQ1])(GD[3] + tidz);

   double (*QQx0)[MQ1] = (double (*)[MQ1])(GQ[0] + tidz);
   double (*QQx1)[MQ1] = (double (*)[MQ1])(GQ[1] + tidz);
   double (*QQy0)[MQ1] = (double (*)[MQ1])(GQ[2] + tidz);
   double (*QQy1)[MQ1] = (double (*)[MQ1])(GQ[3] + tidz);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u[0] += Gt[dx][qx] * QQx0[qy][qx];
            u[1] += Gt[dx][qx] * QQy0[qy][qx];

            v[0] += Bt[dx][qx] * QQx1[qy][qx];
            v[1] += Bt[dx][qx] * QQy1[qy][qx];
         }
         DQxB[dx][qy] = u[0];
         DQyB[dx][qy] = u[1];

         DQxG[dx][qy] = v[0];
         DQyG[dx][qy] = v[1];
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D GradT, stage 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void GradXt(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1*MD1],
                                    const double GD[4][NBZ][MD1*MQ1],
                                    mfem::DeviceTensor<4, double> Y,
                                    const int e)
{
   const int tidz = MFEM_THREAD_ID(z);

   double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
   double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);

   double (*DQxB)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
   double (*DQxG)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);
   double (*DQyB)[MQ1] = (double (*)[MQ1])(GD[2] + tidz);
   double (*DQyG)[MQ1] = (double (*)[MQ1])(GD[3] + tidz);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[2] = {0.0, 0.0};
         double v[2] = {0.0, 0.0};
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u[0] += DQxB[dx][qy] * Bt[dy][qy];
            u[1] += DQyB[dx][qy] * Bt[dy][qy];

            v[0] += DQxG[dx][qy] * Gt[dy][qy];
            v[1] += DQyG[dx][qy] * Gt[dy][qy];
         }
         Y(dx,dy,0,e) += u[0] + v[0];
         Y(dx,dy,1,e) += u[1] + v[1];
      }
   }
}

} // namespace kernels

} // namespace mfem

#endif // TMOP_PA_HPP
