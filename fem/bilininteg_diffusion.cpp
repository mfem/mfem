// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

// PA Diffusion Apply 2D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0>
static bool PADiffusionApply2D(const int NE,
                               const double* b,
                               const double* g,
                               const double* bt,
                               const double* gt,
                               const double* _op,
                               const double* _x,
                               double* _y,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(b, Q1D, D1D);
   const DeviceMatrix G(g, Q1D, D1D);
   const DeviceMatrix Bt(bt, D1D, Q1D);
   const DeviceMatrix Gt(gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 3, Q1D*Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      double grad[MAX_Q1D][MAX_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + (qy*Q1D);

            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[MAX_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
   return true;
}

// Shared memory PA Diffusion Apply 2D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0,
         const int T_NBZ = 0>
static bool SmemPADiffusionApply2D(const int NE,
                                   const double* _b,
                                   const double* _g,
                                   const double* _bt,
                                   const double* _gt,
                                   const double* _op,
                                   const double* _x,
                                   double* _y,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : 1;
   const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   const int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const DeviceMatrix b(_b, Q1D, D1D);
   const DeviceMatrix g(_g, Q1D, D1D);
   const DeviceMatrix bt(_bt, D1D, Q1D);
   const DeviceMatrix gt(_gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 3, Q1D*Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int NBZ = T_NBZ ? T_NBZ : 1;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      const int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];
      double (*Bt)[MQ1] = (double (*)[MQ1]) B;
      double (*Gt)[MQ1] = (double (*)[MQ1]) G;
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + threadIdx(z));
      double (*DQ0)[MD1] = (double (*)[MD1])(GD[0] + threadIdx(z));
      double (*DQ1)[MD1] = (double (*)[MD1])(GD[1] + threadIdx(z));
      double (*QQ0)[MD1] = (double (*)[MD1])(GQ[0] + threadIdx(z));
      double (*QQ1)[MD1] = (double (*)[MD1])(GQ[1] + threadIdx(z));
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               B[qx][dx] = b(qx,dx);
               G[qx][dx] = g(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double coords = X[dy][dx];
               u += B[qx][dx] * coords;
               v += G[qx][dx] * coords;
            }
            DQ0[dy][qx] = u;
            DQ1[dy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               u += DQ1[dy][qx] * B[qy][dy];
               v += DQ0[dy][qx] * G[qy][dy];
            }
            QQ0[qy][qx] = u;
            QQ1[qy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O12 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[qx][dx] * QQ0[qy][qx];
               v += Bt[qx][dx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[qy][dy];
               v += DQ1[qy][dx] * Gt[qy][dy];
            }
            y(dx,dy,e) += (u + v);
         }
      }
   });
   return true;
}

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static bool PADiffusionApply3D(const int NE,
                               const double* b,
                               const double* g,
                               const double* bt,
                               const double* gt,
                               const double* _op,
                               const double* _x,
                               double* _y,
                               int d1d = 0,
                               int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(b, Q1D, D1D);
   const DeviceMatrix G(g, Q1D, D1D);
   const DeviceMatrix Bt(bt, D1D, Q1D);
   const DeviceMatrix Gt(gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 6, Q1D*Q1D*Q1D, NE);
   const DeviceTensor<4> x(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      double grad[MAX_Q1D][MAX_Q1D][MAX_Q1D][4];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[MAX_Q1D][MAX_Q1D][4];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + qy*Q1D + qz*Q1D*Q1D;
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[MAX_D1D][MAX_D1D][4];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double gradX[MAX_D1D][4];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
   return true;
}

// Shared memory PA Diffusion Apply 3D kernel
template<int T_D1D = 0,
         int T_Q1D = 0>
static bool SmemPADiffusionApply3D(const int NE,
                                   const double* _b,
                                   const double* _g,
                                   const double* _bt,
                                   const double* _gt,
                                   const double* _op,
                                   const double* _x,
                                   double* _y,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   const int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const DeviceMatrix b(_b, Q1D, D1D);
   const DeviceMatrix g(_g, Q1D, D1D);
   const DeviceMatrix bt(_bt, D1D, Q1D);
   const DeviceMatrix gt(_gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 6, Q1D*Q1D*Q1D, NE);
   const DeviceTensor<4> x(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      const int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int MDQ = MQ1 > MD1 ? MQ1 : MD1;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               B[qx][dx] = b(qx,dx);
               G[qx][dx] = g(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double coords = X[dz][dy][dx];
                  u += coords * B[qx][dx];
                  v += coords * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ1[dz][dy][qx] * B[qy][dy];
                  v += DDQ0[dz][dy][qx] * G[qy][dy];
                  w += DDQ0[dz][dy][qx] * B[qy][dy];
               }
               DQQ0[dz][qy][qx] = u;
               DQQ1[dz][qy][qx] = v;
               DQQ2[dz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               QQQ0[qz][qy][qx] = u;
               QQQ1[qz][qy][qx] = v;
               QQQ2[qz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               const int q = qx + ((qy*Q1D) + (qz*Q1D*Q1D));
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gX = QQQ0[qz][qy][qx];
               const double gY = QQQ1[qz][qy][qx];
               const double gZ = QQQ2[qz][qy][qx];
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1[qz][qy][qx] = (O12*gX) + (O22*gY) + (O23*gZ);
               QQQ2[qz][qy][qx] = (O13*gX) + (O23*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               Bt[dx][qx] = bt(dx,qx);
               Gt[dx][qx] = gt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ0[qz][qy][qx] * Gt[dx][qx];
                  v += QQQ1[qz][qy][qx] * Bt[dx][qx];
                  w += QQQ2[qz][qy][qx] * Bt[dx][qx];
               }
               QQD0[qz][qy][dx] = u;
               QQD1[qz][qy][dx] = v;
               QQD2[qz][qy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  v += QQD1[qz][qy][dx] * Gt[dy][qy];
                  w += QQD2[qz][qy][dx] * Bt[dy][qy];
               }
               QDD0[qz][dy][dx] = u;
               QDD1[qz][dy][dx] = v;
               QDD2[qz][dy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  v += QDD1[qz][dy][dx] * Bt[dz][qz];
                  w += QDD2[qz][dy][dx] * Gt[dz][qz];
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
   return true;
}

/// PA diffusion kernels
bool PADiffusionApplyKernel(const int dim, const int D1D,
                            const int Q1D, const int NE,
                            const double* B, const double* G,
                            const double* Bt, const double* Gt,
                            const double* op,
                            const double* x, double* y)
{
   if (Device::Allows(Backend::RAJA_CUDA))
   {
      if (dim == 2)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            case 0x22: return PADiffusionApply2D<2,2>(NE,B,G,Bt,Gt,op,x,y);
            case 0x33: return PADiffusionApply2D<3,3>(NE,B,G,Bt,Gt,op,x,y);
            case 0x44: return PADiffusionApply2D<4,4>(NE,B,G,Bt,Gt,op,x,y);
            case 0x55: return PADiffusionApply2D<5,5>(NE,B,G,Bt,Gt,op,x,y);
            case 0x66: return PADiffusionApply2D<6,6>(NE,B,G,Bt,Gt,op,x,y);
            case 0x77: return PADiffusionApply2D<7,7>(NE,B,G,Bt,Gt,op,x,y);
            case 0x88: return PADiffusionApply2D<8,8>(NE,B,G,Bt,Gt,op,x,y);
            case 0x99: return PADiffusionApply2D<9,9>(NE,B,G,Bt,Gt,op,x,y);
            default:   return PADiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
         }
      }
      if (dim == 3)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            case 0x23: return PADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
            case 0x34: return PADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
            case 0x45: return PADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
            case 0x56: return PADiffusionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
            case 0x67: return PADiffusionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
            case 0x78: return PADiffusionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
            case 0x89: return PADiffusionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
            default:   return PADiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
         }
      }
      return false;
   }
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPADiffusionApply2D<2,2,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return SmemPADiffusionApply2D<3,3,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return SmemPADiffusionApply2D<4,4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return SmemPADiffusionApply2D<5,5,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return SmemPADiffusionApply2D<6,6,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return SmemPADiffusionApply2D<7,7,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return SmemPADiffusionApply2D<8,8,1>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return SmemPADiffusionApply2D<9,9,1>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return SmemPADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return SmemPADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return SmemPADiffusionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return SmemPADiffusionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return SmemPADiffusionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return SmemPADiffusionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   return false;
}

} // namespace mfem
