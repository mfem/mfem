// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_MASS_PA_HPP
#define MFEM_BILININTEG_MASS_PA_HPP

#include "../config/config.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

namespace internal
{

template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApply2D_Element(const int e,
                           const int NE,
                           const double *b_,
                           const double *bt_,
                           const double *d_,
                           const double *x_,
                           double *y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto X = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   if (!ACCUMULATE)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx, dy, e) = 0.0;
         }
      }
   }

   constexpr int max_D1D = MAX_D1D;
   constexpr int max_Q1D = MAX_Q1D;
   double sol_xy[max_Q1D][max_Q1D];
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] = 0.0;
      }
   }
   for (int dy = 0; dy < D1D; ++dy)
   {
      double sol_x[max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         sol_x[qy] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double s = X(dx,dy,e);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] += B(qx,dx)* s;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         const double d2q = B(qy,dy);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] += d2q * sol_x[qx];
         }
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] *= D(qx,qy,e);
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      double sol_x[max_D1D];
      for (int dx = 0; dx < D1D; ++dx)
      {
         sol_x[dx] = 0.0;
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const double s = sol_xy[qy][qx];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] += Bt(dx,qx) * s;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         const double q2d = Bt(dy,qy);
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx,dy,e) += q2d * sol_x[dx];
         }
      }
   }
}

template<int T_D1D, int T_Q1D, int T_NBZ, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApply2D_Element(const int e,
                               const int NE,
                               const double *b_,
                               const double *d_,
                               const double *x_,
                               double *y_,
                               int d1d = 0,
                               int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto x = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   const int tidz = MFEM_THREAD_ID(z);

   MFEM_SHARED double BBt[MQ1*MD1];
   double (*B)[MD1] = (double (*)[MD1]) BBt;
   double (*Bt)[MQ1] = (double (*)[MQ1]) BBt;
   MFEM_SHARED double sm0[NBZ][MDQ*MDQ];
   MFEM_SHARED double sm1[NBZ][MDQ*MDQ];
   double (*X)[MD1] = (double (*)[MD1]) (sm0 + tidz);
   double (*DQ)[MQ1] = (double (*)[MQ1]) (sm1 + tidz);
   double (*QQ)[MQ1] = (double (*)[MQ1]) (sm0 + tidz);
   double (*QD)[MD1] = (double (*)[MD1]) (sm1 + tidz);


   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X[dy][dx] = x(dx,dy,e);
      }
   }
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B[q][dy] = b(q,dy);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double dq = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            dq += X[dy][dx] * B[qx][dx];
         }
         DQ[dy][qx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double qq = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            qq += DQ[dy][qx] * B[qy][dy];
         }
         QQ[qy][qx] = qq * D(qx, qy, e);
      }
   }
   MFEM_SYNC_THREAD;
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt[dy][q] = b(q,dy);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double dq = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            dq += QQ[qy][qx] * Bt[dx][qx];
         }
         QD[qy][dx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double dd = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            dd += (QD[qy][dx] * Bt[dy][qy]);
         }
         if (ACCUMULATE)
         {
            Y(dx, dy, e) += dd;
         }
         else
         {
            Y(dx, dy, e) = dd;
         }
      }
   }
}

template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApply3D_Element(const int e,
                           const int NE,
                           const double *b_,
                           const double *bt_,
                           const double *d_,
                           const double *x_,
                           double *y_,
                           const int d1d,
                           const int q1d)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = DeviceTensor<4,const double>(d_, Q1D, Q1D, Q1D, NE);
   auto X = DeviceTensor<4,const double>(x_, D1D, D1D, D1D, NE);
   auto Y = DeviceTensor<4,double>(y_, D1D, D1D, D1D, NE);

   if (!ACCUMULATE)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx, dy, dz, e) = 0.0;
            }
         }
      }
   }

   constexpr int max_D1D = MAX_D1D;
   constexpr int max_Q1D = MAX_Q1D;
   double sol_xyz[max_Q1D][max_Q1D][max_Q1D];
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] = 0.0;
         }
      }
   }
   for (int dz = 0; dz < D1D; ++dz)
   {
      double sol_xy[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[max_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] = 0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,dz,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += wy * sol_x[qx];
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         const double wz = B(qz,dz);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
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
            sol_xyz[qz][qy][qx] *= D(qx,qy,qz,e);
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      double sol_xy[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_xy[dy][dx] = 0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[max_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = sol_xyz[qz][qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] += wy * sol_x[dx];
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         const double wz = Bt(dz,qz);
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
            }
         }
      }
   }
}

template<int T_D1D, int T_Q1D, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApply3D_Element(const int e,
                               const int NE,
                               const double *b_,
                               const double *d_,
                               const double *x_,
                               double *y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   constexpr int D1D = T_D1D ? T_D1D : d1d;
   constexpr int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto d = DeviceTensor<4,const double>(d_, Q1D, Q1D, Q1D, NE);
   auto x = DeviceTensor<4,const double>(x_, D1D, D1D, D1D, NE);
   auto y = DeviceTensor<4,double>(y_, D1D, D1D, D1D, NE);

   MFEM_SHARED double sDQ[MQ1*MD1];
   double (*B)[MD1] = (double (*)[MD1]) sDQ;
   double (*Bt)[MQ1] = (double (*)[MQ1]) sDQ;
   MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
   MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
   double (*X)[MD1][MD1]   = (double (*)[MD1][MD1]) sm0;
   double (*DDQ)[MD1][MQ1] = (double (*)[MD1][MQ1]) sm1;
   double (*DQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm0;
   double (*QQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm1;
   double (*QQD)[MQ1][MD1] = (double (*)[MQ1][MD1]) sm0;
   double (*QDD)[MD1][MD1] = (double (*)[MD1][MD1]) sm1;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            X[dz][dy][dx] = x(dx,dy,dz,e);
         }
      }
      MFEM_FOREACH_THREAD(dx,x,Q1D)
      {
         B[dx][dy] = b(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += X[dz][dy][dx] * B[qx][dx];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ[dz][dy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ[dz][qy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ[qz][qy][qx] = u[qz] * d(qx,qy,qz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(di,y,D1D)
   {
      MFEM_FOREACH_THREAD(q,x,Q1D)
      {
         Bt[di][q] = b(q,di);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD[qz][qy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD[qz][dy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            if (ACCUMULATE)
            {
               y(dx,dy,dz,e) += u[dz];
            }
            else
            {
               y(dx,dy,dz,e) = u[dz];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace internal

} // namespace mfem

#endif
