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

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static bool PAMassApply2D(const int NE,
                          const double* _B,
                          const double* _Bt,
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
   const DeviceMatrix B(_B, Q1D, D1D);
   const DeviceMatrix Bt(_Bt, D1D, Q1D);
   const DeviceTensor<3> op(_op, Q1D, Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      double sol_xy[MAX_Q1D][MAX_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
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
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[MAX_D1D];
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
               y(dx,dy,e) += q2d * sol_x[dx];
            }
         }
      }
   });
   return true;
}

template<const int T_D1D = 0,
         const int T_Q1D = 0,
         const int T_NBZ = 0>
static bool SmemPAMassApply2D(const int NE,
                              const double* _b,
                              const double* _bt,
                              const double* _op,
                              const double* _x,
                              double* _y,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(_b, Q1D, D1D);
   const DeviceMatrix Bt(_bt, D1D, Q1D);
   const DeviceTensor<3> op(_op, Q1D, Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int NBZ = T_NBZ ? T_NBZ : 1;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double buf1[NBZ][MQ1][MQ1];
      MFEM_SHARED double buf2[NBZ][MQ1][MQ1];
      MFEM_SHARED double matrix[MQ1][MQ1];
      double (*sol_x)[MQ1] = (double (*)[MQ1])(buf2 + threadIdx(z));
      double (*sol_xy)[MQ1] = (double (*)[MQ1])(buf1 + threadIdx(z));
      double (*input)[MQ1] = (double (*)[MQ1])(buf1 + threadIdx(z));
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            input[dy][dx] = x(dx,dy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               matrix[dx][qx] = B(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double t = 0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               t += matrix[dx][qx]*input[dy][dx];
            }
            sol_x[dy][qx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double t = 0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               t += matrix[dy][qy]*sol_x[dy][qx];
            }
            sol_xy[qy][qx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int qx = threadIdx(y); qx < Q1D; qx += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               matrix[qx][dx] = Bt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double t = 0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t += matrix[qx][dx] * sol_xy[qy][qx];
            }
            sol_x[qy][dx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double t = 0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               t += matrix[qy][dy] * sol_x[qy][dx];
            }
            y(dx,dy,e) += t;
         }
      }
   });
   return true;
}

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static bool PAMassApply3D(const int NE,
                          const double* _B,
                          const double* _Bt,
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
   const DeviceMatrix B(_B, Q1D, D1D);
   const DeviceMatrix Bt(_Bt, D1D, Q1D);
   const DeviceTensor<4> op(_op, Q1D, Q1D, Q1D,NE);
   const DeviceTensor<4> x(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      double sol_xyz[MAX_Q1D][MAX_Q1D][MAX_Q1D];
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
         double sol_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
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
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double sol_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[MAX_D1D];
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
                  y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
   return true;
}

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static bool SmemPAMassApply3D(const int NE,
                              const double* _b,
                              const double* _bt,
                              const double* _op,
                              const double* _x,
                              double *_y,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(_b, Q1D, D1D);
   const DeviceMatrix Bt(_bt, D1D, Q1D);
   const DeviceTensor<4> op(_op, Q1D, Q1D, Q1D, NE);
   const DeviceTensor<4> X(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> Y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double buf1[MQ1][MQ1][MQ1];
      MFEM_SHARED double buf2[MQ1][MQ1][MQ1];
      MFEM_SHARED double matrix[MQ1][MQ1];
      double (*sol_xyz)[MQ1][MQ1] = buf1;
      double (*sol_xy)[MQ1][MQ1] = buf2;
      double (*sol_x)[MQ1][MQ1] = buf1;
      double (*input)[MQ1][MQ1] = buf2;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               input[dz][dy][dx] = X(dx,dy,dz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               matrix[dx][qx] = B(qx,dx);
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
               double t = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  t += matrix[dx][qx] * input[dz][dy][dx];
               }
               sol_x[dz][dy][qx] = t;
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
               double t = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  t += matrix[dy][qy] * sol_x[dz][dy][qx];
               }
               sol_xy[dz][qy][qx] = t;
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
               double t = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  t += matrix[dz][qz] * sol_xy[dz][qy][qx];
               }
               sol_xyz[qz][qy][qx] = t;
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
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int qx = threadIdx(y); qx < Q1D; qx += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               matrix[qx][dx] = Bt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      sol_x = buf2;
      sol_xy = buf1;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double t = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  t += matrix[qx][dx] * sol_xyz[qz][qy][qx];
               }
               sol_x[qz][qy][dx] = t;
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
               double t = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  t += matrix[qy][dy] * sol_x[qz][qy][dx];
               }
               sol_xy[qz][dy][dx] = t;
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
               double t = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  t += matrix[qz][dz] * sol_xy[qz][dy][dx];
               }
               Y(dx,dy,dz,e) += t;
            }
         }
      }
   });
   return true;
}

/// PA diffusion kernels
bool PAMassApplyKernel(const int dim, const int D1D,
                       const int Q1D, const int NE,
                       const double* B, const double* Bt,
                       const double* op,
                       const double* x, double* y)
{
   if (Device::Allows(Backend::RAJA_CUDA)){
      if (dim == 2)
      {
         switch ((D1D << 4 ) | Q1D)
         {
         case 0x22: return PAMassApply2D<2,2>(NE, B, Bt, op, x, y);
         case 0x33: return PAMassApply2D<3,3>(NE, B, Bt, op, x, y);
         case 0x44: return PAMassApply2D<4,4>(NE, B, Bt, op, x, y);
         case 0x55: return PAMassApply2D<5,5>(NE, B, Bt, op, x, y);
         case 0x66: return PAMassApply2D<6,6>(NE, B, Bt, op, x, y);
         case 0x77: return PAMassApply2D<7,7>(NE, B, Bt, op, x, y);
         case 0x88: return PAMassApply2D<8,8>(NE, B, Bt, op, x, y);
         case 0x99: return PAMassApply2D<9,9>(NE, B, Bt, op, x, y);
         default:   return PAMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D);
         }
      }
      if (dim == 3)
      {
         switch ((D1D << 4 ) | Q1D)
         {
         case 0x23: return PAMassApply3D<2,3>(NE, B, Bt, op, x, y);
         case 0x34: return PAMassApply3D<3,4>(NE, B, Bt, op, x, y);
         case 0x45: return PAMassApply3D<4,5>(NE, B, Bt, op, x, y);
         case 0x56: return PAMassApply3D<5,6>(NE, B, Bt, op, x, y);
         case 0x67: return PAMassApply3D<6,7>(NE, B, Bt, op, x, y);
         case 0x78: return PAMassApply3D<7,8>(NE, B, Bt, op, x, y);
         case 0x89: return PAMassApply3D<8,9>(NE, B, Bt, op, x, y);
         default:   return PAMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D);
         }
      }
      return false;
   }
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPAMassApply2D<2,2,8>(NE, B, Bt, op, x, y);
         case 0x33: return SmemPAMassApply2D<3,3,8>(NE, B, Bt, op, x, y);
         case 0x44: return SmemPAMassApply2D<4,4,2>(NE, B, Bt, op, x, y);
         case 0x55: return SmemPAMassApply2D<5,5,2>(NE, B, Bt, op, x, y);
         case 0x66: return SmemPAMassApply2D<6,6,2>(NE, B, Bt, op, x, y);
         case 0x77: return SmemPAMassApply2D<7,7,2>(NE, B, Bt, op, x, y);
         case 0x88: return SmemPAMassApply2D<8,8,1>(NE, B, Bt, op, x, y);
         case 0x99: return SmemPAMassApply2D<9,9,1>(NE, B, Bt, op, x, y);
         default:   return PAMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPAMassApply3D<2,3>(NE, B, Bt, op, x, y);
         case 0x34: return SmemPAMassApply3D<3,4>(NE, B, Bt, op, x, y);
         case 0x45: return SmemPAMassApply3D<4,5>(NE, B, Bt, op, x, y);
         case 0x56: return SmemPAMassApply3D<5,6>(NE, B, Bt, op, x, y);
         case 0x67: return SmemPAMassApply3D<6,7>(NE, B, Bt, op, x, y);
         case 0x78: return SmemPAMassApply3D<7,8>(NE, B, Bt, op, x, y);
         case 0x89: return SmemPAMassApply3D<8,9>(NE, B, Bt, op, x, y);
         default:   return PAMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
   }
   return false;
}

} // namespace mfem
