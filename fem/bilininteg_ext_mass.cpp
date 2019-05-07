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

const int MAX_Q1D = 10;
const int MAX_NBZ = 10;

template<const int T_D1D = 0, const int T_Q1D = 0, const int T_NBZ = 0> static
bool SmemPAMassApply2D(const int NE,
                       const double* _B,
                       const double* _Bt,
                       const double* _op,
                       const double* _x,
                       double* _y,
                       const int d1d = 0,
                       const int q1d = 0,
                       const int nbz = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : MAX_NBZ;
   const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   MFEM_VERIFY(NBZ <= MAX_NBZ, "");
   const DeviceMatrix B(_B, Q1D, D1D);
   const DeviceMatrix Bt(_Bt, D1D, Q1D);
   const DeviceTensor<3> op(_op, Q1D, Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL_XYZ(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
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
               t += B(qy,dy)*sol_x[dy][qx];
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
            y(dx,dy,e) = t;
         }
      }
   });
   return true;
}

bool SmemPAMassApply(const int dim,
                     const int D1D,
                     const int Q1D,
                     const int NE,
                     const double *B,
                     const double *Bt,
                     const double *op,
                     const double *x,
                     double *y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPAMassApply2D<2,2,8>(NE, B, Bt, op, x, y);
         case 0x33: return SmemPAMassApply2D<3,3,8>(NE, B, Bt, op, x, y);
         case 0x24: return SmemPAMassApply2D<2,4,8>(NE, B, Bt, op, x, y);
         case 0x34: return SmemPAMassApply2D<3,4,8>(NE, B, Bt, op, x, y);
         case 0x35: return SmemPAMassApply2D<3,5,6>(NE, B, Bt, op, x, y);
         case 0x36: return SmemPAMassApply2D<3,6,6>(NE, B, Bt, op, x, y);
         case 0x44: return SmemPAMassApply2D<4,4,2>(NE, B, Bt, op, x, y);
         case 0x45: return SmemPAMassApply2D<4,5,2>(NE, B, Bt, op, x, y);
         case 0x46: return SmemPAMassApply2D<4,6,2>(NE, B, Bt, op, x, y);
         case 0x48: return SmemPAMassApply2D<4,8,2>(NE, B, Bt, op, x, y);
         case 0x55: return SmemPAMassApply2D<5,5,2>(NE, B, Bt, op, x, y);
         case 0x58: return SmemPAMassApply2D<5,8,2>(NE, B, Bt, op, x, y);
         default:   return SmemPAMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D, 1);
      }
   }
   printf("\n\033[33m[SmemPAMassApply] Skipped D1D=%d, Q1D=%d\033[m", D1D, Q1D);
   return false;
}

} // namespace mfem
