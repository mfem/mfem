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

#define QUAD_2D_ID(X, Y) (X + ((Y) * Q1D))

const int MAX_Q1D = 10;
const int MAX_D1D = 10;

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, const int T_NBZ = 0> static
bool SmemPADiffusionApply2D(const int NE,
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
   MFEM_FORALL_XYZ(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];
      MFEM_SHARED double Bt[MD1][MQ1];
      MFEM_SHARED double Gt[MD1][MQ1];
      MFEM_SHARED double bufX[NBZ][MQ1][MQ1];
      MFEM_SHARED double buf1[NBZ][2][MQ1][MQ1];
      MFEM_SHARED double buf2[NBZ][2][MQ1][MQ1];
      double (*X)[MQ1] = (double (*)[MQ1])(bufX + threadIdx(z));
      double (*grad1)[2][MQ1] = (double (*)[2][MQ1])(buf1 + threadIdx(z));
      double (*gradX)[2][MQ1] = (double (*)[2][MQ1])(buf2 + threadIdx(z));
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
               Bt[dx][qx] = bt(dx,qx);
               Gt[dx][qx] = gt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double b = 0.0;
            double g = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double x = X[dy][dx];
               b += G[qx][dx] * x;
               g += B[qx][dx] * x;
            }
            grad1[0][dy][qx] = b;
            grad1[1][dy][qx] = g;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double b = 0.0;
            double g = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               b += grad1[0][dy][qx] * B[qy][dy];
               g += grad1[1][dy][qx] * G[qy][dy];
            }
            gradX[0][qy][qx] = b;
            gradX[1][qy][qx] = g;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            const int q = QUAD_2D_ID(qx, qy);
            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);
            const double gX = gradX[0][qy][qx];
            const double gY = gradX[1][qy][qx];
            gradX[0][qy][qx] = (O11 * gX) + (O12 * gY);
            gradX[1][qy][qx] = (O12 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double b = 0.0;
            double g = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               b += Gt[dx][qx] * gradX[0][qy][qx];
               g += Bt[dx][qx] * gradX[1][qy][qx];
            }
            grad1[0][qy][dx] = b;
            grad1[1][qy][dx] = g;
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double b = 0.0;
            double g = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               b += grad1[0][qy][dx] * Bt[dy][qy];
               g += grad1[1][qy][dx] * Gt[dy][qy];
            }
            y(dx,dy,e) += b + g;
         }
      }
   });
   return true;
}

bool SmemPADiffusionApply(const int dim,
                          const int D1D,
                          const int Q1D,
                          const int NE,
                          const double* B,
                          const double* G,
                          const double* Bt,
                          const double* Gt,
                          const double* op,
                          const double* x,
                          double* y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPADiffusionApply2D<2,2,8>(NE,B,G,Bt,Gt,op,x,y);
            //case 0x33: return SmemPADiffusionApply2D<3,3,8>(NE,B,G,Bt,Gt,op,x,y);
            //case 0x44: return SmemPADiffusionApply2D<4,4,2>(NE,B,G,Bt,Gt,op,x,y);
            //case 0x55: return SmemPADiffusionApply2D<5,5,2>(NE,B,G,Bt,Gt,op,x,y);
         default:   return SmemPADiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   printf("\n\033[33m[SmemPADiffusionApply] Skipped D1D=%d, Q1D=%d\033[m", D1D, Q1D);
   return false;
}

} // namespace mfem
