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

#define MFEM_UNROLL(...)

#define MFEM_FORALL_2D(i,N,X,Y,BZ,...) \
        for (int i = 0; i < N; i++) { __VA_ARGS__; }

#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=0; i<N; i++)

void SmemPADiffusionApply2D_VLA(const int NE,
                                const int D1D,
                                const int Q1D,
                                const int NBZ,
                                const double b[Q1D][D1D],
                                const double g[Q1D][D1D],
                                const double D[NE][3][Q1D*Q1D],
                                const double x[NE][D1D][D1D],
                                double Y[NE][D1D][D1D])
{
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double sBG[2][Q1D*D1D];
      double (*B)[D1D] = (double (*)[D1D]) (sBG+0);
      double (*G)[D1D] = (double (*)[D1D]) (sBG+1);
      double (*Bt)[Q1D] = (double (*)[Q1D]) (sBG+0);
      double (*Gt)[Q1D] = (double (*)[Q1D]) (sBG+1);
      MFEM_SHARED double Xz[NBZ][D1D][D1D];
      MFEM_SHARED double GD[2][NBZ][D1D][Q1D];
      MFEM_SHARED double GQ[2][NBZ][D1D][Q1D];
      double (*X)[D1D] = (double (*)[D1D])(Xz + tidz);
      double (*DQ0)[D1D] = (double (*)[D1D])(GD[0] + tidz);
      double (*DQ1)[D1D] = (double (*)[D1D])(GD[1] + tidz);
      double (*QQ0)[D1D] = (double (*)[D1D])(GQ[0] + tidz);
      double (*QQ1)[D1D] = (double (*)[D1D])(GQ[1] + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x[e][dy][dx];
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][dy] = b[dy][q];
               G[q][dy] = g[dy][q];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_UNROLL(Q1D);
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = D[e][0][q];
            const double O12 = D[e][1][q];
            const double O22 = D[e][2][q];
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O12 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[dy][q] = b[dy][q];
               Gt[dy][q] = g[dy][q];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[dx][qx] * QQ0[qy][qx];
               v += Bt[dx][qx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[dy][qy];
               v += DQ1[qy][dx] * Gt[dy][qy];
            }
            Y[e][dy][dx] += (u + v);
         }
      }
   });
}
