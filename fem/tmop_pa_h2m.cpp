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

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0, int T_MAX = 0>
static void AddMultGradPA_Kernel_2D(const int NE,
                                    const Array<double> &b1d_,
                                    const Array<double> &g1d_,
                                    const DenseMatrix &Jtr,
                                    const Vector &p_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   constexpr int dim = 2;
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto b = Reshape(b1d_.Read(), Q1D, D1D);
   const auto g = Reshape(g1d_.Read(), Q1D, D1D);
   const auto J = Reshape(Jtr.Read(), DIM, DIM);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   const auto dP = Reshape(p_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B1d)[MD1]  = (double (*)[MD1])(s_BG+0);
      double (*G1d)[MD1]  = (double (*)[MD1])(s_BG+1);
      double (*B1dt)[MQ1] = (double (*)[MQ1])(s_BG+0);
      double (*G1dt)[MQ1] = (double (*)[MQ1])(s_BG+1);

      MFEM_SHARED double s_Xx[NBZ][MD1][MD1];
      double (*Xx)[MD1]  = (double (*)[MD1])(s_Xx + tidz);

      MFEM_SHARED double s_Xy[NBZ][MD1][MD1];
      double (*Xy)[MD1]  = (double (*)[MD1])(s_Xy + tidz);

      MFEM_SHARED double s_RDQ[4][NBZ][MD1*MQ1];
      double (*RxB)[MQ1] = (double (*)[MQ1])(s_RDQ[0] + tidz);
      double (*RxG)[MQ1] = (double (*)[MQ1])(s_RDQ[1] + tidz);
      double (*RyB)[MQ1] = (double (*)[MQ1])(s_RDQ[2] + tidz);
      double (*RyG)[MQ1] = (double (*)[MQ1])(s_RDQ[3] + tidz);

      MFEM_SHARED double s_CDQ[4][NBZ][MD1*MQ1];
      double (*CxB)[MQ1] = (double (*)[MQ1])(s_CDQ[0] + tidz);
      double (*CxG)[MQ1] = (double (*)[MQ1])(s_CDQ[1] + tidz);
      double (*CyB)[MQ1] = (double (*)[MQ1])(s_CDQ[2] + tidz);
      double (*CyG)[MQ1] = (double (*)[MQ1])(s_CDQ[3] + tidz);

      MFEM_SHARED double s_RQQ[4][NBZ][MQ1*MQ1];
      double (*Rx0)[MQ1] = (double (*)[MQ1])(s_RQQ[0] + tidz);
      double (*Rx1)[MQ1] = (double (*)[MQ1])(s_RQQ[1] + tidz);
      double (*Ry0)[MQ1] = (double (*)[MQ1])(s_RQQ[2] + tidz);
      double (*Ry1)[MQ1] = (double (*)[MQ1])(s_RQQ[3] + tidz);

      MFEM_SHARED double s_YQQ[4][NBZ][MQ1*MQ1];
      double (*Cx0)[MQ1] = (double (*)[MQ1])(s_YQQ[0] + tidz);
      double (*Cx1)[MQ1] = (double (*)[MQ1])(s_YQQ[1] + tidz);
      double (*Cy0)[MQ1] = (double (*)[MQ1])(s_YQQ[2] + tidz);
      double (*Cy1)[MQ1] = (double (*)[MQ1])(s_YQQ[3] + tidz);

      // Load R(x,y) and X(x,y)
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dy][dx] = X(dx,dy,0,e);
            Xy[dy][dx] = X(dx,dy,1,e);
         }
      }
      // Load B1d and G1d matrices
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B1d[q][d] = b(q,d);
               G1d[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] = {0.0, 0.0};
            double v[2] = {0.0, 0.0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double rx = Xx[dy][dx];
               const double ry = Xy[dy][dx];
               u[0] += B1d[qx][dx] * rx;
               v[0] += G1d[qx][dx] * rx;
               u[1] += B1d[qx][dx] * ry;
               v[1] += G1d[qx][dx] * ry;
            }
            RxB[dy][qx] = u[0];
            RxG[dy][qx] = v[0];
            RyB[dy][qx] = u[1];
            RyG[dy][qx] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] = {0.0, 0.0};
            double v[2] = {0.0, 0.0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               u[0] += RxG[dy][qx] * B1d[qy][dy];
               v[0] += RxB[dy][qx] * G1d[qy][dy];
               u[1] += RyG[dy][qx] * B1d[qy][dy];
               v[1] += RyB[dy][qx] * G1d[qy][dy];
            }
            Rx0[qy][qx] = u[0];
            Rx1[qy][qx] = v[0];
            Ry0[qy][qx] = u[1];
            Ry1[qy][qx] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double A[4], B[4], C[4];

            // Jrt = Jtr^{-1}
            double Jrt[4];
            const double Jtr_p[4] = { J(0,0), J(1,0), J(0,1), J(1,1) };
            kernels::CalcInverse<2>(Jtr_p, Jrt);

            const double GRx0h = Rx0[qy][qx];
            const double GRx1h = Rx1[qy][qx];
            const double GRy0h = Ry0[qy][qx];
            const double GRy1h = Ry1[qy][qx];
            const double hX[4] = {GRx0h, GRy0h, GRx1h, GRy1h};

            // A = X^T . Jrt
            kernels::Mult(2,2,2, hX, Jrt, A);

            // B = A : dP
            for (int r = 0; r < dim; r++)
            {
               for (int c = 0; c < dim; c++)
               {
                  B[r+2*c] = 0.0;
                  for (int i = 0; i < dim; i++)
                  {
                     for (int j = 0; j < dim; j++)
                     {
                        B[r+2*c] += dP(i,j,r,c,qx,qy,e) * A[i+2*j];
                     }
                  }
               }
            }

            // C = Jrt . B
            kernels::MultABt(2,2,2, Jrt, B, C);
            Cx0[qy][qx] = C[0];
            Cy0[qy][qx] = C[2];
            Cx1[qy][qx] = C[1];
            Cy1[qy][qx] = C[3];
         }
      }

      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B1dt[d][q] = b(q,d);
               G1dt[d][q] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[2] = {0.0, 0.0};
            double v[2] = {0.0, 0.0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u[0] += G1dt[dx][qx] * Cx0[qy][qx];
               v[0] += B1dt[dx][qx] * Cx1[qy][qx];
               u[1] += G1dt[dx][qx] * Cy0[qy][qx];
               v[1] += B1dt[dx][qx] * Cy1[qy][qx];
            }
            CxB[dx][qy] = u[0];
            CxG[dx][qy] = v[0];
            CyB[dx][qy] = u[1];
            CyG[dx][qy] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[2] = {0.0, 0.0};
            double v[2] = {0.0, 0.0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u[0] += CxB[dx][qy] * B1dt[dy][qy];
               v[0] += CxG[dx][qy] * G1dt[dy][qy];
               u[1] += CyB[dx][qy] * B1dt[dy][qy];
               v[1] += CyG[dx][qy] * G1dt[dy][qy];
            }
            Y(dx,dy,0,e) += u[0] + v[0];
            Y(dx,dy,1,e) += u[1] + v[1];
         }
      }
   });
}

void TMOP_Integrator::AddMultGradPA_2D(const Vector &X, const Vector &R,
                                       Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseMatrix &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &A = PA.A;

   if (!PA.setup)
   {
      PA.setup = true;
      AssembleGradPA_2D(X);
   }

   switch (id)
   {
      case 0x21: return AddMultGradPA_Kernel_2D<2,1,1>(N,B,G,J,A,R,C);
      case 0x22: return AddMultGradPA_Kernel_2D<2,2,1>(N,B,G,J,A,R,C);
      case 0x23: return AddMultGradPA_Kernel_2D<2,3,1>(N,B,G,J,A,R,C);
      case 0x24: return AddMultGradPA_Kernel_2D<2,4,1>(N,B,G,J,A,R,C);
      case 0x25: return AddMultGradPA_Kernel_2D<2,5,1>(N,B,G,J,A,R,C);
      case 0x26: return AddMultGradPA_Kernel_2D<2,6,1>(N,B,G,J,A,R,C);

      case 0x31: return AddMultGradPA_Kernel_2D<3,1,1>(N,B,G,J,A,R,C);
      case 0x32: return AddMultGradPA_Kernel_2D<3,2,1>(N,B,G,J,A,R,C);
      case 0x33: return AddMultGradPA_Kernel_2D<3,3,1>(N,B,G,J,A,R,C);
      case 0x34: return AddMultGradPA_Kernel_2D<3,4,1>(N,B,G,J,A,R,C);
      case 0x35: return AddMultGradPA_Kernel_2D<3,5,1>(N,B,G,J,A,R,C);
      case 0x36: return AddMultGradPA_Kernel_2D<3,6,1>(N,B,G,J,A,R,C);

      case 0x41: return AddMultGradPA_Kernel_2D<4,1,1>(N,B,G,J,A,R,C);
      case 0x42: return AddMultGradPA_Kernel_2D<4,2,1>(N,B,G,J,A,R,C);
      case 0x43: return AddMultGradPA_Kernel_2D<4,3,1>(N,B,G,J,A,R,C);
      case 0x44: return AddMultGradPA_Kernel_2D<4,4,1>(N,B,G,J,A,R,C);
      case 0x45: return AddMultGradPA_Kernel_2D<4,5,1>(N,B,G,J,A,R,C);
      case 0x46: return AddMultGradPA_Kernel_2D<4,6,1>(N,B,G,J,A,R,C);

      case 0x51: return AddMultGradPA_Kernel_2D<5,1,1>(N,B,G,J,A,R,C);
      case 0x52: return AddMultGradPA_Kernel_2D<5,2,1>(N,B,G,J,A,R,C);
      case 0x53: return AddMultGradPA_Kernel_2D<5,3,1>(N,B,G,J,A,R,C);
      case 0x54: return AddMultGradPA_Kernel_2D<5,4,1>(N,B,G,J,A,R,C);
      case 0x55: return AddMultGradPA_Kernel_2D<5,5,1>(N,B,G,J,A,R,C);
      case 0x56: return AddMultGradPA_Kernel_2D<5,6,1>(N,B,G,J,A,R,C);

      default:
      {
         constexpr int T_MAX = 8;
         MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");
         return AddMultGradPA_Kernel_2D<0,0,0,T_MAX>(N,B,G,J,A,R,C,D1D,Q1D);
      }
   }
}

} // namespace mfem
