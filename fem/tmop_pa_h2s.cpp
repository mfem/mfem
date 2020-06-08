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

MFEM_HOST_DEVICE inline
void Invariant2_dM_2D(const double M[2][2], double dM[2][2])
{
   dM[0][0] =  M[1][1]; dM[1][0] = -M[0][1];
   dM[0][1] = -M[1][0]; dM[1][1] =  M[0][0];
}

MFEM_HOST_DEVICE inline
void Invariant2_dMdM_2D(int i, int j, double dMdM[2][2])
{
   dMdM[0][0] = dMdM[0][1] = dMdM[1][0] = dMdM[1][1] = 0.0;
   dMdM[1-j][1-i] = (i == j) ? 1.0 : -1.0;
}

MFEM_HOST_DEVICE inline
void Invariant1_dMdM_2D(const double *m,
                        const int i, const int j,
                        const int qx, const int qy, const int e,
                        const double weight, DeviceTensor<7,double> P)
{
   const double (*M)[2] = (const double (*)[2])(m);

   double dI[2][2];
   Invariant2_dM_2D(M, dI);
   const double ddet = dI[j][i];
   const double dfnorm2 = 2.0 * M[j][i];

   const double det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
   const double det2 = det * det;
   const double fnorm2 = kernels::FNorm2<2,2>(m);

   double dM[2][2] {{}}; dM[j][i] = 1.0;
   double ddI[2][2];
   Invariant2_dMdM_2D(i, j, ddI);

   for (int r = 0; r < 2; r++)
   {
      for (int c = 0; c < 2; c++)
      {
         P(r,c,i,j,qx,qy,e) =
            (det2 *
             (2.0 * ddet * M[c][r] + 2.0 * det * dM[c][r]
              - dfnorm2 * dI[c][r] - fnorm2 * ddI[c][r])
             - 2.0 * det * ddet *
             (2.0 * det * M[c][r] - fnorm2 * dI[c][r]) ) / (det2 * det2);
         P(r,c,i,j,qx,qy,e) *= weight;
      }
   }

}

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SetupGradPA_2D(const Vector &xe_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseMatrix &j_,
                           Vector &dp_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM);
   const auto X = Reshape(xe_.Read(), D1D, D1D, DIM, NE);
   auto dP = Reshape(dp_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int DIM = 2;
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B)[MD1]  = (double (*)[MD1])(s_BG[0]);
      double (*G)[MD1]  = (double (*)[MD1])(s_BG[1]);

      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      double (*Xx)[MD1]  = (double (*)[MD1])(s_X[0] + tidz);
      double (*Xy)[MD1]  = (double (*)[MD1])(s_X[1] + tidz);

      MFEM_SHARED double s_DQ[4][NBZ][MD1*MQ1];
      double (*XxB)[MQ1] = (double (*)[MQ1])(s_DQ[0] + tidz);
      double (*XxG)[MQ1] = (double (*)[MQ1])(s_DQ[1] + tidz);
      double (*XyB)[MQ1] = (double (*)[MQ1])(s_DQ[2] + tidz);
      double (*XyG)[MQ1] = (double (*)[MQ1])(s_DQ[3] + tidz);

      MFEM_SHARED double s_QQ[4][NBZ][MQ1*MQ1];
      double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[0] + tidz);
      double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1] + tidz);
      double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[2] + tidz);
      double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[3] + tidz);

      // Load X(x,y)
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dy][dx] = X(dx,dy,0,e);
            Xy[dy][dx] = X(dx,dy,1,e);
         }
      }
      // Load B and G matrices
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
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] {};
            double v[2] {};
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] {};
            double v[2] {};
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double ip_weight = W(qx,qy);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtr[4] = {J(0,0), J(1,0), J(0,1), J(1,1)};
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = ip_weight * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            const double GXx0h = Xx0[qy][qx];
            const double GXx1h = Xx1[qy][qx];
            const double GXy0h = Xy0[qy][qx];
            const double GXy1h = Xy1[qy][qx];
            const double Jpr[4] = {GXx0h, GXy0h, GXx1h, GXy1h};

            // Jpt = GX^T.DS = (GX^T.DSh).Jrt = GX.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            const double detJpt = Jpt[0]*Jpt[3] - Jpt[1]*Jpt[2];
            const double sign = detJpt >= 0.0 ? 1.0 : -1.0;

            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  const double w = sign * 0.5 * weight;
                  Invariant1_dMdM_2D(Jpt, i,j, qx,qy,e, w, dP);
               }
            }
         } // qx
      } // qy
   });
}

void TMOP_Integrator::AssembleGradPA_2D(const DenseMatrix &Jtr,
                                        const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &A = PA.A;

   switch (id)
   {
      case 0x21: return SetupGradPA_2D<2,1,1>(X,N,W,B,G,Jtr,A);
      case 0x22: return SetupGradPA_2D<2,2,1>(X,N,W,B,G,Jtr,A);
      case 0x23: return SetupGradPA_2D<2,3,1>(X,N,W,B,G,Jtr,A);
      case 0x24: return SetupGradPA_2D<2,4,1>(X,N,W,B,G,Jtr,A);
      case 0x25: return SetupGradPA_2D<2,5,1>(X,N,W,B,G,Jtr,A);
      case 0x26: return SetupGradPA_2D<2,6,1>(X,N,W,B,G,Jtr,A);

      case 0x31: return SetupGradPA_2D<3,1,1>(X,N,W,B,G,Jtr,A);
      case 0x32: return SetupGradPA_2D<3,2,1>(X,N,W,B,G,Jtr,A);
      case 0x33: return SetupGradPA_2D<3,3,1>(X,N,W,B,G,Jtr,A);
      case 0x34: return SetupGradPA_2D<3,4,1>(X,N,W,B,G,Jtr,A);
      case 0x35: return SetupGradPA_2D<3,5,1>(X,N,W,B,G,Jtr,A);
      case 0x36: return SetupGradPA_2D<3,6,1>(X,N,W,B,G,Jtr,A);

      case 0x41: return SetupGradPA_2D<4,1,1>(X,N,W,B,G,Jtr,A);
      case 0x42: return SetupGradPA_2D<4,2,1>(X,N,W,B,G,Jtr,A);
      case 0x43: return SetupGradPA_2D<4,3,1>(X,N,W,B,G,Jtr,A);
      case 0x44: return SetupGradPA_2D<4,4,1>(X,N,W,B,G,Jtr,A);
      case 0x45: return SetupGradPA_2D<4,5,1>(X,N,W,B,G,Jtr,A);
      case 0x46: return SetupGradPA_2D<4,6,1>(X,N,W,B,G,Jtr,A);

      case 0x51: return SetupGradPA_2D<5,1,1>(X,N,W,B,G,Jtr,A);
      case 0x52: return SetupGradPA_2D<5,2,1>(X,N,W,B,G,Jtr,A);
      case 0x53: return SetupGradPA_2D<5,3,1>(X,N,W,B,G,Jtr,A);
      case 0x54: return SetupGradPA_2D<5,4,1>(X,N,W,B,G,Jtr,A);
      case 0x55: return SetupGradPA_2D<5,5,1>(X,N,W,B,G,Jtr,A);
      case 0x56: return SetupGradPA_2D<5,6,1>(X,N,W,B,G,Jtr,A);

      default:
      {
         MFEM_VERIFY(D1D<=MAX_D1D && Q1D<=MAX_Q1D, "Max size error!");
         return SetupGradPA_2D(X,N,W,B,G,Jtr,A,D1D,Q1D);
      }
   }
}

} // namespace mfem
