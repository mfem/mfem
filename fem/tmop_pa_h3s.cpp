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
#define MFEM_DBG_COLOR 212
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

// *****************************************************************************
// I1b = I1 / det(M)^(2/3).
MFEM_HOST_DEVICE inline
double Dim3Invariant1b(const double *M)
{
   const double fnorm2 = kernels::FNorm2<3,3>(M);
   const double det = fabs(kernels::Det<3>(M));
   const double sign = kernels::Det<3>(M) >= 0.0 ? 1.0 : -1.0;
   return sign * fnorm2 / pow(det, 2.0/3.0);
}

// *****************************************************************************
// I2 = |adj(M)|^2 / det(M)^(4/3).
MFEM_HOST_DEVICE inline
double Dim3Invariant2b(const double *M)
{
   double Madj[9];
   kernels::CalcAdjugate<3>(M, Madj);
   const double fnorm2 = kernels::FNorm2<3,3>(Madj);
   const double det = fabs(kernels::Det<3>(M));
   return fnorm2 / pow(det, 4.0/3.0);
}

// *****************************************************************************
// ddI1b = X1 + X2 + X3, where
// X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
// X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
// X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
MFEM_HOST_DEVICE inline
void Dim3Invariant1b_dMdM(InvariantsEvaluator3D<double> ie,
                          const DeviceMatrix &M, int i, int j,
                          DeviceMatrix &dMdM)
{
   // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
   double X1_p[9], X2_p[9], X3_p[9];
   DeviceMatrix X1(X1_p,3,3);
   const double I3 = ie.Get_I3();
   const double I1b = ie.Get_I1b();
   const double alpha = (2./3.)*I1b/I3;
   DeviceMatrix dI3b((double*)ie.Get_dI3b(),3,3);
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         X1(k,l) = alpha * ((2./3.)*dI3b(i,j) * dI3b(k,l) +
                            dI3b(k,j)*dI3b(i,l));
      }
   }

   // ddI1_ijkl = 2 δ_ik δ_jl
   // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
   DeviceMatrix X2(X2_p,3,3);
   const double beta = ie.Get_I3b_p();
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         const double ddI1_ijkl = (i==k && j==l) ? 2.0 : 0.0;
         X2(k,l) = beta * ddI1_ijkl;
      }
   }

   // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
   DeviceMatrix X3(X3_p,3,3);
   const double I3b = ie.Get_I3b();
   const double gamma = -(4./3.)*ie.Get_I3b_p()/I3b;
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         X3(k,l) = gamma * (M(i,j) * dI3b(k,l) + dI3b(i,j) * M(k,l));
      }
   }

   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         dMdM(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
      }
   }
}

// *****************************************************************************
// ddI2 = x1 + x2 + x3
//    x1_ijkl = (2 I1) δ_ik δ_jl
//    x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
//    x3_ijkl = -2 (J J^t)_ik δ_jl = -2 B_ik δ_jl
MFEM_HOST_DEVICE inline
void Dim3Invariant2_dMdM(InvariantsEvaluator3D<double> ie,
                         const DeviceMatrix &M, int i, int j,
                         DeviceMatrix &dMdM)
{
   double x1_p[9], x2_p[9], x3_p[9];
   DeviceMatrix x1(x1_p,3,3), x2(x2_p,3,3), x3(x3_p,3,3);

   // x1_ijkl = (2 I1) δ_ik δ_jl
   const double I1 = ie.Get_I1();
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         const double ik_jl = (i==k && j==l) ? 1.0 : 0.0;
         x1(k,l) = 2.0 * I1 * ik_jl;
      }
   }

   // x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         x2(k,l) = 0.0;
         for (int u=0; u<3; u++)
         {
            for (int v=0; v<3; v++)
            {
               const double ku_iv = k==u && i==v ? 1.0 : 0.0;
               const double ik_uv = i==k && u==v ? 1.0 : 0.0;
               const double kv_iu = k==v && i==u ? 1.0 : 0.0;
               x2(k,l) += 2.0 * (2.*ku_iv - ik_uv - kv_iu) * M(v,j) * M(u,l);
            }
         }
      }
   }

   //    x3_ijkl = -2 B_ik δ_jl
   double b[9];
   const double *J = M;
   b[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
   b[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
   b[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];

   b[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
   b[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
   b[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   double b_p[9] =
   {
      b[0], b[3], b[4],
      b[3], b[1], b[5],
      b[4], b[5], b[2]
   };
   DeviceMatrix B(b_p,3,3);
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         const double jl = j==l ? 1.0 : 0.0;
         x3(k,l) = -2.0 * B(i,k) * jl;
      }
   }

   // ddI2 = x1 + x2 + x3
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         dMdM(k,l) = x1(k,l) + x2(k,l) + x3(k,l);
      }
   }
}

// *****************************************************************************
// ddI2b = X1 + X2 + X3
//    X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
//               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
//    X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
//    X3_ijkl =      det(J)^{-4/3} ddI2_ijkl
MFEM_HOST_DEVICE inline
void Dim3Invariant2b_dMdM(InvariantsEvaluator3D<double> ie,
                          const DeviceMatrix &M, int i, int j,
                          DeviceMatrix &dMdM)
{
   double X1_p[9], X2_p[9], X3_p[9];
   // X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
   //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
   DeviceMatrix X1(X1_p,3,3);
   const double I3b_p = ie.Get_I3b_p(); // I3b^{-2/3}
   const double I3b = ie.Get_I3b();     // det(J)
   const double I2 = ie.Get_I2();
   const double I3b_p43 = I3b_p*I3b_p;
   const double I3b_p73 = I3b_p*I3b_p/I3b;
   const double I3b_p103 = I3b_p*I3b_p/(I3b*I3b);
   DeviceMatrix dI3b((double*)ie.Get_dI3b(),3,3);
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         const double up = (16./9.)*I3b_p103*I2*dI3b(i,j)*dI3b(k,l);
         const double down = (4./3.)*I3b_p103*I2*dI3b(i,l)*dI3b(k,j);
         X1(k,l) = up + down;
      }
   }

   // X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
   DeviceMatrix X2(X2_p,3,3);
   DeviceMatrix dI2((double*)ie.Get_dI2(),3,3);
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         X2(k,l) = -(4./3.)*I3b_p73*(dI2(i,j)*dI3b(k,l) + dI2(k,l)*dI3b(i,j));
      }
   }

   double ddI2_p[9];
   DeviceMatrix ddI2(ddI2_p,3,3);
   Dim3Invariant2_dMdM(ie, M, i, j, ddI2);

   // X3_ijkl =  det(J)^{-4/3} ddI2_ijkl
   DeviceMatrix X3(X3_p,3,3);
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         X3(k,l) = I3b_p43 * ddI2(k,l);
      }
   }

   // ddI2b = X1 + X2 + X3
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         dMdM(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
      }
   }
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0>
static void SetupGradPA_3D(const Vector &xe_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseMatrix &j_,
                           Vector &dp_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto Jtr = Reshape(j_.Read(), DIM, DIM);
   const auto X = Reshape(xe_.Read(), D1D, D1D, D1D, DIM, NE);
   auto dP = Reshape(dp_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int DIM = 3;
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1])(s_BG[0]);
      double (*G)[MD1] = (double (*)[MD1])(s_BG[1]);

      MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
      double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[0]);
      double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[1]);
      double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[2]);

      MFEM_SHARED double s_DDQ[9][MD1*MD1*MQ1];
      double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[0]);
      double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[1]);
      double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[2]);
      double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[3]);
      double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[4]);
      double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[5]);

      MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
      double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[0]);
      double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[1]);
      double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[2]);
      double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[3]);
      double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[4]);
      double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[5]);
      double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[6]);
      double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[7]);
      double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[8]);

      MFEM_SHARED double s_QQQ[9][MQ1*MQ1*MQ1];
      double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+0);
      double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+1);
      double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+2);
      double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+3);
      double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+4);
      double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+5);
      double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+6);
      double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+7);
      double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+8);

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
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double xx = Xx[dz][dy][dx];
                  const double xy = Xy[dz][dy][dx];
                  const double xz = Xz[dz][dy][dx];
                  const double Bx = B[qx][dx];
                  const double Gx = G[qx][dx];
                  u[0] += Bx * xx;
                  u[1] += Bx * xy;
                  u[2] += Bx * xz;

                  v[0] += Gx * xx;
                  v[1] += Gx * xy;
                  v[2] += Gx * xz;
               }
               XxB[dz][dy][qx] = u[0];
               XyB[dz][dy][qx] = u[1];
               XzB[dz][dy][qx] = u[2];

               XxG[dz][dy][qx] = v[0];
               XyG[dz][dy][qx] = v[1];
               XzG[dz][dy][qx] = v[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double By = B[qy][dy];
                  const double Gy = G[qy][dy];

                  u[0] += XxB[dz][dy][qx] * By;
                  u[1] += XyB[dz][dy][qx] * By;
                  u[2] += XzB[dz][dy][qx] * By;

                  v[0] += XxG[dz][dy][qx] * By;
                  v[1] += XyG[dz][dy][qx] * By;
                  v[2] += XzG[dz][dy][qx] * By;

                  w[0] += XxB[dz][dy][qx] * Gy;
                  w[1] += XyB[dz][dy][qx] * Gy;
                  w[2] += XzB[dz][dy][qx] * Gy;
               }
               XxBB[dz][qy][qx] = u[0];
               XyBB[dz][qy][qx] = u[1];
               XzBB[dz][qy][qx] = u[2];

               XxBG[dz][qy][qx] = v[0];
               XyBG[dz][qy][qx] = v[1];
               XzBG[dz][qy][qx] = v[2];

               XxGB[dz][qy][qx] = w[0];
               XyGB[dz][qy][qx] = w[1];
               XzGB[dz][qy][qx] = w[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double Bz = B[qz][dz];
                  const double Gz = G[qz][dz];

                  u[0] += XxBG[dz][qy][qx] * Bz;
                  u[1] += XyBG[dz][qy][qx] * Bz;
                  u[2] += XzBG[dz][qy][qx] * Bz;

                  v[0] += XxGB[dz][qy][qx] * Bz;
                  v[1] += XyGB[dz][qy][qx] * Bz;
                  v[2] += XzGB[dz][qy][qx] * Bz;

                  w[0] += XxBB[dz][qy][qx] * Gz;
                  w[1] += XyBB[dz][qy][qx] * Gz;
                  w[2] += XzBB[dz][qy][qx] * Gz;
               }
               XxBBG[qz][qy][qx] = u[0];
               XyBBG[qz][qy][qx] = u[1];
               XzBBG[qz][qy][qx] = u[2];

               XxBGB[qz][qy][qx] = v[0];
               XyBGB[qz][qy][qx] = v[1];
               XzBGB[qz][qy][qx] = v[2];

               XxGBB[qz][qy][qx] = w[0];
               XyGBB[qz][qy][qx] = w[1];
               XzGBB[qz][qy][qx] = w[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double ip_weight = W(qx,qy,qz);

               const double Jtrx0 = Jtr(0,0);
               const double Jtrx1 = Jtr(0,1);
               const double Jtrx2 = Jtr(0,2);
               const double Jtry0 = Jtr(1,0);
               const double Jtry1 = Jtr(1,1);
               const double Jtry2 = Jtr(1,2);
               const double Jtrz0 = Jtr(2,0);
               const double Jtrz1 = Jtr(2,1);
               const double Jtrz2 = Jtr(2,2);
               const double Jtr[9] =
               {
                  Jtrx0, Jtry0, Jtrz0,
                  Jtrx1, Jtry1, Jtrz1,
                  Jtrx2, Jtry2, Jtrz2
               };
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight = ip_weight * detJtr;

               // Jrt = Jtr^{-1}
               double Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               const double JprxBBG = XxBBG[qz][qy][qx];
               const double JprxBGB = XxBGB[qz][qy][qx];
               const double JprxGBB = XxGBB[qz][qy][qx];
               const double JpryBBG = XyBBG[qz][qy][qx];
               const double JpryBGB = XyBGB[qz][qy][qx];
               const double JpryGBB = XyGBB[qz][qy][qx];
               const double JprzBBG = XzBBG[qz][qy][qx];
               const double JprzBGB = XzBGB[qz][qy][qx];
               const double JprzGBB = XzGBB[qz][qy][qx];
               const double Jpr[9] =
               {
                  JprxBBG, JpryBBG, JprzBBG,
                  JprxBGB, JpryBGB, JprzBGB,
                  JprxGBB, JpryGBB, JprzGBB
               };

               // J = Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               double Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // metric->AssembleH
               DeviceMatrix Jpt_dm(Jpt,DIM,DIM);
               InvariantsEvaluator3D<double> ie;
               ie.SetJacobian(Jpt);

               //const double I1b = ie.Get_I1b();
               const double I1b = Dim3Invariant1b(Jpt);
               const double I2b = Dim3Invariant2b(Jpt);
               DeviceMatrix dI1b((double*)ie.Get_dI1b(),DIM,DIM);
               DeviceMatrix dI2b((double*)ie.Get_dI2b(),DIM,DIM);
               double dI1b_dMdM_p[9], dI2b_dMdM_p[9];
               DeviceMatrix dI1b_dMdM(dI1b_dMdM_p,DIM,DIM);
               DeviceMatrix dI2b_dMdM(dI2b_dMdM_p,DIM,DIM);
               for (int r = 0; r < DIM; r++)
               {
                  for (int c = 0; c < DIM; c++)
                  {
                     Dim3Invariant1b_dMdM(ie, Jpt_dm, r, c, dI1b_dMdM);
                     Dim3Invariant2b_dMdM(ie, Jpt_dm, r, c, dI2b_dMdM);
                     for (int rr = 0; rr < DIM; rr++)
                     {
                        for (int cc = 0; cc < DIM; cc++)
                        {
                           const double entry_rr_cc =
                              (weight/9.) * (dI1b_dMdM(rr,cc)*I2b
                                             + dI1b(r,c)*dI2b(rr,cc)
                                             + dI1b(rr,cc)*dI2b(r,c)
                                             + dI2b_dMdM(rr,cc)*I1b);
                           dP(rr,cc,r,c,qx,qy,qz,e) = entry_rr_cc;
                        }
                     }
                  }
               }
            } // qx
         } // qy
      } // qz
   });
}

// *****************************************************************************
void TMOP_Integrator::AssembleGradPA_3D(const DenseMatrix &Jtr,
                                        const Vector &Xe) const
{
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   switch (id)
   {
      //case 0x21: { SetupGradPA_3D<2,1>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x22: { SetupGradPA_3D<2,2>(Xe,ne,W,B,G,Jtr,dPpa); break; }/*
      case 0x23: { SetupGradPA_3D<2,3>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x24: { SetupGradPA_3D<2,4>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x25: { SetupGradPA_3D<2,5>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x26: { SetupGradPA_3D<2,6>(Xe,ne,W,B,G,Jtr,dPpa); break; }

      case 0x31: { SetupGradPA_3D<3,1>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x32: { SetupGradPA_3D<3,2>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x33: { SetupGradPA_3D<3,3>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x34: { SetupGradPA_3D<3,4>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x35: { SetupGradPA_3D<3,5>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x36: { SetupGradPA_3D<3,6>(Xe,ne,W,B,G,Jtr,dPpa); break; }

      case 0x41: { SetupGradPA_3D<4,1>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x42: { SetupGradPA_3D<4,2>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x43: { SetupGradPA_3D<4,3>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x44: { SetupGradPA_3D<4,4>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x45: { SetupGradPA_3D<4,5>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x46: { SetupGradPA_3D<4,6>(Xe,ne,W,B,G,Jtr,dPpa); break; }

      case 0x51: { SetupGradPA_3D<5,1>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x52: { SetupGradPA_3D<5,2>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x53: { SetupGradPA_3D<5,3>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x54: { SetupGradPA_3D<5,4>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x55: { SetupGradPA_3D<5,5>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x56: { SetupGradPA_3D<5,6>(Xe,ne,W,B,G,Jtr,dPpa); break; }*/
      default:
      {
         dbg("kernel id: %x", id);
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

} // namespace mfem
