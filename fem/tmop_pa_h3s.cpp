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
#include "../linalg/dinvariants.hpp"

namespace mfem
{

// *****************************************************************************
// dP_302 = (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
static MFEM_HOST_DEVICE inline
void EvalH_302(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9],          ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double dI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(J, B,
                                     nullptr, dI1b, nullptr, ddI1b,
                                     dI2, dI2b,    ddI2, ddI2b,
                                     dI3b, nullptr);
   const double c1 = weight/9.;
   const double I1b = ie.Get_I1b();
   const double I2b = ie.Get_I2b();
   ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
   ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
         ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
         for (int r = 0; r < DIM; r++)
         {
            for (int c = 0; c < DIM; c++)
            {
               const double dp =
                  (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                  + ddi2b(r,c)*I1b
                  + ddi1b(r,c)*I2b;
               dP(r,c,i,j,qx,qy,qz,e) = c1 * dp;
            }
         }
      }
   }
}

// *****************************************************************************
// dP_321 = ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
//               + (1/I3)*ddI2
//               + (6*I2/I3b^4)*(dI3b x dI3b)
//               + (-2*I2/I3b^3)*ddI3b
static MFEM_HOST_DEVICE inline
void EvalH_321(const int e, const int qx, const int qy, const int qz,
               const double weight, const double *J, DeviceTensor<8,double> dP)
{
   double B[9];
   double         dI1b[9], ddI1[9], ddI1b[9];
   double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
   double dI3b[9], ddI3b[9];
   constexpr int DIM = 3;
   kernels::InvariantsEvaluator3D ie(J, B,
                                     nullptr, dI1b, ddI1, ddI1b,
                                     dI2, dI2b, ddI2, ddI2b,
                                     dI3b, ddI3b);
   double sign_detJ;
   const double I2 = ie.Get_I2();
   const double I3b = ie.Get_I3b(sign_detJ);

   ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
   ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);

   const double c0 = 1.0/I3b;
   const double c1 = weight*c0*c0;
   const double c2 = -2*c0*c1;
   const double c3 = c2*I2;

   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
         ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
         ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
         for (int r = 0; r < DIM; r++)
         {
            for (int c = 0; c < DIM; c++)
            {
               const double dp =
                  weight * ddi1(r,c)
                  + c1 * ddi2(r,c)
                  + c3 * ddi3b(r,c)
                  + c2 * ((di2(r,c)*di3b(i,j) + di3b(r,c)*di2(i,j)))
                  -3*c0*c3 * di3b(r,c)*di3b(i,j);
               dP(r,c,i,j,qx,qy,qz,e) = dp;
            }
         }
      }
   }
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0>
static void SetupGradPA_3D(const int mid,
                           const Vector &xe_,
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

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
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
               if (mid == 302) { EvalH_302(e,qx,qy,qz,weight,Jpt,dP); }
               if (mid == 321) { EvalH_321(e,qx,qy,qz,weight,Jpt,dP); }
            } // qx
         } // qy
      } // qz
   });
}

// *****************************************************************************
void TMOP_Integrator::AssembleGradPA_3D(const DenseMatrix &Jtr,
                                        const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int mid = metric->Id();
   const int id = (D1D << 4 ) | Q1D;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &A = PA.A;

   switch (id)
   {
      case 0x21: { SetupGradPA_3D<2,1>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x22: { SetupGradPA_3D<2,2>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x23: { SetupGradPA_3D<2,3>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x24: { SetupGradPA_3D<2,4>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x25: { SetupGradPA_3D<2,5>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x26: { SetupGradPA_3D<2,6>(mid,X,N,W,B,G,Jtr,A); break; }

      case 0x31: { SetupGradPA_3D<3,1>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x32: { SetupGradPA_3D<3,2>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x33: { SetupGradPA_3D<3,3>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x34: { SetupGradPA_3D<3,4>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x35: { SetupGradPA_3D<3,5>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x36: { SetupGradPA_3D<3,6>(mid,X,N,W,B,G,Jtr,A); break; }

      case 0x41: { SetupGradPA_3D<4,1>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x42: { SetupGradPA_3D<4,2>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x43: { SetupGradPA_3D<4,3>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x44: { SetupGradPA_3D<4,4>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x45: { SetupGradPA_3D<4,5>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x46: { SetupGradPA_3D<4,6>(mid,X,N,W,B,G,Jtr,A); break; }

      case 0x51: { SetupGradPA_3D<5,1>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x52: { SetupGradPA_3D<5,2>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x53: { SetupGradPA_3D<5,3>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x54: { SetupGradPA_3D<5,4>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x55: { SetupGradPA_3D<5,5>(mid,X,N,W,B,G,Jtr,A); break; }
      case 0x56: { SetupGradPA_3D<5,6>(mid,X,N,W,B,G,Jtr,A); break; }

      default:
      {
         dbg("kernel id: %x", id);
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

} // namespace mfem
