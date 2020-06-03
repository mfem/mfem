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
// I1 = |M|^2/ det(M)^(2/3).
double Dim3Invariant1(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   const double fnorm = M.FNorm(), det = fabs(M.Det());
   return fnorm * fnorm / pow(det, 2.0/3.0);
}

// *****************************************************************************
// I2 = |adj(M)|^2 / det(M)^(4/3).
double Dim3Invariant2(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);

   const double fnorm = Madj.FNorm(), det = fabs(M.Det());
   return fnorm * fnorm / pow(det, 4.0/3.0);
}

// *****************************************************************************
// dI3_dM = d(det(M))_dM = adj(M)^T.
void Dim3Invariant3_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");
   CalcAdjugateTranspose(M, dM);
}

// *****************************************************************************
// dI1_dM = [ 2 det(M) M - 2/3 |M|^2 det(M)^(-1/3) adj(M)^T ] / det(M)^4/3.
void Dim3Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");
   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);
   const double fnorm = M.FNorm(), det = fabs(M.Det());
   Dim3Invariant3_dM(M, dM);
   dM *= -(2./3.) * fnorm * fnorm * pow(det, -1./3.);
   dM.Add(2.0 * pow(det, 2./3.), M);
   dM *= 1.0 / pow(det, 4./3.);
}

// *****************************************************************************
// dI2_dM = [ -4/3 |adj(M)|^2  det(M)^(1/3) adj(M)^T ] / det(M)^(8/3).
void Dim3Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");
   DenseMatrix Madj(3);
   // dM will have Madj^t because it is the third invariant's derivative.
   CalcAdjugate(M, Madj);
   const double fnorm = Madj.FNorm(), det = fabs(M.Det());
   Dim3Invariant3_dM(M, dM);
   dM *= -(4./3.)* fnorm * fnorm * pow(det, 1./3.);
   dM *= 1.0 / (pow(det, 8./3.));
}

// *****************************************************************************
void Dim3Invariant1_dMdM(const DenseMatrix &M, int i, int j, DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");
   DenseMatrix dI(3);
   Dim3Invariant3_dM(M, dI);
   const double fnorm  = M.FNorm(), det = fabs(M.Det());
   DenseMatrix dM(3); dM = 0.0; dM(i, j) = 1.0;
   for (int r = 0; r < 3; r++)
   {
      for (int c = 0; c < 3; c++)
      {
         dMdM(r,c) = (2.0 * det * det * dM(r,c)
                      + dI(i,j) * (10./9.) * fnorm * fnorm * dI(r,c)
                      - (4./3.) * dI(i,j) * det * M(r,c)
                      - (4./3.) * det * M(i,j) * dI(r,c))
                     / pow(det, 8./3.);
      }
   }
}

// *****************************************************************************
void Dim3Invariant2_dMdM(const DenseMatrix &M, int i, int j, DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");
   DenseMatrix dI(3);
   Dim3Invariant3_dM(M, dI);
   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);
   const double det   = fabs(M.Det());
   const double fnorm = Madj.FNorm();
   DenseMatrix dM(3); dM = 0.0; dM(i, j) = 1.0;
   for (int r = 0; r < 3; r++)
   {
      for (int c = 0; c < 3; c++)
      {
         dMdM(r,c) = (28./9.) * fnorm * fnorm *
                     det * det * dI(i,j) * dI(r,c) / pow(det, 16./3.);
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
               double J[9];
               kernels::Mult(3,3,3, Jpr, Jrt, J);

               const double detJpt = kernels::Det<3>(J);
               const double sign = detJpt >= 0.0 ? 1.0 : -1.0;

               // metric->AssembleH(Jpt, DS, weight_m, elmat);
               DenseMatrix Jpt(J,DIM,DIM);
               const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
               DenseMatrix dI1_dM(DIM), dI1_dMdM(DIM), dI2_dM(DIM), dI2_dMdM(DIM);

               Dim3Invariant1_dM(Jpt, dI1_dM);
               Dim3Invariant2_dM(Jpt, dI2_dM);

               for (int r = 0; r < DIM; r++)
               {
                  for (int c = 0; c < DIM; c++)
                  {
                     Dim3Invariant1_dMdM(Jpt, r,c, dI1_dMdM);
                     Dim3Invariant2_dMdM(Jpt, r,c, dI2_dMdM);
                     // Compute each entry of d(Prc)_dJ.
                     for (int rr = 0; rr < DIM; rr++)
                     {
                        for (int cc = 0; cc < DIM; cc++)
                        {
                           const double entry_rr_cc =
                              (1./9.)*(dI1_dMdM(rr,cc)*I2
                                       + dI1_dM(r,c)*dI2_dM(rr,cc)
                                       + dI1_dM(rr,cc)*dI2_dM(r,c)
                                       + dI2_dMdM(rr,cc)*I1);
                           dP(rr,cc,r,c,qx,qy,qz,e) = sign * weight * entry_rr_cc;
                           //dbg("dP: %.15e", dP(rr,cc,r,c,qx,qy,qz,e));
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
      case 0x21: { SetupGradPA_3D<2,1>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      case 0x22: { SetupGradPA_3D<2,2>(Xe,ne,W,B,G,Jtr,dPpa); break; }
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
      case 0x56: { SetupGradPA_3D<5,6>(Xe,ne,W,B,G,Jtr,dPpa); break; }
      default:
      {
         dbg("kernel id: %x", id);
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

} // namespace mfem
