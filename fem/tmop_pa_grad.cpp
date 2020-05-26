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
#define MFEM_DBG_COLOR 211
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

// *****************************************************************************
// dI2_dM = d(det(M))_dM = adj(M)^T.
static void Dim2Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");
   dM(0, 0) =  M(1, 1); dM(0, 1) = -M(1, 0);
   dM(1, 0) = -M(0, 1); dM(1, 1) =  M(0, 0);
}

// *****************************************************************************
static
void Dim2Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                         DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");
   dMdM = 0.0;
   dMdM(1-i,1-j) = (i == j) ? 1.0 : -1.0;
}

// *****************************************************************************
// (dI1_dM)_d(Mij) = d[(2 det(M) M - |M|^2 adj(M)^T) / det(M)^2]_d[Mij].
static
void Dim2Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                         DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   // Compute d(det(M))_d(Mij), d(|M|^2)_d(Mij).
   DenseMatrix dI(2);
   Dim2Invariant2_dM(M, dI);
   const double ddet   = dI(i,j);
   const double dfnorm2 = 2.0 * M(i,j);

   const double det    = M.Det();
   const double det2   = det * det;
   const double fnorm2 = M.FNorm2();

   DenseMatrix dM(2); dM = 0.0; dM(i, j) = 1.0;
   DenseMatrix ddI(2);
   Dim2Invariant2_dMdM(M, i, j, ddI);
   for (int r = 0; r < 2; r++)
   {
      for (int c = 0; c < 2; c++)
      {
         dMdM(r,c) =
            (det2 *
             (2.0 * ddet * M(r,c) + 2.0 * det * dM(r,c)
              - dfnorm2 * dI(r,c) - fnorm2 * ddI(r,c))
             - 2.0 * det * ddet *
             (2.0 * det * M(r,c) - fnorm2 * dI(r,c)) ) / (det2 * det2);
      }
   }
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SetupGradPA_2D(const Vector &xe_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b1d_,
                           const Array<double> &g1d_,
                           const DenseMatrix &Jtr,
                           Vector &p_,
                           Vector &g_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   dbg("");
   constexpr int dim = 2;
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int dof = D1D*D1D;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b1d = Reshape(b1d_.Read(), Q1D, D1D);
   const auto g1d = Reshape(g1d_.Read(), Q1D, D1D);
   const auto J = Reshape(Jtr.Read(), VDIM, VDIM);
   const auto X = Reshape(xe_.Read(), D1D, D1D, VDIM, NE);
   auto P = Reshape(p_.Write(), VDIM, VDIM, VDIM, VDIM, Q1D, Q1D, NE);
   auto G = Reshape(g_.Write(), Q1D, Q1D, dof*VDIM, dof*VDIM, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B1d)[MD1]  = (double (*)[MD1])(s_BG[0]);
      double (*G1d)[MD1]  = (double (*)[MD1])(s_BG[1]);

      MFEM_SHARED double s_X[NBZ][MD1*MD1];
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
      // Load B1d and G1d matrices
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B1d[q][d] = b1d(q,d);
               G1d[q][d] = g1d(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double xx = Xx[dy][dx];
               const double xy = Xy[dy][dx];
               xu[0] += B1d[qx][dx] * xx;
               xv[0] += G1d[qx][dx] * xx;
               xu[1] += B1d[qx][dx] * xy;
               xv[1] += G1d[qx][dx] * xy;
            }
            XxB[dy][qx] = xu[0];
            XxG[dy][qx] = xv[0];
            XyB[dy][qx] = xu[1];
            XyG[dy][qx] = xv[1];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               xu[0] += XxG[dy][qx] * B1d[qy][dy];
               xv[0] += XxB[dy][qx] * G1d[qy][dy];
               xu[1] += XyG[dy][qx] * B1d[qy][dy];
               xv[1] += XyB[dy][qx] * G1d[qy][dy];
            }
            Xx0[qy][qx] = xu[0];
            Xx1[qy][qx] = xv[0];
            Xy0[qy][qx] = xu[1];
            Xy1[qy][qx] = xv[1];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double weight = W(qx,qy);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = J(0,0);
            const double Jtrx1 = J(0,1);
            const double Jtry0 = J(1,0);
            const double Jtry1 = J(1,1);
            double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};

            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double weight_detJtr = weight * detJtr;

            // Jrt = Jtr^{-1}
            DenseMatrix Jrt(dim);
            kernels::CalcInverse<2>(Jtr_p, Jrt.HostWrite());

            // Compute DSh (dof x dim)
            const int dof = D1D*D1D;
            DenseMatrix DSh(dof, dim);
            for (int i1 = 0; i1 < D1D; ++i1)
            {
               for (int i2 = 0; i2 < D1D; ++i2)
               {
                  const double bg = G1d[qx][i1] * B1d[qy][i2];
                  const double gb = B1d[qx][i1] * G1d[qy][i2];
                  const int dof = i2 + i1*D1D;
                  DSh(dof, 0) = bg;
                  DSh(dof, 1) = gb;
               }
            }

            // Compute DS = DSh Jrt
            DenseMatrix DS(dof, dim);
            Mult(DSh, Jrt, DS);

            // GX = X^T.DSh
            const double GXx0h = Xx0[qy][qx];
            const double GXx1h = Xx1[qy][qx];
            const double GXy0h = Xy0[qy][qx];
            const double GXy1h = Xy1[qy][qx];
            double GXh_p[4] = {GXx0h, GXy0h, GXx1h, GXy1h};
            DenseMatrix GXh(GXh_p, dim, dim);

            // Jpt = GX^T.DS = (GX^T.DSh).Jrt = GX.Jrt
            DenseMatrix Jpt(dim);
            Mult(GXh, Jrt, Jpt);
            const double sign = Jpt.Det() < 0.0 ? -1.0 : 1.0;

            for (int r = 0; r < dim; r++)
            {
               for (int c = 0; c < dim; c++)
               {
                  DenseMatrix dP(&P(0,0,r,c,qx,qy,e),dim,dim);
                  Dim2Invariant1_dMdM(Jpt,r,c,dP);
                  dP *= sign * 0.5 * weight_detJtr;
                  for (int rr = 0; rr < dim; rr++)
                  {
                     for (int cc = 0; cc < dim; cc++)
                     {
                        const double dp = dP(rr,cc);
                        for (int i = 0; i < dof; i++)
                        {
                           for (int j = 0; j < dof; j++)
                           {
                              const double ds = DS(i, c) * DS(j, cc);
                              G(qx, qy, i+r*dof, j+rr*dof, e) += ds * dp;
                           }
                        }
                     }
                  }
               }
            }
         } // qx
      } // qy
      MFEM_SYNC_THREAD;
   });
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultGradPA_Kernel_2D(const Vector &xe_,
                                    const int NE,
                                    const Array<double> &w_,
                                    const Array<double> &b1d_,
                                    const Array<double> &g1d_,
                                    const DenseMatrix &Jtr,
                                    const Vector &p_,
                                    const Vector &g_,
                                    const Vector &re_,
                                    Vector &ce_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   constexpr int dim = 2;
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int dof = D1D*D1D;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b1d_.Read(), Q1D, D1D);
   const auto g = Reshape(g1d_.Read(), Q1D, D1D);
   const auto J = Reshape(Jtr.Read(), VDIM, VDIM);
   const auto X = Reshape(xe_.Read(), D1D, D1D, VDIM, NE);
   const auto R = Reshape(re_.Read(), D1D, D1D, VDIM, NE);
   const auto paP = Reshape(p_.Read(), VDIM, VDIM, VDIM, VDIM, Q1D, Q1D, NE);
   const auto G = Reshape(g_.Read(), Q1D, Q1D, dof*VDIM, dof*VDIM, NE);
   auto C = Reshape(ce_.ReadWrite(), D1D, D1D, VDIM, NE);
   //dbg("D1D:%d, Q1D:%d, nq:%d", D1D, Q1D, Q1D*Q1D);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B1d)[MD1]  = (double (*)[MD1])(s_BG[0]);
      double (*G1d)[MD1]  = (double (*)[MD1])(s_BG[1]);
      double (*B1dt)[MQ1] = (double (*)[MQ1])(s_BG[0]);
      double (*G1dt)[MQ1] = (double (*)[MQ1])(s_BG[1]);

      MFEM_SHARED double s_R[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      double (*Rx)[MD1]  = (double (*)[MD1])(s_R[0] + tidz);
      double (*Ry)[MD1]  = (double (*)[MD1])(s_R[1] + tidz);
      double (*Xx)[MD1]  = (double (*)[MD1])(s_X[0] + tidz);
      double (*Xy)[MD1]  = (double (*)[MD1])(s_X[1] + tidz);

      MFEM_SHARED double s_DQ[3][4][NBZ][MD1*MQ1];
      double (*RxB)[MQ1] = (double (*)[MQ1])(s_DQ[0][0] + tidz);
      double (*RxG)[MQ1] = (double (*)[MQ1])(s_DQ[0][1] + tidz);
      double (*RyB)[MQ1] = (double (*)[MQ1])(s_DQ[0][2] + tidz);
      double (*RyG)[MQ1] = (double (*)[MQ1])(s_DQ[0][3] + tidz);

      double (*XxB)[MQ1] = (double (*)[MQ1])(s_DQ[1][0] + tidz);
      double (*XxG)[MQ1] = (double (*)[MQ1])(s_DQ[1][1] + tidz);
      double (*XyB)[MQ1] = (double (*)[MQ1])(s_DQ[1][2] + tidz);
      double (*XyG)[MQ1] = (double (*)[MQ1])(s_DQ[1][3] + tidz);

      double (*CxB)[MQ1] = (double (*)[MQ1])(s_DQ[2][0] + tidz);
      double (*CxG)[MQ1] = (double (*)[MQ1])(s_DQ[2][1] + tidz);
      double (*CyB)[MQ1] = (double (*)[MQ1])(s_DQ[2][2] + tidz);
      double (*CyG)[MQ1] = (double (*)[MQ1])(s_DQ[2][3] + tidz);

      MFEM_SHARED double s_QQ[3][4][NBZ][MQ1*MQ1];
      double (*Rx0)[MQ1] = (double (*)[MQ1])(s_QQ[0][0] + tidz);
      double (*Rx1)[MQ1] = (double (*)[MQ1])(s_QQ[0][1] + tidz);
      double (*Ry0)[MQ1] = (double (*)[MQ1])(s_QQ[0][2] + tidz);
      double (*Ry1)[MQ1] = (double (*)[MQ1])(s_QQ[0][3] + tidz);

      double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[1][0] + tidz);
      double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1][1] + tidz);
      double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[1][2] + tidz);
      double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[1][3] + tidz);

      double (*Cx0)[MQ1] = (double (*)[MQ1])(s_QQ[2][0] + tidz);
      double (*Cx1)[MQ1] = (double (*)[MQ1])(s_QQ[2][1] + tidz);
      double (*Cy0)[MQ1] = (double (*)[MQ1])(s_QQ[2][2] + tidz);
      double (*Cy1)[MQ1] = (double (*)[MQ1])(s_QQ[2][3] + tidz);

      // Load R(x,y) and X(x,y)
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Rx[dy][dx] = R(dx,dy,0,e);
            Ry[dy][dx] = R(dx,dy,1,e);
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
            double ru[2] = {0};
            double rv[2] = {0};
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double rx = Rx[dy][dx];
               const double ry = Ry[dy][dx];
               ru[0] += B1d[qx][dx] * rx;
               rv[0] += G1d[qx][dx] * rx;
               ru[1] += B1d[qx][dx] * ry;
               rv[1] += G1d[qx][dx] * ry;
               const double xx = Xx[dy][dx];
               const double xy = Xy[dy][dx];
               xu[0] += B1d[qx][dx] * xx;
               xv[0] += G1d[qx][dx] * xx;
               xu[1] += B1d[qx][dx] * xy;
               xv[1] += G1d[qx][dx] * xy;
            }
            RxB[dy][qx] = ru[0];
            RxG[dy][qx] = rv[0];
            RyB[dy][qx] = ru[1];
            RyG[dy][qx] = rv[1];

            XxB[dy][qx] = xu[0];
            XxG[dy][qx] = xv[0];
            XyB[dy][qx] = xu[1];
            XyG[dy][qx] = xv[1];

         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double ru[2] = {0};
            double rv[2] = {0};
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               ru[0] += RxG[dy][qx] * B1d[qy][dy];
               rv[0] += RxB[dy][qx] * G1d[qy][dy];
               ru[1] += RyG[dy][qx] * B1d[qy][dy];
               rv[1] += RyB[dy][qx] * G1d[qy][dy];

               xu[0] += XxG[dy][qx] * B1d[qy][dy];
               xv[0] += XxB[dy][qx] * G1d[qy][dy];
               xu[1] += XyG[dy][qx] * B1d[qy][dy];
               xv[1] += XyB[dy][qx] * G1d[qy][dy];
            }
            Rx0[qy][qx] = ru[0];
            Rx1[qy][qx] = rv[0];
            Ry0[qy][qx] = ru[1];
            Ry1[qy][qx] = rv[1];

            Xx0[qy][qx] = xu[0];
            Xx1[qy][qx] = xv[0];
            Xy0[qy][qx] = xu[1];
            Xy1[qy][qx] = xv[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = J(0,0);
            const double Jtrx1 = J(0,1);
            const double Jtry0 = J(1,0);
            const double Jtry1 = J(1,1);
            double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};

            // Jrt = Jtr^{-1}
            DenseMatrix Jrt(dim);
            kernels::CalcInverse<2>(Jtr_p, Jrt.HostWrite());

            const double GRx0h = Rx0[qy][qx];
            const double GRx1h = Rx1[qy][qx];
            const double GRy0h = Ry0[qy][qx];
            const double GRy1h = Ry1[qy][qx];
            double hGR_p[4] = {GRx0h, GRy0h, GRx1h, GRy1h};
            DenseMatrix hGR(hGR_p, dim, dim);

            // GR = R^T . Jrt
            DenseMatrix GR(dim);
            Mult(hGR,Jrt,GR);

            // GZ = GR : dP
            DenseMatrix GZ(dim);
            for (int r = 0; r < dim; r++)
            {
               for (int c = 0; c < dim; c++)
               {
                  GZ(r,c) = 0.0;
                  DenseMatrix dP((double*)&paP(0,0,r,c,qx,qy,e),dim,dim);
                  for (int rr = 0; rr < dim; rr++)
                  {
                     for (int cc = 0; cc < dim; cc++)
                     {
                        GZ(r,c) += dP(rr,cc) * GR(rr,cc);
                     }
                  }
               } // c
            } // r

            // A = Jrt . GZ
            double A_p[4];
            DenseMatrix A(A_p, dim, dim);
            MultABt(Jrt, GZ, A);

            Cx0[qy][qx] = A(0,0);
            Cy0[qy][qx] = A(0,1);
            Cx1[qy][qx] = A(1,0);
            Cy1[qy][qx] = A(1,1);
         } // qx
      } // qy

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
            double cu[2] = {0};
            double cv[2] = {0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               cu[0] += G1dt[dx][qx] * Cx0[qy][qx];
               cv[0] += B1dt[dx][qx] * Cx1[qy][qx];
               cu[1] += G1dt[dx][qx] * Cy0[qy][qx];
               cv[1] += B1dt[dx][qx] * Cy1[qy][qx];
            }
            CxB[dx][qy] = cu[0];
            CxG[dx][qy] = cv[0];
            CyB[dx][qy] = cu[1];
            CyG[dx][qy] = cv[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double cu[2] = {0};
            double cv[2] = {0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               cu[0] += CxB[dx][qy] * B1dt[dy][qy];
               cv[0] += CxG[dx][qy] * G1dt[dy][qy];
               cu[1] += CyB[dx][qy] * B1dt[dy][qy];
               cv[1] += CyG[dx][qy] * G1dt[dy][qy];
            }
            C(dx,dy,0,e) += cu[0] + cv[0];
            C(dx,dy,1,e) += cu[1] + cv[1];
         }
      }
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultGradPA(const Vector &Xe, const Vector &Re,
                                    Vector &Ce) const
{
   //dbg("Xe: %d, Re:%d, Ce:%d", Xe.Size(), Re.Size(), Ce.Size());
   //dbg("Xe: %.15e, Re: %.15e", Xe*Xe, Re*Re);
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B1d = maps->B;
   const Array<double> &G1d = maps->G;
   const int id = (D1D << 4 ) | Q1D;


   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
   // Get Wideal into Jtr
#if 1
   const FiniteElement *fe = fes->GetFE(0);
   const Geometry::Type geom_type = fe->GetGeomType();
   const DenseMatrix Jtr = Geometries.GetGeomToPerfGeomJac(geom_type);
   MFEM_VERIFY(Jtr.Det() == 1.0 ,"");
   {
      MFEM_VERIFY(Jtr(0,0)==1.0 && Jtr(1,1)==1.0 &&
                  Jtr(1,0)==0.0 && Jtr(0,1)==0.0,"");
   }
#elif 0
   DenseMatrix Jtr(dim);
   Jtr(0,0) = 1.0;
   Jtr(0,1) = 0.0;
   Jtr(1,0) = 0.0;
   Jtr(1,1) = 1.0;
#elif 0
   DenseMatrix Jtr(dim);
   Jtr(0,0) = 2.0;
   Jtr(0,1) = 0.0;
   Jtr(1,0) = 0.0;
   Jtr(1,1) = -1.0;
#else
   DenseMatrix Jtr(dim);
   Jtr(0,0) = 2.0;
   Jtr(0,1) = +1.123;
   Jtr(1,0) = -1.456;
   Jtr(1,1) = 1.0;
#endif
   //dbg("Jtr:"); Jtr.Print();

   /*
      Array<int> vdofs;
      DenseTensor Jtr(dim, dim, ir->GetNPoints());
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const FiniteElement *el = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
     }*/
   if (!setup)
   {
      Gpa = 0.0;
      setup = true;
      switch (id)
      {
         case 0x32: { SetupGradPA_2D<3,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa,Gpa); break; }
         default:
         {
            dbg("kernel id: %x", id);
            MFEM_ABORT("Unknown kernel.");
         }
      }
   }

   switch (id)
   {
      /*case 0x21: return AddMultGradPA_Kernel_2D<2,1,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x22: return AddMultGradPA_Kernel_2D<2,2,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x23: return AddMultGradPA_Kernel_2D<2,3,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x24: return AddMultGradPA_Kernel_2D<2,4,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x25: return AddMultGradPA_Kernel_2D<2,5,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);*/

      //case 0x31: return AddMultGradPA_Kernel_2D<3,1,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x32:
         return AddMultGradPA_Kernel_2D<3,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa,Gpa,Re,Ce);/*
      case 0x33: return AddMultGradPA_Kernel_2D<3,3,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x34: return AddMultGradPA_Kernel_2D<3,4,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x35: return AddMultGradPA_Kernel_2D<3,5,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);

      case 0x41: return AddMultGradPA_Kernel_2D<4,1,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x42: return AddMultGradPA_Kernel_2D<4,2,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x43: return AddMultGradPA_Kernel_2D<4,3,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x44: return AddMultGradPA_Kernel_2D<4,4,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x45: return AddMultGradPA_Kernel_2D<4,5,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);

      case 0x51: return AddMultGradPA_Kernel_2D<5,1,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x52: return AddMultGradPA_Kernel_2D<5,2,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x53: return AddMultGradPA_Kernel_2D<5,3,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x54: return AddMultGradPA_Kernel_2D<5,4,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);
      case 0x55: return AddMultGradPA_Kernel_2D<5,5,1>(Xe,ne,W,B1d,G1d,Jtr,G,Re,Ce);*/
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
