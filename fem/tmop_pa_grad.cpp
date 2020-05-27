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
#include "../linalg/dtensor.hpp"

namespace mfem
{

// *****************************************************************************
MFEM_HOST_DEVICE inline
void Invariant2_dM_2D(const double M[2][2], double dM[2][2])
{
   dM[0][0] =  M[1][1]; dM[1][0] = -M[0][1];
   dM[0][1] = -M[1][0]; dM[1][1] =  M[0][0];
}

// *****************************************************************************
MFEM_HOST_DEVICE inline
void Invariant2_dMdM_2D(int i, int j, double dMdM[2][2])
{
   dMdM[0][0] = dMdM[0][1] = dMdM[1][0] = dMdM[1][1] = 0.0;
   dMdM[1-j][1-i] = (i == j) ? 1.0 : -1.0;
}

// *****************************************************************************
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

   double dM[2][2] = {0.0}; dM[j][i] = 1.0;
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

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SetupGradPA_2D(const Vector &xe_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseMatrix &j_,
                           Vector &p_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   dbg("");
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM);
   const auto X = Reshape(xe_.Read(), D1D, D1D, DIM, NE);

   auto P = Reshape(p_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int DIM = 2;
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int DOF = T_D1D ? T_D1D*T_D1D : MAX_D1D*MAX_D1D;

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
            double u[2] = {0};
            double v[2] = {0};
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
            double u[2] = {0};
            double v[2] = {0};
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
            const double weight = W(qx,qy);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtr_p[4] = {J(0,0), J(1,0), J(0,1), J(1,1)};
            const double detJtr = J(0,0)*J(1,1) - J(1,0)*J(0,1);
            const double weight_detJtr = weight * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt_p[4];
            kernels::CalcInverse<2>(Jtr_p, Jrt_p);

            // Compute DSh (dof x dim)
            double DSh_p[DOF*DIM];
            for (int i = 0; i < D1D; ++i)
            {
               for (int j = 0; j < D1D; ++j)
               {
                  const double bg = G[qx][i] * B[qy][j];
                  const double gb = B[qx][i] * G[qy][j];
                  const int dof = j + i*D1D;
                  DSh_p[dof*DIM + 0] = bg;
                  DSh_p[dof*DIM + 1] = gb;
               }
            }

            // Compute DS = DSh Jrt
            double DS_p[DOF*DIM];
            kernels::Mult(DOF,DIM,DIM, DSh_p, Jrt_p, DS_p);

            // GX = X^T.DSh
            const double GXx0h = Xx0[qy][qx];
            const double GXx1h = Xx1[qy][qx];
            const double GXy0h = Xy0[qy][qx];
            const double GXy1h = Xy1[qy][qx];
            double GXh_p[4] = {GXx0h, GXy0h, GXx1h, GXy1h};

            // Jpt = GX^T.DS = (GX^T.DSh).Jrt = GX.Jrt
            double Jpt_p[4];
            kernels::Mult(2,2,2,GXh_p,Jrt_p, Jpt_p);

            const double detJpt = Jpt_p[0]*Jpt_p[3] - Jpt_p[1]*Jpt_p[2];
            const double sign = detJpt >= 0.0 ? 1.0 : -1.0;

            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  const double w = sign * 0.5 * weight_detJtr;
                  Invariant1_dMdM_2D(Jpt_p, i,j, qx,qy,e, w, P);
               }
            }
         } // qx
      } // qy
   });
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
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
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

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
            double u[2] = {0};
            double v[2] = {0};
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
            double u[2] = {0};
            double v[2] = {0};
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
            double u[2] = {0};
            double v[2] = {0};
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
            double u[2] = {0};
            double v[2] = {0};
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

// *****************************************************************************
void TMOP_Integrator::AddMultGradPA(const Vector &Xe, const Vector &Re,
                                    Vector &Ce) const
{
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
   DenseMatrix Jtr(dim);
   static bool RAND = getenv("RAND");
   if (!RAND)
   {
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      Jtr = Geometries.GetGeomToPerfGeomJac(geom_type);
      MFEM_VERIFY(Jtr.Det() == 1.0 ,"");
      {
         MFEM_VERIFY(Jtr(0,0)==1.0 && Jtr(1,1)==1.0 &&
                     Jtr(1,0)==0.0 && Jtr(0,1)==0.0,"");
      }
   }
   else
   {
      Jtr(0,0) = 1.0;
      Jtr(0,1) = 0.123;
      Jtr(1,0) = 0.456;
      Jtr(1,1) = 1.0;
   }
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
      setup = true;
      switch (id)
      {
         case 0x21: { SetupGradPA_2D<2,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x22: { SetupGradPA_2D<2,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x23: { SetupGradPA_2D<2,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x24: { SetupGradPA_2D<2,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x25: { SetupGradPA_2D<2,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

         case 0x31: { SetupGradPA_2D<3,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x32: { SetupGradPA_2D<3,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x33: { SetupGradPA_2D<3,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x34: { SetupGradPA_2D<3,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x35: { SetupGradPA_2D<3,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

         case 0x41: { SetupGradPA_2D<4,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x42: { SetupGradPA_2D<4,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x43: { SetupGradPA_2D<4,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x44: { SetupGradPA_2D<4,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x45: { SetupGradPA_2D<4,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

         case 0x51: { SetupGradPA_2D<5,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x52: { SetupGradPA_2D<5,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x53: { SetupGradPA_2D<5,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x54: { SetupGradPA_2D<5,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         case 0x55: { SetupGradPA_2D<5,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
         default:
         {
            dbg("kernel id: %x", id);
            MFEM_ABORT("Unknown kernel.");
         }
      }
   }

   switch (id)
   {
      case 0x21: return AddMultGradPA_Kernel_2D<2,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x22: return AddMultGradPA_Kernel_2D<2,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x23: return AddMultGradPA_Kernel_2D<2,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x24: return AddMultGradPA_Kernel_2D<2,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x25: return AddMultGradPA_Kernel_2D<2,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x31: return AddMultGradPA_Kernel_2D<3,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x32: return AddMultGradPA_Kernel_2D<3,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x33: return AddMultGradPA_Kernel_2D<3,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x34: return AddMultGradPA_Kernel_2D<3,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x35: return AddMultGradPA_Kernel_2D<3,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x41: return AddMultGradPA_Kernel_2D<4,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x42: return AddMultGradPA_Kernel_2D<4,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x43: return AddMultGradPA_Kernel_2D<4,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x44: return AddMultGradPA_Kernel_2D<4,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x45: return AddMultGradPA_Kernel_2D<4,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x51: return AddMultGradPA_Kernel_2D<5,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x52: return AddMultGradPA_Kernel_2D<5,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x53: return AddMultGradPA_Kernel_2D<5,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x54: return AddMultGradPA_Kernel_2D<5,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x55: return AddMultGradPA_Kernel_2D<5,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
