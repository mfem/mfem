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
#define MFEM_DBG_COLOR 190
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dinvariants.hpp"

namespace mfem
{

// *****************************************************************************
// mu_302 = I1b * I2b / 9 - 1
static MFEM_HOST_DEVICE inline
double EvalW_302(const double *J)
{
   double B[9];
   kernels::InvariantsEvaluator3D ie(J,B);
   return ie.Get_I1b()*ie.Get_I2b()/9. - 1.;
}

// *****************************************************************************
// mu_321 = I1 + I2/I3 - 6
static MFEM_HOST_DEVICE inline
double EvalW_321(const double *J)
{
   double B[9];
   kernels::InvariantsEvaluator3D ie(J,B);
   return ie.Get_I1() + ie.Get_I2()/ie.Get_I3() - 6.0;
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0>
static double EnergyPA_3D(const int mid,
                          const int NE,
                          const double metric_normal,
                          const DenseMatrix &j_,
                          const Array<double> &w_,
                          const Array<double> &b_,
                          const Array<double> &g_,
                          const Vector &x_,
                          Vector &e_,
                          Vector &o_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   constexpr int dim = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto Jid = Reshape(j_.Read(), dim, dim, dim);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, dim, NE);

   auto E = Reshape(e_.Write(), Q1D, Q1D, Q1D, NE);
   auto O = Reshape(o_.Write(), Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1])(s_BG+0);
      double (*G)[MD1] = (double (*)[MD1])(s_BG+1);

      MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
      double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD+0);
      double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD+1);
      double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD+2);

      MFEM_SHARED double s_DDQ[6][MD1*MD1*MQ1];
      double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+0);
      double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+1);
      double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+2);
      double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+3);
      double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+4);
      double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ+5);

      MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
      double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+0);
      double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+1);
      double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+2);
      double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+3);
      double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+4);
      double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+5);
      double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+6);
      double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+7);
      double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ+8);

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

      // Load X(x,y,z)
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
      // Load B1d and G1d matrices
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
               double u[3] = {0};
               double v[3] = {0};
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
               double u[3] = {0};
               double v[3] = {0};
               double w[3] = {0};
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
               double u[3] = {0};
               double v[3] = {0};
               double w[3] = {0};
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
               // Jtr = targetC->ComputeElementTargets
               const double *Jtr = Jid;
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight = W(qx,qy,qz) * detJtr;

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
               const double EvalW = mid == 302 ? EvalW_302(J) :
                                    mid == 321 ? EvalW_321(J) :
                                    0.0;
               E(qx,qy,qz,e) = weight * EvalW;
               O(qx,qy,qz,e) = 1.0;
            }
         }
      }
   });
   return e_ * o_; // Energy * One
}

// *****************************************************************************
double
TMOP_Integrator::GetGridFunctionEnergyPA_3D(const FiniteElementSpace &fes,
                                            const Vector &x) const
{
   //dbg("");
   const IntegrationRule *ir = IntRule;
   MFEM_VERIFY(ir,"");

   const int NE = fes.GetMesh()->GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;

   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
   const FiniteElement *fe = fes.GetFE(0);
   const Geometry::Type geom_type = fe->GetGeomType();
   const DenseMatrix &Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
   const DenseMatrix J = Wideal;

   const double mn = metric_normal;
   MFEM_VERIFY(metric_normal == 1.0, "");

   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;

   if (elem_restrict_lex) { elem_restrict_lex->Mult(x, Xpa); }
   else { MFEM_ABORT("Not yet implemented!"); }

   const int mid = metric->Id();
   const int id = (D1D << 4 ) | Q1D;
   switch (id)
   {
      case 0x21: return EnergyPA_3D<2,1>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x22: return EnergyPA_3D<2,2>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x23: return EnergyPA_3D<2,3>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x24: return EnergyPA_3D<2,4>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x25: return EnergyPA_3D<2,5>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x26: return EnergyPA_3D<2,6>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);

      case 0x31: return EnergyPA_3D<3,1>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x32: return EnergyPA_3D<3,2>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x33: return EnergyPA_3D<3,3>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x34: return EnergyPA_3D<3,4>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x35: return EnergyPA_3D<3,5>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x36: return EnergyPA_3D<3,6>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);

      case 0x41: return EnergyPA_3D<4,1>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x42: return EnergyPA_3D<4,2>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x43: return EnergyPA_3D<4,3>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x44: return EnergyPA_3D<4,4>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x45: return EnergyPA_3D<4,5>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x46: return EnergyPA_3D<4,6>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);

      case 0x51: return EnergyPA_3D<5,1>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x52: return EnergyPA_3D<5,2>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x53: return EnergyPA_3D<5,3>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x54: return EnergyPA_3D<5,4>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x55: return EnergyPA_3D<5,5>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);
      case 0x56: return EnergyPA_3D<5,6>(mid,NE,mn,J,W,B,G,Xpa,Epa,Opa);

      default: break;
         //return EnergyPA_3D(NE, m_n, J, W, B, G, Xpa, Epa, Opa, D1D, Q1D);
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
   return 0.0;
}

} // namespace mfem
