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
#include "../linalg/invariants.hpp"

namespace mfem
{

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static double EnergyPA_2D(const int NE,
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
   constexpr int dim = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto Jtr = Reshape(j_.Read(), dim, dim);
   const auto b1d = Reshape(b_.Read(), Q1D, D1D);
   const auto g1d = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, dim, NE);

   auto E = Reshape(e_.Write(), Q1D, Q1D, NE);
   auto O = Reshape(o_.Write(), Q1D, Q1D, NE);

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
            double u[2] = {0};
            double v[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double xx = Xx[dy][dx];
               const double xy = Xy[dy][dx];
               u[0] += B1d[qx][dx] * xx;
               v[0] += G1d[qx][dx] * xx;
               u[1] += B1d[qx][dx] * xy;
               v[1] += G1d[qx][dx] * xy;
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
               u[0] += XxG[dy][qx] * B1d[qy][dy];
               v[0] += XxB[dy][qx] * G1d[qy][dy];
               u[1] += XyG[dy][qx] * B1d[qy][dy];
               v[1] += XyB[dy][qx] * G1d[qy][dy];
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
            // Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = Jtr(0,0);
            const double Jtrx1 = Jtr(0,1);
            const double Jtry0 = Jtr(1,0);
            const double Jtry1 = Jtr(1,1);
            const double Jtr[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};
            const double detJtr = (Jtr[0]*Jtr[3])-(Jtr[1]*Jtr[2]);

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            const double Jprx0 = Xx0[qy][qx];
            const double Jprx1 = Xx1[qy][qx];
            const double Jpry0 = Xy0[qy][qx];
            const double Jpry1 = Xy1[qy][qx];
            const double Jpr[4] = {Jprx0, Jpry0, Jprx1, Jpry1};

            // J = Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
            double J[4];
            kernels::Mult(2,2,2, Jpr, Jrt, J);

            // TMOP_Metric_002::EvalW: 0.5 * ie.Get_I1b() - 1.0;
            const double weight = W(qx,qy) * detJtr;
            const double I1 = J[0]*J[0] + J[1]*J[1] +
                              J[2]*J[2] + J[3]*J[3];
            const double detJpt = J[0]*J[3] - J[1]*J[2];
            const double sign_detJpt = ScalarOps<double>::sign(detJpt);
            const double I2b = sign_detJpt*detJpt;
            const double I1b = I1 / I2b;
            const double metric_EvalW = 0.5 * I1b - 1.0;
            const double EvalW = metric_normal * metric_EvalW;
            E(qx,qy,e) = weight * EvalW;
            O(qx,qy,e) = 1.0;
         }
      }
   });
   return e_ * o_; // Energy * One
}

// *****************************************************************************
double
TMOP_Integrator::GetGridFunctionEnergyPA_2D(const FiniteElementSpace &fes,
                                            const Vector &x) const
{
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

   const double m_n = metric_normal;
   MFEM_VERIFY(metric_normal == 1.0, "");

   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;

   if (elem_restrict_lex) { elem_restrict_lex->Mult(x, Xpa); }
   else { MFEM_ABORT("Not yet implemented!"); }

   const int id = (D1D << 4 ) | Q1D;
   switch (id)
   {
      case 0x21: return EnergyPA_2D<2,1,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x22: return EnergyPA_2D<2,2,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x23: return EnergyPA_2D<2,3,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x24: return EnergyPA_2D<2,4,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x25: return EnergyPA_2D<2,5,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x26: return EnergyPA_2D<2,6,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);

      case 0x31: return EnergyPA_2D<3,1,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x32: return EnergyPA_2D<3,2,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x33: return EnergyPA_2D<3,3,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x34: return EnergyPA_2D<3,4,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x35: return EnergyPA_2D<3,5,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x36: return EnergyPA_2D<3,6,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);

      case 0x41: return EnergyPA_2D<4,1,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x42: return EnergyPA_2D<4,2,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x43: return EnergyPA_2D<4,3,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x44: return EnergyPA_2D<4,4,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x45: return EnergyPA_2D<4,5,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x46: return EnergyPA_2D<4,6,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);

      case 0x51: return EnergyPA_2D<5,1,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x52: return EnergyPA_2D<5,2,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x53: return EnergyPA_2D<5,3,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x54: return EnergyPA_2D<5,4,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x55: return EnergyPA_2D<5,5,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      case 0x56: return EnergyPA_2D<5,6,1>(NE, m_n, J, W, B, G, Xpa, Epa, Opa);
      default: break;
         //return EnergyPA_2D(NE, m_n, J, W, B, G, Xpa, Epa, Opa, D1D, Q1D);
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
   return 0.0;
}

} // namespace mfem
