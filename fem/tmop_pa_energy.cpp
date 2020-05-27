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
#define MFEM_DBG_COLOR 201
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/invariants.hpp"

namespace mfem
{

// *****************************************************************************
double TMOP_Integrator::GetGridFunctionEnergyPA(const FiniteElementSpace &fes,
                                                const Vector &x) const
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      dbg("");
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");
   const int NE = fes.GetMesh()->GetNE();
   const int NQ = ir->GetNPoints();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;

   DenseTensor JtrQ(dim, dim, NQ*NE);
   DenseTensor JptQ(dim, dim, NQ*NE);

   x.HostRead();
   for (int e = 0; e < NE; e++) // NonlinearForm::GetGridFunctionEnergy
   {
      Vector el_x;
      Array<int> vdofs;
      const FiniteElement *fe = fes.GetFE(e);
      fes.GetElementVDofs(e, vdofs);
      ElementTransformation &T = *fes.GetElementTransformation(e);
      x.GetSubVector(vdofs, el_x);
      {
         // TMOP_Integrator::GetElementEnergy
         // ... fe => el, el_x => elfun
         const FiniteElement &el = *fe;
         const Vector &elfun = el_x;
         const int dof = el.GetDof(), dim = el.GetDim();

         DSh.SetSize(dof, dim);
         Jrt.SetSize(dim);
         Jpr.SetSize(dim);
         Jpt.SetSize(dim);
         PMatI.UseExternalData(elfun.GetData(), dof, dim);
         DenseTensor Jtr(dim, dim, NQ);
         targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
         for (int q = 0; q < NQ; q++)
         {
            //dbg("Jtr(%d):",q); Jtr(q).Print();
            JtrQ(e*NQ+q) = Jtr(q);
            MFEM_VERIFY(Jtr(q).Det() == 1.0 ,"");
            {
               MFEM_VERIFY(Jtr(q)(0,0)==1.0 && Jtr(q)(1,1)==1.0 &&
                           Jtr(q)(1,0)==0.0 && Jtr(q)(0,1)==0.0,"");
            }
         }
         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir->IntPoint(q);
            const DenseMatrix &Jtr_i = Jtr(q);
            //metric->SetTargetJacobian(Jtr_i);
            CalcInverse(Jtr_i, Jrt);
            el.CalcDShape(ip, DSh);
            MultAtB(PMatI, DSh, Jpr);
            Mult(Jpr, Jrt, Jpt);
            JptQ(e*NQ+q) = Jpt;
            //dbg("Jpt(%d,%d):",e,q); Jpt.Print();
         }
      }
   }

   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
   const FiniteElement *fe = fes.GetFE(0);
   const Geometry::Type geom_type = fe->GetGeomType();
   const DenseMatrix &Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);

   const auto b1d = Reshape(maps->B.Read(), Q1D, D1D);
   const auto g1d = Reshape(maps->G.Read(), Q1D, D1D);

   const auto W = ir->GetWeights().Read();
   const auto Jtr = Reshape(JtrQ.Read(), dim, dim, NE*NQ);
   const auto Jpt = Reshape(JptQ.Read(), dim, dim, NE*NQ);
   const auto Jid = Reshape(Wideal.Read(), dim, dim);

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *elem_restrict_lex = fes.GetElementRestriction(ordering);
   Vector xe;
   if (elem_restrict_lex)
   {
      xe.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      elem_restrict_lex->Mult(x, xe);
   }
   else {MFEM_ABORT("Not implemented!");}

   const auto X = Reshape(xe.Read(), D1D, D1D, dim, NE);
   MFEM_VERIFY(NQ == Q1D*Q1D, "");
   Vector energy(NE*NQ), one(NE*NQ);
   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);
   auto O = Reshape(one.Write(), Q1D, Q1D, NE);
   const double metric_normal_d = metric_normal;
   MFEM_VERIFY(metric_normal == 1.0, "");
   MFEM_VERIFY(D1D==3,"");
   MFEM_VERIFY(Q1D==2,"");
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      constexpr int NBZ = 1;
      constexpr int D1D = 3;
      constexpr int Q1D = 2;
      constexpr int MD1 = D1D;
      constexpr int MQ1 = Q1D;

      const int tidz = MFEM_THREAD_ID(z);

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
            const int q = qx + qy * Q1D;

            // Jtr = targetC->ComputeElementTargets
            const double Jtr11 = Jtr(0,0,e*NQ+q);
            const double Jtr12 = Jtr(1,0,e*NQ+q);
            const double Jtr21 = Jtr(0,1,e*NQ+q);
            const double Jtr22 = Jtr(1,1,e*NQ+q);
            const double detJtr = (Jtr11*Jtr22)-(Jtr21*Jtr12);

            const double weight = W[q] * detJtr;

            const double J[4] =
            {
               Jpt(0,0,e*NQ+q), Jpt(1,0,e*NQ+q),
               Jpt(0,1,e*NQ+q), Jpt(1,1,e*NQ+q)
            };
            //dbg("Jpt1: %f, %f, %f, %f", J[0], J[1], J[2], J[3]);

            {
               // Jtr = targetC->ComputeElementTargets
               const double Jtrx0 = Jid(0,0); MFEM_VERIFY(Jtrx0==Jtr11,"");
               const double Jtrx1 = Jid(0,1); MFEM_VERIFY(Jtrx1==Jtr12,"");
               const double Jtry0 = Jid(1,0); MFEM_VERIFY(Jtry0==Jtr21,"");
               const double Jtry1 = Jid(1,1); MFEM_VERIFY(Jtry1==Jtr22,"");
               double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};
               DenseMatrix JTR(Jtr_p, dim, dim);

               // Jrt = Jtr^{-1}
               DenseMatrix Jrt(dim);
               kernels::CalcInverse<2>(Jtr_p, Jrt.HostWrite());
               MFEM_VERIFY(Jrt.Det() == detJtr, "");
               //dbg("Jrt:"); Jrt.Print();

               // Jpr = X^T.DSh
               const double Jprx0 = Xx0[qy][qx];
               const double Jprx1 = Xx1[qy][qx];
               const double Jpry0 = Xy0[qy][qx];
               const double Jpry1 = Xy1[qy][qx];
               double Jpr_p[4] = {Jprx0, Jpry0, Jprx1, Jpry1};
               DenseMatrix Jpr(Jpr_p, dim, dim);
               //dbg("Jpr(%d,%d):",e,q); Jpr.Print();

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               DenseMatrix J2(dim);
               Mult(Jpr, Jrt, J2);
               //dbg("Jpt(%d,%d):",e,q); J2.Print();
               //dbg("Jpt2: %f, %f, %f, %f", J2(0,0), J2(1,0), J2(0,1), J2(1,1));
               const double EPS = 1.e-14;
               MFEM_VERIFY(fabs(J2(0,0)-J[0])<EPS,"");
               MFEM_VERIFY(fabs(J2(1,0)-J[1])<EPS,"");
               MFEM_VERIFY(fabs(J2(0,1)-J[2])<EPS,"");
               MFEM_VERIFY(fabs(J2(1,1)-J[3])<EPS,"");
            }

            // TMOP_Metric_002::EvalW: 0.5 * ie.Get_I1b() - 1.0;
            const double I1 = J[0]*J[0] + J[1]*J[1] +
                              J[2]*J[2] + J[3]*J[3];
            const double detJpt = J[0]*J[3] - J[1]*J[2];
            const double sign_detJpt = ScalarOps<double>::sign(detJpt);
            const double I2b = sign_detJpt*detJpt;
            const double I1b = I1 / I2b;
            const double metric_EvalW = 0.5 * I1b - 1.0;
            const double EvalW = metric_normal_d * metric_EvalW;
            E(qx,qy,e) = weight * EvalW;

            O(qx,qy,e) = 1.0;
         }
      }
   });
   return energy * one;
}

} // namespace mfem
