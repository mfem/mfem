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
#define MFEM_DBG_COLOR 221
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

// *****************************************************************************
double TMOP_Integrator::GetGridFunctionEnergyPA(const FiniteElementSpace &fes,
                                                const Vector &x) const
{
   dbg("");
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
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();

   DenseTensor Jtr_E(dim, dim, NQ*NE);
   DenseTensor Jpt_E(dim, dim, NQ*NE);

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
         for (int i = 0; i < NQ; i++) { Jtr_E(e*NQ+i) = Jtr(i); }
         for (int i = 0; i < NQ; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const DenseMatrix &Jtr_i = Jtr(i);
            metric->SetTargetJacobian(Jtr_i);
            CalcInverse(Jtr_i, Jrt);
            el.CalcDShape(ip, DSh);
            MultAtB(PMatI, DSh, Jpr);
            Mult(Jpr, Jrt, Jpt);
            Jpt_E(e*NQ+i) = Jpt;
         }
      }
   }

   const auto W = ir->GetWeights().Read();
   const auto Jtr = Reshape(Jtr_E.Read(), dim, dim, NE*NQ);
   const auto Jpt = Reshape(Jpt_E.Read(), dim, dim, NE*NQ);
   MFEM_VERIFY(NQ == Q1D*Q1D, "");
   Vector energy(NE*NQ), one(NE*NQ);
   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);
   auto O = Reshape(one.Write(), Q1D, Q1D, NE);
   const double metric_normal_d = metric_normal;
   MFEM_VERIFY(metric_normal == 1.0, "");
   //InvariantsEvaluator2D<double> ie;
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int i = qx + qy * Q1D;
            //const IntegrationPoint &ip = ir->IntPoint(i);
            const double J11 = Jtr(0,0,e*NQ+i);
            const double J12 = Jtr(1,0,e*NQ+i);
            const double J21 = Jtr(0,1,e*NQ+i);
            const double J22 = Jtr(1,1,e*NQ+i);
            const double Jtr_i_Det = (J11*J22)-(J21*J12);
            const double weight = W[i]* Jtr_i_Det;
            double JPT[4];
            DenseMatrix Jpt_a(dim);
            {
               JPT[0] = Jpt(0,0,e*NQ+i);
               JPT[1] = Jpt(1,0,e*NQ+i);
               JPT[2] = Jpt(0,1,e*NQ+i);
               JPT[3] = Jpt(1,1,e*NQ+i);
               Jpt_a.UseExternalData(JPT, dim, dim);
            }
            const double val = metric_normal_d * metric->EvalW(Jpt_a);
            // TMOP_Metric_002::EvalW: 0.5 * ie.Get_I1b() - 1.0;
            // Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
            //ie.SetJacobian(Jpt.GetData());
            //const double metric_EvalW = 0.5 * ie.Get_I1b() - 1.0;
            //const double val = metric_normal_d * metric_EvalW;
            E(qx,qy,e) = weight * val;
            O(qx,qy,e) = 1.0;
         }
      }
   });
   return energy * one;
}

// *****************************************************************************
// Setup dim, ne, nq, maps, (geom) & fes
void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fespace)
{
   dbg("");
   fes = &fespace;
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh const *mesh = fes->GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(IntRule,"");
   MFEM_VERIFY(dim == 2, "");
   nq = IntRule->GetNPoints();
   ne = fes->GetMesh()->GetNE();
   const IntegrationRule &ir = *IntRule;
   maps = &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   //const int flags = GeometricFactors::COORDINATES|GeometricFactors::JACOBIANS;
   //geom = mesh->GetGeometricFactors(ir, flags);
   D.SetSize(dim * dim * nq * ne, Device::GetDeviceMemoryType());
   const int dof = fes->GetFE(0)->GetDof();
   JrtD.SetSize(dof * dim * nq * ne, Device::GetDeviceMemoryType());
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultPA_Kernel_2D(const int NE,
                                const Array<double> &w_,
                                const Array<double> &b_,
                                const Array<double> &g_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   dbg("");
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, VDIM, VDIM, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   dbg("D1D:%d, Q1D:%d, nq:%d", D1D, Q1D, Q1D*Q1D);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[2][NBZ][MD1*MD1];
      MFEM_SHARED double GD[4][NBZ][MD1*MQ1];
      MFEM_SHARED double GQ[4][NBZ][MQ1*MQ1];
      double (*Xx)[MD1]   = (double (*)[MD1])(Xz[0] + tidz);
      double (*Xy)[MD1]   = (double (*)[MD1])(Xz[1] + tidz);

      double (*DQxB)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
      double (*DQxG)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);
      double (*DQyB)[MQ1] = (double (*)[MQ1])(GD[2] + tidz);
      double (*DQyG)[MQ1] = (double (*)[MQ1])(GD[3] + tidz);

      double (*QQx0)[MQ1] = (double (*)[MQ1])(GQ[0] + tidz);
      double (*QQx1)[MQ1] = (double (*)[MQ1])(GQ[1] + tidz);
      double (*QQy0)[MQ1] = (double (*)[MQ1])(GQ[2] + tidz);
      double (*QQy1)[MQ1] = (double (*)[MQ1])(GQ[3] + tidz);

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dy][dx] = X(dx,dy,0,e);
            Xy[dy][dx] = X(dx,dy,1,e);
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
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] = {0};
            double v[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double cx = Xx[dy][dx];
               const double cy = Xy[dy][dx];
               u[0] += B[qx][dx] * cx;
               v[0] += G[qx][dx] * cx;
               u[1] += B[qx][dx] * cy;
               v[1] += G[qx][dx] * cy;
            }
            DQxB[dy][qx] = u[0];
            DQxG[dy][qx] = v[0];
            DQyB[dy][qx] = u[1];
            DQyG[dy][qx] = v[1];
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
               u[0] += DQxG[dy][qx] * B[qy][dy];
               v[0] += DQxB[dy][qx] * G[qy][dy];
               u[1] += DQyG[dy][qx] * B[qy][dy];
               v[1] += DQyB[dy][qx] * G[qy][dy];
            }
            QQx0[qy][qx] = u[0];
            QQx1[qy][qx] = v[0];
            QQy0[qy][qx] = u[1];
            QQy1[qy][qx] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double weight = W(qx,qy);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = D(qx,qy,0,0,e);
            const double Jtrx1 = D(qx,qy,0,1,e);
            const double Jtry0 = D(qx,qy,1,0,e);
            const double Jtry1 = D(qx,qy,1,1,e);
            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double w = weight * detJtr;

            // Jrt = Jtr^{-1}
            const double Jrt0x =  Jtry1 / detJtr;
            const double Jrt0y = -Jtrx1 / detJtr;
            const double Jrt1x = -Jtry0 / detJtr;
            const double Jrt1y =  Jtrx0 / detJtr;
            /*{
               const double detJrt = (Jrt0x*Jrt1y)-(Jrt0y*Jrt1x);
               dbg("\033[0mdetJrt: %.15e", detJrt);
               dbg("Jrt: %.15e %.15e",Jrt0x,Jrt0y);
               dbg("Jrt: %.15e %.15e",Jrt1x,Jrt1y);
            }*/

            // G = X{^T}.DSh
            const double Gx0 = QQx0[qy][qx];
            const double Gx1 = QQx1[qy][qx];
            const double Gy0 = QQy0[qy][qx];
            const double Gy1 = QQy1[qy][qx];
            /*{
               const double detG = Gx0*Gy1 - Gx1*Gy0;
               dbg("");
               dbg("\033[0mdetG: %.15e",detG);
               dbg("G: %.15e %.15e",Gx0,Gx1);
               dbg("G: %.15e %.15e",Gy0,Gy1);
            }*/


            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = G.Jrt
            //             |Jrt0x Jrt0y|
            //             |Jrt1x Jrt1y|
            //   |Gx0 Gx1| |Jptxx Jptxy|
            //   |Gy0 Gy1| |Jptyx Jptyy|
            const double Jptxx = ((Gx0 * Jrt0x) + (Gx1 * Jrt1x));
            const double Jptxy = ((Gx0 * Jrt0y) + (Gx1 * Jrt1y));
            const double Jptyx = ((Gy0 * Jrt0x) + (Gy1 * Jrt1x));
            const double Jptyy = ((Gy0 * Jrt0y) + (Gy1 * Jrt1y));
            /*{
               const double detJpt = Jptxx*Jptyy - Jptxy*Jptyx;
               dbg("\033[0mdetJpt: %.15e",detJpt);
               dbg("Jpt: %.15e %.15e",Jptxx,Jptxy);
               dbg("Jpt: %.15e %.15e",Jptyx,Jptyy);
            }*/

            // metric->EvalP(Jpt, P);
            const double J[4]= {Jptxx, Jptyx, Jptxy, Jptyy};
            InvariantsEvaluator2D<double> ie;
            ie.SetJacobian(J);
            DenseMatrix P(2);
            P.Set(0.5, ie.Get_dI1b());
            /*{
               const double detP = P(0,0)*P(1,1) - P(0,1)*P(1,0);
               dbg("\033[0mdetP %.15e",detP);
               dbg("P: %.15e %.15e",P(0,0),P(0,1));
               dbg("P: %.15e %.15e",P(1,0),P(1,1));
            }*/

            // P(0,0) = Jptxx; P(0,1) = Jptxy; P(1,0) = Jptyx; P(1,1) = Jptyy;

            const double Pxx = w * P(0,0);
            const double Pxy = w * P(0,1);
            const double Pyx = w * P(1,0);
            const double Pyy = w * P(1,1);
            /*{
               const double detP = Pxx*Pyy - Pxy*Pyx;
               dbg("\033[0mdetP %.15e",detP);
               dbg("P: %.15e %.15e",Pxx,Pxy);
               dbg("P: %.15e %.15e",Pyx,Pyy);
            }*/

            // PMatO +=  DS . P^t += DSh . (Jrt . (P==Jpt)^t)
            // Jrt . Jpt^t:
            // |Pxx Pxy|^{T}  => |Pxx Pyx|
            // |Pyx Pyy|         |Pxy Pyy|
            //     |Jrt0x Jrt0y|  A0x A0y
            //     |Jrt1x Jrt1y|  A1x A1y
            const double A0x = Jrt0x*Pxx + Jrt0y*Pxy;
            const double A0y = Jrt0x*Pyx + Jrt0y*Pyy;
            const double A1x = Jrt1x*Pxx + Jrt1y*Pxy;
            const double A1y = Jrt1x*Pyx + Jrt1y*Pyy;
            QQx0[qy][qx] = A0x;
            QQy0[qy][qx] = A0y;
            QQx1[qy][qx] = A1x;
            QQy1[qy][qx] = A1y;
            /*{
               dbg("\033[0mdetA: %.15e", A0x*A1y - A1x*A0y);
               dbg("A: %.15e %.15e",A0x,A0y);
               dbg("A: %.15e %.15e",A1x,A1y);
            }*/
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = b(q,d);
               Gt[d][q] = g(q,d);
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
               u[0] += Gt[dx][qx] * QQx0[qy][qx];
               v[0] += Bt[dx][qx] * QQx1[qy][qx];
               u[1] += Gt[dx][qx] * QQy0[qy][qx];
               v[1] += Bt[dx][qx] * QQy1[qy][qx];
            }
            DQxB[dx][qy] = u[0];
            DQxG[dx][qy] = v[0];
            DQyB[dx][qy] = u[1];
            DQyG[dx][qy] = v[1];
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
               u[0] += DQxB[dx][qy] * Bt[dy][qy];
               v[0] += DQxG[dx][qy] * Gt[dy][qy];
               u[1] += DQyB[dx][qy] * Bt[dy][qy];
               v[1] += DQyG[dx][qy] * Gt[dy][qy];
            }
            Y(dx,dy,0,e) += u[0] + v[0];
            Y(dx,dy,1,e) += u[1] + v[1];
         }
      }
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultPA(const Vector &X, Vector &Y) const
{
   dbg("X: %.15e", X*X);
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   {
      // Jtr setup:
      //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
      //  - Jtr(i) == Wideal
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      const DenseMatrix Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
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
      const auto Jtr = Reshape(Wideal.Read(), dim, dim);
      auto G = Reshape(D.Write(), Q1D, Q1D, dim, dim, ne);
      MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               G(qx,qy,0,0,e) = Jtr(0,0);
               G(qx,qy,0,1,e) = Jtr(0,1);
               G(qx,qy,1,0,e) = Jtr(1,0);
               G(qx,qy,1,1,e) = Jtr(1,1);
            }
         }
      });
   }

   switch (id)
   {

      case 0x21: return AddMultPA_Kernel_2D<2,1,1>(ne,W,B,G,D,X,Y);/*
      case 0x23: return AddMultPA_Kernel_2D<2,3,1>(ne,W,B,G,D,X,Y);

      case 0x31: return AddMultPA_Kernel_2D<3,1,1>(ne,W,B,G,D,X,Y);
      case 0x32: return AddMultPA_Kernel_2D<3,2,1>(ne,W,B,G,D,X,Y);
      case 0x33: return AddMultPA_Kernel_2D<3,3,1>(ne,W,B,G,D,X,Y);
      case 0x35: return AddMultPA_Kernel_2D<3,5,1>(ne,W,B,G,D,X,Y);

      case 0x41: return AddMultPA_Kernel_2D<4,1,1>(ne,W,B,G,D,X,Y);
      case 0x42: return AddMultPA_Kernel_2D<4,2,1>(ne,W,B,G,D,X,Y);
      case 0x43: return AddMultPA_Kernel_2D<4,3,1>(ne,W,B,G,D,X,Y);
      case 0x44: return AddMultPA_Kernel_2D<4,4,1>(ne,W,B,G,D,X,Y);

      case 0x52: return AddMultPA_Kernel_2D<5,2,1>(ne,W,B,G,D,X,Y);
      case 0x55: return AddMultPA_Kernel_2D<5,5,1>(ne,W,B,G,D,X,Y);
      case 0x57: return AddMultPA_Kernel_2D<5,7,1>(ne,W,B,G,D,X,Y);*/
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}


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
static void AddMultGradPA_Kernel_2D(const Vector &GradX,
                                    const int NE,
                                    const Array<double> &w_,
                                    const Array<double> &b_,
                                    const Array<double> &g_,
                                    const Vector &d_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   dbg("");
   constexpr int dim =2;
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   MFEM_VERIFY(D1D == 2,"");
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(Q1D == 1,"");
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, VDIM, VDIM, NE);
   auto GX = Reshape(GradX.Read(), D1D, D1D, VDIM, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   dbg("D1D:%d, Q1D:%d, nq:%d", D1D, Q1D, Q1D*Q1D);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[2][2][NBZ][MD1*MD1];
      MFEM_SHARED double GXz[2][2][NBZ][MD1*MD1];
      MFEM_SHARED double GD[2][4][NBZ][MD1*MQ1];
      MFEM_SHARED double GQ[2][4][NBZ][MQ1*MQ1];
      double (*Xx)[MD1]   = (double (*)[MD1])(Xz[0] + tidz);
      double (*Xy)[MD1]   = (double (*)[MD1])(Xz[1] + tidz);
      double (*GXx)[MD1]  = (double (*)[MD1])(GXz[0] + tidz);
      double (*GXy)[MD1]  = (double (*)[MD1])(GXz[1] + tidz);

      double (*DQxB)[MQ1] = (double (*)[MQ1])(GD[0][0] + tidz);
      double (*DQxG)[MQ1] = (double (*)[MQ1])(GD[0][1] + tidz);
      double (*DQyB)[MQ1] = (double (*)[MQ1])(GD[0][2] + tidz);
      double (*DQyG)[MQ1] = (double (*)[MQ1])(GD[0][3] + tidz);

      double (*QQx0)[MQ1] = (double (*)[MQ1])(GQ[0][0] + tidz);
      double (*QQx1)[MQ1] = (double (*)[MQ1])(GQ[0][1] + tidz);
      double (*QQy0)[MQ1] = (double (*)[MQ1])(GQ[0][2] + tidz);
      double (*QQy1)[MQ1] = (double (*)[MQ1])(GQ[0][3] + tidz);

      double (*GDQxB)[MQ1] = (double (*)[MQ1])(GD[1][0] + tidz);
      double (*GDQxG)[MQ1] = (double (*)[MQ1])(GD[1][1] + tidz);
      double (*GDQyB)[MQ1] = (double (*)[MQ1])(GD[1][2] + tidz);
      double (*GDQyG)[MQ1] = (double (*)[MQ1])(GD[1][3] + tidz);

      double (*GQQx0)[MQ1] = (double (*)[MQ1])(GQ[1][0] + tidz);
      double (*GQQx1)[MQ1] = (double (*)[MQ1])(GQ[1][1] + tidz);
      double (*GQQy0)[MQ1] = (double (*)[MQ1])(GQ[1][2] + tidz);
      double (*GQQy1)[MQ1] = (double (*)[MQ1])(GQ[1][3] + tidz);

      for (int _i_ = 0; _i_ < dim; _i_++)
      {
         for (int _j_ = 0; _j_ < dim; _j_++)
         {
            // Load X(x,y) and GradX(x,y)
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  Xx[dy][dx] = X(dx,dy,0,e);
                  Xy[dy][dx] = X(dx,dy,1,e);
                  GXx[dy][dx] = GX(dx,dy,0,e);
                  GXy[dy][dx] = GX(dx,dy,1,e);
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
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u[2]  = {0};
                  double v[2]  = {0};
                  double gu[2] = {0};
                  double gv[2] = {0};
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double cx = Xx[dy][dx];
                     const double cy = Xy[dy][dx];
                     //dbg("X(%f,%f)",cx,cy);
                     u[0] += B[qx][dx] * cx;
                     v[0] += G[qx][dx] * cx;
                     u[1] += B[qx][dx] * cy;
                     v[1] += G[qx][dx] * cy;
                     const double gcx = GXx[dy][dx];
                     const double gcy = GXy[dy][dx];
                     gu[0] += B[qx][dx] * gcx;
                     gv[0] += G[qx][dx] * gcx;
                     gu[1] += B[qx][dx] * gcy;
                     gv[1] += G[qx][dx] * gcy;
                  }
                  DQxB[dy][qx]  = u[0];
                  DQxG[dy][qx]  = v[0];
                  DQyB[dy][qx]  = u[1];
                  DQyG[dy][qx]  = v[1];

                  GDQxB[dy][qx] = gu[0];
                  GDQxG[dy][qx] = gv[0];
                  GDQyB[dy][qx] = gu[1];
                  GDQyG[dy][qx] = gv[1];

               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u[2]  = {0};
                  double v[2]  = {0};
                  double gu[2] = {0};
                  double gv[2] = {0};
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u[0] += DQxG[dy][qx] * B[qy][dy];
                     v[0] += DQxB[dy][qx] * G[qy][dy];
                     u[1] += DQyG[dy][qx] * B[qy][dy];
                     v[1] += DQyB[dy][qx] * G[qy][dy];

                     gu[0] += GDQxG[dy][qx] * B[qy][dy];
                     gv[0] += GDQxB[dy][qx] * G[qy][dy];
                     gu[1] += GDQyG[dy][qx] * B[qy][dy];
                     gv[1] += GDQyB[dy][qx] * G[qy][dy];
                  }
                  QQx0[qy][qx]  = u[0];
                  QQx1[qy][qx]  = v[0];
                  QQy0[qy][qx]  = u[1];
                  QQy1[qy][qx]  = v[1];

                  GQQx0[qy][qx] = gu[0];
                  GQQx1[qy][qx] = gv[0];
                  GQQy0[qy][qx] = gu[1];
                  GQQy1[qy][qx] = gv[1];
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double weight = W(qx,qy);

                  //  Jtr = targetC->ComputeElementTargets
                  const double Jtrx0 = D(qx,qy,0,0,e);
                  const double Jtrx1 = D(qx,qy,0,1,e);
                  const double Jtry0 = D(qx,qy,1,0,e);
                  const double Jtry1 = D(qx,qy,1,1,e);
                  const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
                  const double weight_m = weight * detJtr;
                  dbg("\033[7;31mQ(%d,%d): weight_m:%f",qx,qy,weight_m);

                  // Jrt = Jtr^{-1}
                  const double Jrt0x =  Jtry1 / detJtr;
                  const double Jrt0y = -Jtrx1 / detJtr;
                  const double Jrt1x = -Jtry0 / detJtr;
                  const double Jrt1y =  Jtrx0 / detJtr;
                  DenseMatrix Jrt(VDIM);
                  Jrt(0,0) = Jrt0x; Jrt(0,1) = Jrt0y;
                  Jrt(1,0) = Jrt1x; Jrt(1,1) = Jrt1y;
                  {
                     const double detJrt = (Jrt0x*Jrt1y)-(Jrt0y*Jrt1x);
                     dbg("\033[0mdetJrt: %.15e", detJrt);
                     dbg("Jrt: %.15e %.15e",Jrt0x,Jrt0y);
                     dbg("Jrt: %.15e %.15e",Jrt1x,Jrt1y);
                  }

                  // Compute DSh (dof x dim)
                  const int dof = D1D*D1D;
                  DenseMatrix DSh(dof, dim);
                  for (int i1 = 0; i1 < D1D; ++i1)
                  {
                     for (int i2 = 0; i2 < D1D; ++i2)
                     {
                        const double bg = G[qx][i1] * B[qy][i2];
                        const double gb = B[qx][i1] * G[qy][i2];
                        const int dof = i2 + i1*D1D;
                        DSh(dof, 1) = bg;
                        DSh(dof, 0) = gb;
                     }
                  }
                  //dbg("DSh:"); DSh.Print(); //exit(0);

                  // Compute DS = DSh Jrt
                  DenseMatrix DS(dof, dim);
                  Mult(DSh, Jrt, DS);
                  dbg("DS:"); DS.Print(); //exit(0);

                  // G = X^T.DSh
                  const double Gx0 = QQx0[qy][qx];
                  const double Gx1 = QQx1[qy][qx];
                  const double Gy0 = QQy0[qy][qx];
                  const double Gy1 = QQy1[qy][qx];
                  {
                     const double detG = Gx0*Gy1 - Gx1*Gy0;
                     dbg("\033[0mdetG: %.15e",detG);
                     dbg("G: %.15e %.15e",Gx0,Gx1);
                     dbg("G: %.15e %.15e",Gy0,Gy1);
                     //dbg("G: %.15e %.15e",Gx0,Gy0);
                  }

                  // GG = GX^T.DSh
                  const double GGx0 = GQQx0[qy][qx];
                  const double GGx1 = GQQx1[qy][qx];
                  const double GGy0 = GQQy0[qy][qx];
                  const double GGy1 = GQQy1[qy][qx];
                  {
                     const double detGG = GGx0*GGy1 - GGx1*GGy0;
                     dbg("\033[0mdetGG: %.15e",detGG);
                     dbg("GG: %.15e %.15e",GGx0,GGx1);
                     dbg("GG: %.15e %.15e",GGy0,GGy1);
                  }

                  // GJpt = GX^T.DS = (GX^T.DSh).Jrt = GG.Jrt
                  //                |Jrt0x Jrt0y|
                  //                |Jrt1x Jrt1y|
                  //   |GGx0 GGx1| |GJptxx GJptxy|
                  //   |GGy0 GGy1| |GJptyx GJptyy|
                  const double GJptxx = ((GGx0 * Jrt0x) + (GGx1 * Jrt1x));
                  const double GJptxy = ((GGx0 * Jrt0y) + (GGx1 * Jrt1y));
                  const double GJptyx = ((GGy0 * Jrt0x) + (GGy1 * Jrt1x));
                  const double GJptyy = ((GGy0 * Jrt0y) + (GGy1 * Jrt1y));
                  {
                     const double detGJpt = GJptxx*GJptyy - GJptxy*GJptyx;
                     dbg("\033[0mdetGJpt: %.15e",detGJpt);
                     dbg("GJpt: %.15e %.15e",GJptxx,GJptxy);
                     dbg("GJpt: %.15e %.15e",GJptyx,GJptyy);
                  }
                  double GJpt_p[4] = {GJptxx, GJptyx, GJptxy, GJptyy};
                  DenseMatrix GJpt(GJpt_p, dim, dim);

                  //metric->AssembleH(GJpt, DS, weight_m, elmat);
                  InvariantsEvaluator2D<double> ie;
                  ie.SetJacobian(GJpt_p);
                  ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
                  DenseMatrix elmat(dof*dim);
                  elmat = 0.0;
                  ie.Assemble_ddI1b(0.5*weight_m, elmat.GetData());
                  dbg("ELMAT:"); elmat.Print();

                  DenseMatrix P(dim);
                  DenseMatrix Pelmat(dof*dim);
                  Pelmat = 0.0;
                  DenseMatrix dI1_dMdM(dim);
                  // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
                  for (int r = 0; r < dim; r++)
                  {
                     for (int c = 0; c < dim; c++)
                     {
                        Dim2Invariant1_dMdM(GJpt, r, c, dI1_dMdM);
                        // Compute each entry of d(Prc)_dJ.
                        for (int rr = 0; rr < dim; rr++)
                        {
                           for (int cc = 0; cc < dim; cc++)
                           {
                              const double entry_rr_cc = 0.5 * dI1_dMdM(rr,cc);
                              for (int i = 0; i < dof; i++)
                              {
                                 for (int j = 0; j < dof; j++)
                                 {
                                    const double ds = DS(i, c) * DS(j, cc);
                                    //dbg("ds[(%d,%d),(%d,%d)]=%.15e",i,c,j,cc,ds);
                                    Pelmat(i+r*dof, j+rr*dof) +=
                                       weight_m * ds * entry_rr_cc;
                                 }
                              }
                           }
                        }
                     }
                  }

                  const double EPS = 1.e-8;
                  const bool flip = GJpt.Det() < 0.0;
                  Pelmat *= flip ? -1.0 : 1.0;
                  dbg("P_ELMAT:"); Pelmat.Print();
                  for (int i = 0; i < dim*dof; i++)
                  {
                     for (int j = 0; j < dim*dof; j++)
                     {
                        if (fabs(elmat(i,j)-Pelmat(i,j)) > EPS)
                        {
                           dbg("\033[31m%.15e", elmat(i,j));
                           dbg("\033[31m%.15e", Pelmat(i,j));
                        }
                        MFEM_VERIFY(fabs(elmat(i,j)-Pelmat(i,j)) < EPS,"");
                     }
                  }

                  Dim2Invariant1_dMdM(GJpt, _i_, _j_, P);
                  P *= -0.5 * weight_m;
                  P = 0.5 * weight_m * 0.33;//((_i_+1.0)*(_j_+1.0));
                  {
                     const double detP = P.Det();
                     dbg("\033[0mdetP %.15e",detP);
                     dbg("P: %.15e %.15e",P(0,0),P(0,1));
                     dbg("P: %.15e %.15e",P(1,0),P(1,1));
                  }
                  //dbg("P:"); P.Print();

                  // Y += DS . P^t
                  // Y += (DSh . Jrt) . P^t
                  // Y += DSh . (Jrt . P^t), with P = dMdM_GJpt    ?????
                  // Y += DSh . (Jrt . (dMdM_GJpt . X)^t)
                  // Y += DSh . (Jrt . (X^t . dMdM_GJpt^t))

                  //             | P00 P01 |
                  //             | P10 P11 |
                  // | Gx0 Gx1 |   Pxx Pxy
                  // | Gy0 Gy1 |   Pyx Pyy
                  /*const double Pxx = Gx0*P(0,0) + Gx1*P(1,0);
                  const double Pxy = Gx0*P(0,1) + Gx1*P(1,1);
                  const double Pyx = Gy0*P(0,0) + Gy1*P(1,0);
                  const double Pyy = Gy0*P(0,1) + Gy1*P(1,1);*/

                  //             | Gx0 Gx1 |
                  //             | Gy0 Gy1 |
                  // | P00 P01 |   Pxx Pxy
                  // | P10 P11 |   Pyx Pyy
                  const double Pxx = P(0,0)*Gx0 + P(0,1)*Gy0;
                  const double Pxy = P(0,0)*Gx1 + P(0,1)*Gy1;
                  const double Pyx = P(1,0)*Gx0 + P(1,1)*Gy0;
                  const double Pyy = P(1,0)*Gx1 + P(1,1)*Gy1;

                  /*QQx0[qy][qx] = Pxx;
                  QQy0[qy][qx] = Pyx;
                  QQx1[qy][qx] = Pxy;
                  QQy1[qy][qx] = Pyy;*/

                  const double A0x = Jrt0x*Pxx + Jrt0y*Pxy;
                  const double A0y = Jrt0x*Pyx + Jrt0y*Pyy;
                  const double A1x = Jrt1x*Pxx + Jrt1y*Pxy;
                  const double A1y = Jrt1x*Pyx + Jrt1y*Pyy;
                  QQx0[qy][qx] = A0x;
                  QQy0[qy][qx] = A0y;
                  QQx1[qy][qx] = A1x;
                  QQy1[qy][qx] = A1y;

                  /*QQx0[qy][qx] = Pxx;
                  QQy0[qy][qx] = Pxy;
                  QQx1[qy][qx] = Pyx;
                  QQy1[qy][qx] = Pyy;*/

                  /* const double Pxx = P(0,0);
                   const double Pxy = P(0,1);
                   const double Pyx = P(1,0);
                   const double Pyy = P(1,1);*/

                  //               |Gx0|
                  //               |Gy0|
                  //   | P00 P01 |  Pxx
                  //   | P10 P11 |  Pyy
                  //const double Pxx = P(0,0)*Gx0 + P(0,1)*Gy0;
                  //const double Pyy = P(1,0)*Gx0 + P(1,1)*Gy0;

                  //            | P00 P01 |
                  //            | P10 P11 |
                  // |Gx0 Gy0|    Pxx Pyy
                  //const double Pxx = P(0,0)*Gx0 + P(1,0)*Gy0;
                  //const double Pyy = P(0,1)*Gx0 + P(1,1)*Gy0;

                  //                   |Pxx|
                  //                   |Pyy|
                  //     |Jrt0x Jrt0y|  Axx
                  //     |Jrt1x Jrt1y|  Ayy
                  //QQx0[qy][qx] = Jrt0x*Pxx + Jrt0y*Pyy;
                  //QQy0[qy][qx] = Jrt1x*Pxx + Jrt1y*Pyy;

                  //             |Jrt0x Jrt0y|
                  //             |Jrt1x Jrt1y|
                  //   |Pxx Pyy|
                  //QQx0[qy][qx] = Jrt0x*Pxx + Jrt1x*Pyy;
                  //QQy0[qy][qx] = Jrt0y*Pxx + Jrt1y*Pyy;

                  //dbg("QQx,y: %.15e,%.15e",QQx0[qy][qx], QQy0[qy][qx]);
               }
            }
            MFEM_SYNC_THREAD;
            if (tidz == 0)
            {
               MFEM_FOREACH_THREAD(d,y,D1D)
               {
                  MFEM_FOREACH_THREAD(q,x,Q1D)
                  {
                     Bt[d][q] = b(q,d);
                     Gt[d][q] = g(q,d);
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
                     u[0] += Gt[dx][qx] * QQx0[qy][qx];
                     v[0] += Bt[dx][qx] * QQx1[qy][qx];
                     u[1] += Gt[dx][qx] * QQy0[qy][qx];
                     v[1] += Bt[dx][qx] * QQy1[qy][qx];
                  }
                  DQxB[dx][qy] = u[0];
                  DQxG[dx][qy] = v[0];
                  DQyB[dx][qy] = u[1];
                  DQyG[dx][qy] = v[1];
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
                     u[0] += DQxB[dx][qy] * Bt[dy][qy];
                     v[0] += DQxG[dx][qy] * Gt[dy][qy];
                     u[1] += DQyB[dx][qy] * Bt[dy][qy];
                     v[1] += DQyG[dx][qy] * Gt[dy][qy];
                  }
                  Y(dx,dy,0,e) += u[0] + v[0];
                  Y(dx,dy,1,e) += u[1] + v[1];
               }
            }
         } // _j_
      } // _i_
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultGradPA(const Vector &GradX,
                                    const Vector &X, Vector &Y) const
{
   dbg("x:%d, y:%d", X.Size(), Y.Size());
   dbg("GradX: %.15e, X: %.15e", GradX*GradX, X*X);
   dbg("X:"); X.Print();
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   {
      // Jtr setup:
      //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
      //  - Jtr(i) == Wideal
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      const DenseMatrix Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
      //Wideal.Print();
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
      const auto Jtr = Reshape(Wideal.Read(), dim, dim);
      auto G = Reshape(D.Write(), Q1D, Q1D, dim, dim, ne);
      MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               G(qx,qy,0,0,e) = Jtr(0,0);
               G(qx,qy,0,1,e) = Jtr(0,1);
               G(qx,qy,1,0,e) = Jtr(1,0);
               G(qx,qy,1,1,e) = Jtr(1,1);
            }
         }
      });
   }

   switch (id)
   {
      case 0x21: return AddMultGradPA_Kernel_2D<2,1,1>(GradX,ne,W,B,G,D,X,Y);/*
      case 0x23: return AddMultGradPA_Kernel_2D<2,3,1>(GradX,ne,W,B,G,D,X,Y);

      case 0x31: return AddMultGradPA_Kernel_2D<3,1,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x32: return AddMultGradPA_Kernel_2D<3,2,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x33: return AddMultGradPA_Kernel_2D<3,3,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x35: return AddMultGradPA_Kernel_2D<3,5,1>(GradX,ne,W,B,G,D,X,Y);

      case 0x41: return AddMultGradPA_Kernel_2D<4,1,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x42: return AddMultGradPA_Kernel_2D<4,2,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x43: return AddMultGradPA_Kernel_2D<4,3,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x44: return AddMultGradPA_Kernel_2D<4,4,1>(GradX,ne,W,B,G,D,X,Y);

      case 0x52: return AddMultGradPA_Kernel_2D<5,2,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x55: return AddMultGradPA_Kernel_2D<5,5,1>(GradX,ne,W,B,G,D,X,Y);
      case 0x57: return AddMultGradPA_Kernel_2D<5,7,1>(GradX,ne,W,B,G,D,X,Y);*/
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
