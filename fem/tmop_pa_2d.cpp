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

namespace mfem
{

// *****************************************************************************
// dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
MFEM_HOST_DEVICE inline
void Eval_dI1b(const double w, const double *J, double *dI1b)
{
   const double det = J[0]*J[3] - J[1]*J[2];
   const double sign_detJ = det >= 0.0 ? 1.0 : -1.0;
   const double I2b = sign_detJ * det;

   const double I1 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2] + J[3]*J[3];
   const double I1b = I1/I2b;

   const double c1 = 2.0/I2b;
   const double c2 = I1b/2.0;

   const double dI2b0 =  sign_detJ*J[3];
   const double dI2b1 = -sign_detJ*J[2];
   const double dI2b2 = -sign_detJ*J[1];
   const double dI2b3 =  sign_detJ*J[0];

   dI1b[0] = w * c1 * (J[0] - c2*dI2b0);
   dI1b[1] = w * c1 * (J[1] - c2*dI2b1);
   dI1b[2] = w * c1 * (J[2] - c2*dI2b2);
   dI1b[3] = w * c1 * (J[3] - c2*dI2b3);
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
            const double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};
            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double weight_detJtr = weight * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt_p[4];
            kernels::CalcInverse<2>(Jtr_p, Jrt_p);

            // G = X{^T}.DSh
            const double Gx0 = QQx0[qy][qx];
            const double Gx1 = QQx1[qy][qx];
            const double Gy0 = QQy0[qy][qx];
            const double Gy1 = QQy1[qy][qx];
            double G_p[4] = {Gx0, Gy0, Gx1, Gy1};

            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = G.Jrt
            double Jpt_p[4];
            kernels::Mult(2,2,2, G_p, Jrt_p, Jpt_p);

            // metric->EvalP(Jpt, P);
            double P[4];
            Eval_dI1b(0.5*weight_detJtr, Jpt_p, P);

            // PMatO +=  DS . P^t += DSh . (Jrt . (P==Jpt)^t)
            double A[4];
            kernels::MultABt(2,2,2, Jrt_p, P, A);
            QQx0[qy][qx] = A[0];
            QQy0[qy][qx] = A[2];
            QQx1[qy][qx] = A[1];
            QQy1[qy][qx] = A[3];
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
void TMOP_Integrator::AddMultPA_2D(const Vector &X, Vector &Y) const
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
   DenseMatrix Wideal(dim);
   static bool RAND = getenv("RAND");
   if (!RAND)
   {
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
      MFEM_VERIFY(Wideal.Det() == 1.0 ,"");
      {
         MFEM_VERIFY(Wideal(0,0)==1.0 && Wideal(1,1)==1.0 &&
                     Wideal(1,0)==0.0 && Wideal(0,1)==0.0,"");
      }
   }
   else
   {
      Wideal(0,0) = 1.0;
      Wideal(0,1) = 0.123;
      Wideal(1,0) = 0.456;
      Wideal(1,1) = 1.0;
   }
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
   auto J = Reshape(Dpa.Write(), Q1D, Q1D, dim, dim, ne);
   MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            J(qx,qy,0,0,e) = Jtr(0,0);
            J(qx,qy,0,1,e) = Jtr(0,1);
            J(qx,qy,1,0,e) = Jtr(1,0);
            J(qx,qy,1,1,e) = Jtr(1,1);
         }
      }
   });

   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_2D<2,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x22: return AddMultPA_Kernel_2D<2,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x23: return AddMultPA_Kernel_2D<2,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x24: return AddMultPA_Kernel_2D<2,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x25: return AddMultPA_Kernel_2D<2,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x26: return AddMultPA_Kernel_2D<2,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x31: return AddMultPA_Kernel_2D<3,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x32: return AddMultPA_Kernel_2D<3,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x33: return AddMultPA_Kernel_2D<3,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x34: return AddMultPA_Kernel_2D<3,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x35: return AddMultPA_Kernel_2D<3,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x36: return AddMultPA_Kernel_2D<3,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x41: return AddMultPA_Kernel_2D<4,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x42: return AddMultPA_Kernel_2D<4,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x43: return AddMultPA_Kernel_2D<4,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x44: return AddMultPA_Kernel_2D<4,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x45: return AddMultPA_Kernel_2D<4,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x46: return AddMultPA_Kernel_2D<4,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x51: return AddMultPA_Kernel_2D<5,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x52: return AddMultPA_Kernel_2D<5,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x53: return AddMultPA_Kernel_2D<5,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x54: return AddMultPA_Kernel_2D<5,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x55: return AddMultPA_Kernel_2D<5,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x56: return AddMultPA_Kernel_2D<5,6,1>(ne,W,B1d,G1d,Dpa,X,Y);
      default:  break;
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
