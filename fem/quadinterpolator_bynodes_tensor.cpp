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

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template<int T_VDIM, int T_ND, int T_NQ,
         const int MAX_ND3D = QuadratureInterpolator::MAX_ND3D,
         const int MAX_NQ3D = QuadratureInterpolator::MAX_NQ3D,
         const int MAX_VDIM3D = QuadratureInterpolator::MAX_VDIM3D,
         const int VALUES = QuadratureInterpolator::VALUES,
         const int DETERMINANTS = QuadratureInterpolator::DETERMINANTS,
         const int DERIVATIVES = QuadratureInterpolator::DERIVATIVES>
static void Eval3D(const int NE,
                   const int vdim,
                   const DofToQuad &maps,
                   const Vector &e_vec,
                   Vector &q_val,
                   Vector &q_der,
                   Vector &q_det,
                   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND3D, "");
   MFEM_VERIFY(NQ <= MAX_NQ3D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 3, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 3, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND3D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      double s_E[max_VDIM*max_ND];
      for (int d = 0; d < ND; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,e) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES) || (eval_flags & DETERMINANTS))
         {
            // use MAX_VDIM3D to avoid "subscript out of range" warnings
            double D[MAX_VDIM3D*3];
            for (int i = 0; i < 3*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               const double wz = G(q,2,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
                  D[c+VDIM*2] += s_e * wz;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
                  der(q,c,2,e) = D[c+VDIM*2];
               }
            }
            if (VDIM == 3 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 3) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 3).
               det(q,e) = kernels::Det<3>(D);
            }
         }
      }
   });
}

template<int T_VDIM, int T_D1D, int T_Q1D,
         const int MAX_ND3D = QuadratureInterpolator::MAX_ND3D,
         const int MAX_NQ3D = QuadratureInterpolator::MAX_NQ3D>
static void EvalTensor3D(const int NE,
                         const double *b_,
                         const double *x_,
                         double *y_,
                         const int vdim = 1,
                         const int d1d = 0,
                         const int q1d = 0)
{
   dbg();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_NQ3D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_ND3D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]   = (double (*)[MD1][MD1]) sm0;
      double (*DDQ)[MD1][MQ1] = (double (*)[MD1][MQ1]) sm1;
      double (*DQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm0;

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; c++)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  X[dz][dy][dx] = x(dx,dy,dz,c,e);
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
                  double u = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     u += B[qx][dx] * X[dz][dy][dx];
                  }
                  DDQ[dz][dy][qx] = u;
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
                  double u = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ[dz][qy][qx] = u;
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
                  double u = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ[dz][qy][qx] * B[qz][dz];
                  }
                  y(qx,qy,qz,c,e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM, int T_D1D, int T_Q1D,
         const int MAX_ND3D = QuadratureInterpolator::MAX_ND3D,
         const int MAX_NQ3D = QuadratureInterpolator::MAX_NQ3D,
         const int MAX_VDIM3D = QuadratureInterpolator::MAX_VDIM3D>
static void GradTensor3D(const int NE,
                         const double *b_,
                         const double *g_,
                         const double *x_,
                         double *y_,
                         const int vdim = 1,
                         const int d1d = 0,
                         const int q1d = 0)
{
   dbg();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, Q1D, Q1D, Q1D, VDIM, 3, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_NQ3D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_ND3D;

      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double s_B[MQ1][MD1];
      MFEM_SHARED double s_G[MQ1][MD1];
      DeviceTensor<2,double> B((double*)(s_B+0), Q1D, D1D);
      DeviceTensor<2,double> G((double*)(s_G+0), Q1D, D1D);

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      DeviceTensor<3,double> X((double*)(sm0+2), MD1, MD1, MD1);
      DeviceTensor<3,double> DDQ0((double*)(sm0+0), MD1, MD1, MQ1);
      DeviceTensor<3,double> DDQ1((double*)(sm0+1), MD1, MD1, MQ1);
      DeviceTensor<3,double> DQQ0((double*)(sm1+0), MD1, MQ1, MQ1);
      DeviceTensor<3,double> DQQ1((double*)(sm1+1), MD1, MQ1, MQ1);
      DeviceTensor<3,double> DQQ2((double*)(sm1+2), MD1, MQ1, MQ1);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B(q,d) = b(q,d);
               G(q,d) = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  X(dx,dy,dz) = x(dx,dy,dz,c,e);
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
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double input = X(dx,dy,dz);
                     u += input * B(qx,dx);
                     v += input * G(qx,dx);
                  }
                  DDQ0(dz,dy,qx) = u;
                  DDQ1(dz,dy,qx) = v;
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
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ1(dz,dy,qx) * B(qy,dy);
                     v += DDQ0(dz,dy,qx) * G(qy,dy);
                     w += DDQ0(dz,dy,qx) * B(qy,dy);
                  }
                  DQQ0(dz,qy,qx) = u;
                  DQQ1(dz,qy,qx) = v;
                  DQQ2(dz,qy,qx) = w;
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
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ0(dz,qy,qx) * B(qz,dz);
                     v += DQQ1(dz,qy,qx) * B(qz,dz);
                     w += DQQ2(dz,qy,qx) * G(qz,dz);
                  }
                  y(qx,qy,qz,c,0,e) = u;
                  y(qx,qy,qz,c,1,e) = v;
                  y(qx,qy,qz,c,2,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void QuadratureInterpolator::MultByNodesTensor(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det) const
{
   dbg();
   MFEM_VERIFY(use_tensor_products, "");
   MFEM_VERIFY (!(eval_flags & DETERMINANTS),"");
   const int NE = fespace->GetNE();
   if (NE == 0) { return; }
   const int vdim = fespace->GetVDim();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps_t = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps_t.ndof;
   const int Q1D = maps_t.nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;
   const double *B = maps_t.B.Read();
   const double *G = maps_t.G.Read();
   const double *X = e_vec.Read();
   double *Y_val = q_val.Write();
   double *Y_der = q_der.Write();
   if (id == 0x333)
   {
#if 0
      Eval3D<3,3,3>(NE,vdim,maps_f,e_vec,q_val,q_der,q_det,eval_flags);
#else
      if (eval_flags & VALUES) { EvalTensor3D<3,3,3>(NE, B, X, Y_val); }
      if (eval_flags & DERIVATIVES) { GradTensor3D<3,3,3>(NE, B, G, X, Y_der); }
#endif
      return;
   }

   if (id == 0x335)
   {
#if 0
      const DofToQuad &maps_f = fe->GetDofToQuad(*ir, DofToQuad::FULL);
      Eval3D<3,3,3>(NE,vdim,maps_f,e_vec,q_val,q_der,q_det,eval_flags);
#else
      if (eval_flags & VALUES) { EvalTensor3D<3,3,5>(NE, B, X, Y_val); }
      if (eval_flags & DERIVATIVES) { GradTensor3D<3,3,5>(NE, B, G, X, Y_der); }
#endif
      return;
   }
   dbg("0x%x",id);
   MFEM_ABORT("");
}

} // namespace mfem
