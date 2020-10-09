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

namespace mfem
{

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir)
{
   fespace = &fes;
   qspace = NULL;
   IntRule = &ir;
   q_layout = QVectorLayout::byNODES;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const QuadratureSpace &qs)
{
   fespace = &fes;
   qspace = &qs;
   IntRule = NULL;
   q_layout = QVectorLayout::byNODES;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Eval2D(
   const int NE,
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
   MFEM_VERIFY(ND <= MAX_ND2D, "");
   MFEM_VERIFY(NQ <= MAX_NQ2D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 2, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 2, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND2D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
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
            // use MAX_VDIM2D to avoid "subscript out of range" warnings
            double D[MAX_VDIM2D*2];
            for (int i = 0; i < 2*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
               }
            }
            if (VDIM == 2 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 2) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 2).
               det(q,e) = kernels::Det<2>(D);
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Eval3D(
   const int NE,
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

void QuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det) const
{
   if (q_layout == QVectorLayout::byVDIM)
   {
      if (eval_flags & VALUES) { Values(e_vec, q_val); }
      if (eval_flags & DERIVATIVES) { Derivatives(e_vec, q_der); }
      if (eval_flags & DETERMINANTS)
      {
         MFEM_ABORT("evaluation of determinants with 'byVDIM' output layout"
                    " is not implemented yet!");
      }
      return;
   }

   // q_layout == QVectorLayout::byNODES
   const int ne = fespace->GetNE();
   if (ne == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::FULL);
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   void (*eval_func)(
      const int NE,
      const int vdim,
      const DofToQuad &maps,
      const Vector &e_vec,
      Vector &q_val,
      Vector &q_der,
      Vector &q_det,
      const int eval_flags) = NULL;
   if (vdim == 1)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q0
            case 101: eval_func = &Eval2D<1,1,1>; break;
            case 104: eval_func = &Eval2D<1,1,4>; break;
            // Q1
            case 404: eval_func = &Eval2D<1,4,4>; break;
            case 409: eval_func = &Eval2D<1,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<1,9,9>; break;
            case 916: eval_func = &Eval2D<1,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<1,16,16>; break;
            case 1625: eval_func = &Eval2D<1,16,25>; break;
            case 1636: eval_func = &Eval2D<1,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<1,25,25>; break;
            case 2536: eval_func = &Eval2D<1,25,36>; break;
            case 2549: eval_func = &Eval2D<1,25,49>; break;
            case 2564: eval_func = &Eval2D<1,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<1>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q0
            case 1001: eval_func = &Eval3D<1,1,1>; break;
            case 1008: eval_func = &Eval3D<1,1,8>; break;
            // Q1
            case 8008: eval_func = &Eval3D<1,8,8>; break;
            case 8027: eval_func = &Eval3D<1,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<1,27,27>; break;
            case 27064: eval_func = &Eval3D<1,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<1,64,64>; break;
            case 64125: eval_func = &Eval3D<1,64,125>; break;
            case 64216: eval_func = &Eval3D<1,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<1,125,125>; break;
            case 125216: eval_func = &Eval3D<1,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<1>;
         }
      }
   }
   else if (vdim == 3 && dim == 2)
   {
      switch (100*nd + nq)
      {
         // Q0
         case 101: eval_func = &Eval2D<3,1,1>; break;
         case 104: eval_func = &Eval2D<3,1,4>; break;
         // Q1
         case 404: eval_func = &Eval2D<3,4,4>; break;
         case 409: eval_func = &Eval2D<3,4,9>; break;
         // Q2
         case 904: eval_func = &Eval2D<3,9,4>; break;
         case 909: eval_func = &Eval2D<3,9,9>; break;
         case 916: eval_func = &Eval2D<3,9,16>; break;
         case 925: eval_func = &Eval2D<3,9,25>; break;
         // Q3
         case 1616: eval_func = &Eval2D<3,16,16>; break;
         case 1625: eval_func = &Eval2D<3,16,25>; break;
         case 1636: eval_func = &Eval2D<3,16,36>; break;
         // Q4
         case 2525: eval_func = &Eval2D<3,25,25>; break;
         case 2536: eval_func = &Eval2D<3,25,36>; break;
         case 2549: eval_func = &Eval2D<3,25,49>; break;
         case 2564: eval_func = &Eval2D<3,25,64>; break;
         default:   eval_func = &Eval2D<3>;
      }
   }
   else if (vdim == dim)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q1
            case 404: eval_func = &Eval2D<2,4,4>; break;
            case 409: eval_func = &Eval2D<2,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<2,9,9>; break;
            case 916: eval_func = &Eval2D<2,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<2,16,16>; break;
            case 1625: eval_func = &Eval2D<2,16,25>; break;
            case 1636: eval_func = &Eval2D<2,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<2,25,25>; break;
            case 2536: eval_func = &Eval2D<2,25,36>; break;
            case 2549: eval_func = &Eval2D<2,25,49>; break;
            case 2564: eval_func = &Eval2D<2,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<2>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q1
            case 8008: eval_func = &Eval3D<3,8,8>; break;
            case 8027: eval_func = &Eval3D<3,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<3,27,27>; break;
            case 27064: eval_func = &Eval3D<3,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<3,64,64>; break;
            case 64125: eval_func = &Eval3D<3,64,125>; break;
            case 64216: eval_func = &Eval3D<3,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<3,125,125>; break;
            case 125216: eval_func = &Eval3D<3,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<3>;
         }
      }
   }
   if (eval_func)
   {
      eval_func(ne, vdim, maps, e_vec, q_val, q_der, q_det, eval_flags);
   }
   else
   {
      MFEM_ABORT("case not supported yet");
   }
}

void QuadratureInterpolator::MultTranspose(
   unsigned eval_flags, const Vector &q_val, const Vector &q_der,
   Vector &e_vec) const
{
   MFEM_ABORT("this method is not implemented yet");
}


template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void D2QValues2D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int vdim = 1,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      const int zid = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];

      MFEM_SHARED double DDz[NBZ][MD1*MD1];
      double (*DD)[MD1] = (double (*)[MD1])(DDz + zid);

      MFEM_SHARED double DQz[NBZ][MD1*MQ1];
      double (*DQ)[MQ1] = (double (*)[MQ1])(DQz + zid);

      if (zid == 0)
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
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               DD[dy][dx] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double dq = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  dq += B[qx][dx] * DD[dy][dx];
               }
               DQ[dy][qx] = dq;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double qq = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  qq += DQ[dy][qx] * B[qy][dy];
               }
               y(c,qx,qy,e) = qq;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static void D2QValues3D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int vdim = 1,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
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
                  y(c,qx,qy,qz,e) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void D2QValues(const FiniteElementSpace &fes,
                      const DofToQuad *maps,
                      const Vector &e_vec,
                      Vector &q_val)
{
   const int dim = fes.GetMesh()->Dimension();
   const int vdim = fes.GetVDim();
   const int NE = fes.GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x124: return D2QValues2D<1,2,4,8>(NE, maps->B, e_vec, q_val);
         case 0x136: return D2QValues2D<1,3,6,4>(NE, maps->B, e_vec, q_val);
         case 0x148: return D2QValues2D<1,4,8,2>(NE, maps->B, e_vec, q_val);
         case 0x224: return D2QValues2D<2,2,4,8>(NE, maps->B, e_vec, q_val);
         case 0x236: return D2QValues2D<2,3,6,4>(NE, maps->B, e_vec, q_val);
         case 0x248: return D2QValues2D<2,4,8,2>(NE, maps->B, e_vec, q_val);
         default:
         {
            MFEM_VERIFY(D1D <= MAX_D1D, "Orders higher than " << MAX_D1D-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MAX_Q1D, "Quadrature rules with more than "
                        << MAX_Q1D << " 1D points are not supported!");
            D2QValues2D(NE, maps->B, e_vec, q_val, vdim, D1D, Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x124: return D2QValues3D<1,2,4>(NE, maps->B, e_vec, q_val);
         case 0x136: return D2QValues3D<1,3,6>(NE, maps->B, e_vec, q_val);
         case 0x148: return D2QValues3D<1,4,8>(NE, maps->B, e_vec, q_val);
         case 0x324: return D2QValues3D<3,2,4>(NE, maps->B, e_vec, q_val);
         case 0x336: return D2QValues3D<3,3,6>(NE, maps->B, e_vec, q_val);
         case 0x348: return D2QValues3D<3,4,8>(NE, maps->B, e_vec, q_val);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than " << MQ
                        << " 1D points are not supported!");
            D2QValues3D<0,0,0,MD,MQ>(NE, maps->B, e_vec, q_val, vdim, D1D, Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

void QuadratureInterpolator::Values(const Vector &e_vec, Vector &q_val) const
{
   if (q_layout == QVectorLayout::byNODES)
   {
      Vector empty;
      Mult(e_vec, VALUES, q_val, empty, empty);
      return;
   }

   // q_layout == QVectorLayout::byVDIM
   if (fespace->GetNE() == 0) { return; }
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QValues(*fespace, &d2q, e_vec, q_val);
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void D2QGrad2D(const int NE,
                      const double *b_,
                      const double *g_,
                      const double *x_,
                      double *y_,
                      const int vdim = 1,
                      const int d1d = 0,
                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 2, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);

      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      double (*DQ0)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
      double (*DQ1)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);

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

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               X[dx][dy] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static  void D2QGrad3D(const int NE,
                       const double *b_,
                       const double *g_,
                       const double *x_,
                       double *y_,
                       const int vdim = 1,
                       const int d1d = 0,
                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);

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

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  X[dx][dy][dz] = x(dx,dy,dz,c,e);
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
                     const double coords = X[dx][dy][dz];
                     u += coords * B[qx][dx];
                     v += coords * G[qx][dx];
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
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
                     u += DDQ1[dz][dy][qx] * B[qy][dy];
                     v += DDQ0[dz][dy][qx] * G[qy][dy];
                     w += DDQ0[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
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
                     u += DQQ0[dz][qy][qx] * B[qz][dz];
                     v += DQQ1[dz][qy][qx] * B[qz][dz];
                     w += DQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  y(c,0,qx,qy,qz,e) = u;
                  y(c,1,qx,qy,qz,e) = v;
                  y(c,2,qx,qy,qz,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void D2QGrad(const FiniteElementSpace &fes,
                    const DofToQuad *maps,
                    const Vector &e_vec,
                    Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int vdim = fes.GetVDim();
   const int NE = fes.GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;
   const double *B = maps->B.Read();
   const double *G = maps->G.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();
   if (dim == 2)
   {
      switch (id)
      {
         case 0x134: return D2QGrad2D<1,3,4,8>(NE, B, G, X, Y);
         case 0x146: return D2QGrad2D<1,4,6,4>(NE, B, G, X, Y);
         case 0x158: return D2QGrad2D<1,5,8,2>(NE, B, G, X, Y);
         case 0x234: return D2QGrad2D<2,3,4,8>(NE, B, G, X, Y);
         case 0x246: return D2QGrad2D<2,4,6,4>(NE, B, G, X, Y);
         case 0x258: return D2QGrad2D<2,5,8,2>(NE, B, G, X, Y);
         default:
         {
            MFEM_VERIFY(D1D <= MAX_D1D, "Orders higher than " << MAX_D1D-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MAX_Q1D, "Quadrature rules with more than "
                        << MAX_Q1D << " 1D points are not supported!");
            D2QGrad2D(NE, B, G, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x134: return D2QGrad3D<1,3,4>(NE, B, G, X, Y);
         case 0x146: return D2QGrad3D<1,4,6>(NE, B, G, X, Y);
         case 0x158: return D2QGrad3D<1,5,8>(NE, B, G, X, Y);
         case 0x334: return D2QGrad3D<3,3,4>(NE, B, G, X, Y);
         case 0x346: return D2QGrad3D<3,4,6>(NE, B, G, X, Y);
         case 0x358: return D2QGrad3D<3,5,8>(NE, B, G, X, Y);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than " << MQ
                        << " 1D points are not supported!");
            D2QGrad3D<0,0,0,MD,MQ>(NE, B, G, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void D2QPhysGrad2D(const int NE,
                          const double *b_,
                          const double *g_,
                          const double *j_,
                          const double *x_,
                          double *y_,
                          const int vdim = 1,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto j = Reshape(j_, Q1D, Q1D, 2, 2, NE);
   auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 2, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);

      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      double (*DQ0)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
      double (*DQ1)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);

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

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               X[dx][dy] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               double Jloc[4], Jinv[4];
               Jloc[0] = j(qx,qy,0,0,e);
               Jloc[1] = j(qx,qy,1,0,e);
               Jloc[2] = j(qx,qy,0,1,e);
               Jloc[3] = j(qx,qy,1,1,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               y(c,0,qx,qy,e) = Jinv[0]*u + Jinv[1]*v;
               y(c,1,qx,qy,e) = Jinv[2]*u + Jinv[3]*v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D = 0, int MAX_Q = 0>
static  void D2QPhysGrad3D(const int NE,
                           const double *b_,
                           const double *g_,
                           const double *j_,
                           const double *x_,
                           double *y_,
                           const int vdim = 1,
                           const int d1d = 0,
                           const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto j = Reshape(j_, Q1D, Q1D, Q1D, 3, 3, NE);
   auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);

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

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {

                  X[dx][dy][dz] = x(dx,dy,dz,c,e);
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
                     const double coords = X[dx][dy][dz];
                     u += coords * B[qx][dx];
                     v += coords * G[qx][dx];
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
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
                     u += DDQ1[dz][dy][qx] * B[qy][dy];
                     v += DDQ0[dz][dy][qx] * G[qy][dy];
                     w += DDQ0[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
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
                     u += DQQ0[dz][qy][qx] * B[qz][dz];
                     v += DQQ1[dz][qy][qx] * B[qz][dz];
                     w += DQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  double Jloc[9], Jinv[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = j(qx,qy,qz,row,col,e);
                     }
                  }
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  y(c,0,qx,qy,qz,e) = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                  y(c,1,qx,qy,qz,e) = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                  y(c,2,qx,qy,qz,e) = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}


static void D2QPhysGrad(const FiniteElementSpace &fes,
                        const GeometricFactors *geom,
                        const DofToQuad *maps,
                        const Vector &e_vec,
                        Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int vdim = fes.GetVDim();
   const int NE = fes.GetNE();
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const int id = (vdim<<8) | (D1D<<4) | Q1D;
   const double *B = maps->B.Read();
   const double *G = maps->G.Read();
   const double *J = geom->J.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();
   if (dim == 2)
   {
      switch (id)
      {
         case 0x134: return D2QPhysGrad2D<1,3,4,8>(NE, B, G, J, X, Y);
         case 0x146: return D2QPhysGrad2D<1,4,6,4>(NE, B, G, J, X, Y);
         case 0x158: return D2QPhysGrad2D<1,5,8,2>(NE, B, G, J, X, Y);
         case 0x234: return D2QPhysGrad2D<2,3,4,8>(NE, B, G, J, X, Y);
         case 0x246: return D2QPhysGrad2D<2,4,6,4>(NE, B, G, J, X, Y);
         case 0x258: return D2QPhysGrad2D<2,5,8,2>(NE, B, G, J, X, Y);
         default:
         {
            MFEM_VERIFY(D1D <= MAX_D1D, "Orders higher than " << MAX_D1D-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MAX_Q1D, "Quadrature rules with more than "
                        << MAX_Q1D << " 1D points are not supported!");
            D2QPhysGrad2D(NE, B, G, J, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x134: return D2QPhysGrad3D<1,3,4>(NE, B, G, J, X, Y);
         case 0x146: return D2QPhysGrad3D<1,4,6>(NE, B, G, J, X, Y);
         case 0x158: return D2QPhysGrad3D<1,5,8>(NE, B, G, J, X, Y);
         case 0x334: return D2QPhysGrad3D<3,3,4>(NE, B, G, J, X, Y);
         case 0x346: return D2QPhysGrad3D<3,4,6>(NE, B, G, J, X, Y);
         case 0x358: return D2QPhysGrad3D<3,5,8>(NE, B, G, J, X, Y);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than " << MQ
                        << " 1D points are not supported!");
            D2QPhysGrad3D<0,0,0,MD,MQ>(NE, B, G, J, X, Y, vdim, D1D, Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

void QuadratureInterpolator::Derivatives(const Vector &e_vec,
                                         Vector &q_der) const
{
   if (q_layout == QVectorLayout::byNODES)
   {
      Vector empty;
      Mult(e_vec, DERIVATIVES, empty, q_der, empty);
      return;
   }

   // q_layout == QVectorLayout::byVDIM
   if (fespace->GetNE() == 0) { return; }
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QGrad(*fespace, &d2q, e_vec, q_der);
}

void QuadratureInterpolator::PhysDerivatives(const Vector &e_vec,
                                             Vector &q_der) const
{
   if (q_layout == QVectorLayout::byNODES)
   {
      MFEM_ABORT("evaluation of physical derivatives with 'byNODES' output"
                 " layout is not implemented yet!");
      return;
   }

   // q_layout == QVectorLayout::byVDIM
   Mesh *mesh = fespace->GetMesh();
   if (mesh->GetNE() == 0) { return; }
   // mesh->DeleteGeometricFactors(); // This should be done outside
   const IntegrationRule &ir = *IntRule;
   const GeometricFactors *geom =
      mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QPhysGrad(*fespace, geom, &d2q, e_vec, q_der);
}

} // namespace mfem
