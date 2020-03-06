// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/blas.hpp"

namespace mfem
{

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir)
{
   fespace = &fes;
   qspace = NULL;
   IntRule = &ir;
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
               det(q,e) = D[0]*D[3] - D[1]*D[2];
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
               det(q,e) = D[0] * (D[4] * D[8] - D[5] * D[7]) +
                          D[3] * (D[2] * D[7] - D[1] * D[8]) +
                          D[6] * (D[1] * D[5] - D[2] * D[4]);
            }
         }
      }
   });
}

void QuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det) const
{
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


template<int T_D1D =0, int T_Q1D =0, int T_NBZ =0>
static void D2QValues2D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int d1d =0,
                        const int q1d =0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
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

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            DD[dy][dx] = x(dx,dy,e);
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
            y(qx,qy,e) = qq;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int T_D1D =0, int T_Q1D =0, int MAX_D =0, int MAX_Q =0>
static void D2QValues3D(const int NE,
                        const Array<double> &b_,
                        const Vector &x_,
                        Vector &y_,
                        const int d1d =0,
                        const int q1d =0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
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

      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
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
               y(qx,qy,qz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

static void D2QValues(const FiniteElementSpace &fes,
                      const DofToQuad *maps,
                      const IntegrationRule& ir,
                      const Vector &e_vec,
                      Vector &q_val)
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int D1D = fes.GetFE(0)->GetOrder() + 1;
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int id = (D1D<<4) | Q1D;

   if (dim==2)
   {
      switch (id)
      {
         case 0x24: return D2QValues2D<2,4,8>(NE, maps->B, e_vec, q_val);
         case 0x36: return D2QValues2D<3,6,4>(NE, maps->B, e_vec, q_val);
         case 0x48: return D2QValues2D<4,8,2>(NE, maps->B, e_vec, q_val);
         default: return D2QValues2D(NE, maps->B, e_vec, q_val, D1D, Q1D);
      }
   }
   if (dim==3)
   {
      switch (id)
      {
         case 0x24: return D2QValues3D<2,4>(NE, maps->B, e_vec, q_val);
         case 0x36: return D2QValues3D<3,6>(NE, maps->B, e_vec, q_val);
         case 0x48: return D2QValues3D<4,8>(NE, maps->B, e_vec, q_val);
         default:
         {
            MFEM_ASSERT(D1D<=8 && Q1D <=8, "Kernel needs the order to be <=8");
            return D2QValues3D<0,0,8,8>(NE, maps->B, e_vec, q_val, D1D, Q1D);
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

void QuadratureInterpolator::Values(const Vector &e_vec, Vector &q_val) const
{
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QValues(*fespace, &d2q,ir, e_vec, q_val);
}

template<int T_D1D =0, int T_Q1D =0, int T_NBZ =0>
static void D2QGrad2D(const int NE,
                      const double *b_,
                      const double *g_,
                      const double *x_,
                      double *y_,
                      const int d1d =0,
                      const int q1d =0)
{
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
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

      for (int c = 0; c < 2; ++c)
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

template<int T_D1D =0, int T_Q1D =0, int MAX_D =0, int MAX_Q =0>
static  void D2QGrad3D(const int NE,
                       const double *b_,
                       const double *g_,
                       const double *x_,
                       double *y_,
                       const int d1d =0,
                       const int q1d =0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
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
                    const IntegrationRule& ir,
                    const Vector &e_vec,
                    Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int D1D = fes.GetFE(0)->GetOrder() + 1;
   const int Q1D = maps->nqpt;
   const int id = (D1D<<4) | Q1D;
   const double *B = maps->B.Read();
   const double *G = maps->G.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();
   if (dim==2)
   {
      switch (id)
      {
         case 0x34: return D2QGrad2D<3,4,8>(NE, B, G, X, Y);
         case 0x46: return D2QGrad2D<4,6,4>(NE, B, G, X, Y);
         case 0x58: return D2QGrad2D<5,8,2>(NE, B, G, X, Y);
         default: return D2QGrad2D(NE, B, G, X, Y, D1D, Q1D);
      }
   }
   if (dim==3)
   {
      switch (id)
      {
         case 0x34: return D2QGrad3D<3,4>(NE, B, G, X, Y);
         case 0x46: return D2QGrad3D<4,6>(NE, B, G, X, Y);
         case 0x58: return D2QGrad3D<5,8>(NE, B, G, X, Y);
         default:
         {
            MFEM_ASSERT(D1D<=8 && Q1D <=8, "Kernel needs the order to be <=8");
            return D2QGrad3D<0,0,8,8>(NE, B, G, X, Y, D1D, Q1D);
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

template<int T_D1D =0, int T_Q1D =0, int T_NBZ =0>
static void D2QPhysGrad2D(const int NE,
                          const double *b_,
                          const double *g_,
                          const double *j_,
                          const double *x_,
                          double *y_,
                          const int d1d =0,
                          const int q1d =0)
{
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto j = Reshape(j_, Q1D, Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int VDIM = 2;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      double Yloc[VDIM][VDIM];
      double Jloc[VDIM][VDIM];
      double Jinv[VDIM][VDIM];

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

      for (int c = 0; c < 2; ++c)
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
               //Possible optimization:
               //Use exclusive memory to
               //store data at quad points
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }

      //Transfer to physical space
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {

            for (int r=0; r<VDIM; ++r)
            {
               for (int c=0; c<VDIM; ++c)
               {
                  Yloc[c][r] = y(c,r,qx,qy,e);
                  Jloc[c][r] = j(qx,qy,c,r,e);
               }
            }

            blas::CalcInverse<2>((&Jloc)[0][0], (&Jinv)[0][0]);

            for (int r=0; r<VDIM; ++r)
            {
               for (int c=0; c<VDIM; ++c)
               {
                  double dot(0.0);
                  for (int k=0; k<VDIM; ++k)
                  {
                     dot += Yloc[r][k]*Jinv[k][c];
                  }
                  y(r,c,qx,qy,e) = dot;
               }
            }

         }
      }
   });
}

template<int T_D1D =0, int T_Q1D =0, int MAX_D =0, int MAX_Q =0>
static  void D2QPhysGrad3D(const int NE,
                           const double *b_,
                           const double *g_,
                           const double *j_,
                           const double *x_,
                           double *y_,
                           const int d1d =0,
                           const int q1d =0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto j = Reshape(j_, Q1D, Q1D, Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_, VDIM, VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D;
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];

      double Yloc[VDIM][VDIM];
      double Jloc[VDIM][VDIM];
      double Jinv[VDIM][VDIM];

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
                  //Possible optimization:
                  //Use exclusive memory to
                  //store data at quad points
                  y(c,0,qx,qy,qz,e) = u;
                  y(c,1,qx,qy,qz,e) = v;
                  y(c,2,qx,qy,qz,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }

      //Transfer to physical space
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {

               for (int r=0; r<VDIM; ++r)
               {
                  for (int c=0; c<VDIM; ++c)
                  {
                     Yloc[c][r] = y(c,r,qx,qy,qz,e);
                     Jloc[c][r] = j(qx,qy,qz,c,r,e);
                  }
               }

               blas::CalcInverse<3>((&Jloc)[0][0], (&Jinv)[0][0]);

               for (int r=0; r<VDIM; ++r)
               {
                  for (int c=0; c<VDIM; ++c)
                  {

                     double dot(0.0);
                     for (int k=0; k<VDIM; ++k)
                     {
                        dot += Yloc[r][k]*Jinv[k][c];
                     }
                     y(r,c,qx,qy,qz,e) = dot;
                  }
               }

            }
         }
      }
   });
}


static void D2QPhysGrad(const FiniteElementSpace &fes,
                        const GeometricFactors *geom,
                        const DofToQuad *maps,
                        const IntegrationRule& ir,
                        const Vector &e_vec,
                        Vector &q_der)
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int D1D = fes.GetFE(0)->GetOrder() + 1;
   const int Q1D = maps->nqpt;
   const int id = (D1D<<4) | Q1D;
   const double *B = maps->B.Read();
   const double *G = maps->G.Read();
   const double *J = geom->J.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();
   if (dim==2)
   {
      switch (id)
      {
         case 0x34: return D2QPhysGrad2D<3,4,8>(NE, B, G, J, X, Y);
         case 0x46: return D2QPhysGrad2D<4,6,4>(NE, B, G, J, X, Y);
         case 0x58: return D2QPhysGrad2D<5,8,2>(NE, B, G, J, X, Y);
         default: return D2QPhysGrad2D(NE, B, G, J, X, Y, D1D, Q1D);
      }
   }
   if (dim==3)
   {
      switch (id)
      {
         case 0x34: return D2QPhysGrad3D<3,4>(NE, B, G, J, X, Y);
         case 0x46: return D2QPhysGrad3D<4,6>(NE, B, G, J, X, Y);
         case 0x58: return D2QPhysGrad3D<5,8>(NE, B, G, J, X, Y);
         default:
         {
            MFEM_ASSERT(D1D<=8 && Q1D <=8, "Kernel needs the order to be <=8");
            return D2QPhysGrad3D<0,0,8,8>(NE, B, G, J, X, Y, D1D, Q1D);
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel");
}

void QuadratureInterpolator::Derivatives(const Vector &e_vec,
                                         Vector &q_der) const
{
   const IntegrationRule &ir = *IntRule;
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QGrad(*fespace, &d2q, ir, e_vec, q_der);
}

void QuadratureInterpolator::PhysDerivatives(const Vector &e_vec,
                                             Vector &q_der) const
{

   Mesh *mesh = fespace->GetMesh();
   mesh->DeleteGeometricFactors();
   const IntegrationRule &ir = *IntRule;
   const GeometricFactors *geom =
      mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &d2q = fespace->GetFE(0)->GetDofToQuad(ir, mode);
   D2QPhysGrad(*fespace, geom, &d2q, ir, e_vec, q_der);
}


/// Returns the sign to apply to the normals on each face to point from e1 to e2.
static void GetSigns(const FiniteElementSpace &fes, const FaceType type,
                     Array<bool> &signs)
{
   const int dim = fes.GetMesh()->SpaceDimension();
   int e1, e2;
   int inf1, inf2;
   int face_id;
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      face_id = inf1 / 64;
      if ( (type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
           (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         if (dim==2)
         {
            if (face_id==2 || face_id==3)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         else if (dim==3)
         {
            if (face_id==0 || face_id==3 || face_id==4)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         f_ind++;
      }
   }
}

FaceQuadratureInterpolator::FaceQuadratureInterpolator(const FiniteElementSpace
                                                       &fes,
                                                       const IntegrationRule &ir, FaceType type_)
   : type(type_), nf(fes.GetNFbyType(type)), signs(nf)
{
   fespace = &fes;
   IntRule = &ir;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   GetSigns(*fespace, type, signs);
   const FiniteElement *fe = fespace->GetFE(0);
   const ScalarFiniteElement *sfe = dynamic_cast<const ScalarFiniteElement*>(fe);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(sfe != NULL, "Only scalar finite elements are supported");
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in FaceQuadratureInterpolator.");
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::Eval2D(
   const int NF,
   const int vdim,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &f_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(f_vec.Read(), ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = Reshape(q_val.Write(), NQ1D, VDIM, NF);
   // auto der = Reshape(q_der.Write(), NQ1D, VDIM, NF);//Only tangential der
   auto det = Reshape(q_det.Write(), NQ1D, NF);
   auto n   = Reshape(q_nor.Write(), NQ1D, VDIM, NF);
   MFEM_VERIFY(eval_flags | DERIVATIVES,
               "Derivatives on the faces are not yet supported.");
   //if Gauss-Lobatto
   MFEM_FORALL(f, NF,
   {
      const int ND1D = T_ND1D ? T_ND1D : nd;
      const int NQ1D = T_NQ1D ? T_NQ1D : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      double r_F[max_ND1D][max_VDIM];
      for (int d = 0; d < ND1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            r_F[d][c] = F(d,c,f);
         }
      }
      for (int q = 0; q < NQ1D; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*r_F[d][c]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,f) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES)
             || (eval_flags & DETERMINANTS)
             || (eval_flags & NORMALS))
         {
            double D[max_VDIM];
            for (int i = 0; i < VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const double w = G(q,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = r_F[d][c];
                  D[c] += s_e * w;
               }
            }
            if (VDIM == 2 &&
                ((eval_flags & NORMALS)
                 || (eval_flags & DETERMINANTS)))
            {
               const double norm = sqrt(D[0]*D[0]+D[1]*D[1]);
               if (eval_flags & DETERMINANTS)
               {
                  det(q,f) = norm;
               }
               if (eval_flags & NORMALS)
               {
                  const double s = sign[f] ? -1.0 : 1.0;
                  n(q,0,f) =  s*D[1]/norm;
                  n(q,1,f) = -s*D[0]/norm;
               }
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::Eval3D(
   const int NF,
   const int vdim,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(e_vec.Read(), ND1D, ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = Reshape(q_val.Write(), NQ1D, NQ1D, VDIM, NF);
   // auto der = Reshape(q_der.Write(), NQ1D, VDIM, 3, NF);
   auto det = Reshape(q_det.Write(), NQ1D, NQ1D, NF);
   auto nor = Reshape(q_nor.Write(), NQ1D, NQ1D, 3, NF);
   MFEM_VERIFY(eval_flags | DERIVATIVES,
               "Derivatives on the faces are not yet supported.");
   MFEM_FORALL(f, NF,
   {
      const int ND1D = T_ND1D ? T_ND1D : nd;
      const int NQ1D = T_NQ1D ? T_NQ1D : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_NQ1D = T_NQ1D ? T_NQ1D : MAX_NQ1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      double r_F[max_ND1D][max_ND1D][max_VDIM];
      for (int d1 = 0; d1 < ND1D; d1++)
      {
         for (int d2 = 0; d2 < ND1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               r_F[d1][d2][c] = F(d1,d2,c,f);
            }
         }
      }
      if (eval_flags & VALUES)
      {
         double Bu[max_NQ1D][max_ND1D][VDIM];
         for (int d2 = 0; d2 < ND1D; ++d2)
         {
            for (int q = 0; q < NQ1D; ++q)
            {
               for (int c = 0; c < VDIM; c++) { Bu[q][d2][c] = 0.0; }
               for (int d1 = 0; d1 < ND1D; ++d1)
               {
                  const double b = B(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     Bu[q][d2][c] += b*r_F[d1][d2][c];
                  }
               }
            }
         }
         double BBu[max_NQ1D][max_NQ1D][VDIM];
         for (int q2 = 0; q2 < NQ1D; ++q2)
         {
            for (int q1 = 0; q1 < NQ1D; ++q1)
            {
               for (int c = 0; c < VDIM; c++) { BBu[q2][q1][c] = 0.0; }
               for (int d2 = 0; d2 < ND1D; ++d2)
               {
                  const double b = B(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BBu[q2][q1][c] += b*Bu[q1][d2][c];
                  }
               }
               for (int c = 0; c < VDIM; c++)
               {
                  val(q1,q2,c,f) = BBu[q2][q1][c];
               }
            }
         }
      }
      if ((eval_flags & DERIVATIVES)
          || (eval_flags & DETERMINANTS)
          || (eval_flags & NORMALS))
      {
         //We only compute the tangential derivatives
         double Gu[max_NQ1D][max_ND1D][VDIM];
         double Bu[max_NQ1D][max_ND1D][VDIM];
         for (int d2 = 0; d2 < ND1D; ++d2)
         {
            for (int q = 0; q < NQ1D; ++q)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  Gu[q][d2][c] = 0.0;
                  Bu[q][d2][c] = 0.0;
               }
               for (int d1 = 0; d1 < ND1D; ++d1)
               {
                  const double b = B(q,d1);
                  const double g = G(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     const double u = r_F[d1][d2][c];
                     Gu[q][d2][c] += g*u;
                     Bu[q][d2][c] += b*u;
                  }
               }
            }
         }
         double BGu[max_NQ1D][max_NQ1D][VDIM];
         double GBu[max_NQ1D][max_NQ1D][VDIM];
         for (int q2 = 0; q2 < NQ1D; ++q2)
         {
            for (int q1 = 0; q1 < NQ1D; ++q1)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  BGu[q2][q1][c] = 0.0;
                  GBu[q2][q1][c] = 0.0;
               }
               for (int d2 = 0; d2 < ND1D; ++d2)
               {
                  const double b = B(q2,d2);
                  const double g = G(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BGu[q2][q1][c] += b*Gu[q1][d2][c];
                     GBu[q2][q1][c] += g*Bu[q1][d2][c];
                  }
               }
            }
         }
         if (VDIM == 3 && ((eval_flags & NORMALS) || (eval_flags & DETERMINANTS)))
         {
            double n[3];
            for (int q2 = 0; q2 < NQ1D; ++q2)
            {
               for (int q1 = 0; q1 < NQ1D; ++q1)
               {
                  const double s = sign[f] ? -1.0 : 1.0;
                  n[0] = s*( BGu[q2][q1][1]*GBu[q2][q1][2]-GBu[q2][q1][1]*BGu[q2][q1][2] );
                  n[1] = s*(-BGu[q2][q1][0]*GBu[q2][q1][2]+GBu[q2][q1][0]*BGu[q2][q1][2] );
                  n[2] = s*( BGu[q2][q1][0]*GBu[q2][q1][1]-GBu[q2][q1][0]*BGu[q2][q1][1] );
                  const double norm = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
                  if (eval_flags & DETERMINANTS) { det(q1,q2,f) = norm; }
                  if (eval_flags & NORMALS)
                  {
                     nor(q1,q2,0,f) = n[0]/norm;
                     nor(q1,q2,1,f) = n[1]/norm;
                     nor(q1,q2,2,f) = n[2]/norm;
                  }
               }
            }
         }
      }
   });
}

void FaceQuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det, Vector &q_nor) const
{
   if (nf == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe =
      fespace->GetTraceElement(0, fespace->GetMesh()->GetFaceBaseGeometry(0));
   const IntegrationRule *ir = IntRule;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   void (*eval_func)(
      const int NF,
      const int vdim,
      const DofToQuad &maps,
      const Array<bool> &signs,
      const Vector &e_vec,
      Vector &q_val,
      Vector &q_der,
      Vector &q_det,
      Vector &q_nor,
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
      eval_func(nf, vdim, maps, signs, e_vec, q_val, q_der, q_det, q_nor, eval_flags);
   }
   else
   {
      MFEM_ABORT("case not supported yet");
   }
}

} // namespace mfem
