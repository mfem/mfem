// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{
// PA DG Trace Integrator

// PA DG Trace Assemble 2D kernel for constant velocity
static void PADGTraceSetup2D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const double rho,
                             const Vector &u,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
   // const Operator *elem_restr =
   //    fespace->GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   // elem_restr->Mult(*nodes, Fnodes);//TODO replace with L2FaceRestriction

   const int VDIM = 2;

   auto d = Reshape(det.Read(), Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, VDIM, NF);
   auto c = Reshape(u.Read(), Q1D, VDIM, NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, 2, 2, NF);

   MFEM_FORALL(f, NF,//can be optimized with Q1D thread for NF blocks
   {
      for (int q = 0; q < Q1D; ++q)
      {
         const double dot = n(q,0,f) * c(q,0,f) + n(q,1,f) * c(q,1,f);
         const double abs = dot > 0.0 ? dot : -dot;
         const double w = W[q]*d(q,f);
         qd(q,0,0,f) = w*( alpha/2 * dot + beta * abs );
         qd(q,1,0,f) = w*( alpha/2 * dot - beta * abs );
         qd(q,0,1,f) = w*(-alpha/2 * dot - beta * abs );
         qd(q,1,1,f) = w*(-alpha/2 * dot + beta * abs );
      }
   });
}

// PA DG Trace Assemble 2D kernel for constant velocity
static void PADGTraceSetup3D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const double rho,
                             const Vector &u,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
   // const Operator *elem_restr =
   //    fespace->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   // elem_restr->Mult(*nodes, Enodes);//TODO replace with L2FaceRestriction

   const int VDIM = 3;
   const int Q = Q1D*Q1D;

   auto d = Reshape(det.Read(), Q, NF);
   auto n = Reshape(nor.Read(), Q, VDIM, NF);
   auto c = Reshape(u.Read(), Q, VDIM, NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q, 2, 2, NF);

   MFEM_FORALL(f, NF,//can be optimized with Q1D*Q1D threads for NF blocks
   {
      for (int q = 0; q < Q; ++q)
      {
         const double dot = n(q,0,f) * c(q,0,f) + n(q,1,f) * c(q,1,f) + n(q,2,f) * c(q,2,f);
         const double abs = dot > 0.0 ? dot : -dot;
         const double w = W[q]*d(q,f);
         qd(q,0,0,f) = w*( alpha/2 * dot + beta * abs );
         qd(q,1,0,f) = w*( alpha/2 * dot - beta * abs );
         qd(q,0,1,f) = w*(-alpha/2 * dot - beta * abs );
         qd(q,1,1,f) = w*(-alpha/2 * dot + beta * abs );
      }
   });
}

static void PADGTraceSetup(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &W,
                           const Vector &det,
                           const Vector &nor,
                           const double rho,
                           const Vector &u,
                           const double alpha,
                           const double beta,
                           Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGTraceSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
   if (dim == 3)
   {
      PADGTraceSetup3D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
}

void DGTraceIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFaceElement(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   const int dims = el.GetDim();
   const int symmDims = 4;
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   nf = mesh->GetNFaces();
   geom = mesh->GetFaceGeometricFactors(*ir,
      FaceGeometricFactors::DETERMINANTS & FaceGeometricFactors::NORMALS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
   ConstantCoefficient *c_rho = dynamic_cast<ConstantCoefficient*>(rho);
   MFEM_VERIFY(c_rho != NULL, "only ConstantCoefficient is supported!");
   const double r = c_rho->constant;
   VectorConstantCoefficient *c_u = dynamic_cast<VectorConstantCoefficient*>(u);
   MFEM_VERIFY(c_u != NULL, "only ConstantCoefficient is supported!");
   const Vector& Q = c_u->GetVec();
   PADGTraceSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(),
                  geom->detJ, geom->normal, r, Q,
                  alpha, beta, pa_data);
}

// PA DGTrace Apply 2D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApply2D(const int NF,
                      const Array<double> &b,
                      const Array<double> &bt,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y,
                      const int d1d = 0,
                      const int q1d = 0)
{
   const int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][VDIM];
      double u1[max_D1D][VDIM];
      for (int d = 0; d < D1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            u0[d][c] = x(d,c,0,f);
            u1[d][c] = x(d,c,1,f);
         }
      }
      double Bu0[max_Q1D][VDIM];
      double Bu1[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            Bu0[q][c] = 0.0;
            Bu1[q][c] = 0.0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const int b = B(q,d);
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][c] += b*u0[d][c];
               Bu1[q][c] += b*u1[d][c];
            }
         }
      }
      double DBu[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            DBu[q][c] = op(q,0,0,f)*Bu0[q][c] + op(q,1,0,f)*Bu1[q][c];
         }
      }
      double BDBu[max_D1D][VDIM];
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; c++)
         {
            BDBu[d][c] = 0.0;
         }
         for (int q = 0; q < Q1D; ++q)
         {
            const int b = Bt(d,q);
            for (int c = 0; c < VDIM; c++)
            {
               BDBu[d][c] += b*DBu[q][c];
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            y(d,c,0,f) =  BDBu[d][c];
            y(d,c,1,f) = -BDBu[d][c];
         }
      }
   });
}

// PA DGTrace Apply 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApply3D(const int NF,
                      const Array<double> &b,
                      const Array<double> &bt,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y,
                      const int d1d = 0,
                      const int q1d = 0)
{
   const int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, 2, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, 2, NF);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, 2, NF);

   MFEM_FORALL(f, NF,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double u0[max_D1D][max_D1D][VDIM];
      double u1[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; d1++)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               u0[d1][d2][c] = x(d1,d2,c,0,f);
               u1[d1][d2][c] = x(d1,d2,c,1,f);
            }
         }
      }
      double Bu0[max_Q1D][max_D1D][VDIM];
      double Bu1[max_Q1D][max_D1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][d2][c] = 0.0;
               Bu1[q][d2][c] = 0.0;
            }
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const int b = B(q,d1);
               for (int c = 0; c < VDIM; c++)
               {
                  Bu0[q][d2][c] += b*u0[d1][d2][c];
                  Bu1[q][d2][c] += b*u1[d1][d2][c];
               }
            }
         }
      }
      double BBu0[max_Q1D][max_Q1D][VDIM];
      double BBu1[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; q2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBu0[q1][q2][c] = 0.0;
               BBu1[q1][q2][c] = 0.0;
            }
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const int b = B(q2,d2);
               for (int c = 0; c < VDIM; c++)
               {
                  BBu0[q1][q2][c] += b*Bu0[q1][d2][c];
                  BBu1[q1][q2][c] += b*Bu1[q1][d2][c];
               }
            }
         }
      }
      double DBBu[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; q2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               DBBu[q1][q2][c] = op(q1,q2,0,0,f)*BBu0[q1][q2][c] + op(q1,q2,1,0,f)*BBu1[q1][q2][c];
            }
         }
      }
      double BDBBu[max_Q1D][max_D1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BDBBu[q1][d2][c] = 0.0;
            }
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const int b = B(q2,d2);
               for (int c = 0; c < VDIM; c++)
               {
                  BDBBu[q1][d2][c] += b*DBBu[q1][q2][c];
               }
            }
         }
      }
      double BBDBBu[max_D1D][max_D1D][VDIM];
      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBDBBu[d1][d2][c] = 0.0;
            }
            for (int q1 = 0; q1 < Q1D; ++q1)
            {
               const int b = B(q1,d1);
               for (int c = 0; c < VDIM; c++)
               {
                  BBDBBu[d1][d2][c] += b*BDBBu[q1][d2][c];
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               y(d1,d2,c,0,f) =  BBDBBu[d1][d2][c];
               y(d1,d2,c,1,f) = -BBDBBu[d1][d2][c];
            }
         }
      }
   });
}

static void PADGTraceApply(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NF,
                             const Array<double> &B,
                             const Array<double> &Bt,
                             const Vector &op,
                             const Vector &x,
                             Vector &y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADGTraceApply2D<2,2>(NF,B,Bt,op,x,y);
         case 0x33: return PADGTraceApply2D<3,3>(NF,B,Bt,op,x,y);
         case 0x44: return PADGTraceApply2D<4,4>(NF,B,Bt,op,x,y);
         case 0x55: return PADGTraceApply2D<5,5>(NF,B,Bt,op,x,y);
         case 0x66: return PADGTraceApply2D<6,6>(NF,B,Bt,op,x,y);
         case 0x77: return PADGTraceApply2D<7,7>(NF,B,Bt,op,x,y);
         case 0x88: return PADGTraceApply2D<8,8>(NF,B,Bt,op,x,y);
         case 0x99: return PADGTraceApply2D<9,9>(NF,B,Bt,op,x,y);
         default:   return PADGTraceApply2D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PADGTraceApply3D<2,3>(NF,B,Bt,op,x,y);
         case 0x34: return PADGTraceApply3D<3,4>(NF,B,Bt,op,x,y);
         case 0x45: return PADGTraceApply3D<4,5>(NF,B,Bt,op,x,y);
         case 0x56: return PADGTraceApply3D<5,6>(NF,B,Bt,op,x,y);
         case 0x67: return PADGTraceApply3D<6,7>(NF,B,Bt,op,x,y);
         case 0x78: return PADGTraceApply3D<7,8>(NF,B,Bt,op,x,y);
         case 0x89: return PADGTraceApply3D<8,9>(NF,B,Bt,op,x,y);
         default:   return PADGTraceApply3D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA DGTraceIntegrator Apply kernel
void DGTraceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PADGTraceApply(dim, dofs1D, quad1D, nf,
                  maps->B, maps->Bt,
                  pa_data, x, y);
}

#if 0
   MFEM_FORALL_3D(e, NE, Q1D, 1, 1, //Iterate over faces?
   {
      MFEM_SHARED double s_E[VDIM][D1D][D1D];
      MFEM_SHARED double s_Bu[FACES][VDIM][D1D];
      MFEM_FOREACH_THREAD(x,x,D1D) {
         for (int y = 0; y < D1D; y++) {
            for (int c = 0; c < VDIM; c++) {
               s_E[c][y][x] = E(x,y,c,e);//Use restriction instead?
            }
         }
      }
      MFEM_SYNC_THREAD;
      double B0[D1D],B1[D1D];
      for (int i = 0; i < D1D; ++i) {
         B0[i] = B(0,i);
         B1[i] = B(Q1D-1,i);
      }
      double Bu[FACES][VDIM];
      for (int i = 0; i < D1D; i++) {
         for(int face = LEFT; face <= TOP; face++) {
            Bu[face][d] = 0.0;
         }
      }
      MFEM_FOREACH_THREAD(x,x,D1D) {
         for (int y = 0; y < D1D; ++y) {
            for (int c = 0; c < VDIM; c++) {
               const double e = s_e[c][y][x];
               Bu[BOTTOM][c] += B0[y] * e;
               Bu[TOP]   [c] += B1[y] * e;
            }
         }
         s_Bu[BOTTOM][c][x] = Bu[BOTTOM][c]
         s_Bu[TOP]   [c][x] = Bu[TOP]   [c]
      }
      MFEM_FOREACH_THREAD(y,x,D1D) {
         for (int x = 0; x < D1D; ++x) {
            for (int c = 0; c < VDIM; c++) {
               const double e = s_e[c][y][x];
               s_Bu[LEFT] [c][y] += B0[x] * e;
               s_Bu[RIGHT][c][y] += B1[x] * e;
            }
         }
         s_Bu[LEFT] [c][y] = Bu[LEFT] [c];
         s_Bu[RIGHT][c][y] = Bu[RIGHT][c];
      }
      MFEM_SYNC_THREAD;
      double GBu[FACES][VDIM];
      for (int d = 0; d < VDIM; d++) {
         for(int face = LEFT; face <= TOP; face++) {
            GBu[face][d] = 0.0;
         }
      }
      MFEM_FOREACH_THREAD(q,x,Q1D) {
         for (int i = 0; i < D1D; ++i) {
            const double w = G(q,i);
            for (int c = 0; c < VDIM; c++) {
               for(int face = LEFT; face <= TOP; face++) {
                  GBu[face][c] += w * s_Bu[face][c][i];
               }
            }
         }
      }
      MFEM_FOREACH_THREAD(q,x,Q1D) {
         for(int face = LEFT; face <= TOP; face++) {
            //Normal contains |J|
            normal[0] = -GBu[face][1];
            normal[1] =  GBu[face][0];
            dot = normal[0] * vx + normal[1] * vy;
            abs = dot > 0.0 ? dot : -dot;
            qd(q,0,0,face,e) = W[q]*( alpha/2 * dot + beta * abs );
            qd(q,1,0,face,e) = W[q]*( alpha/2 * dot - beta * abs );
            qd(q,0,1,face,e) = W[q]*(-alpha/2 * dot - beta * abs );
            qd(q,1,1,face,e) = W[q]*(-alpha/2 * dot + beta * abs );
         }
      }
   });
}

//Version over faces
static void PADGTraceSetup2D(const int Q1D,
                             const int D1D,
                             const int NE,
                             const Array<double> &w,
                             const Vector &j,
                             const double COEFF,
                             Vector &op)
{
   const Operator *elem_restr =
      fespace->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   elem_restr->Mult(*nodes, Enodes);

   const int VDIM = 2;
   const int FACES = 4;
   enum face {LEFT, RIGHT, BOTTOM, TOP};

   const int nbFaces = mesh->GetNFaces();

   auto E = Reshape(Enodes.Read(), D1D, D1D, VDIM, NE);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, 2, 2, FACES, nbFaces);

   //TODO initialize Face to (Elem1, Elem2) array? and Face info

   MFEM_FORALL(f, nbFaces,
   {
      const int ref_elem = ref_face(f);//Elem1
      double u[VDIM][D1D];
      LoadFace(u, ref_elem);//u==Bu
      double GBu[VDIM][Q1D];
      for (int d = 0; d < VDIM; d++) {
         for (int q = 0; q < Q1D; ++q) {
            GBu[d][q] = 0.0;
         }
      }
      for (int q = 0; q < Q1D; ++q) {
         for (int i = 0; i < D1D; ++i) {
            const double w = G(q,i);
            for (int d = 0; d < VDIM; d++) {
               GBu[d][q] += w * u[d][i];
            }
         }
      }
      for (int q = 0; q < Q1D; ++q) {
         //Normal contains |J|
         normal[0] = -GBu[1][q];
         normal[1] =  GBu[0][q];
         dot = normal[0] * vx + normal[1] * vy;
         abs = dot > 0.0 ? dot : -dot;
         const int q2 = perm(q,info_face(f));
         qd(q,0,0,f) = W[q]*( alpha/2 * dot + beta * abs );
         qd(q,1,0,f) = W[q]*( alpha/2 * dot - beta * abs );
         qd(q2,0,1,f) = W[q]*(-alpha/2 * dot - beta * abs );//necessary? for coalescing?
         qd(q2,1,1,f) = W[q]*(-alpha/2 * dot + beta * abs );//necessary?
      }
   });
}

// PA DGTrace Apply 2D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGTraceApply2D(const int NE,
                      const Array<double> &b,
                      const Array<double> &g,
                      const Array<double> &bt,
                      const Array<double> &gt,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y,
                      const int d1d = 0,
                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   // Compute fluxes
   MFEM_FORALL(f, nbFaces,
   {
      // Compute interpolation on the face
      // Get Elem1 assume face dofs are enough
      // Get Elem2 reordered
      // Get qData
      for (int q = 0; q < Q1D; ++q)
      {
         for (int d = 0; d < VDIM; ++d)
         {
            flux(q,f) = qd[0][q]*u1[q]+qd[1][q]*u2[q];
         }
      }
   });
   // Apply fluxes
   MFEM_FORALL(e, NE,
   {
      //load dofs
      for (int i = 0; i < D1D; ++i)
      {
         for (int j = 0; j < D1D; ++j)
         {
            s_x(i,j) = x(i,j,e);
         }
      }
      //SOUTH flux
      for (int q = 0; q < Q1D; ++q)
      {
         Bu[q] = 0.0;
         for (int i = 0; i < D1D; ++i)
         {
            Bu[q] += B(q,i) * s_x(i,0);
         }
      }
      bool isRefElem = true;
      for (int q = 0; q < Q1D; ++q)
      {
         f[q] = isRefElem? flux(q,f):flux(perm(q,info_face(f)),f2);
      }
      for (int i = 0; i < D1D; ++i)
      {
         BDBu[i] = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            BDBu[i] += B(q,i) * f(q);
         }
      }
      for (int i = 0; i < D1D; ++i)
      {
         y(i,0) += BDBu[i];
      }
      //NORTH flux
      for (int q = 0; q < Q1D; ++q)
      {
         Bu[q] = 0.0;
         for (int i = 0; i < D1D; ++i)
         {
            Bu[q] += B(q,i) * s_x(i,D1D-1);
         }
      }
      bool isRefElem = true;
      for (int q = 0; q < Q1D; ++q)
      {
         f[q] = isRefElem? flux(q,f):flux(perm(q,info_face(f)),f2);
      }
      for (int i = 0; i < D1D; ++i)
      {
         BDBu[i] = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            BDBu[i] += B(q,i) * f(q);
         }
      }
      for (int i = 0; i < D1D; ++i)
      {
         y(i,D1D-1) += BDBu[i];
      }
      //EAST flux
      for (int q = 0; q < Q1D; ++q)
      {
         Bu[q] = 0.0;
         for (int i = 0; i < D1D; ++i)
         {
            Bu[q] += B(q,i) * s_x(D1D-1,i);
         }
      }
      bool isRefElem = true;
      for (int q = 0; q < Q1D; ++q)
      {
         f[q] = isRefElem? flux(q,f):flux(perm(q,info_face(f)),f2);
      }
      for (int i = 0; i < D1D; ++i)
      {
         BDBu[i] = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            BDBu[i] += B(q,i) * f(q);
         }
      }
      for (int i = 0; i < D1D; ++i)
      {
         y(D1D-1,i) += BDBu[i];
      }
      //WEST flux
      for (int q = 0; q < Q1D; ++q)
      {
         Bu[q] = 0.0;
         for (int i = 0; i < D1D; ++i)
         {
            Bu[q] += B(q,i) * s_x(0,i);
         }
      }
      bool isRefElem = true;
      for (int q = 0; q < Q1D; ++q)
      {
         f[q] = isRefElem? flux(q,f):flux(perm(q,info_face(f)),f2);
      }
      for (int i = 0; i < D1D; ++i)
      {
         BDBu[i] = 0.0;
         for (int q = 0; q < Q1D; ++q)
         {
            BDBu[i] += B(q,i) * f(q);
         }
      }
      for (int i = 0; i < D1D; ++i)
      {
         y(0,i) += BDBu[i];
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PADGTraceSetup3D(const int Q1D,
                             const int NE,
                             const Array<double> &w,
                             const Vector &j,
                             const double COEFF,
                             Vector &op)
{
   const Operator *elem_restr =
      fespace->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   elem_restr->Mult(*nodes, Enodes);

   const int VDIM = 3;
   const int FACES = 6;
   enum face {LEFT, RIGHT, BOTTOM, TOP, FRONT, BACK};

   auto E = Reshape(Enodes.Read(), D1D, D1D, D1D, VDIM, NE);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, Q1D, 2, 2, FACES, NE);

   //TODO use MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   MFEM_FORALL(e, NE,
   {
      MFEM_SHARED double s_E[VDIM][D1D][D1D][D1D];
      MFEM_SHARED double s_Bu[FACES][VDIM][D1D];
      MFEM_SHARED double s_GBu[FACES][VDIM][Q1D];
      for (int x = 0; x < D1D; x++) {
         for (int y = 0; y < D1D; y++) {
            for (int z = 0; z < D1D; z++) {
               for (int c = 0; c < VDIM; c++) {
                  s_E[c][z][y][x] = E(x,y,z,c,e);
               }
            }
         }
      }
      //Normal contains |J|
      double B0[D1D],B1[D1D];
      for (int i = 0; i < D1D; ++i) {
         B0[i] = B(0,i);
         B1[i] = B(Q1D-1,i);
      }
      for (int i = 0; i < D1D; i++) {
         for (int j = 0; j < D1D; j++) {
            for (int d = 0; d < VDIM; d++) {
               for(int face = LEFT; face <= BACK; face++) {
                  s_Bu[face][d][j][i] = 0.0;
               }
            }
         }
      }
      for (int x = 0; x < D1D; ++x) {
         for (int y = 0; y < D1D; ++y) {
            for (int z = 0; z < D1D; ++z) {
               for (int c = 0; c < VDIM; c++) {
                  const double e = s_e[c][y][x];
                  s_Bu[LEFT]  [c][z][y] += B0[x] * e;
                  s_Bu[RIGHT] [c][z][y] += B1[x] * e;
                  s_Bu[BOTTOM][c][y][x] += B0[z] * e;
                  s_Bu[TOP]   [c][y][x] += B1[z] * e;
                  s_Bu[FRONT] [c][z][x] += B0[y] * e;
                  s_Bu[BACK]  [c][z][x] += B1[y] * e;
               }
            }
         }
      }
      //TODO
      for (int i = 0; i < Q1D; i++) {
         for (int j = 0; j < Q1D; j++) {
            for (int d = 0; d < VDIM; d++) {
               for(int face = LEFT; face <= BACK; face++) {
                  s_BBu[face][d][j][i] = 0.0;
                  s_GBu[face][d][j][i] = 0.0;
               }
            }
         }
      }
      for (int q = 0; q < Q1D; ++q) {
         for (int i = 0; i < D1D; ++i) {
            const double w = G(q,i);
            for (int c = 0; c < VDIM; c++) {
               for(int face = LEFT; face <= BACK; face++) {
                  s_GBu[face][c][q] += w * s_Bu[face][c][i];
               }
            }
         }
      }
      for (int q = 0; q < Q1D; ++q) {
         for(int face = LEFT; face <= TOP; face++) {
            normal[0] = -s_GBu[face][1][q];
            normal[1] =  s_GBu[face][0][q];
            dot = normal[0] * vx + normal[1] * vy;
            abs = dot > 0.0 ? dot : -dot;
            //Should only store half of it
            qd(q,0,0,face,e) = weight[q]*( alpha/2 * dot + beta * abs );
            qd(q,1,0,face,e) = weight[q]*( alpha/2 * dot - beta * abs );
            qd(q,0,1,face,e) = weight[q]*(-alpha/2 * dot - beta * abs );
            qd(q,1,1,face,e) = weight[q]*(-alpha/2 * dot + beta * abs );
         }
      }
   });
}

#endif

} // namespace mfem
