// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include "bilininteg.hpp"
#include "../general/forall.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template<int D1D, int Q1D> static
void Kernel2D(const int ND,
              const int NE,
              const double *marks,
              const double *b,
              const double *g,
              const int *idx,
              const double *jacobians,
              const double *weights,
              const Vector &coeff,
              double* __restrict y)
{
   constexpr int DIM = 2;
   constexpr int VDIM = 2;

   const bool constant_coeff = coeff.Size() == 1;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = constant_coeff ?
                  Reshape(F,1,1,1,1,1):
                  Reshape(F,DIM,VDIM,Q1D,Q1D,NE);
   auto Y = Reshape(y,ND,VDIM);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][D1D*Q1D];
      MFEM_SHARED double sm0[2][Q1D*Q1D];
      MFEM_SHARED double sm1[2][Q1D*Q1D];

      double (*Bt)[Q1D] = (double (*)[Q1D]) sBG[0];
      double (*Gt)[Q1D] = (double (*)[Q1D]) sBG[1];

      double (*QQ0)[Q1D] = (double (*)[Q1D]) sm0[0];
      double (*QQ1)[Q1D] = (double (*)[Q1D]) sm0[1];

      double (*DQ0)[Q1D] = (double (*)[Q1D]) sm1[0];
      double (*DQ1)[D1D] = (double (*)[D1D]) sm1[1];

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            Bt[dy][qx] = B(qx,dy);
            Gt[dy][qx] = G(qx,dy);
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++ c)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               double Jloc[4],Jinv[4];
               Jloc[0] = J(qx,qy,0,0,e);
               Jloc[1] = J(qx,qy,1,0,e);
               Jloc[2] = J(qx,qy,0,1,e);
               Jloc[3] = J(qx,qy,1,1,e);
               const double detJ = kernels::Det<2>(Jloc);
               kernels::CalcInverse<2>(Jloc, Jinv);
               const double weight = W(qx,qy);
               const double u = constant_coeff ? C(0,0,0,0,0) : C(c,0,qx,qy,e);
               const double v = constant_coeff ? C(0,0,0,0,0) : C(c,1,qx,qy,e);
               QQ0[qy][qx] = Jinv[0]*u + Jinv[2]*v;
               QQ1[qy][qx] = Jinv[1]*u + Jinv[3]*v;
               QQ0[qy][qx] *= weight * detJ;
               QQ1[qy][qx] *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0, v = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += Gt[dx][qx] * QQ0[qy][qx];
                  v += Bt[dx][qx] * QQ1[qy][qx];
               }
               DQ0[dx][qy] = u;
               DQ1[dx][qy] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0, v = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += DQ0[dx][qy] * Bt[dy][qy];
                  v += DQ1[dx][qy] * Gt[dy][qy];
               }
               const double sum = u + v;
               const int gid = I(dx,dy,e);
               const int idx = gid >= 0 ? gid : -1-gid;
               AtomicAdd(Y(idx,c), sum);
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
}

template<int D1D, int Q1D> static
void Kernel3D(const int ND,
              const int NE,
              const double *marks,
              const double *b,
              const double *g,
              const int *idx,
              const double *jacobians,
              const double *weights,
              const Vector &coeff,
              double* __restrict y)
{
   constexpr int DIM = 3;
   constexpr int VDIM = 3;

   const bool constant_coeff = coeff.Size() == 1;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = constant_coeff ?
                  Reshape(F,1,1,1,1,1,1):
                  Reshape(F,DIM,VDIM,Q1D,Q1D,Q1D,NE);
   auto Y = Reshape(y,ND,VDIM);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][Q1D*D1D];
      double (*Bt)[Q1D] = (double (*)[Q1D]) sBG[0];
      double (*Gt)[Q1D] = (double (*)[Q1D]) sBG[1];

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);
      double (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);
      double (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);
      double (*QQD0)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+0);
      double (*QQD1)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+1);
      double (*QQD2)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+2);
      double (*QDD0)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+0);
      double (*QDD1)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+1);
      double (*QDD2)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+2);

      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            const int i = qi(q,d,Q1D);
            const int j = dj(q,d,D1D);
            const int k = qk(q,d,Q1D);
            const int l = dl(q,d,D1D);
            Bt[j][i] = B(q,d);
            Gt[l][k] = G(q,d) * sign(q,d);
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++ c)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  double Jloc[9],Jinv[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = J(qx,qy,qz,row,col,e);
                     }
                  }
                  const double detJ = kernels::Det<3>(Jloc);
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  const double weight = W(qx,qy,qz);
                  const double u = constant_coeff ? C(0,0,0,0,0) : C(0,c,qx,qy,qz,c,e);
                  const double v = constant_coeff ? C(0,0,0,0,0) : C(1,c,qx,qy,qz,c,e);
                  const double w = constant_coeff ? C(0,0,0,0,0) : C(2,c,qx,qy,qz,c,e);
                  QQQ0[qz][qy][qx] = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQQ1[qz][qy][qx] = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQQ2[qz][qy][qx] = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;
                  QQQ0[qz][qy][qx] *= weight * detJ;
                  QQQ1[qz][qy][qx] *= weight * detJ;
                  QQQ2[qz][qy][qx] *= weight * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u[Q1D], v[Q1D], w[Q1D];
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const int i = qi(qx,dx,Q1D);
                  const int j = dj(qx,dx,D1D);
                  const int k = qk(qx,dx,Q1D);
                  const int l = dl(qx,dx,D1D);
                  const double s = sign(qx,dx);
                  MFEM_UNROLL(Q1D)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += QQQ0[qz][qy][qx] * Gt[l][k] * s;
                     v[qz] += QQQ1[qz][qy][qx] * Bt[j][i];
                     w[qz] += QQQ2[qz][qy][qx] * Bt[j][i];
                  }
               }
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  QQD0[qz][qy][dx] = u[qz];
                  QQD1[qz][qy][dx] = v[qz];
                  QQD2[qz][qy][dx] = w[qz];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u[Q1D], v[Q1D], w[Q1D];
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const int i = qi(qy,dy,Q1D);
                  const int j = dj(qy,dy,D1D);
                  const int k = qk(qy,dy,Q1D);
                  const int l = dl(qy,dy,D1D);
                  const double s = sign(qy,dy);
                  MFEM_UNROLL(Q1D)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += QQD0[qz][qy][dx] * Bt[j][i];
                     v[qz] += QQD1[qz][qy][dx] * Gt[l][k] * s;
                     w[qz] += QQD2[qz][qy][dx] * Bt[j][i];
                  }
               }
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  QDD0[qz][dy][dx] = u[qz];
                  QDD1[qz][dy][dx] = v[qz];
                  QDD2[qz][dy][dx] = w[qz];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u[D1D], v[D1D], w[D1D];
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  MFEM_UNROLL(D1D)
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     const int i = qi(qz,dz,Q1D);
                     const int j = dj(qz,dz,D1D);
                     const int k = qk(qz,dz,Q1D);
                     const int l = dl(qz,dz,D1D);
                     const double s = sign(qz,dz);
                     u[dz] += QDD0[qz][dy][dx] * Bt[j][i];
                     v[dz] += QDD1[qz][dy][dx] * Bt[j][i];
                     w[dz] += QDD2[qz][dy][dx] * Gt[l][k] * s;
                  }
               }
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double sum = u[dz] + v[dz] + w[dz];
                  const int gid = I(dx,dy,dz,e);
                  const int idx = gid >= 0 ? gid : -1-gid;
                  AtomicAdd(Y(idx,c), sum);
               }
            }
         }
      }
   });
}

void VectorDomainLFGradIntegrator::AssemblePA(const FiniteElementSpace &fes,
                                              const Vector &mark,
                                              Vector &b)
{
   const MemoryType mt = Device::GetDeviceMemoryType();
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();

   const FiniteElement &el = *fes.GetFE(0);
   const Geometry::Type geom_type = el.GetGeomType();
   const int qorder = 2.0 * el.GetOrder(); // as in AssembleRHSElementVect
   const IntegrationRule *ir =
      IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
   const int flags = GeometricFactors::JACOBIANS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_VERIFY(ER, "Not supported!");

   const double *M = mark.Read();
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const int *I = ER->GatherMap().Read();
   const double *J = geom->J.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = b.ReadWrite();

   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int ND = fes.GetNDofs();
   const int NE = fes.GetMesh()->GetNE();
   const int NQ = ir->GetNPoints();

   Vector coeff;
   const int qvdim = Q.GetVDim();

   if (ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(&Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = cQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == NE*NQ,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else
   {
      Vector Qvec(qvdim);
      coeff.SetSize(qvdim * NQ * NE);
      auto C = Reshape(coeff.HostWrite(), qvdim, NQ, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            Q.Eval(Qvec, T, ir->IntPoint(q));
            for (int c=0; c<qvdim; ++c)
            {
               C(c,q,e) = Qvec[c];
            }
         }
      }
   }

   const int id = (dim<<8) |(D1D << 4) | Q1D;

   void (*Ker)(const int ND,
               const int NE,
               const double *marks,
               const double *b,
               const double *g,
               const int *idx,
               const double *jacobians,
               const double *weights,
               const Vector &C,
               double *Y) = nullptr;

   switch (id) // orders 1~6
   {
      // 2D kernels
      case 0x222: Ker=Kernel2D<2,2>; break; // 1
      /*      case 0x233: Ker=Kernel2D<3,3>; break; // 2
            case 0x244: Ker=Kernel2D<4,4>; break; // 3
            case 0x255: Ker=Kernel2D<5,5>; break; // 4
            case 0x266: Ker=Kernel2D<6,6>; break; // 5
            case 0x277: Ker=Kernel2D<7,7>; break; // 6

            // 3D kernels
            case 0x322: Ker=Kernel3D<2,2>; break; // 1
            case 0x333: Ker=Kernel3D<3,3>; break; // 2
            case 0x344: Ker=Kernel3D<4,4>; break; // 3
            case 0x355: Ker=Kernel3D<5,5>; break; // 4
            case 0x366: Ker=Kernel3D<6,6>; break; // 5
            case 0x377: Ker=Kernel3D<7,7>; break; // 6
      */
      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Ker(ND,NE,M,B,G,I,J,W,coeff,Y);
}

} // namespace mfem
