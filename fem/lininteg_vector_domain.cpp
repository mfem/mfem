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

namespace mfem
{

template<int D1D, int Q1D> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const int ND,
                                        const int NE,
                                        const double *marks,
                                        const double *d2q,
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double * __restrict y)
{
   constexpr int DIM = 2;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(d2q, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1):
                  Reshape(F,vdim,Q1D,Q1D,NE);

   auto Y = Reshape(y,vdim,ND);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) return;

      MFEM_SHARED double Bt[D1D][Q1D];
      MFEM_SHARED double sm0[Q1D*Q1D];
      MFEM_SHARED double sm1[Q1D*Q1D];
      double (*QQ)[Q1D] = (double (*)[Q1D]) (sm0);
      double (*QD)[D1D] = (double (*)[D1D]) (sm1);

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            Bt[dy][qx] = B(qx,dy);
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < vdim; ++ c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const double J11 = J(qx,qy,0,0,e);
               const double J21 = J(qx,qy,1,0,e);
               const double J12 = J(qx,qy,0,1,e);
               const double J22 = J(qx,qy,1,1,e);
               const double detJ = (J11*J22)-(J21*J12);
               const double coeff_val = cst_coeff ? cst_val : C(c,qx,qy,e);
               QQ[qy][qx] = W(qx,qy) * coeff_val * detJ;

            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double dq = 0.0;
               for (int qx = 0; qx < Q1D; ++qx) { dq += QQ[qy][qx] * Bt[dx][qx]; }
               QD[qy][dx] = dq;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double dd = 0.0;
               for (int qy = 0; qy < Q1D; ++qy) { dd += QD[qy][dx] * Bt[dy][qy]; }
               const int gid = I(dx,dy,e);
               const int idx = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(Y(c,idx), dd);
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int D1D, int Q1D> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const int ND,
                                        const int NE,
                                        const double *marks,
                                        const double *d2q,
                                        const int *idx,
                                        const double *jacobians,
                                        const double *weights,
                                        const Vector &coeff,
                                        double * __restrict y)
{
   constexpr int DIM = 3;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(d2q, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,vdim,1,1,1,1) :
                  Reshape(F,vdim,Q1D,Q1D,Q1D,NE);
   auto Y = Reshape(y,vdim,ND);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) return;

      double u[Q1D];
      MFEM_SHARED double s_B[Q1D][D1D];
      MFEM_SHARED double s_q[Q1D][Q1D][Q1D];

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            s_B[qx][dy] = B(qx,dy);
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < vdim; ++ c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double J11 = J(qx,qy,qz,0,0,e);
                  const double J21 = J(qx,qy,qz,1,0,e);
                  const double J31 = J(qx,qy,qz,2,0,e);
                  const double J12 = J(qx,qy,qz,0,1,e);
                  const double J22 = J(qx,qy,qz,1,1,e);
                  const double J32 = J(qx,qy,qz,2,1,e);
                  const double J13 = J(qx,qy,qz,0,2,e);
                  const double J23 = J(qx,qy,qz,1,2,e);
                  const double J33 = J(qx,qy,qz,2,2,e);
                  const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                      J21 * (J12 * J33 - J32 * J13) +
                                      J31 * (J12 * J23 - J22 * J13);
                  const double coeff_val = cst_coeff ? cst_val : C(c,qx,qy,qz,e);
                  s_q[qz][qy][qx] = W(qx,qy,qz) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               // Zt projection
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double ZYX = s_q[qz][qy][qx];
                  MFEM_UNROLL(D1D)
                  for (int dz = 0; dz < D1D; ++dz) { u[dz] += ZYX * s_B[qz][dz]; }
               }
               MFEM_UNROLL(D1D)
               for (int dz = 0; dz < D1D; ++dz) { s_q[dz][qy][qx] = u[dz]; }
            }
         }
         MFEM_SYNC_THREAD;

         // Yt projection
         MFEM_FOREACH_THREAD(dz,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; ++dy) { u[dy] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double zYX = s_q[dz][qy][qx];
                  MFEM_UNROLL(D1D)
                  for (int dy = 0; dy < D1D; ++dy) { u[dy] += zYX * s_B[qy][dy]; }
               }
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; ++dy) { s_q[dz][dy][qx] = u[dy]; }
            }
         }
         MFEM_SYNC_THREAD;

         // Xt projection & save output
         MFEM_FOREACH_THREAD(dz,y,D1D)
         {
            MFEM_FOREACH_THREAD(dy,x,D1D)
            {
               MFEM_UNROLL(D1D)
               for (int dx = 0; dx < D1D; ++dx) { u[dx] = 0.0; }
               MFEM_UNROLL(Q1D)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double zyX = s_q[dz][dy][qx];
                  MFEM_UNROLL(D1D)
                  for (int dx = 0; dx < D1D; ++dx) { u[dx] += zyX * s_B[qx][dx]; }
               }
               MFEM_UNROLL(D1D)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double output = u[dx];
                  const int gid = I(dx,dy,dz,e);
                  const int idx = gid >= 0 ? gid : -1 - gid;
                  AtomicAdd(Y(c,idx), output);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void VectorDomainLFIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                            const Vector &mark,
                                            Vector &b)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   const int dim = mesh->Dimension();

   const FiniteElement &el = *fes.GetFE(0);
   const Geometry::Type geom_type = el.GetGeomType();
   const int qorder = 2.0 * el.GetOrder(); // as in AssembleRHSElementVect
   const IntegrationRule *ir =
      IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
   const int flags = GeometricFactors::JACOBIANS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_VERIFY(ER, "Not supported!");

   const double *M = mark.Read();
   const double *B = maps.B.Read();
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

   if (VectorConstantCoefficient *vcQ =
          dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      coeff = vcQ->GetVec();
   }
   else if (VectorQuadratureFunctionCoefficient *vQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = vQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == vdim*NE*NQ,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else
   {
      coeff.SetSize(vdim * NQ * NE);
      auto C = Reshape(coeff.HostWrite(), vdim, NQ, NE);
      Vector Qvec(vdim);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            Q.Eval(Qvec, T, ir->IntPoint(q));
            for (int i=0; i<vdim; ++i) { C(i,q,e) = Qvec[i]; }
         }
      }
   }

   const int id = (dim<<8) | (D1D << 4) | Q1D;

   void (*Ker)(const int vdim,
               const int ND,
               const int NE,
               const double *marks,
               const double *d2q,
               const int *idx,
               const double *jacobians,
               const double *weights,
               const Vector &C,
               double *Y) = nullptr;

   switch (id) // orders 1~6
   {
      // 2D kernels, p=q
      case 0x222: Ker=VectorDomainLFIntegratorAssemble2D<2,2>; break; // 1
      case 0x233: Ker=VectorDomainLFIntegratorAssemble2D<3,3>; break; // 2
      case 0x244: Ker=VectorDomainLFIntegratorAssemble2D<4,4>; break; // 3
      case 0x255: Ker=VectorDomainLFIntegratorAssemble2D<5,5>; break; // 4
      case 0x266: Ker=VectorDomainLFIntegratorAssemble2D<6,6>; break; // 5
      case 0x277: Ker=VectorDomainLFIntegratorAssemble2D<7,7>; break; // 6

      // 2D kernels
      case 0x223: Ker=VectorDomainLFIntegratorAssemble2D<2,3>; break; // 1
      case 0x234: Ker=VectorDomainLFIntegratorAssemble2D<3,4>; break; // 2
      case 0x245: Ker=VectorDomainLFIntegratorAssemble2D<4,5>; break; // 3
      case 0x256: Ker=VectorDomainLFIntegratorAssemble2D<5,6>; break; // 4
      case 0x267: Ker=VectorDomainLFIntegratorAssemble2D<6,7>; break; // 5
      case 0x278: Ker=VectorDomainLFIntegratorAssemble2D<7,8>; break; // 6

      // 3D kernels, p=q
      case 0x322: Ker=VectorDomainLFIntegratorAssemble3D<2,2>; break; // 1
      case 0x333: Ker=VectorDomainLFIntegratorAssemble3D<3,3>; break; // 2
      case 0x344: Ker=VectorDomainLFIntegratorAssemble3D<4,4>; break; // 3
      case 0x355: Ker=VectorDomainLFIntegratorAssemble3D<5,5>; break; // 4
      case 0x366: Ker=VectorDomainLFIntegratorAssemble3D<6,6>; break; // 5
      case 0x377: Ker=VectorDomainLFIntegratorAssemble3D<7,7>; break; // 6

      // 3D kernels
      case 0x323: Ker=VectorDomainLFIntegratorAssemble3D<2,3>; break; // 1
      case 0x334: Ker=VectorDomainLFIntegratorAssemble3D<3,4>; break; // 2
      case 0x345: Ker=VectorDomainLFIntegratorAssemble3D<4,5>; break; // 3
      case 0x356: Ker=VectorDomainLFIntegratorAssemble3D<5,6>; break; // 4
      case 0x367: Ker=VectorDomainLFIntegratorAssemble3D<6,7>; break; // 5
      case 0x378: Ker=VectorDomainLFIntegratorAssemble3D<7,8>; break; // 6
      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Ker(vdim,ND,NE,M,B,I,J,W,coeff,Y);
}

} // namespace mfem
