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

namespace mfem
{

template<int D1D, int Q1D> static
void VectorDomainLFGradIntegratorAssemble2D(const int vdim,
                                            const int ND,
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

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,vdim/DIM,1,1,1):
                  Reshape(F,DIM,vdim/DIM,Q1D,Q1D,NE);

   auto Y = Reshape(y,vdim/DIM,ND);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][D1D*Q1D];
      const DeviceMatrix Bt(sBG[0],D1D,Q1D);
      const DeviceMatrix Gt(sBG[1],D1D,Q1D);

      MFEM_SHARED double sm0[2][Q1D*Q1D];
      const DeviceMatrix QQ0(sm0[0],Q1D,Q1D);
      const DeviceMatrix QQ1(sm0[1],Q1D,Q1D);

      MFEM_SHARED double sm1[2][Q1D*Q1D];
      const DeviceMatrix DQ0(sm1[0],D1D,Q1D);
      const DeviceMatrix DQ1(sm1[1],D1D,Q1D);

      for (int c = 0; c < vdim/DIM; ++ c)
      {
         const double cst_val0 = C(0,c,0,0,0);
         const double cst_val1 = C(1,c,0,0,0);

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
               const double u = cst_coeff ? cst_val0 : C(0,c,qx,qy,e);
               const double v = cst_coeff ? cst_val1 : C(1,c,qx,qy,e);
               QQ0(qy,qx) = Jinv[0]*u + Jinv[2]*v;
               QQ1(qy,qx) = Jinv[1]*u + Jinv[3]*v;
               QQ0(qy,qx) *= weight * detJ;
               QQ1(qy,qx) *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::LoadBGt(D1D,Q1D,B,G,Bt,Gt);
         kernels::internal::Atomic2DGradTranspose(D1D,Q1D,Bt,Gt,
                                                  QQ0,QQ1,DQ0,DQ1,
                                                  I,Y,c,e);
      }
   });
}

template<int D1D, int Q1D> static
void VectorDomainLFGradIntegratorAssemble3D(const int vdim,
                                            const int ND,
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

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,vdim/DIM,1,1,1,1):
                  Reshape(F,DIM,vdim/DIM,Q1D,Q1D,Q1D,NE);

   auto Y = Reshape(y,vdim/DIM,ND);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][Q1D*D1D];
      const DeviceMatrix Bt(sBG[0],D1D,Q1D);
      const DeviceMatrix Gt(sBG[1],D1D,Q1D);
      kernels::internal::LoadBGt(D1D,Q1D,B,G,Bt,Gt);

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      const DeviceCube QQ0(sm0[0],Q1D,Q1D,Q1D);
      const DeviceCube QQ1(sm0[1],Q1D,Q1D,Q1D);
      const DeviceCube QQ2(sm0[2],Q1D,Q1D,Q1D);

      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      const DeviceCube QD0(sm1[0],Q1D,Q1D,D1D);
      const DeviceCube QD1(sm1[1],Q1D,Q1D,D1D);
      const DeviceCube QD2(sm1[2],Q1D,Q1D,D1D);

      const DeviceCube DD0(sm0[0],Q1D,D1D,D1D);
      const DeviceCube DD1(sm0[1],Q1D,D1D,D1D);
      const DeviceCube DD2(sm0[2],Q1D,D1D,D1D);

      for (int c = 0; c < vdim/DIM; ++ c)
      {
         const double cst_val_0 = C(0,c,0,0,0,0);
         const double cst_val_1 = C(1,c,0,0,0,0);
         const double cst_val_2 = C(2,c,0,0,0,0);

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
                  const double u = cst_coeff ? cst_val_0 : C(0,c,qx,qy,qz,e);
                  const double v = cst_coeff ? cst_val_1 : C(1,c,qx,qy,qz,e);
                  const double w = cst_coeff ? cst_val_2 : C(2,c,qx,qy,qz,e);
                  QQ0(qz,qy,qx) = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQ1(qz,qy,qx) = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQ2(qz,qy,qx) = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;
                  QQ0(qz,qy,qx) *= weight * detJ;
                  QQ1(qz,qy,qx) *= weight * detJ;
                  QQ2(qz,qy,qx) *= weight * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;

         kernels::internal::Atomic3DGrad(D1D,Q1D,Bt,Gt,
                                         QQ0,QQ1,QQ2,
                                         QD0,QD1,QD2,
                                         DD0,DD1,DD2,
                                         I,Y,c,e);
      }
   });
}

void VectorDomainLFGradIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                                const Vector &mark,
                                                Vector &b)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   const int dim = mesh->Dimension();
   const MemoryType mt = Device::GetDeviceMemoryType();

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

   if (VectorConstantCoefficient *vcQ =
          dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      coeff = vcQ->GetVec();
   }
   else if (QuadratureFunctionCoefficient *qfQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = qfQ->GetQuadFunction();
      MFEM_VERIFY(qfun.Size() == NE*NQ,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qfun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.\n");
      qfun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction&>(qfun),0);
   }
   else if (VectorQuadratureFunctionCoefficient* vqfQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qFun = vqfQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == vdim * NQ * NE,
                  "Incompatible QuadratureFunction dimension \n");
      MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
   }
   else
   {
      Vector Qvec(vdim);
      coeff.SetSize(vdim * NQ * NE);
      auto C = Reshape(coeff.HostWrite(), vdim, NQ, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation &Tr = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            Q.Eval(Qvec, Tr, ir->IntPoint(q));
            for (int c = 0; c<vdim; ++c) { C(c,q,e) = Qvec[c]; }
         }
      }
   }

   const int id = (dim<<8) | (D1D << 4) | Q1D;

   void (*Ker)(const int vdim,
               const int ND,
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
      // 2D kernels, p=q
      case 0x222: Ker=VectorDomainLFGradIntegratorAssemble2D<2,2>; break; // 1
      case 0x233: Ker=VectorDomainLFGradIntegratorAssemble2D<3,3>; break; // 2
      case 0x244: Ker=VectorDomainLFGradIntegratorAssemble2D<4,4>; break; // 3
      case 0x255: Ker=VectorDomainLFGradIntegratorAssemble2D<5,5>; break; // 4
      case 0x266: Ker=VectorDomainLFGradIntegratorAssemble2D<6,6>; break; // 5
      case 0x277: Ker=VectorDomainLFGradIntegratorAssemble2D<7,7>; break; // 6

      // 2D kernels
      case 0x223: Ker=VectorDomainLFGradIntegratorAssemble2D<2,3>; break; // 1
      case 0x234: Ker=VectorDomainLFGradIntegratorAssemble2D<3,4>; break; // 2
      case 0x245: Ker=VectorDomainLFGradIntegratorAssemble2D<4,5>; break; // 3
      case 0x256: Ker=VectorDomainLFGradIntegratorAssemble2D<5,6>; break; // 4
      case 0x267: Ker=VectorDomainLFGradIntegratorAssemble2D<6,7>; break; // 5
      case 0x278: Ker=VectorDomainLFGradIntegratorAssemble2D<7,8>; break; // 6

      // 3D kernels, p=q
      case 0x322: Ker=VectorDomainLFGradIntegratorAssemble3D<2,2>; break; // 1
      case 0x333: Ker=VectorDomainLFGradIntegratorAssemble3D<3,3>; break; // 2
      case 0x344: Ker=VectorDomainLFGradIntegratorAssemble3D<4,4>; break; // 3
      case 0x355: Ker=VectorDomainLFGradIntegratorAssemble3D<5,5>; break; // 4
      case 0x366: Ker=VectorDomainLFGradIntegratorAssemble3D<6,6>; break; // 5
      case 0x377: Ker=VectorDomainLFGradIntegratorAssemble3D<7,7>; break; // 6

      // 3D kernels
      case 0x323: Ker=VectorDomainLFGradIntegratorAssemble3D<2,3>; break; // 1
      case 0x334: Ker=VectorDomainLFGradIntegratorAssemble3D<3,4>; break; // 2
      case 0x345: Ker=VectorDomainLFGradIntegratorAssemble3D<4,5>; break; // 3
      case 0x356: Ker=VectorDomainLFGradIntegratorAssemble3D<5,6>; break; // 4
      case 0x367: Ker=VectorDomainLFGradIntegratorAssemble3D<6,7>; break; // 5
      case 0x378: Ker=VectorDomainLFGradIntegratorAssemble3D<7,8>; break; // 6

      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Ker(vdim,ND,NE,M,B,G,I,J,W,coeff,Y);
}

} // namespace mfem
