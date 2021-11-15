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
void DomainLFIntegratorAssemble2D(const int NE,
                                  const double *marks,
                                  const double *d2q,
                                  const int *idx,
                                  const double *jacobians,
                                  const double *weights,
                                  const Vector &coeff,
                                  double* __restrict Y)
{
   constexpr int DIM = 2;

   const bool cst_coeff = coeff.Size() == 1;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto b = Reshape(d2q, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = cst_coeff ? Reshape(F,1,1,1):Reshape(F,Q1D,Q1D,NE);

   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sB[D1D*Q1D];
      MFEM_SHARED double sQQ[Q1D*Q1D];
      MFEM_SHARED double sQD[Q1D*D1D];

      const DeviceMatrix B(sB,Q1D,D1D);
      const DeviceMatrix QQ(sQQ,Q1D,Q1D);
      const DeviceMatrix QD(sQD,Q1D,D1D);

      const double cst_val = C(0,0,0);

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            double Jloc[4];
            Jloc[0] = J(qx,qy,0,0,e);
            Jloc[1] = J(qx,qy,1,0,e);
            Jloc[2] = J(qx,qy,0,1,e);
            Jloc[3] = J(qx,qy,1,1,e);
            const double detJ = kernels::Det<2>(Jloc);
            const double coeff_val = cst_coeff ? cst_val : C(qx,qy,e);
            QQ(qy,qx) = W(qx,qy) * coeff_val * detJ;
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadB(D1D,Q1D,b,B);
      kernels::internal::Atomic2DEvalTranspose(D1D,Q1D,B,QQ,QD,I,Y,e);
   });
}

template<int D1D, int Q1D> static
void DomainLFIntegratorAssemble3D(const int NE,
                                  const double *marks,
                                  const double *d2q,
                                  const int *idx,
                                  const double *jacobians,
                                  const double *weights,
                                  const Vector &coeff,
                                  double* __restrict Y)
{
   const bool constant_coeff = coeff.Size() == 1;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto b = Reshape(d2q, Q1D, D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,3,3,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = constant_coeff ? Reshape(F,1,1,1,1):Reshape(F,Q1D,Q1D,Q1D,NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      const double mark = M(e);
      assert(mark > 0.0);
      if (mark < 1.0) return;

      double u[Q1D];
      const double constant_val = C(0,0,0,0);

      MFEM_SHARED double s_B[Q1D*D1D];
      MFEM_SHARED double s_q[Q1D*Q1D*Q1D];

      const DeviceMatrix B(s_B,Q1D,D1D);
      const DeviceCube Q(s_q,Q1D,Q1D,Q1D);

      kernels::internal::LoadB(D1D,Q1D,b,B);

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            for (int qz = 0; qz < Q1D; ++qz)
            {
               double Jloc[9];
               for (int col = 0; col < 3; col++)
               {
                  for (int row = 0; row < 3; row++)
                  {
                     Jloc[row+3*col] = J(qx,qy,qz,row,col,e);
                  }
               }
               const double detJ = kernels::Det<3>(Jloc);
               const double coeff_val =
                  constant_coeff ? constant_val : C(qx,qy,qz,e);
               Q(qz,qy,qx) = W(qx,qy,qz) * coeff_val * detJ;
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::Atomic3DEvalTranspose(D1D,Q1D,u,B,Q,I,Y,e);
   });
}

void DomainLFIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                      const Vector &mark,
                                      Vector &b)
{
   const MemoryType mt = Device::GetDeviceMemoryType();
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();

   const FiniteElement &el = *fes.GetFE(0);
   const Geometry::Type geom_type = el.GetGeomType();
   const int qorder = oa * el.GetOrder() + ob; // as in AssembleRHSElementVect
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
   const int *I = ER->GatherMap().Read();
   const double *J = geom->J.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = b.ReadWrite();

   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int NE = fes.GetMesh()->GetNE();
   const int NQ = ir->GetNPoints();

   Vector coeff;

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
      coeff.SetSize(NQ * NE);
      auto C = Reshape(coeff.HostWrite(), NQ, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            C(q,e) = Q.Eval(T, ir->IntPoint(q));
         }
      }
   }

   const int id = (dim<<8) | (D1D << 4) | Q1D;

   void (*Ker)(const int NE,
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
      case 0x222: Ker=DomainLFIntegratorAssemble2D<2,2>; break; // 1
      case 0x233: Ker=DomainLFIntegratorAssemble2D<3,3>; break; // 2
      case 0x244: Ker=DomainLFIntegratorAssemble2D<4,4>; break; // 3
      case 0x255: Ker=DomainLFIntegratorAssemble2D<5,5>; break; // 4
      case 0x266: Ker=DomainLFIntegratorAssemble2D<6,6>; break; // 5
      case 0x277: Ker=DomainLFIntegratorAssemble2D<7,7>; break; // 6

      // 2D kernels
      case 0x223: Ker=DomainLFIntegratorAssemble2D<2,3>; break; // 1
      case 0x234: Ker=DomainLFIntegratorAssemble2D<3,4>; break; // 2
      case 0x245: Ker=DomainLFIntegratorAssemble2D<4,5>; break; // 3
      case 0x256: Ker=DomainLFIntegratorAssemble2D<5,6>; break; // 4
      case 0x267: Ker=DomainLFIntegratorAssemble2D<6,7>; break; // 5
      case 0x278: Ker=DomainLFIntegratorAssemble2D<7,8>; break; // 6

      // 3D kernels
      case 0x322: Ker=DomainLFIntegratorAssemble3D<2,2>; break; // 1
      case 0x333: Ker=DomainLFIntegratorAssemble3D<3,3>; break; // 2
      case 0x344: Ker=DomainLFIntegratorAssemble3D<4,4>; break; // 3
      case 0x355: Ker=DomainLFIntegratorAssemble3D<5,5>; break; // 4
      case 0x366: Ker=DomainLFIntegratorAssemble3D<6,6>; break; // 5
      case 0x377: Ker=DomainLFIntegratorAssemble3D<7,7>; break; // 6

      // 3D kernels
      case 0x323: Ker=DomainLFIntegratorAssemble3D<2,3>; break; // 1
      case 0x334: Ker=DomainLFIntegratorAssemble3D<3,4>; break; // 2
      case 0x345: Ker=DomainLFIntegratorAssemble3D<4,5>; break; // 3
      case 0x356: Ker=DomainLFIntegratorAssemble3D<5,6>; break; // 4
      case 0x367: Ker=DomainLFIntegratorAssemble3D<6,7>; break; // 5
      case 0x378: Ker=DomainLFIntegratorAssemble3D<7,8>; break; // 6

      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Ker(NE,M,B,I,J,W,coeff,Y);
}

} // namespace mfem
