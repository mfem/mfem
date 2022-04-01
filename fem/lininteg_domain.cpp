// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#include "../fem/kernels.hpp"
#include "../general/forall.hpp"

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
template<int T_D1D = 0, int T_Q1D = 0> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double *detJ,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int NBZ = 1;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto DetJ = Reshape(detJ, q,q, NE);
   const auto W = Reshape(weights, q,q);
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,NE);

   auto Y = Reshape(y, d,d, vdim, NE);

   MFEM_FORALL_3D(e, NE, q,q, NBZ,
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBt[Q*D];
      MFEM_SHARED double sQQ[Q*Q];
      MFEM_SHARED double sQD[Q*D];

      const DeviceMatrix Bt(sBt, d,q);
      kernels::internal::LoadB<D,Q>(d,q,B,sBt);

      const DeviceMatrix QQ(sQQ, q,q);
      const DeviceMatrix QD(sQD, q,d);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               const double detJ = DetJ(x,y,e);
               const double coeff_val = cst_coeff ? cst_val : C(c,x,y,e);
               QQ(y,x) = W(x,y) * coeff_val * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::EvalYt(d,q,Bt,QQ,QD);
         kernels::internal::EvalXt(d,q,Bt,QD,Y,c,e);
      }
   });
}

/// Internal assembly kernel for the 2D (Vector)DomainLFIntegrator
template<int T_D1D = 0, int T_Q1D = 0> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double *detJ,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr int NBZ = 1;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto DetJ = Reshape(detJ, q,q,q, NE);
   const auto W = Reshape(weights, q,q,q);
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1,1):Reshape(F,vdim,q,q,q,NE);

   auto Y = Reshape(y, d,d,d, vdim, NE);

   MFEM_FORALL_3D(e, NE, q,q, NBZ,
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;

      double u[Q];

      MFEM_SHARED double sBt[Q*D];
      const DeviceMatrix Bt(sBt, d,q);
      kernels::internal::LoadB<D,Q>(d,q,B,sBt);

      MFEM_SHARED double sQQQ[Q*Q*Q];
      const DeviceCube QQQ(sQQQ, q,q,q);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const double detJ = DetJ(x,y,z,e);
                  const double coeff_val = cst_coeff ? cst_val : C(c,x,y,z,e);
                  QQQ(z,y,x) = W(x,y,z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::EvalZt(d,q,u,Bt,QQQ);
         kernels::internal::EvalYt(d,q,u,Bt,QQQ);
         kernels::internal::EvalXt(d,q,u,Bt,QQQ,Y,c,e);
      }
   });
}

void DomainLFIntegrator::DeviceAssemble(const FiniteElementSpace &fes,
                                        const Array<int> &markers,
                                        Vector &y)
{
   const int vdim = fes.GetVDim();

   MFEM_VERIFY(vdim == 1, "vdim should be equal to 1!");

   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type geom_type = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule :
                               &IntRules.Get(geom_type, qorder);

   Vector coeff;
   const int NQ = ir->GetNPoints();
   const int NE = fes.GetMesh()->GetNE();

   if (ConstantCoefficient *cQ =
          dynamic_cast<ConstantCoefficient*>(&Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (QuadratureFunctionCoefficient *qfQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = qfQ->GetQuadFunction();
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

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof;
   const int q = maps.nqpt;
   const int id = (dim << 8) | (d << 4) | q;

   void (*ker)(const int vdim,
               const int NE,
               const int d,
               const int q,
               const int *markers,
               const double *b,
               const double *detJ,
               const double *weights,
               const Vector &coeff,
               double *y) = nullptr;

   if (dim==2) { ker=VectorDomainLFIntegratorAssemble2D; }
   if (dim==3) { ker=VectorDomainLFIntegratorAssemble3D; }

   switch (id)
   {
      // 2D kernels, q=p+1
      case 0x222: ker=VectorDomainLFIntegratorAssemble2D<2,2>; break;
      case 0x233: ker=VectorDomainLFIntegratorAssemble2D<3,3>; break;
      case 0x244: ker=VectorDomainLFIntegratorAssemble2D<4,4>; break;
      case 0x255: ker=VectorDomainLFIntegratorAssemble2D<5,5>; break;

      // 2D kernels, q=p+2
      case 0x223: ker=VectorDomainLFIntegratorAssemble2D<2,3>; break;
      case 0x234: ker=VectorDomainLFIntegratorAssemble2D<3,4>; break;
      case 0x245: ker=VectorDomainLFIntegratorAssemble2D<4,5>; break;
      case 0x256: ker=VectorDomainLFIntegratorAssemble2D<5,6>; break;

      // 3D kernels, q=p+1, BENCH_FULL_DomainLF_3D_GLL
      case 0x322: ker=VectorDomainLFIntegratorAssemble3D<2,2>; break;
      case 0x333: ker=VectorDomainLFIntegratorAssemble3D<3,3>; break;
      case 0x344: ker=VectorDomainLFIntegratorAssemble3D<4,4>; break;
      case 0x355: ker=VectorDomainLFIntegratorAssemble3D<5,5>; break;
      case 0x366: ker=VectorDomainLFIntegratorAssemble3D<6,6>; break;

      // 3D kernels, q=p+2, BENCH_FULL_DomainLF_3D_GL
      case 0x323: ker=VectorDomainLFIntegratorAssemble3D<2,3>; break;
      case 0x334: ker=VectorDomainLFIntegratorAssemble3D<3,4>; break;
      case 0x345: ker=VectorDomainLFIntegratorAssemble3D<4,5>; break;
      case 0x356: ker=VectorDomainLFIntegratorAssemble3D<5,6>; break;
      case 0x367: ker=VectorDomainLFIntegratorAssemble3D<6,7>; break;
   }
   MFEM_VERIFY(ker, "Unexpected kernel " << std::hex << id << std::dec);

   constexpr int flags = GeometricFactors::JACOBIANS |
                         GeometricFactors::DETERMINANTS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);

   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *detJ = geom->detJ.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();

   ker(vdim, NE, d, q, M, B, detJ, W, coeff, Y);
}

////////////////////////////////////////////////////////////////////////////////
void VectorDomainLFIntegrator::DeviceAssemble(const FiniteElementSpace &fes,
                                              const Array<int> &markers,
                                              Vector &y)
{
   const int vdim = fes.GetVDim();

   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type geom_type = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule :
                               &IntRules.Get(geom_type, qorder);

   Vector coeff;
   const int NQ = ir->GetNPoints();
   const int NE = fes.GetMesh()->GetNE();

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
      Vector qvec(vdim);
      coeff.SetSize(vdim * NQ * NE);
      auto C = Reshape(coeff.HostWrite(), vdim, NQ, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < NQ; ++q)
         {
            Q.Eval(qvec, T, ir->IntPoint(q));
            for (int c=0; c<vdim; ++c) { C(c,q,e) = qvec[c]; }
         }
      }
   }

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof;
   const int q = maps.nqpt;

   const int id = (dim << 8) | (d << 4) | q;

   void (*ker)(const int vdim,
               const int NE,
               const int d,
               const int q,
               const int *markers,
               const double *b,
               const double *detJ,
               const double *weights,
               const Vector &coeff,
               double *y) = nullptr;

   if (dim==2) { ker = VectorDomainLFIntegratorAssemble2D<>; }
   if (dim==3) { ker = VectorDomainLFIntegratorAssemble3D<>; }

   switch (id)
   {
      // 2D kernels, q=p+1
      case 0x222: ker=VectorDomainLFIntegratorAssemble2D<2,2>; break;
      case 0x233: ker=VectorDomainLFIntegratorAssemble2D<3,3>; break;
      case 0x244: ker=VectorDomainLFIntegratorAssemble2D<4,4>; break;
      case 0x255: ker=VectorDomainLFIntegratorAssemble2D<5,5>; break;

      // 2D kernels, q=p+2
      case 0x223: ker=VectorDomainLFIntegratorAssemble2D<2,3>; break;
      case 0x234: ker=VectorDomainLFIntegratorAssemble2D<3,4>; break;
      case 0x245: ker=VectorDomainLFIntegratorAssemble2D<4,5>; break;
      case 0x256: ker=VectorDomainLFIntegratorAssemble2D<5,6>; break;

      // 3D kernels, q=p+1
      case 0x322: ker=VectorDomainLFIntegratorAssemble3D<2,2>; break;
      case 0x333: ker=VectorDomainLFIntegratorAssemble3D<3,3>; break;
      case 0x344: ker=VectorDomainLFIntegratorAssemble3D<4,4>; break;
      case 0x355: ker=VectorDomainLFIntegratorAssemble3D<5,5>; break;

      // 3D kernels, q=p+2
      case 0x323: ker=VectorDomainLFIntegratorAssemble3D<2,3>; break;
      case 0x334: ker=VectorDomainLFIntegratorAssemble3D<3,4>; break;
      case 0x345: ker=VectorDomainLFIntegratorAssemble3D<4,5>; break;
      case 0x356: ker=VectorDomainLFIntegratorAssemble3D<5,6>; break;
   }
   MFEM_VERIFY(ker, "Unexpected kernel " << std::hex << id << std::dec);

   constexpr int flags = GeometricFactors::JACOBIANS |
                         GeometricFactors::DETERMINANTS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);

   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *detJ = geom->detJ.Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();

   ker(vdim, NE, d, q, M, B, detJ, W, coeff, Y);
}

} // namespace mfem
