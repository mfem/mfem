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
#include "lininteg_domain.hpp"

#include "../general/forall.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

using namespace internal::linearform_extension;

void DomainLFIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                      const Vector &mark,
                                      Vector &b)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   MFEM_ASSERT(vdim==1, "vdim != 1");
   const int dim = mesh->Dimension();
   const bool byVDIM = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &el = *fes.GetFE(0);
   const Geometry::Type geom_type = el.GetGeomType();
   const int qorder = oa * el.GetOrder() + ob; // as in AssembleRHSElementVect
   const IntegrationRule *ir =
      IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
   const int flags = GeometricFactors::JACOBIANS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_ASSERT(ER, "Not supported!");

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

   if (ConstantCoefficient *cQ =
          dynamic_cast<ConstantCoefficient*>(&Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(&Q))
   {
      const QuadratureFunction &qfun = cQ->GetQuadFunction();
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

   const int id = (dim << 8) | (D1D << 4) | Q1D;

   void (*Ker)(const int vdim,
               const bool byVDIM,
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

      // 3D kernels
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
   Ker(vdim,byVDIM,ND,NE,M,B,I,J,W,coeff,Y);
}

} // namespace mfem
