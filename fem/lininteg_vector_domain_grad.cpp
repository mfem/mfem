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
#include "lininteg_domain_grad.hpp"

namespace mfem
{

using namespace internal::linearform_extension;

void VectorDomainLFGradIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                                const Vector &mark,
                                                Vector &y)
{
   const int vdim = fes.GetVDim();
   GetOrder_f gof = [](const int el_order) { return 2.0 * el_order; };
   const IntegrationRule *ir = GetIntegrationRule(fes, IntRule, gof);

   Vector coeff;
   const int NQ = ir->GetNPoints();
   const int NE = fes.GetMesh()->GetNE();

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

   Kernel_f ker = nullptr;
   const int id = GetKernelId(fes,ir);

   switch (id) // orders 1~6
   {
      // 2D kernels, p=q
      case 0x222: ker=VectorDomainLFGradIntegratorAssemble2D<2,2>; break; // 1
      case 0x233: ker=VectorDomainLFGradIntegratorAssemble2D<3,3>; break; // 2
      case 0x244: ker=VectorDomainLFGradIntegratorAssemble2D<4,4>; break; // 3
      case 0x255: ker=VectorDomainLFGradIntegratorAssemble2D<5,5>; break; // 4
      case 0x266: ker=VectorDomainLFGradIntegratorAssemble2D<6,6>; break; // 5
      case 0x277: ker=VectorDomainLFGradIntegratorAssemble2D<7,7>; break; // 6

      // 2D kernels
      case 0x223: ker=VectorDomainLFGradIntegratorAssemble2D<2,3>; break; // 1
      case 0x234: ker=VectorDomainLFGradIntegratorAssemble2D<3,4>; break; // 2
      case 0x245: ker=VectorDomainLFGradIntegratorAssemble2D<4,5>; break; // 3
      case 0x256: ker=VectorDomainLFGradIntegratorAssemble2D<5,6>; break; // 4
      case 0x267: ker=VectorDomainLFGradIntegratorAssemble2D<6,7>; break; // 5
      case 0x278: ker=VectorDomainLFGradIntegratorAssemble2D<7,8>; break; // 6

      // 3D kernels, p=q
      case 0x322: ker=VectorDomainLFGradIntegratorAssemble3D<2,2>; break; // 1
      case 0x333: ker=VectorDomainLFGradIntegratorAssemble3D<3,3>; break; // 2
      case 0x344: ker=VectorDomainLFGradIntegratorAssemble3D<4,4>; break; // 3
      case 0x355: ker=VectorDomainLFGradIntegratorAssemble3D<5,5>; break; // 4
      case 0x366: ker=VectorDomainLFGradIntegratorAssemble3D<6,6>; break; // 5
      case 0x377: ker=VectorDomainLFGradIntegratorAssemble3D<7,7>; break; // 6

      // 3D kernels
      case 0x323: ker=VectorDomainLFGradIntegratorAssemble3D<2,3>; break; // 1
      case 0x334: ker=VectorDomainLFGradIntegratorAssemble3D<3,4>; break; // 2
      case 0x345: ker=VectorDomainLFGradIntegratorAssemble3D<4,5>; break; // 3
      case 0x356: ker=VectorDomainLFGradIntegratorAssemble3D<5,6>; break; // 4
      case 0x367: ker=VectorDomainLFGradIntegratorAssemble3D<6,7>; break; // 5
      case 0x378: ker=VectorDomainLFGradIntegratorAssemble3D<7,8>; break; // 6

      default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Launch(ker,fes,ir,coeff,mark,y);
}

} // namespace mfem
