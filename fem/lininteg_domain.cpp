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

namespace mfem
{

using namespace internal::linearform_extension;

void DomainLFIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                      const Vector &mark,
                                      Vector &y)
{
   const int vdim = fes.GetVDim();
   MFEM_ASSERT(vdim==1, "vdim != 1");
   GetOrder_f gof = [&](int el_order) { return oa * el_order + ob; };
   const IntegrationRule *ir = GetIntegrationRule(fes, IntRule, gof);

   Vector coeff;
   const int NQ = ir->GetNPoints();
   const int NE = fes.GetMesh()->GetNE();

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

   Kernel_f ker = nullptr;
   const int id = GetKernelId(fes,ir);

   switch (id) // orders 1~6
   {
      // 2D kernels, p=q
      //case 0x222: ker=VectorDomainLFIntegratorAssemble2D<2,2>; break; // 1
      //case 0x233: ker=VectorDomainLFIntegratorAssemble2D<3,3>; break; // 2
      case 0x244: ker=VectorDomainLFIntegratorAssemble2D<4,4>; break; // 3
      case 0x255: ker=VectorDomainLFIntegratorAssemble2D<5,5>; break; // 4
      //case 0x266: ker=VectorDomainLFIntegratorAssemble2D<6,6>; break; // 5
      //case 0x277: ker=VectorDomainLFIntegratorAssemble2D<7,7>; break; // 6

      // 2D kernels
      //case 0x223: ker=VectorDomainLFIntegratorAssemble2D<2,3>; break; // 1
      //case 0x234: ker=VectorDomainLFIntegratorAssemble2D<3,4>; break; // 2
      case 0x245: ker=VectorDomainLFIntegratorAssemble2D<4,5>; break; // 3
      case 0x256: ker=VectorDomainLFIntegratorAssemble2D<5,6>; break; // 4
      //case 0x267: ker=VectorDomainLFIntegratorAssemble2D<6,7>; break; // 5
      //case 0x278: ker=VectorDomainLFIntegratorAssemble2D<7,8>; break; // 6

      // 3D kernels
      //case 0x322: ker=VectorDomainLFIntegratorAssemble3D<2,2>; break; // 1
      case 0x333: ker=VectorDomainLFIntegratorAssemble3D<3,3>; break; // 2
      case 0x344: ker=VectorDomainLFIntegratorAssemble3D<4,4>; break; // 3
      //case 0x355: ker=VectorDomainLFIntegratorAssemble3D<5,5>; break; // 4
      //case 0x366: ker=VectorDomainLFIntegratorAssemble3D<6,6>; break; // 5
      //case 0x377: ker=VectorDomainLFIntegratorAssemble3D<7,7>; break; // 6

      // 3D kernels
      //case 0x323: ker=VectorDomainLFIntegratorAssemble3D<2,3>; break; // 1
      case 0x334: ker=VectorDomainLFIntegratorAssemble3D<3,4>; break; // 2
      case 0x345: ker=VectorDomainLFIntegratorAssemble3D<4,5>; break; // 3
      //case 0x356: ker=VectorDomainLFIntegratorAssemble3D<5,6>; break; // 4
      //case 0x367: ker=VectorDomainLFIntegratorAssemble3D<6,7>; break; // 5
      //case 0x378: ker=VectorDomainLFIntegratorAssemble3D<7,8>; break; // 6

      default: ker = ((id&0xF00) == 0x200)?
                        VectorDomainLFIntegratorAssemble2D<>:
                        VectorDomainLFIntegratorAssemble3D<>;
         //default: MFEM_ABORT("Unknown kernel 0x" << std::hex << id << std::dec);
   }
   Launch(ker,fes,ir,coeff,mark,y);
}

} // namespace mfem
