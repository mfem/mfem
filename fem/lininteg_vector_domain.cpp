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
#include "lininteg_domain.hpp"

namespace mfem
{

using namespace internal::linearform_extension;

void VectorDomainLFIntegrator::AssembleFull(const FiniteElementSpace &fes,
                                            const Array<int> &markers,
                                            Vector &y)
{
   const int vdim = fes.GetVDim();
   GetOrder_f gof = [](const int el_order) { return 2.0 * el_order; };
   const IntegrationRule *ir = GetIntRuleFromOrder(fes, IntRule, gof);

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

   LinearFormExtensionKernel_f ker = nullptr;
   const int id = GetKernelId(fes,ir);
   const int dim = fes.GetMesh()->Dimension();

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
   MFEM_VERIFY(ker, "Unexpected kernel error!");
   Launch(ker,fes,ir,coeff,markers,y);
}

} // namespace mfem
