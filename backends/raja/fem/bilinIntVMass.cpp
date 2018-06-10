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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{
// *****************************************************************************
RajaVectorMassIntegrator::RajaVectorMassIntegrator(const RajaCoefficient &
                                                   coeff_)
   :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   coeff.SetName("COEFF");
}

RajaVectorMassIntegrator::~RajaVectorMassIntegrator() {}

// *****************************************************************************
std::string RajaVectorMassIntegrator::GetName()
{
   return "VectorMassIntegrator";
}

// *****************************************************************************
void RajaVectorMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

// *****************************************************************************
void RajaVectorMassIntegrator::Setup()
{
   assert(false);/*
 ::raja::properties kernelProps = props;

   coeff.Setup(*this, kernelProps);

   // Setup assemble and mult kernels
   assembleKernel = GetAssembleKernel(kernelProps);
   multKernel     = GetMultAddKernel(kernelProps);*/
}

// *****************************************************************************
void RajaVectorMassIntegrator::Assemble()
{
   assert(false);/*
const int elements = trialFESpace->GetNE();
   const int quadraturePoints = ir->GetNPoints();

   RajaGeometry geom = GetGeometry(RajaGeometry::Jacobian);

   assembledOperator.Resize<double>(quadraturePoints * elements, NULL);

   assembleKernel((int) mesh->GetNE(),
                  maps.quadWeights,
                  geom.J,
                  coeff,
                  assembledOperator.RajaMem());*/
}
   
// *****************************************************************************
void RajaVectorMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   assert(false);/*
 multKernel((int) mesh->GetNE(),
              maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.RajaMem(),
              x.RajaMem(), y.RajaMem());*/
}
   
} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
