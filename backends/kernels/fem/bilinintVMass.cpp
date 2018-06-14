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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{
// *****************************************************************************
KernelsVectorMassIntegrator::KernelsVectorMassIntegrator(const KernelsCoefficient &
                                                   coeff_)
   :
   KernelsIntegrator(coeff_.KernelsEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.KernelsEngine(), 0)))
{
   coeff.SetName("COEFF");
}

KernelsVectorMassIntegrator::~KernelsVectorMassIntegrator() {}

// *****************************************************************************
std::string KernelsVectorMassIntegrator::GetName()
{
   return "VectorMassIntegrator";
}

// *****************************************************************************
void KernelsVectorMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

// *****************************************************************************
void KernelsVectorMassIntegrator::Setup()
{
   assert(false);/*
 ::kernels::properties kernelProps = props;

   coeff.Setup(*this, kernelProps);

   // Setup assemble and mult kernels
   assembleKernel = GetAssembleKernel(kernelProps);
   multKernel     = GetMultAddKernel(kernelProps);*/
}

// *****************************************************************************
void KernelsVectorMassIntegrator::Assemble()
{
   assert(false);/*
const int elements = trialFESpace->GetNE();
   const int quadraturePoints = ir->GetNPoints();

   KernelsGeometry geom = GetGeometry(KernelsGeometry::Jacobian);

   assembledOperator.Resize<double>(quadraturePoints * elements, NULL);

   assembleKernel((int) mesh->GetNE(),
                  maps.quadWeights,
                  geom.J,
                  coeff,
                  assembledOperator.KernelsMem());*/
}

// *****************************************************************************
void KernelsVectorMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   assert(false);/*
 multKernel((int) mesh->GetNE(),
              maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.KernelsMem(),
              x.KernelsMem(), y.KernelsMem());*/
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
