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
KernelsMassIntegrator::KernelsMassIntegrator(const mfem::Engine &ng) :
   KernelsIntegrator(ng.As<kernels::Engine>()),
   coeff(ng.As<kernels::Engine>(),1.0),
   assembledOperator(*(new Layout(ng.As<kernels::Engine>(), 0)))
{
   push();
   pop();
}

// *****************************************************************************
KernelsMassIntegrator::KernelsMassIntegrator(const KernelsCoefficient &coeff_) :
   KernelsIntegrator(coeff_.KernelsEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.KernelsEngine(), 0)))
{
   push();
   coeff.SetName("COEFF");
   pop();
}

KernelsMassIntegrator::~KernelsMassIntegrator() {}

std::string KernelsMassIntegrator::GetName()
{
   return "MassIntegrator";
}

// *****************************************************************************
void KernelsMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

// *****************************************************************************
void KernelsMassIntegrator::Setup()
{
   push();
   pop();
}

// *****************************************************************************
void KernelsMassIntegrator::Assemble() {
   push();
   pop();
}

// *****************************************************************************
void KernelsMassIntegrator::SetOperator(mfem::Vector &v)
{
   push();
   op = v;
   assembledOperator.PushData(v.GetData());
   pop();
}

// *****************************************************************************
void KernelsMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   push();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = trialFESpace->GetFE(0)->GetOrder() + 1;
   rMassMultAdd(dim,
                dofs1D,
                quad1D,
                mesh->GetNE(),
                maps->dofToQuad,
                maps->dofToQuadD,
                maps->quadToDof,
                maps->quadToDofD,
                op.GetData(),
                (const double*)x.KernelsMem().ptr(),
                (double*)y.KernelsMem().ptr());
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
