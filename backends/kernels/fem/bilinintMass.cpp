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
   engine(ng),
   coeff(ng.As<kernels::Engine>(),1.0)
{
   nvtx_push();
   nvtx_pop();
}

// *****************************************************************************
KernelsMassIntegrator::KernelsMassIntegrator(const KernelsCoefficient &coeff_) :
   KernelsIntegrator(coeff_.KernelsEngine()),
   engine(coeff_.KernelsEngine()),
   coeff(coeff_)
{
   nvtx_push();
   coeff.SetName("COEFF");
   nvtx_pop();
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
   nvtx_push();
   nvtx_pop();
}

// *****************************************************************************
void KernelsMassIntegrator::Assemble()
{
   nvtx_push();
   nvtx_pop();
}

// *****************************************************************************
void KernelsMassIntegrator::SetOperator(mfem::Vector &v)
{
   nvtx_push();
   op = v;
   op.Resize(engine.MakeLayout(v.Size()));
   op.PushData(v.GetData());
   nvtx_pop();
}

// *****************************************************************************
void KernelsMassIntegrator::MultAdd(kernels::Vector &x,
                                    kernels::Vector &y)
{
   nvtx_push();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = trialFESpace->GetFE(0)->GetOrder() + 1;
   kernels::Vector kop = op.Get_PVector()->As<kernels::Vector>();
   rMassMultAdd(dim,
                dofs1D,
                quad1D,
                mesh->GetNE(),
                maps->dofToQuad,
                maps->dofToQuadD,
                maps->quadToDof,
                maps->quadToDofD,
                (const double*)kop.KernelsMem().ptr(),
                (const double*)x.KernelsMem().ptr(),
                (double*)y.KernelsMem().ptr());
   nvtx_pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
