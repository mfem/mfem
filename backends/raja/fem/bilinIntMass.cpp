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
RajaMassIntegrator::RajaMassIntegrator(const mfem::Engine &ng) :
   RajaIntegrator(ng.As<raja::Engine>()),
   coeff(ng.As<raja::Engine>(),1.0),
   assembledOperator(*(new Layout(ng.As<raja::Engine>(), 0)))
{
   push();
   pop();
}

// *****************************************************************************
RajaMassIntegrator::RajaMassIntegrator(const RajaCoefficient &coeff_) :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   push();
   coeff.SetName("COEFF");
   pop();
}

RajaMassIntegrator::~RajaMassIntegrator() {}

std::string RajaMassIntegrator::GetName()
{
   return "MassIntegrator";
}

// *****************************************************************************
void RajaMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

// *****************************************************************************
void RajaMassIntegrator::Setup()
{
   push();
   pop();
}

// *****************************************************************************
void RajaMassIntegrator::Assemble() { }

// *****************************************************************************
void RajaMassIntegrator::SetOperator(mfem::Vector &v)
{
   op = v;
   assembledOperator.PushData(v.GetData());
}

// *****************************************************************************
void RajaMassIntegrator::MultAdd(Vector &x, Vector &y)
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
                //(double*)assembledOperator.RajaMem().ptr(),
                (const double*)x.RajaMem().ptr(),
                (double*)y.RajaMem().ptr());
   pop();
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
