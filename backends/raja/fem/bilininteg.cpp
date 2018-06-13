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
RajaIntegrator::RajaIntegrator(const raja::Engine &e)
   : engine(&e),
     bform(),
     mesh(),
     rtrialFESpace(),
     rtestFESpace(),
     trialFESpace(),
     testFESpace(),
     itype(DomainIntegrator),
     ir(NULL),
     hasTensorBasis(false) { }

// *****************************************************************************
RajaIntegrator::~RajaIntegrator() {}

// *****************************************************************************
void RajaIntegrator::SetupMaps()
{
   push();
   maps = RajaDofQuadMaps::Get(*rtrialFESpace->GetFESpace(),
                               *rtestFESpace->GetFESpace(),
                               *ir);

   mapsTranspose = RajaDofQuadMaps::Get(*rtestFESpace->GetFESpace(),
                                        *rtrialFESpace->GetFESpace(),
                                        *ir);
   pop();
}

// *****************************************************************************
RajaFiniteElementSpace& RajaIntegrator::GetTrialRajaFESpace() const
{
   return *rtrialFESpace;
}

RajaFiniteElementSpace& RajaIntegrator::GetTestRajaFESpace() const
{
   return *rtestFESpace;
}

mfem::FiniteElementSpace& RajaIntegrator::GetTrialFESpace() const
{
   return *trialFESpace;
}

mfem::FiniteElementSpace& RajaIntegrator::GetTestFESpace() const
{
   return *testFESpace;
}

void RajaIntegrator::SetIntegrationRule(const mfem::IntegrationRule &ir_)
{
   ir = &ir_;
}

const mfem::IntegrationRule& RajaIntegrator::GetIntegrationRule() const
{
   assert(ir);
   return *ir;
}

RajaDofQuadMaps *RajaIntegrator::GetDofQuadMaps()
{
   return maps;
}

// *****************************************************************************
void RajaIntegrator::SetupIntegrator(RajaBilinearForm &bform_,
                                     const RajaIntegratorType itype_)
{
   push();
   MFEM_ASSERT(engine == &bform_.RajaEngine(), "");
   bform     = &bform_;
   mesh      = &(bform_.GetMesh());

   rtrialFESpace = &(bform_.GetTrialRajaFESpace());
   rtestFESpace  = &(bform_.GetTestRajaFESpace());

   trialFESpace = &(bform_.GetTrialFESpace());
   testFESpace  = &(bform_.GetTestFESpace());

   hasTensorBasis = rtrialFESpace->hasTensorBasis();

   itype = itype_;

   if (ir == NULL)
   {
      SetupIntegrationRule();
   }
   SetupMaps();
   Setup();
   pop();
}

// *****************************************************************************
RajaGeometry *RajaIntegrator::GetGeometry(const int flags)
{
   push();
   pop();
   return RajaGeometry::Get(*rtrialFESpace, *ir);
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
