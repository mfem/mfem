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
KernelsIntegrator::KernelsIntegrator(const kernels::Engine &e)
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
KernelsIntegrator::~KernelsIntegrator() {}

// *****************************************************************************
void KernelsIntegrator::SetupMaps()
{
   push();
   maps = KernelsDofQuadMaps::Get(*rtrialFESpace->GetFESpace(),
                                  *rtestFESpace->GetFESpace(),
                                  *ir);

   mapsTranspose = KernelsDofQuadMaps::Get(*rtestFESpace->GetFESpace(),
                                           *rtrialFESpace->GetFESpace(),
                                           *ir);
   pop();
}

// *****************************************************************************
kFiniteElementSpace& KernelsIntegrator::GetTrialKernelsFESpace() const
{
   return *rtrialFESpace;
}

kFiniteElementSpace& KernelsIntegrator::GetTestKernelsFESpace() const
{
   return *rtestFESpace;
}

mfem::FiniteElementSpace& KernelsIntegrator::GetTrialFESpace() const
{
   return *trialFESpace;
}

mfem::FiniteElementSpace& KernelsIntegrator::GetTestFESpace() const
{
   return *testFESpace;
}

void KernelsIntegrator::SetIntegrationRule(const mfem::IntegrationRule &ir_)
{
   ir = &ir_;
}

const mfem::IntegrationRule& KernelsIntegrator::GetIntegrationRule() const
{
   assert(ir);
   return *ir;
}

KernelsDofQuadMaps *KernelsIntegrator::GetDofQuadMaps()
{
   return maps;
}

// *****************************************************************************
void KernelsIntegrator::SetupIntegrator(kBilinearForm &bform_,
                                        const KernelsIntegratorType itype_)
{
   push();
   MFEM_ASSERT(engine == &bform_.KernelsEngine(), "");
   bform     = &bform_;
   mesh      = &(bform_.GetMesh());

   rtrialFESpace = &(bform_.GetTrialKernelsFESpace());
   rtestFESpace  = &(bform_.GetTestKernelsFESpace());

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
kGeometry *KernelsIntegrator::GetGeometry(const int flags)
{
   push();
   pop();
   return kGeometry::Get(*rtrialFESpace, *ir);
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
