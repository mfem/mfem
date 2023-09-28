// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_DIFF_HPP
#define MFEM_LIBCEED_DIFF_HPP

#include "../../interface/integrator.hpp"
#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/// Represent a DiffusionIntegrator with AssemblyLevel::Partial using libCEED.
class PADiffusionIntegrator : public PAIntegrator
{
public:
   PADiffusionIntegrator(const mfem::FiniteElementSpace &fes,
                         const mfem::IntegrationRule &ir,
                         mfem::Coefficient *Q);
};

class MixedPADiffusionIntegrator : public MixedIntegrator<PAIntegrator>
{
public:
   MixedPADiffusionIntegrator(const DiffusionIntegrator &integ,
                              const mfem::FiniteElementSpace &fes,
                              mfem::Coefficient *Q);

   MixedPADiffusionIntegrator(const VectorDiffusionIntegrator &integ,
                              const mfem::FiniteElementSpace &fes,
                              mfem::Coefficient *Q);
};

/// Represent a DiffusionIntegrator with AssemblyLevel::None using libCEED.
class MFDiffusionIntegrator : public MFIntegrator
{
public:
   MFDiffusionIntegrator(const mfem::FiniteElementSpace &fes,
                         const mfem::IntegrationRule &ir,
                         mfem::Coefficient *Q);
};

class MixedMFDiffusionIntegrator : public MixedIntegrator<MFIntegrator>
{
public:
   MixedMFDiffusionIntegrator(const DiffusionIntegrator &integ,
                              const mfem::FiniteElementSpace &fes,
                              mfem::Coefficient *Q);

   MixedMFDiffusionIntegrator(const VectorDiffusionIntegrator &integ,
                              const mfem::FiniteElementSpace &fes,
                              mfem::Coefficient *Q);
};

}

}

#endif // MFEM_LIBCEED_DIFF_HPP
