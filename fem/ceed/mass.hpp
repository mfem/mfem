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

#ifndef MFEM_LIBCEED_MASS_HPP
#define MFEM_LIBCEED_MASS_HPP

#include "integrator.hpp"
#include "../fespace.hpp"

namespace mfem
{

namespace ceed
{

/// Represent a MassIntegrator with AssemblyLevel::Partial using libCEED.
class PAMassIntegrator : public PAIntegrator
{
public:
   PAMassIntegrator(const mfem::FiniteElementSpace &fes,
                    const mfem::IntegrationRule &irm,
                    mfem::Coefficient *Q);
};

/// Represent a MassIntegrator with AssemblyLevel::None using libCEED.
class MFMassIntegrator : public MFIntegrator
{
public:
   MFMassIntegrator(const mfem::FiniteElementSpace &fes,
                    const mfem::IntegrationRule &irm,
                    mfem::Coefficient *Q);
};

}

}

#endif // MFEM_LIBCEED_MASS_HPP
