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

#ifndef MFEM_LIBCEED_DIVDIV_HPP
#define MFEM_LIBCEED_DIVDIV_HPP

#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/// Represent a DivDivIntegrator with AssemblyLevel::Partial using libCEED.
class PADivDivIntegrator : public MixedIntegrator
{
public:
   PADivDivIntegrator(const mfem::DivDivIntegrator &integ,
                      const mfem::FiniteElementSpace &fes,
                      mfem::Coefficient *Q,
                      const bool use_bdr = false);
};

/// Represent a DivDivIntegrator with AssemblyLevel::None using libCEED.
class MFDivDivIntegrator : public MixedIntegrator
{
public:
   MFDivDivIntegrator(const mfem::DivDivIntegrator &integ,
                      const mfem::FiniteElementSpace &fes,
                      mfem::Coefficient *Q,
                      const bool use_bdr = false);
};

}

}

#endif // MFEM_LIBCEED_DIVDIV_HPP
