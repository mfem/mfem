// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include "ceed.hpp"
#include "pa_integrator.hpp"
#include "mf_integrator.hpp"
#include "../fespace.hpp"

namespace mfem
{

class CeedPADiffusionIntegrator : public CeedPAIntegrator
{
public:
   CeedPADiffusionIntegrator(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm,
                        Coefficient *coeff);
};

class CeedMFDiffusionIntegrator : public CeedMFIntegrator
{
public:
   CeedMFDiffusionIntegrator(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm,
                        Coefficient *coeff);
};

}

#endif // MFEM_LIBCEED_DIFF_HPP
