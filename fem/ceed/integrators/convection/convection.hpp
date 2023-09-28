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

#ifndef MFEM_LIBCEED_CONV_HPP
#define MFEM_LIBCEED_CONV_HPP

#include "../../interface/integrator.hpp"
#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/// Represent a ConvectionIntegrator with AssemblyLevel::Partial using libCEED.
class PAConvectionIntegrator : public PAIntegrator
{
public:
   PAConvectionIntegrator(const mfem::FiniteElementSpace &fes,
                          const mfem::IntegrationRule &ir,
                          mfem::VectorCoefficient *Q,
                          const double alpha);
};

class MixedPAConvectionIntegrator : public MixedIntegrator<PAIntegrator>
{
public:
   MixedPAConvectionIntegrator(const ConvectionIntegrator &integ,
                               const mfem::FiniteElementSpace &fes,
                               mfem::VectorCoefficient *Q,
                               const double alpha);
};

/// Represent a ConvectionIntegrator with AssemblyLevel::None using libCEED.
class MFConvectionIntegrator : public MFIntegrator
{
public:
   MFConvectionIntegrator(const mfem::FiniteElementSpace &fes,
                          const mfem::IntegrationRule &ir,
                          mfem::VectorCoefficient *Q,
                          const double alpha);
};

class MixedMFConvectionIntegrator : public MixedIntegrator<MFIntegrator>
{
public:
   MixedMFConvectionIntegrator(const ConvectionIntegrator &integ,
                               const mfem::FiniteElementSpace &fes,
                               mfem::VectorCoefficient *Q,
                               const double alpha);
};

}

}

#endif // MFEM_LIBCEED_CONV_HPP
