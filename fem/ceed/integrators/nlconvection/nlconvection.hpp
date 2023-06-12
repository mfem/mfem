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

#ifndef MFEM_LIBCEED_NLCONV_HPP
#define MFEM_LIBCEED_NLCONV_HPP

#include "../../interface/integrator.hpp"
#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/** Represent a VectorConvectionNLFIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAVectorConvectionNLFIntegrator : public PAIntegrator
{
public:
   PAVectorConvectionNLFIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *coeff);
};

class MixedPAVectorConvectionNLIntegrator : public MixedIntegrator<PAIntegrator>
{
public:
   MixedPAVectorConvectionNLIntegrator(
      const VectorConvectionNLFIntegrator &integ,
      const mfem::FiniteElementSpace &fes,
      mfem::Coefficient *Q);
};

/** Represent a VectorConvectionNLFIntegrator with AssemblyLevel::None
    using libCEED. */
class MFVectorConvectionNLFIntegrator : public MFIntegrator
{
public:
   MFVectorConvectionNLFIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *coeff);
};

class MixedMFVectorConvectionNLIntegrator : public MixedIntegrator<MFIntegrator>
{
public:
   MixedMFVectorConvectionNLIntegrator(
      const VectorConvectionNLFIntegrator &integ,
      const mfem::FiniteElementSpace &fes,
      mfem::Coefficient *Q);
};

}

}

#endif // MFEM_LIBCEED_NLCONV_HPP
