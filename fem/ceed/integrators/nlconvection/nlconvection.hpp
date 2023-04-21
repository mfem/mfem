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

#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/** Represent a VectorConvectionNLFIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAVectorConvectionNLIntegrator : public MixedIntegrator
{
public:
   PAVectorConvectionNLIntegrator(
      const mfem::VectorConvectionNLFIntegrator &integ,
      const mfem::FiniteElementSpace &fes,
      mfem::Coefficient *Q,
      const bool use_bdr = false);
};

/** Represent a VectorConvectionNLFIntegrator with AssemblyLevel::None
    using libCEED. */
class MFVectorConvectionNLIntegrator : public MixedIntegrator
{
public:
   MFVectorConvectionNLIntegrator(
      const mfem::VectorConvectionNLFIntegrator &integ,
      const mfem::FiniteElementSpace &fes,
      mfem::Coefficient *Q,
      const bool use_bdr = false);
};

}

}

#endif // MFEM_LIBCEED_NLCONV_HPP
