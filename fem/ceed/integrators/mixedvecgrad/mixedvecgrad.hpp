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

#ifndef MFEM_LIBCEED_MIXEDVECGRAD_HPP
#define MFEM_LIBCEED_MIXEDVECGRAD_HPP

#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/** Represent a MixedVectorGradientIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAMixedVectorGradientIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   PAMixedVectorGradientIntegrator(
      const mfem::MixedVectorGradientIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorGradientIntegrator with AssemblyLevel::None
    using libCEED. */
class MFMixedVectorGradientIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   MFMixedVectorGradientIntegrator(
      const mfem::MixedVectorGradientIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorWeakDivergenceIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAMixedVectorWeakDivergenceIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   PAMixedVectorWeakDivergenceIntegrator(
      const mfem::MixedVectorWeakDivergenceIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorWeakDivergenceIntegrator with AssemblyLevel::None
    using libCEED. */
class MFMixedVectorWeakDivergenceIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   MFMixedVectorWeakDivergenceIntegrator(
      const mfem::MixedVectorWeakDivergenceIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

}

}

#endif // MFEM_LIBCEED_MIXEDVECGRAD_HPP
