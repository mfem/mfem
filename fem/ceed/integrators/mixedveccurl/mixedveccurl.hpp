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

#ifndef MFEM_LIBCEED_MIXEDVECCURL_HPP
#define MFEM_LIBCEED_MIXEDVECCURL_HPP

#include "../../interface/integrator.hpp"
#include "../../interface/mixed_operator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/** Represent a MixedVectorCurlIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAMixedVectorCurlIntegrator : public MixedOperator<Integrator>
{
public:
   template <typename CoeffType>
   PAMixedVectorCurlIntegrator(
      const mfem::MixedVectorCurlIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorCurlIntegrator with AssemblyLevel::None
    using libCEED. */
class MFMixedVectorCurlIntegrator : public MixedOperator<Integrator>
{
public:
   template <typename CoeffType>
   MFMixedVectorCurlIntegrator(
      const mfem::MixedVectorCurlIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorWeakCurlIntegrator with AssemblyLevel::Partial
    using libCEED. */
class PAMixedVectorWeakCurlIntegrator : public MixedOperator<Integrator>
{
public:
   template <typename CoeffType>
   PAMixedVectorWeakCurlIntegrator(
      const mfem::MixedVectorWeakCurlIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

/** Represent a MixedVectorWeakCurlIntegrator with AssemblyLevel::None
    using libCEED. */
class MFMixedVectorWeakCurlIntegrator : public MixedOperator<Integrator>
{
public:
   template <typename CoeffType>
   MFMixedVectorWeakCurlIntegrator(
      const mfem::MixedVectorWeakCurlIntegrator &integ,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes,
      CoeffType *Q,
      const bool use_bdr = false);
};

}

}

#endif // MFEM_LIBCEED_MIXEDVECCURL_HPP
