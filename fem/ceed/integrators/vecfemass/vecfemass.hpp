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

#ifndef MFEM_LIBCEED_VECFEMASS_HPP
#define MFEM_LIBCEED_VECFEMASS_HPP

#include "../../interface/mixed_integrator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/// Represent a VectorFEMassIntegrator with AssemblyLevel::Partial using libCEED.
class PAVectorFEMassIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   PAVectorFEMassIntegrator(const mfem::VectorFEMassIntegrator &integ,
                            const mfem::FiniteElementSpace &fes,
                            CoeffType *Q,
                            const bool use_bdr = false);
};

/// Represent a VectorFEMassIntegrator with AssemblyLevel::None using libCEED.
class MFVectorFEMassIntegrator : public MixedIntegrator
{
public:
   template <typename CoeffType>
   MFVectorFEMassIntegrator(const mfem::VectorFEMassIntegrator &integ,
                            const mfem::FiniteElementSpace &fes,
                            CoeffType *Q,
                            const bool use_bdr = false);
};

}

}

#endif // MFEM_LIBCEED_VECFEMASS_HPP
