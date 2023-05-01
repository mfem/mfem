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

#ifndef MFEM_LIBCEED_INTERP_HPP
#define MFEM_LIBCEED_INTERP_HPP

#include "../../interface/integrator.hpp"
#include "../../interface/mixed_operator.hpp"
#include "../../../fespace.hpp"

namespace mfem
{

namespace ceed
{

/** Represent DiscreteInterpolator classes with AssemblyLevel::Partial
    using libCEED. */
class PADiscreteInterpolator : public MixedOperator<Interpolator>
{
public:
   PADiscreteInterpolator(
      const mfem::DiscreteInterpolator &interp,
      const mfem::FiniteElementSpace &trial_fes,
      const mfem::FiniteElementSpace &test_fes);
};

}

}

#endif // MFEM_LIBCEED_INTERP_HPP
