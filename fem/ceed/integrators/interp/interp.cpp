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

#include "interp.hpp"

#include "../../../../config/config.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DiscreteInterpolatorOperatorInfo : public OperatorInfo
{
   DiscreteInterpolatorOperatorInfo()
   {
      // Discrete interpolators use a built-in QFunction
      header = "";
      header = "";
      build_func = "";
      build_qf = nullptr;
      apply_func = "";
      apply_qf = nullptr;
      apply_func_mf = "";
      apply_qf_mf = nullptr;
      trial_op = EvalMode::Interp;
      test_op = EvalMode::None;
      qdatasize = 0;
   }
};
#endif

PADiscreteInterpolator::PADiscreteInterpolator(
   const mfem::DiscreteInterpolator &interp,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes)
{
#ifdef MFEM_USE_CEED
   DiscreteInterpolatorOperatorInfo info;
   Assemble(interp, info, trial_fes, test_fes, (mfem::Coefficient *)nullptr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
