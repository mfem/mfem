// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mass.hpp"

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "mass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct MassOperatorInfo : public OperatorInfo
{
   MassContext ctx;
   MassOperatorInfo()
   {
      header = "/mass_qf.h";
      build_func_const = ":f_build_mass_const";
      build_qf_const = &f_build_mass_const;
      build_func_quad = ":f_build_mass_quad";
      build_qf_quad = &f_build_mass_quad;
      apply_func = ":f_apply_mass";
      apply_qf = &f_apply_mass;
      apply_func_mf_const = ":f_apply_mass_mf_const";
      apply_qf_mf_const = &f_apply_mass_mf_const;
      apply_func_mf_quad = ":f_apply_mass_mf_quad";
      apply_qf_mf_quad = &f_apply_mass_mf_quad;
      trial_op = EvalMode::Interp;
      test_op = EvalMode::Interp;
      qdatasize = 1;
   }
};
#endif

PAMassIntegrator::PAMassIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *Q,
                                   mfem::ActionType &action_type)
   : PAIntegrator()
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info;
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
   action_type = mfem::ActionType::L2L;
}

MFMassIntegrator::MFMassIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *Q)
   : MFIntegrator()
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info;
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
