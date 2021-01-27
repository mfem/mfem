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
struct MassOperatorInfo
{
   static constexpr const char *header = "/mass_qf.h";
   static constexpr const char *build_func_const = ":f_build_mass_const";
   static constexpr const char *build_func_quad = ":f_build_mass_quad";
   static constexpr const char *apply_func = ":f_apply_mass";
   static constexpr const char *apply_func_mf_const = ":f_apply_mass_mf_const";
   static constexpr const char *apply_func_mf_quad = ":f_apply_mass_mf_quad";
   static constexpr CeedQFunctionUser build_qf_const = &f_build_mass_const;
   static constexpr CeedQFunctionUser build_qf_quad = &f_build_mass_quad;
   static constexpr CeedQFunctionUser apply_qf = &f_apply_mass;
   static constexpr CeedQFunctionUser apply_qf_mf_const = &f_apply_mass_mf_const;
   static constexpr CeedQFunctionUser apply_qf_mf_quad = &f_apply_mass_mf_quad;
   static constexpr EvalMode trial_op = EvalMode::Interp;
   static constexpr EvalMode test_op = EvalMode::Interp;
   static constexpr int qdatasize = 1;
   MassContext ctx;
};
#endif

PAMassIntegrator::PAMassIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *Q)
   : PAIntegrator()
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info;
   PAOperator op = InitPA(info, fes, irm, Q);
   Assemble(op, info.ctx);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFMassIntegrator::MFMassIntegrator(const mfem::FiniteElementSpace &fes,
                                   const mfem::IntegrationRule &irm,
                                   mfem::Coefficient *Q)
   : MFIntegrator()
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info;
   MFOperator op = InitMF(info, fes, irm, Q);
   Assemble(op, info.ctx);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
