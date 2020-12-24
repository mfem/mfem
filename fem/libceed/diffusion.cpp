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

#include "diffusion.hpp"

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "diffusion.h"
#endif

namespace mfem
{

#ifdef MFEM_USE_CEED
struct CeedDiffusionInfo
{
   static constexpr const char *header = "/diffusion.h";
   static constexpr const char *build_func_const = ":f_build_diff_const";
   static constexpr const char *build_func_quad = ":f_build_diff_quad";
   static constexpr const char *apply_func = ":f_apply_diff";
   static constexpr const char *apply_func_mf_const = ":f_apply_diff_mf_const";
   static constexpr const char *apply_func_mf_quad = ":f_apply_diff_mf_quad";
   static constexpr CeedQFunctionUser build_qf_const = &f_build_diff_const;
   static constexpr CeedQFunctionUser build_qf_quad = &f_build_diff_quad;
   static constexpr CeedQFunctionUser apply_qf = &f_apply_diff;
   static constexpr CeedQFunctionUser apply_qf_mf_const = &f_apply_diff_mf_const;
   static constexpr CeedQFunctionUser apply_qf_mf_quad = &f_apply_diff_mf_quad;
   static constexpr EvalMode trial_op = EvalMode::Grad;
   static constexpr EvalMode test_op = EvalMode::Grad;
   const int qdatasize;
   DiffusionContext ctx;
   CeedDiffusionInfo(int dim) : qdatasize(dim*(dim+1)/2) { }
};

#endif

CeedPADiffusionIntegrator::CeedPADiffusionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   Coefficient *Q)
   : CeedPAIntegrator()
{
#ifdef MFEM_USE_CEED
   CeedDiffusionInfo info(fes.GetMesh()->Dimension());
   CeedPAOperator op = InitPA(info, fes, irm, Q);
   Assemble(op, info.ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

CeedMFDiffusionIntegrator::CeedMFDiffusionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   Coefficient *Q)
   : CeedMFIntegrator()
{
#ifdef MFEM_USE_CEED
   CeedDiffusionInfo info(fes.GetMesh()->Dimension());
   CeedMFOperator op = InitMF(info, fes, irm, Q);
   Assemble(op, info.ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace mfem
