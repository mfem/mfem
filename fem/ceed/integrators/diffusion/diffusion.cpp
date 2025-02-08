// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "diffusion_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DiffusionOperatorInfo : public OperatorInfo
{
   DiffusionContext ctx;
   DiffusionOperatorInfo(int dim)
   {
      header = "/integrators/diffusion/diffusion_qf.h";
      build_func_const = ":f_build_diff_const";
      build_qf_const = &f_build_diff_const;
      build_func_quad = ":f_build_diff_quad";
      build_qf_quad = &f_build_diff_quad;
      apply_func = ":f_apply_diff";
      apply_qf = &f_apply_diff;
      apply_func_mf_const = ":f_apply_diff_mf_const";
      apply_qf_mf_const = &f_apply_diff_mf_const;
      apply_func_mf_quad = ":f_apply_diff_mf_quad";
      apply_qf_mf_quad = &f_apply_diff_mf_quad;
      trial_op = EvalMode::Grad;
      test_op = EvalMode::Grad;
      qdatasize = dim*(dim+1)/2;
   }
};
#endif

PADiffusionIntegrator::PADiffusionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   mfem::Coefficient *Q)
   : PAIntegrator()
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const DiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::Coefficient *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const VectorDiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::Coefficient *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFDiffusionIntegrator::MFDiffusionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   mfem::Coefficient *Q)
   : MFIntegrator()
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const DiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::Coefficient *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const VectorDiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::Coefficient *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes.GetMesh()->Dimension());
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
