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

#include "convection.hpp"

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "convection_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct ConvectionOperatorInfo : public OperatorInfo
{
   ConvectionContext ctx;
   ConvectionOperatorInfo(int dim, double alpha)
   {
      header = "/convection_qf.h";
      build_func_const = ":f_build_conv_const";
      build_qf_const = &f_build_conv_const;
      build_func_quad = ":f_build_conv_quad";
      build_qf_quad = &f_build_conv_quad;
      apply_func = ":f_apply_conv";
      apply_qf = &f_apply_conv;
      apply_func_mf_const = ":f_apply_conv_mf_const";
      apply_qf_mf_const = &f_apply_conv_mf_const;
      apply_func_mf_quad = ":f_apply_conv_mf_quad";
      apply_qf_mf_quad = &f_apply_conv_mf_quad;
      trial_op = EvalMode::Grad;
      test_op = EvalMode::Interp;
      qdatasize = dim * (dim + 1) / 2;
      ctx.alpha = alpha;
   }
};
#endif

PAConvectionIntegrator::PAConvectionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   mfem::VectorCoefficient *Q,
   const double alpha)
   : PAIntegrator()
{
#ifdef MFEM_USE_CEED
   ConvectionOperatorInfo info(fes.GetMesh()->Dimension(), alpha);
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFConvectionIntegrator::MFConvectionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   mfem::VectorCoefficient *Q,
   const double alpha)
   : MFIntegrator()
{
#ifdef MFEM_USE_CEED
   ConvectionOperatorInfo info(fes.GetMesh()->Dimension(), alpha);
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
