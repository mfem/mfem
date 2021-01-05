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

#include "convection.hpp"

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "convection_qf.h"
#endif

namespace mfem
{

#ifdef MFEM_USE_CEED
struct CeedConvectionInfo
{
   static constexpr const char *header = "/convection_qf.h";
   static constexpr const char *build_func_const = ":f_build_conv_const";
   static constexpr const char *build_func_quad = ":f_build_conv_quad";
   static constexpr const char *apply_func = ":f_apply_conv";
   static constexpr const char *apply_func_mf_const = ":f_apply_conv_mf_const";
   static constexpr const char *apply_func_mf_quad = ":f_apply_conv_mf_quad";
   static constexpr CeedQFunctionUser build_qf_const = &f_build_conv_const;
   static constexpr CeedQFunctionUser build_qf_quad = &f_build_conv_quad;
   static constexpr CeedQFunctionUser apply_qf = &f_apply_conv;
   static constexpr CeedQFunctionUser apply_qf_mf_const = &f_apply_conv_mf_const;
   static constexpr CeedQFunctionUser apply_qf_mf_quad = &f_apply_conv_mf_quad;
   static constexpr EvalMode trial_op = EvalMode::Grad;
   static constexpr EvalMode test_op = EvalMode::Interp;
   const int qdatasize;
   ConvectionContext ctx;
   CeedConvectionInfo(int dim) : qdatasize(dim * (dim + 1) / 2) { }
};
#endif

CeedPAConvectionIntegrator::CeedPAConvectionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   VectorCoefficient *Q,
   const double alpha)
   : CeedPAIntegrator()
{
#ifdef MFEM_USE_CEED
   CeedConvectionInfo info(fes.GetMesh()->Dimension());
   CeedPAOperator op = InitPA(info, fes, irm, Q);
   info.ctx.alpha = alpha;
   Assemble(op, info.ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

CeedMFConvectionIntegrator::CeedMFConvectionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   VectorCoefficient *Q,
   const double alpha)
   : CeedMFIntegrator()
{
#ifdef MFEM_USE_CEED
   CeedConvectionInfo info(fes.GetMesh()->Dimension());
   CeedMFOperator op = InitMF(info, fes, irm, Q);
   info.ctx.alpha = alpha;
   Assemble(op, info.ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace mfem
