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

namespace ceed
{

#ifdef MFEM_USE_CEED
struct ConvectionOperatorInfo : public OperatorInfo
{
   ConvectionContext ctx;
   ConvectionOperatorInfo(int dim)
   : OperatorInfo{"/convection_qf.h",
                  ":f_build_conv_const",
                  ":f_build_conv_quad",
                  ":f_apply_conv",
                  ":f_apply_conv_mf_const",
                  ":f_apply_conv_mf_quad",
                  &f_build_conv_const,
                  &f_build_conv_quad,
                  &f_apply_conv,
                  &f_apply_conv_mf_const,
                  &f_apply_conv_mf_quad,
                  EvalMode::Grad,
                  EvalMode::Interp,
                  dim * (dim + 1) / 2}
   { }
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
   ConvectionOperatorInfo info(fes.GetMesh()->Dimension());
   info.ctx.alpha = alpha;
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
   ConvectionOperatorInfo info(fes.GetMesh()->Dimension());
   info.ctx.alpha = alpha;
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
