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

#ifdef MFEM_USE_CEED
#include "convection.h"
#endif

namespace mfem
{

struct ConvectionContext : public BuildContext
{
   CeedScalar alpha;
};

CeedPAConvectionIntegrator::CeedPAConvectionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   VectorCoefficient *Q,
   const double alpha)
: CeedPAIntegrator()
{
   Mesh &mesh = *fes.GetMesh();
   // Perform checks for some assumptions made in the Q-functions.
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(), "case not supported");
   MFEM_VERIFY(fes.GetVDim() == 1 || fes.GetVDim() == mesh.Dimension(),
               "case not supported");
   int dim = mesh.Dimension();
   InitCeedVecCoeff(Q, mesh, irm, coeff_type, coeff);
   CeedPAOperator convOp = {fes, irm,
                           dim * (dim + 1) / 2, "/convection.h",
                           "", nullptr,
                           "", nullptr,
                           ":f_build_conv_const", f_build_conv_const,
                           ":f_build_conv_quad", f_build_conv_quad,
                           ":f_apply_conv", f_apply_conv,
                           EvalMode::Grad,
                           EvalMode::Interp
                           };
   ConvectionContext ctx;
   ctx.alpha = alpha;
   Assemble(convOp, ctx);
}

CeedMFConvectionIntegrator::CeedMFConvectionIntegrator(
   const FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   VectorCoefficient *Q,
   const double alpha)
: CeedMFIntegrator()
{
   Mesh &mesh = *fes.GetMesh();
   InitCeedVecCoeff(Q, mesh, irm, coeff_type, coeff);
   CeedMFOperator convOp = {fes, irm,
                           "/convection.h",
                           "", nullptr,
                           "", nullptr,
                           ":f_apply_conv_mf_const", f_apply_conv_mf_const,
                           ":f_apply_conv_mf_quad", f_apply_conv_mf_quad,
                           EvalMode::Grad,
                           EvalMode::Interp
                        };
   ConvectionContext ctx;
   ctx.alpha = alpha;
   Assemble(convOp, ctx);
}

} // namespace mfem
