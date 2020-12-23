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
#include "mass.h"
#endif

namespace mfem
{

CeedPAMassIntegrator::CeedPAMassIntegrator(const FiniteElementSpace &fes,
                                           const mfem::IntegrationRule &irm,
                                           Coefficient *Q)
   : CeedPAIntegrator()
{
#ifdef MFEM_USE_CEED
   Mesh &mesh = *fes.GetMesh();
   // Perform checks for some assumptions made in the Q-functions.
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(), "case not supported");
   MFEM_VERIFY(fes.GetVDim() == 1 || fes.GetVDim() == mesh.Dimension(),
               "case not supported");
   MassContext ctx;
   InitCeedCoeff(Q, mesh, irm, coeff, ctx);
   bool const_coeff = coeff->IsConstant();
   std::string build_func = const_coeff ? ":f_build_mass_const" : ":f_build_mass_quad";
   CeedQFunctionUser build_qf = const_coeff ? f_build_mass_const : f_build_mass_quad;
   CeedPAOperator massOp = {fes, irm,
                            1, "/mass.h",
                            build_func, build_qf,
                            ":f_apply_mass", f_apply_mass,
                            EvalMode::Interp,
                            EvalMode::Interp
                           };
   Assemble(massOp, ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

CeedMFMassIntegrator::CeedMFMassIntegrator(const FiniteElementSpace &fes,
                                           const mfem::IntegrationRule &irm,
                                           Coefficient *Q)
   : CeedMFIntegrator()
{
#ifdef MFEM_USE_CEED
   Mesh &mesh = *fes.GetMesh();
   MassContext ctx;
   InitCeedCoeff(Q, mesh, irm, coeff, ctx);
   bool const_coeff = coeff->IsConstant();
   std::string apply_func = const_coeff ? ":f_apply_mass_mf_const" : ":f_apply_mass_mf_quad";
   CeedQFunctionUser apply_qf = const_coeff ? f_apply_mass_mf_const : f_apply_mass_mf_quad;
   CeedMFOperator massOp = {fes, irm,
                            "/mass.h",
                            apply_func, apply_qf,
                            EvalMode::Interp,
                            EvalMode::Interp
                           };
   Assemble(massOp, ctx);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace mfem
