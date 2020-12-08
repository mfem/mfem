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
   Mesh &mesh = *fes.GetMesh();
   // Perform checks for some assumptions made in the Q-functions.
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(), "case not supported");
   MFEM_VERIFY(fes.GetVDim() == 1 || fes.GetVDim() == mesh.Dimension(),
               "case not supported");
   InitCeedCoeff(Q, mesh, irm, coeff_type, coeff);
   CeedPAOperator massOp = {fes, irm,
                            1, "/mass.h",
                            ":f_build_mass_const", f_build_mass_const,
                            ":f_build_mass_quad", f_build_mass_quad,
                            "", nullptr,
                            "", nullptr,
                            ":f_apply_mass", f_apply_mass,
                            EvalMode::Interp,
                            EvalMode::Interp
                           };
   MassContext ctx;
   Assemble(massOp, ctx);
}

CeedMFMassIntegrator::CeedMFMassIntegrator(const FiniteElementSpace &fes,
                                           const mfem::IntegrationRule &irm,
                                           Coefficient *Q)
: CeedMFIntegrator()
{
   Mesh &mesh = *fes.GetMesh();
   InitCeedCoeff(Q, mesh, irm, coeff_type, coeff);
   CeedMFOperator massOp = {fes, irm,
                            "/mass.h",
                            ":f_apply_mass_mf_const", f_apply_mass_mf_const,
                            ":f_apply_mass_mf_quad", f_apply_mass_mf_quad,
                            "", nullptr,
                            "", nullptr,
                            EvalMode::Interp,
                            EvalMode::Interp
                           };
   MassContext ctx;
   Assemble(massOp, ctx);
}

} // namespace mfem

