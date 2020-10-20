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

#include "ceed.hpp"
#ifdef MFEM_USE_CEED
#include "mass.h"
#endif

namespace mfem
{

void CeedPAMassAssemble(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm, CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   Mesh &mesh = *fes.GetMesh();
   // Perform checks for some assumptions made in the Q-functions.
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(), "case not supported");
   MFEM_VERIFY(1 <= fes.GetVDim() && fes.GetVDim() <= 3, "case not supported");
   CeedPAOperator massOp = {fes, irm,
                            1, "/mass.h",
                            ":f_build_mass_const", f_build_mass_const,
                            ":f_build_mass_quad", f_build_mass_quad,
                            ":f_apply_mass", f_apply_mass,
                            CEED_EVAL_INTERP,
                            CEED_EVAL_INTERP
                           };
   CeedPAAssemble(massOp, ceedData);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedMFMassAssemble(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm, CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   CeedMFOperator massOp = {fes, irm,
                            "/mass.h",
                            ":f_apply_mass_mf_const", f_apply_mass_mf_const,
                            ":f_apply_mass_mf_quad", f_apply_mass_mf_quad,
                            CEED_EVAL_INTERP,
                            CEED_EVAL_INTERP
                           };
   CeedMFAssemble(massOp, ceedData);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace mfem

