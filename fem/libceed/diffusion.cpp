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

#ifdef MFEM_USE_CEED
#include "ceed.hpp"
#include "diffusion.h"

namespace mfem
{

void CeedPADiffusionAssemble(const FiniteElementSpace &fes,
                             const mfem::IntegrationRule &irm, CeedData& ceedData)
{
   CeedInt dim = fes.GetMesh()->SpaceDimension();
   CeedPAOperator diffOp = {fes, irm,
                            dim * (dim + 1) / 2, "/diffusion.h",
                            ":f_build_diff_const", f_build_diff_const,
                            ":f_build_diff_grid", f_build_diff_grid,
                            ":f_apply_diff", f_apply_diff,
                            CEED_EVAL_GRAD,
                            CEED_EVAL_GRAD
                           };
   CeedPAAssemble(diffOp, ceedData);
}

} // namespace mfem

#endif // MFEM_USE_CEED
