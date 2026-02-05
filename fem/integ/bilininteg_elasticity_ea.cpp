// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "bilininteg_elasticity_kernels.hpp"

namespace mfem
{
void ElasticityComponentIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                               Vector &emat,
                                               const bool add)
{
   AssemblePA(fes);
   const auto &ir = parent.q_space->GetIntRule(0);
   internal::ElasticityAssembleEA(parent.vdim, i_block, j_block, parent.ndofs, ir,
                                  *parent.lambda_quad, *parent.mu_quad,
                                  *geom, *maps, emat);
}
}
