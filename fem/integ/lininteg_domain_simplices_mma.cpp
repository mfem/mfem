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

#include "lininteg_domain_simplices_mma.hpp"
#include "simplex_mma_keys.hxx"

namespace mfem
{

void DomainLFIntegrator::RegisterSimplexMmaKernels()
{
   // Key is (D1D, simplex nq), not 1D Q1D.
#define MFEM_SIMPLEX_MMA_ADD(DIM, D1D, NQ) \
   AddSimplexMmaSpecialization<DIM, D1D, NQ>();
   MFEM_SIMPLEX_MMA_FOR_EACH_COMMON_KEY(MFEM_SIMPLEX_MMA_ADD)
#undef MFEM_SIMPLEX_MMA_ADD
}

} // namespace mfem
