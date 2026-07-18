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

#include "bilininteg_mass_pa_simplices_mma.hpp"
#include "simplex_mma_keys.hxx"

namespace mfem
{

void MassIntegrator::RegisterSimplexMmaKernels()
{
   // Key is (D1D, simplex nq), not 1D Q1D.
#define MFEM_SIMPLEX_MMA_ADD(DIM, D1D, NQ) \
   AddSimplexMmaSpecialization<DIM, D1D, NQ>();
   MFEM_SIMPLEX_MMA_FOR_EACH_COMMON_KEY(MFEM_SIMPLEX_MMA_ADD)
#undef MFEM_SIMPLEX_MMA_ADD

   // Mass-only extras (BP1 q=2p+3, BP7, …) not shared with DomainLF.
   AddSimplexMmaSpecialization<2,2,7>();   // BP1tri p=1, q=2p+3
   AddSimplexMmaSpecialization<2,4,7>();   // BP7tri p=3, q=2p-1
   AddSimplexMmaSpecialization<2,5,15>();  // BP7tri p=4
   AddSimplexMmaSpecialization<2,6,19>();  // BP7tri p=5
   AddSimplexMmaSpecialization<2,7,28>();  // BP7tri p=6
   AddSimplexMmaSpecialization<2,8,37>();  // BP7tri p=7

   AddSimplexMmaSpecialization<3,2,14>();  // BP1/3tet p=1, q=2p+3
   AddSimplexMmaSpecialization<3,3,8>();   // BP7tet p=2, q=2p-1
   AddSimplexMmaSpecialization<3,4,14>();  // BP7tet p=3
   AddSimplexMmaSpecialization<3,4,59>();  // BP1/3tet p=3, q=2p+3
   AddSimplexMmaSpecialization<3,5,35>();  // BP7tet p=4
   AddSimplexMmaSpecialization<3,6,59>();  // BP7tet p=5
   AddSimplexMmaSpecialization<3,6,145>(); // BP1/3tet p=5, q=2p+3
   AddSimplexMmaSpecialization<3,7,96>();  // BP7tet p=6
   AddSimplexMmaSpecialization<3,8,145>(); // BP7tet p=7
}

} // namespace mfem
