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

namespace mfem
{

void MassIntegrator::RegisterSimplexMmaKernels()
{
   // Key is (D1D, simplex nq), not 1D Q1D.
   // IntRules TRIANGLE counts for order 2p and 2p+4, p = 1..7.
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,7>();   // BP1tri p=1, q=2p+3
   AddSimplexMmaSpecialization<2,2,12>();
   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,15>(); // GLL BP1tri p=2
   AddSimplexMmaSpecialization<2,3,16>();
   AddSimplexMmaSpecialization<2,4,7>();   // BP7tri p=3, q=2p-1
   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,19>(); // GLL BP1tri p=3
   AddSimplexMmaSpecialization<2,4,25>();
   AddSimplexMmaSpecialization<2,5,15>();  // BP7tri p=4
   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,28>(); // GLL BP1tri p=4
   AddSimplexMmaSpecialization<2,5,33>();
   AddSimplexMmaSpecialization<2,6,19>();  // BP7tri p=5
   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,37>(); // GLL BP1tri p=5
   AddSimplexMmaSpecialization<2,6,42>();
   AddSimplexMmaSpecialization<2,7,28>();  // BP7tri p=6
   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>(); // GLL BP1tri p=6
   AddSimplexMmaSpecialization<2,7,55>();
   AddSimplexMmaSpecialization<2,8,37>();  // BP7tri p=7
   AddSimplexMmaSpecialization<2,8,42>();  // BP5tri p=7, q=2p
   AddSimplexMmaSpecialization<2,8,60>(); // GLL BP1tri p=7

   // 3D (GLL tet): BP1tet uses IntRules order 2p; tests use 2p+4.
   AddSimplexMmaSpecialization<3,2,4>();   // BP p=1 order 2
   AddSimplexMmaSpecialization<3,2,14>();  // BP1/3tet p=1, q=2p+3
   AddSimplexMmaSpecialization<3,2,24>();  // test p=1 order 6
   AddSimplexMmaSpecialization<3,3,8>();   // BP7tet p=2, q=2p-1
   AddSimplexMmaSpecialization<3,3,14>();  // BP p=2 order 4
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();  // test p=2 order 8
   AddSimplexMmaSpecialization<3,4,14>();  // BP7tet p=3
   AddSimplexMmaSpecialization<3,4,24>();  // BP p=3 order 6
   AddSimplexMmaSpecialization<3,4,59>();  // BP1/3tet p=3, q=2p+3
   AddSimplexMmaSpecialization<3,4,81>();  // test p=3 order 10
   AddSimplexMmaSpecialization<3,5,35>();  // BP7tet p=4
   AddSimplexMmaSpecialization<3,5,46>();  // BP p=4 order 8
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>(); // test p=4 order 12
   AddSimplexMmaSpecialization<3,6,59>();  // BP7tet p=5
   AddSimplexMmaSpecialization<3,6,81>();  // BP p=5 order 10
   AddSimplexMmaSpecialization<3,6,145>(); // BP1/3tet p=5, q=2p+3
   AddSimplexMmaSpecialization<3,6,175>(); // test p=5 order 14
   AddSimplexMmaSpecialization<3,7,96>();  // BP7tet p=6
   AddSimplexMmaSpecialization<3,7,123>(); // BP p=6 order 12
   AddSimplexMmaSpecialization<3,7,209>();
   AddSimplexMmaSpecialization<3,7,248>(); // test p=6 order 16
   AddSimplexMmaSpecialization<3,8,145>(); // BP7tet p=7
   AddSimplexMmaSpecialization<3,8,175>(); // BP p=7 order 14
   AddSimplexMmaSpecialization<3,8,284>();
}

} // namespace mfem
