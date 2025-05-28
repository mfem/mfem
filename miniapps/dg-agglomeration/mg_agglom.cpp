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

#include "mg_agglom.hpp"

namespace mfem
{

std::vector<std::vector<int>> Agglomerate(Mesh &mesh)
{
   std::vector<std::vector<int>> E;

   // Recursive METIS partitioning to create 'E' data structure

   return E;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");

   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh());

   // Populate the arrays: operators, smoothers, ownedOperators, ownedSmoothers
   // from the MultigridBase class. (All smoothers are owned, all operators
   // except the finest are owned).

   // Populate the arrays: prolongations, ownedProlongations from the Multigrid
   // class. All prolongations are owned.

   // Create the prolongations using 'E' using the SparseMatrix class

   // Create the operator hierarchy using 'RAP' (setting R = P^T)

   // Create the smoothers (using BlockILU for now)
}

} // namespace mfem
