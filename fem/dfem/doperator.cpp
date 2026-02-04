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

#include "doperator.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;

DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &infds,
   const std::vector<FieldDescriptor> &outfds,
   const ParMesh &mesh) :
   mesh(mesh),
   infds(infds),
   outfds(outfds)
{
   unionfds.clear();
   unionfds.insert(unionfds.end(), infds.begin(), infds.end());
   unionfds.insert(unionfds.end(), outfds.begin(), outfds.end());
   std::sort(unionfds.begin(), unionfds.end());
   auto last = std::unique(unionfds.begin(), unionfds.end());
   unionfds.erase(last, unionfds.end());

   fields_e.resize(infds.size());
   fields_l.resize(infds.size());
}

#endif // MFEM_USE_MPI
