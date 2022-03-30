// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../config/config.hpp"

#include "tools.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 206
#include "../debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

namespace jit
{

#ifndef MFEM_USE_MPI
bool Root() { return true;}
#else // MFEM_USE_MPI
bool Root()
{
   int world_rank = 0;
   if (MPI_Inited()) { MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); }
   return world_rank == 0;
}

int MPI_Size()
{
   int size = 1;
   if (MPI_Inited()) { MPI_Comm_size(MPI_COMM_WORLD, &size); }
   return size;
}

bool MPI_Inited()
{
   int ini = false;
   MPI_Initialized(&ini);
   return ini ? true : false;
}
#endif // MFEM_USE_MPI

/// \brief GetRuntimeVersion
/// \param increment
/// \return the current runtime version
int GetRuntimeVersion(bool increment)
{
   static int version = 0;
   const int actual = version;
   if (increment) { version += 1; }
   return actual;
}

} // namespace jit

} // namespace mfem

