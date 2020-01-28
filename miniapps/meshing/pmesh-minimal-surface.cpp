// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//              --------------------------------
//              Parallel Minimal Surface Miniapp
//              --------------------------------
//
// Description:  See mesh-minimal-surface.cpp description.
//
// Compile with: make pmesh-minimal-surface
//
// Sample runs:  mesh-minimal-surface -vis
//
// Device sample runs:
//               mesh-minimal-surface -d cuda

// Tie X-MPI classes and construct to MFEM_USE_MPI's ones.
#define XMesh ParMesh
#define XGridFunction ParGridFunction
#define XBilinearForm ParBilinearForm
#define XFiniteElementSpace ParFiniteElementSpace
#define XMeshConstructor(this) ParMesh(MPI_COMM_WORLD, *this)
#define XInit(num_procs, myid){\
  MPI_Init(&argc, &argv);\
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);\
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);}
#define XCGArguments (MPI_COMM_WORLD)
#define XPreconditioner new HypreBoomerAMG
#define XFinalize MPI_Finalize

// Re-use mesh-minimal-surface.cpp code.
#include "mesh-minimal-surface.cpp"
