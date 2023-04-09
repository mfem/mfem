// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define CATCH_CONFIG_NOSTDOUT
#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

#ifndef MFEM_USE_MPI
#error "This test should be disabled without MFEM_USE_MPI!"
#endif

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::Mpi::Init();
   mfem::Hypre::Init();
#endif

   // Only run tests that are labeled with Parallel.
   return RunCatchSession(argc, argv, {"[Parallel]"}, Root());
}
