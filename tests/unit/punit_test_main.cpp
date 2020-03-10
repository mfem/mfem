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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "catch.hpp"

#ifdef MFEM_USE_MPI
mfem::MPI_Session *GlobalMPISession;
#endif

int main(int argc, char *argv[])
{
   // There must be exactly one instance.
   Catch::Session session;

   // Apply provided command line arguments.
   int r = session.applyCommandLine(argc, argv);
   if (r != 0)
   {
      return r;
   }

#ifdef MFEM_USE_MPI
   mfem::MPI_Session mpi;
   GlobalMPISession = &mpi;

   // Exclude all tests that are not labeled with Parallel.
   auto cfg = session.configData();
   cfg.testsOrTags.push_back("[Parallel]");
   session.useConfigData(cfg);

   if (mpi.Root())
   {
      mfem::out
            << "WARNING: Only running the [Parallel] label."
            << std::endl;
   }
#endif

   int result = session.run();

   return result;
}
