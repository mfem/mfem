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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "catch.hpp"

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::MPI_Session mpi;
#endif

   // There must be exactly one instance.
   Catch::Session session;

   // Apply provided command line arguments.
   int r = session.applyCommandLine(argc, argv);
   if (r != 0)
   {
      return r;
   }

#ifdef MFEM_USE_MPI
   // Force to only run parallel tests if the tests
   // are executed on multiple processors.
   if (mpi.WorldSize() > 1)
   {
      auto cfg = session.configData();
      cfg.testsOrTags.push_back("[Parallel]");
      session.useConfigData(cfg);
      if (mpi.Root())
      {
         mfem::out
               << "WARNING: Only running the [Parallel] label on MPI ranks > 1." << std::endl;
      }
   }
#endif

   int result = session.run();

   return result;
}
