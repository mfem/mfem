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

#ifndef MFEM_RUN_UNIT_TEST
#define MFEM_RUN_UNIT_TEST

#include "unit_tests.hpp"

bool launch_all_non_regression_tests = false;
bool verbose_tests = false;
std::string mfem_data_dir;

static int RunCatchSession(int argc, char *argv[],
                           const std::vector<std::string> &testsOrTags,
                           bool root=true)
{
   // There must be exactly one instance.
   Catch::Session session;

   // Build a new command line parser on top of Catch's
   using namespace Catch::clara;
   bool enable_mfem_out = false;
   auto cli = session.cli()
              | Opt(launch_all_non_regression_tests) ["--all"] ("all tests")
              | Opt(mfem_data_dir, "") ["--data"] ("mfem/data repository")
              | Opt(enable_mfem_out) ["--enable-output"] ("enable mfem::out")
              | Opt(verbose_tests) ["--verbose"] ("verbose output");
   session.cli(cli);

   // For floating point comparisons, print 8 digits for single precision
   // values, and 16 digits for double precision values.
   Catch::StringMaker<float>::precision = 8;
   Catch::StringMaker<double>::precision = 16;

   // Apply provided command line arguments.
   int r = session.applyCommandLine(argc, argv);
   if (r != 0) { return r; }

   auto cfg = session.configData();
   cfg.testsOrTags.insert(cfg.testsOrTags.end(), testsOrTags.begin(),
                          testsOrTags.end());
   if (mfem_data_dir == "") { cfg.testsOrTags.push_back("~[MFEMData]"); }
   session.useConfigData(cfg);

   if (!verbose_tests && !enable_mfem_out)
   {
      mfem::out.Disable();
   }

   if (root)
   {
      std::cout << "INFO: Test filter: ";
      for (std::string &filter : cfg.testsOrTags)
      {
         std::cout << filter << " ";
      }
      std::cout << std::endl;
   }

   int result = session.run();

   return result;
}

static inline bool Root()
{
#ifdef MFEM_USE_MPI
   return mfem::Mpi::IsInitialized() ? mfem::Mpi::Root() : true;
#else
   return true;
#endif
}

#ifdef CATCH_CONFIG_NOSTDOUT

namespace Catch
{

std::ofstream null_stream;

std::ostream& cout()
{
   if (Root()) { return std::cout; }
   else { return null_stream; }
}
std::ostream& cerr()
{
   if (Root()) { return std::cerr; }
   else { return null_stream; }
}
std::ostream& clog()
{
   if (Root()) { return std::clog; }
   else { return null_stream; }
}

} // namespace Catch

#endif

#endif
