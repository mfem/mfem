// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <functional>
#include <memory>
#include "unit_tests.hpp"

bool launch_all_non_regression_tests = false;
bool verbose_tests = false;
std::string mfem_data_dir;

static inline bool Root()
{
#ifdef MFEM_USE_MPI
   return mfem::Mpi::IsInitialized() ? mfem::Mpi::Root() : true;
#else
   return true;
#endif
}

/// Options and extension hooks for executables using the shared Catch runner.
struct CatchSessionOptions
{
   bool root = true;
   bool configure_cpu = false;
   // Offer --parallel in MPI builds and select serial/parallel test tags.
   bool enable_parallel = false;
   bool enable_all = true;
   bool enable_data = true;
   // A fixed device is always configured and cannot be changed from the CLI.
   const char *fixed_device = nullptr;
   // Called before parsing; append executable-specific options to session.cli().
   std::function<void(Catch::Session &)> configure_cli;
   // Called after device setup; a nonzero result skips running the session.
   std::function<int()> before_tests;
};

static int RunCatchSession(int argc, char *argv[],
                           const std::vector<std::string> &testsOrTags,
                           const CatchSessionOptions &options = {})
{
   // There must be exactly one instance.
   Catch::Session session;

   // Build a new command line parser on top of Catch's
   using namespace Catch::clara;
   bool enable_mfem_out = false;
   bool parallel = false;
   std::string device_config = options.fixed_device ? options.fixed_device : "cpu";
   auto cli = session.cli()
              | Opt(enable_mfem_out) ["--enable-output"] ("enable mfem::out")
              | Opt(verbose_tests) ["--verbose"] ("verbose output");
   // Catch reserves -d for test durations.
   if (!options.fixed_device)
   {
      cli |= Opt(device_config, "device") ["--device"]
             ("MFEM device backend (default: cpu)");
   }
   if (options.enable_all)
   {
      cli |= Opt(launch_all_non_regression_tests) ["--all"] ("all tests");
   }
   if (options.enable_data)
   {
      cli |= Opt(mfem_data_dir, "path") ["--data"] ("mfem/data repository");
   }
#ifdef MFEM_USE_MPI
   if (options.enable_parallel)
   {
      cli |= Opt(parallel) ["--parallel"] ("run parallel tests using MPI");
   }
#endif
   session.cli(cli);
   if (options.configure_cli) { options.configure_cli(session); }

   // For floating point comparisons, print 8 digits for single precision
   // values, and 16 digits for double precision values.
   Catch::StringMaker<float>::precision = 8;
   Catch::StringMaker<double>::precision = 16;

   // Apply provided command line arguments.
   int r = session.applyCommandLine(argc, argv);
   if (r != 0) { return r; }

   bool root = options.root;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      mfem::Mpi::Init();
      mfem::Hypre::Init();
      root = Root();
   }
#endif

   // Serial CPU tests may select their own device or memory types. Parallel
   // CPU tests need an explicit device so hypre uses the CPU when possible.
   std::unique_ptr<mfem::Device> device;
   if (device_config != "cpu" || options.configure_cpu || options.fixed_device ||
       parallel)
   {
      device.reset(new mfem::Device(device_config.c_str()));
   }

   if (options.before_tests)
   {
      r = options.before_tests();
      if (r != 0) { return r; }
   }

   auto cfg = session.configData();
   cfg.testsOrTags.insert(cfg.testsOrTags.end(), testsOrTags.begin(),
                          testsOrTags.end());
   if (options.enable_parallel)
   {
      cfg.testsOrTags.push_back(parallel ? "[Parallel]" : "~[Parallel]");
   }
   if (device_config != "cpu" && !options.fixed_device)
   {
      cfg.testsOrTags.push_back("[GPU]");
   }
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
