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

#include "mfem.hpp"
#include "general/jit/jit.hpp"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char *argv[])
{
   Catch::Session session;
   std::string device_str("cpu");
   auto cli = session.cli()
              | Catch::clara::Opt(device_str, "device_string")
              ["--device"]("device string (default: cpu)");
   session.cli(cli);
   if (session.applyCommandLine(argc, argv) != 0) { return EXIT_FAILURE; }
   mfem::Device device(device_str.c_str());
   mfem::Jit::Configure("mjit_test_seq", ".", false);
   int result = session.run();
   mfem::Jit::Finalize();
   return result;
}
