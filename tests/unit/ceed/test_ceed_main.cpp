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

int main(int argc, char *argv[])
{
   // There must be exactly one instance.
   Catch::Session session;
   std::string device_str("ceed-cpu");
   using namespace Catch::clara;
   auto cli = session.cli()
      | Opt(device_str, "device_string")
        ["--device"]
        ("CEED device string (default: ceed-cpu)");
   session.cli(cli);
   int result = session.applyCommandLine( argc, argv );
   if (result != 0)
   {
      return result;
   }
   mfem::Device device(device_str.c_str());
   result = session.run();
   return result;
}
