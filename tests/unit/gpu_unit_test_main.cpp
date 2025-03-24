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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   std::cout << "\nThe serial GPU unit tests are not supported in single"
             " precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   mfem::Device device("gpu");

   // Include only tests labeled with GPU. Exclude parallel tests.
   return RunCatchSession(argc, argv, {"[GPU]", "~[Parallel]"});
}
