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
   // unit/general/test_umpire_mem.cpp has some strange requirements, isolate it
   // from all other tests

   // set the default host and device memory types; they will be made dual to
   // each other
   mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST,
                                mfem::MemoryType::DEVICE_UMPIRE);

   // update some dual memory types
   mfem::MemoryManager::SetDualMemoryType(mfem::MemoryType::DEVICE_UMPIRE_2,
                                          mfem::MemoryType::HOST);
   mfem::MemoryManager::SetDualMemoryType(mfem::MemoryType::HOST_PINNED,
                                          mfem::MemoryType::DEVICE_UMPIRE);
   mfem::Device device("gpu");

   // Include only tests labeled with GPU. Exclude parallel tests.
   int res = RunCatchSession(argc, argv, {"[GPU]", "~[Parallel]"});
   return res;
}
