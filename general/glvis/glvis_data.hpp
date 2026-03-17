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
#pragma once

#include <condition_variable>
#include <istream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

struct GLVisData
{
   std::mutex mutex;
   std::condition_variable cond;
   std::atomic<bool> running {false}, ready {false}, update {false};
   /////////////////////////////////////
   std::stringstream stream;
   std::size_t streamsize;
   /////////////////////////////////////
   bool serial {true};
   size_t shared_size {0}; // should be equal to mpi_size in parallel
   size_t offset[32];
   size_t total_size {0};
   /////////////////////////////////////
   std::vector<std::unique_ptr<std::stringstream>> bbs {};
   static constexpr bool fix_elem_orien = true, save_coloring = true;
   static constexpr bool headless = false;
   std::vector<std::unique_ptr<std::istream>> streams {};
   std::string type { "unknown" };
   static constexpr int GLVIS_MAX_WAIT_SECONDS = 3600;
};