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
#include <vector>

#include <boost/interprocess/streams/bufferstream.hpp>
namespace bip = boost::interprocess;

using char_stream_t = bip::basic_bufferstream<char>;
using char_stream_uptr = std::unique_ptr<char_stream_t>;

///////////////////////////////////////////////////////////////////////////////
constexpr size_t RNK_SIZE = 8*1024*1024;
constexpr size_t SHM_SIZE = 64*1024*1024;
constexpr size_t BIP_SIZE = 4*1024;
constexpr size_t SHM_DELTA_SIZE = SHM_SIZE - BIP_SIZE;

constexpr int GLVIS_MAX_HOOK_SECONDS = 1;
constexpr int GLVIS_MAX_WAIT_SECONDS = 3600;

struct GLVisData
{
   std::mutex mutex;
   std::condition_variable cond;
   std::atomic<bool> running {false};
   std::atomic<bool> ready {false}, update {false};
   std::size_t streamsize;
   char buffer[SHM_DELTA_SIZE];
   bool serial {true};
   size_t shared_size {0}; // should be equal to mpi_size in parallel
   size_t offset[32];
   size_t total_size {0};
   /////////////////////////////////////
   std::vector<char_stream_uptr> bbs {};
   void *win {nullptr};
   const bool fix_elem_orien = true, save_coloring = true;
   const std::string plot_caption = {"GLVis from MFEM" };
   const bool headless = false;
   std::vector<std::unique_ptr<std::istream>> streams {};
   std::string type { "unknown" };
};