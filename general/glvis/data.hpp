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

#include <sstream>
#include <vector>

using StreamCollection = std::vector<std::unique_ptr<std::istream>>;
using StreamUniqueVector = StreamCollection;

struct GLVisData
{
   const bool serial;
   const int mpi_size, mpi_rank, mpi_root;
   std::stringstream stream;
   std::vector<std::stringstream> streams;
   size_t total_size;
   std::vector<size_t> offsets;
   std::string type;

   GLVisData(const bool serial, const size_t size,
             const size_t rank, const bool root):
      serial(serial), mpi_size(size), mpi_rank(rank), mpi_root(root),
      stream(), streams(), total_size(0), offsets(), type({}) {}
};