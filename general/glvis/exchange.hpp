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

#include "../../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_MPI

#include <mpi.h>

#include "../../general/glvis/data.hpp"

namespace mfem
{

class GLVisExchanger
{
   int tmp = 0;
   const size_t stream_size;
   const std::size_t buffer_size;
   char *buffer;
   const int mpi_size, mpi_rank;
   const bool mpi_root;
   MPI_Comm shared_comm;
   int shared_rank, shared_size;

   MPI_Win win;
   void* base_ptr = nullptr;

public:
   GLVisExchanger(const std::shared_ptr<GLVisData> &data);
};

} // namespace mfem

#endif // MFEM_USE_MPI