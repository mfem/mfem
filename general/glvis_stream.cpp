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

#include <cassert>
#include <istream>

#include "../config/config.hpp"

#ifdef MFEM_USE_GLVIS

#include "glvis_stream.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <limits>
#include <numeric>
#endif

#include "../fem/geom.hpp"
thread_local mfem::GeometryRefiner GLVisGeometryRefiner;

// Use local declaration to avoid circular dependency when fetching GLVis
extern int GLVisStreamSession(
   bool fix_elem_orient,
   bool save_coloring,
   bool keep_attr,
   bool headless,
   const std::string& plot_caption,
   const std::string& data_type,
   std::vector<std::unique_ptr<std::istream>>&& streams);

namespace mfem
{

#ifdef MFEM_USE_MPI
namespace
{
glvis_data MakeGlVisData()
{
   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   return glvis_data(size == 1, size, rank == 0);
}
} // namespace
#endif

glvis_stream::glvis_stream(): std::iostream(nullptr),
#ifdef MFEM_USE_MPI
   data(MakeGlVisData())
#else
   data(true, 1, true)
#endif
{
   std::iostream::rdbuf(data.stream.rdbuf());
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   pf(static_cast<std::ostream&>(*this));
   this->flush();
   this->operator()();
   return *this;
}

void glvis_stream::operator()()
{
   if (data.serial)
   {
      const auto size = this->size();
      data.offsets.resize(2);
      data.offsets[0] = 0, data.offsets[1] = size;
      data.total_size = size;
   }
   else
   {
      serialize();
   }

   this->reset(); // reset the local buffer for reuse

   if (data.mpi_root)
   {
      MFEM_VERIFY(data.mpi_size >= 0 &&
                  (size_t) data.mpi_size == data.offsets.size() - 1,
                  "Invalid MPI size");

      data.streams.clear();
      data.type.clear();

      // loop over all input streams
      for (int k = 0; k < data.mpi_size; ++k)
      {
         const size_t offset = data.offsets[k];
         const size_t size = data.offsets[k+1] - data.offsets[k];

         // add a new stream for this rank's data
         data.streams.emplace_back(std::make_unique<std::stringstream>());
         data.streams.back()->write(data.stream.str().data() + offset, size);

         auto stream = data.streams.back().get();
         if (!(*stream)) { break; }

         *stream >> std::ws >> data.type >> std::ws;

         if (data.type == "parallel") // Handle parallel data
         {
            int is_mpi_size, is_mpi_rank;
            *stream >> is_mpi_size >> is_mpi_rank;
            assert(is_mpi_size == static_cast<int>(data.mpi_size));
            assert(is_mpi_rank == static_cast<int>(k));
         }
         else if (data.type != "mesh" && data.type != "solution")
         {
            MFEM_ABORT("Stream: unknown command: " << data.type);
         }
      }
   }

   if (!data.mpi_root) { return; }

   constexpr bool fix_elem_orien = true;
   constexpr bool save_coloring = true;
   constexpr bool keep_attr = false;
   constexpr bool headless = false;
   const std::string plot_caption {};
   std::vector<std::unique_ptr<std::istream>> istreams;
   istreams.reserve(data.streams.size());
   for (auto &s : data.streams) { istreams.push_back(std::move(s)); }
   GLVisStreamSession(fix_elem_orien,
                      save_coloring,
                      keep_attr,
                      headless,
                      plot_caption,
                      data.type,
                      std::move(istreams));
}

void glvis_stream::serialize()
{
#ifdef MFEM_USE_MPI
   const std::string local = data.stream.str();
   MFEM_VERIFY(local.size() <= static_cast<size_t>
               (std::numeric_limits<int>::max()),
               "GLVis stream is too large for MPI_Gatherv");
   const int local_size = static_cast<int>(local.size());

   std::vector<int> sizes(data.mpi_size);
   MFEM_VERIFY(MPI_Allgather(&local_size, 1, MPI_INT,
                             sizes.data(), 1, MPI_INT,
                             MPI_COMM_WORLD) == MPI_SUCCESS,
               "MPI_Allgather failed");

   if (data.mpi_root)
   {
      data.offsets.resize(data.mpi_size + 1);
      data.offsets[0] = 0;
      std::partial_sum(sizes.begin(), sizes.end(), data.offsets.begin() + 1);
      data.total_size = data.offsets[data.mpi_size];
   }

   std::vector<char> recvbuf(data.mpi_root ? data.total_size : 0);
   MFEM_VERIFY(MPI_Gatherv(local.data(), local_size, MPI_CHAR,
                           data.mpi_root ? recvbuf.data() : nullptr,
                           data.mpi_root ? sizes.data() : nullptr,
                           data.mpi_root ? data.offsets.data() : nullptr,
                           MPI_CHAR, 0, MPI_COMM_WORLD) == MPI_SUCCESS,
               "MPI_Gatherv failed");

   if (data.mpi_root)
   {
      reset();
      data.stream.write(recvbuf.data(), data.total_size);
   }
#endif // MFEM_USE_MPI
}

} // namespace mfem

#endif // MFEM_USE_GLVIS
