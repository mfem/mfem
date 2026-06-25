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
#include <limits>

#include "../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_GLVIS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "glvis_stream.hpp"

///////////////////////////////////////////////////////////////////////////////
using StreamCollection = std::vector<std::unique_ptr<std::istream>>;

#include "../fem/geom.hpp" // GeometryRefiner

thread_local mfem::GeometryRefiner GLVisGeometryRefiner;

extern int GLVisStreamSession(const bool fix_elem_orient,
                              const bool save_coloring,
                              const bool keep_attr,
                              const bool headless,
                              const std::string &plot_caption,
                              const std::string &data_type,
                              StreamCollection &&streams);

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
inline bool IsMpiInitialized()
{
   int flag;
   MPI_Initialized(&flag);
   return flag != 0;
};

inline int MpiSize()
{
   int size = 1;
   if (IsMpiInitialized()) { MPI_Comm_size(MPI_COMM_WORLD, &size); }
   return size;
}

inline int MpiRank()
{
   int rank = 0;
   if (IsMpiInitialized()) { MPI_Comm_rank(MPI_COMM_WORLD, &rank); }
   return rank;
}

/////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream(): glvis_stream(nullptr, 0) {}

glvis_stream::glvis_stream(std::streambuf*): glvis_stream(nullptr, 0) {}

glvis_stream::glvis_stream(const char*, int): std::iostream(nullptr),
   data(MpiSize() == 1, MpiSize(), MpiRank(), MpiRank() == 0)
{
   // Sets the associated stream buffer to the data stream
   std::iostream::rdbuf(data.stream.rdbuf());
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   // this->flush(); // will trigger the GLVis update
   pf(static_cast<std::ostream&>(*this));
   this->flush();
   this->glvis();
   return *this;
}

void glvis_stream::glvis()
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
      MpiGather();
   }

   this->reset(); // reset the local buffer for reuse

   if (data.mpi_root)
   {
      assert(data.mpi_size >= 0 &&
             (size_t) data.mpi_size == data.offsets.size() - 1);
      if (data.mpi_root) { assert(data.offsets[data.mpi_size] == data.total_size); }

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
         else if (data.type == "mesh") { /**/ }
         else if (data.type == "solution") { /**/ }
         else
         {
            MFEM_ABORT("Stream: unknown command: " << data.type);
         }
      }
   }

   if (!(data.serial || data.mpi_root)) { return; }

   constexpr bool fix_elem_orien = true;
   constexpr bool save_coloring = true;
   constexpr bool keep_attr = false;
   constexpr bool headless = false;
   const std::string plot_caption {};
   const auto to_istream_vector =
      [](std::vector<std::unique_ptr<std::stringstream>> &&streams)
      ->std::vector<std::unique_ptr<std::istream>>
   {
      std::vector<std::unique_ptr<std::istream>> istreams;
      istreams.reserve(streams.size());
      for (auto& s : streams) { istreams.push_back(std::move(s)); }
      return istreams;
   };
   GLVisStreamSession(fix_elem_orien,
                      save_coloring,
                      keep_attr,
                      headless,
                      plot_caption,
                      data.type,
                      to_istream_vector(std::move(data.streams)));
}

void glvis_stream::MpiGather()
{
#ifdef MFEM_USE_MPI
   // Gather sizes from ALL ranks
   std::vector<uint64_t> all_sizes(data.mpi_size);
   uint64_t local_size_u = data.stream.tellp();
   int status = MPI_Allgather(&local_size_u, 1, MPI_UINT64_T,
                              all_sizes.data(), 1, MPI_UINT64_T,
                              MPI_COMM_WORLD);
   MFEM_VERIFY(status == MPI_SUCCESS, "MPI_Allgather failed");
   if (data.mpi_root)
   {
      for (int r = 0; r < data.mpi_size; ++r)
      {
         mfem::out << "Rank " << r << " size: " << all_sizes[r] << std::endl;
      }
   }

   // Compute offsets and total size on root only
   data.offsets.resize(data.mpi_size + 1);
   data.offsets[0] = 0;
   uint64_t total_size = 0;
   for (int i = 1; i <= data.mpi_size; ++i)
   {
      total_size += all_sizes[i - 1];
      if (data.mpi_root)
      {
         data.offsets[i] = total_size;
      }
   }
   if (data.mpi_root)
   {
      data.total_size = total_size;
      mfem::out << "Total size: " << data.total_size << std::endl;
   }

   // Root allocates a temporary buffer for the full data
   std::vector<char> recvbuf;
   if (data.mpi_root) { recvbuf.resize(total_size); }

   // Prepare Gatherv arguments (counts & displacements)
   std::vector<int> recvcounts(data.mpi_size, 0);
   std::vector<int> displs(data.mpi_size, 0);
   if (data.mpi_root)
   {
      for (int i = 0; i < data.mpi_size; ++i)
      {
         assert(all_sizes[i] <= static_cast<uint64_t>(std::numeric_limits<int>::max()));
         recvcounts[i] = static_cast<int>(all_sizes[i]);
         displs[i]     = static_cast<int>(data.offsets[i]);
      }
   }

   // Everyone sends their data to the root
   const std::string local = data.stream.str();
   MFEM_VERIFY(local.size() <= static_cast<size_t>
               (std::numeric_limits<int>::max()),
               "GLVis stream is too large for MPI_Gatherv");
   const int sendcount = static_cast<int>(local.size());
   status = MPI_Gatherv(local.data(), sendcount, MPI_CHAR,
                        (data.mpi_root ? recvbuf.data() : nullptr),
                        recvcounts.data(), displs.data(),
                        MPI_CHAR, 0, MPI_COMM_WORLD);
   MFEM_VERIFY(status == MPI_SUCCESS, "MPI_Gatherv failed");

   // Root writes the concatenated result back into its stream
   if (data.mpi_root)
   {
      reset();
      data.stream.write(recvbuf.data(), total_size);
   }

   MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_MPI
}

} // namespace mfem

#endif // MFEM_USE_GLVIS
