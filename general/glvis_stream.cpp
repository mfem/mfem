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
#define NVTX_COLOR ::nvtx::kCyan

#include <cassert>
#include <istream>

#include "../config/config.hpp" // IWYU pragma: keep dbg

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
                              const bool headless,
                              const std::string &plot_caption,
                              const std::string &data_type,
                              StreamCollection &&streams);

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
inline bool IsMpiInitialized()
{
#ifdef MFEM_USE_MPI
   int flag;
   static bool initialized = (MPI_Initialized(&flag), flag != 0);
   return initialized;
#endif
   return false;
};

inline int MpiSize()
{
#ifdef MFEM_USE_MPI
   if (IsMpiInitialized())
   {
      int tmp = 0;
      static int size = (MPI_Comm_size(MPI_COMM_WORLD, &tmp), tmp);
      return size;
   }
#endif
   return 1;
}

inline int MpiRank()
{
#ifdef MFEM_USE_MPI
   if (IsMpiInitialized())
   {
      int tmp = 0;
      static int rank = (MPI_Comm_rank(MPI_COMM_WORLD, &tmp), tmp);
      return rank;
   }
#endif
   return 0;
}

/////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream(const char*, int, int rank): std::iostream(nullptr),
   data((rank < 0), MpiSize(), MpiRank(), !IsMpiInitialized() || MpiRank() == 0)
{
   // Sets the associated stream buffer to the data stream
   this->rdbuf(data.stream.rdbuf());
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   dbg();
   this->flush(); // will trigger the GLVis update
   this->glvis();
   return *this;
}

void glvis_stream::glvis()
{
   if (data.serial)
   {
      dbg("Serial");
      const auto size = this->size();
      dbg("stream size: {}", size);
      data.offsets.resize(2);
      data.offsets[0] = 0, data.offsets[1] = size;
      data.total_size = size;
   }
   else
   {
      MpiGather();
      dbg("Parallel, data.total_size: {}", data.total_size);
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
         dbg("Creating bufferstream #{}, size: {}, offset: {}",
             k, size, offset);

         // add a new stream for this rank's data
         data.streams.emplace_back(std::make_unique<std::stringstream>());
         data.streams.back()->write(data.stream.str().data() + offset, size);

         auto stream = data.streams.back().get();
         if (!(*stream)) { break; }

         *stream >> std::ws >> data.type >> std::ws;

         if (data.type == "parallel") // Handle parallel data
         {
            dbg("<parallel>");
            int is_mpi_size, is_mpi_rank;
            *stream >> is_mpi_size >> is_mpi_rank;
            assert(is_mpi_size == static_cast<int>(data.mpi_size));
            assert(is_mpi_rank == static_cast<int>(k));
         }
         else if (data.type == "mesh") { dbg("<mesh>"); }
         else if (data.type == "solution") { dbg("<solution>"); }
         else
         {
            dbg("Unknown data_type: '{}'", data.type);
            MFEM_ABORT("Stream: unknown command: " << data.type);
         }
      }
      dbg("✅");
   }

   if (!(data.serial || data.mpi_root)) { dbg("🔴 No GLVisStreamSession 🔴"); return; }

   constexpr bool fix_elem_orien = true;
   constexpr bool save_coloring = true;
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
                      headless,
                      plot_caption,
                      data.type,
                      to_istream_vector(std::move(data.streams)));
   dbg("✅");
}

void glvis_stream::MpiGather()
{
#ifdef MFEM_USE_MPI
   dbg("MPI size: {}, rank: {}", data.mpi_size, data.mpi_rank);

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
         dbg("Rank {} size: {}", r, all_sizes[r]);
      }
   }

   // Compute offsets and total size on root only
   data.offsets.resize(data.mpi_size + 1);
   data.offsets[0] = 0;
   if (data.mpi_root) { dbg("Offsets[{}]: {}", 0, data.offsets[0]); }
   uint64_t total_size = 0;
   for (int i = 1; i <= data.mpi_size; ++i)
   {
      total_size += all_sizes[i - 1];
      if (data.mpi_root)
      {
         data.offsets[i] = total_size;
         dbg("Offsets[{}]: {}", i, data.offsets[i]);
      }
   }
   if (data.mpi_root)
   {
      data.total_size = total_size;
      dbg("Total size: {}", total_size);
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
   status = MPI_Gatherv(data.stream.str().data(),
                        data.stream.tellp(), MPI_CHAR,
                        recvbuf.data(),
                        recvcounts.data(),
                        displs.data(),
                        MPI_CHAR, 0, MPI_COMM_WORLD);
   assert(status == MPI_SUCCESS);

   // Root writes the concatenated result back into its stream
   if (data.mpi_root)
   {
      reset();
      dbg("data.stream size before write: {}", (int)data.stream.tellp());
      data.stream.write(recvbuf.data(), total_size);
      dbg("data.stream size after write: {}", (int)data.stream.tellp());
   }

   MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_MPI
}

} // namespace mfem
