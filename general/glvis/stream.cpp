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
#include <string>
#include <iostream>

#include "../../config/config.hpp" // IWYU pragma: keep dbg
#include "../../fem/geom.hpp" // GeometryRefiner

#include "../../general/glvis/stream.hpp"

#ifdef MFEM_USE_MPI
#include "../../general/glvis/exchange.hpp"
#endif

thread_local mfem::GeometryRefiner GLVisGeometryRefiner;

extern int GLVisLibWindow(bool fix_elem_orient,
                          bool save_coloring, bool headless,
                          std::istream &stream,
                          std::string data_type);

extern int GLVisLibWindow(bool fix_elem_orient,
                          bool save_coloring, bool headless,
                          StreamCollection &&streams);

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
int glvis_stream::MpiSize() const
{
#ifdef MFEM_USE_MPI
   if (mpi_initialized)
   {
      int tmp = 0;
      static int size = (MPI_Comm_size(MPI_COMM_WORLD, &tmp), tmp);
      return size;
   }
#endif
   return 1;
}

int glvis_stream::MpiRank() const
{
#ifdef MFEM_USE_MPI
   if (mpi_initialized)
   {
      int tmp = 0;
      static int rank = (MPI_Comm_rank(MPI_COMM_WORLD, &tmp), tmp);
      return rank;
   }
#endif
   return 0;
}

const auto IsMpiInitialized = []()
{
   int flag;
   MPI_Initialized(&flag);
   return flag != 0;
};

/////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream(const char*, int, int rank):
   std::iostream((dbg(rank >= 0 ? "Parallel" : "Serial"), nullptr)),
   mpi_initialized(IsMpiInitialized()),
   mpi_size(MpiSize()),
   mpi_rank(MpiRank()),
   serial(rank < 0),
   mpi_root(!mpi_initialized || mpi_rank == 0),
   data(serial, mpi_size, mpi_rank, mpi_root)
{
   // Sets the associated stream buffer to the data stream one
   this->rdbuf(data.stream.rdbuf());
}

/////////////////////////////////////////////////////////////////////
glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   dbg();
   this->flush(); // will trigger the GLVis update
   std::iostream::operator<<(pf); // optional
   this->glvis();
   return *this;
}

void glvis_stream::flush() { std::iostream::flush(); }

void glvis_stream::reset()
{
   data.stream.clear();
   data.stream.seekg(0, std::ios::beg);
   data.stream.seekp(0, std::ios::beg);
}

void glvis_stream::glvis()
{
   if (data.serial)
   {
      dbg("Serial");
      const auto size = this->size();
      dbg("stream size: {}", size);
      assert(size > 0);
      assert(data.mpi_size == 1);
      data.offsets.resize(2);
      data.offsets[0] = 0, data.offsets[1] = size;
      data.total_size = size;
   }
   else
   {
      dbg("Parallel, data.mpi_size: {}", data.mpi_size);
      dbg("Parallel, data.total_size: {}", data.total_size);
      MpiExchange();
   }

   this->reset(); // reset the local buffer for reuse

   if (mpi_root)
   {
      assert(mpi_size >= 0 && (size_t)mpi_size <= data.offsets.size());
      if (data.mpi_root)
      {
         for (int i = 0; i < mpi_size; ++i)
         {
            dbg("\x1b[33mdata->offset[{}]: {}", i, data.offsets[i]);
         }
         dbg("\x1b[33mdata->total_size: {}", data.total_size);
         assert(data.offsets[mpi_size] == data.total_size);
      }

      const int data_mpi_size = data.mpi_size;
      dbg("\x1b[37m data_mpi_size: {}", data_mpi_size);
      dbg("\x1b[37m data.serial: {}", serial);

      if (serial)
      {
         dbg("\x1b[37m Serial mode");
         assert(data.mpi_size == 1);
      }
      else
      {
         dbg("\x1b[37m Parallel mode");
         assert(data.mpi_size >= 1);
      }

      data.streams.clear();
      data.type.clear();

      // loop over all input streams
      for (int k = 0; k < data_mpi_size; ++k)
      {
         const size_t offset = data.offsets[k];
         const size_t size = data.offsets[k+1] - data.offsets[k];
         dbg("\x1b[37m Creating bufferstream #{}, size: {}, offset: {}",
             k, size, offset);

         // add a new stream for this rank's data
         data.streams.emplace_back();
         data.streams.back().write(data.stream.str().data() + offset, size);

         auto isock = &data.streams.back();
         if (!(*isock)) { dbg("\x1b[37mdone"); break; }

         dbg("\x1b[37m Get data_type");
         *isock >> std::ws >> data.type >> std::ws;
         dbg("\x1b[37m data_type: '{}'", data.type);

         if (data.type == "parallel") // Handle parallel data
         {
            dbg("\x1b[37m <parallel>");

            int is_mpi_size, is_mpi_rank;
            *isock >> is_mpi_size >> is_mpi_rank;
            dbg("\x1b[37m is_mpi_size: {}, is_mpi_rank: {}", is_mpi_size, is_mpi_rank);
            assert(is_mpi_size == static_cast<int>(data.mpi_size));
            assert(is_mpi_rank == static_cast<int>(k));

            dbg("\x1b[37m Nothing done with the streams");
         }
         else if (data.type == "mesh" || data.type == "solution")
         {
            if (data.type == "mesh")
            {
               dbg("\x1b[37m <mesh>");
            }
            else if (data.type == "solution")
            {
               dbg("\x1b[37m <solution>");
            }
            else { MFEM_ABORT("Unknown identifier");}
         }
         else
         {
            dbg("\x1b[37m Unknown data_type: '{}'", data.type);
            MFEM_ABORT("\x1b[31mStream: unknown command: " << data.type);
         }
      }
      dbg("\x1b[37m ✅");
   }

   constexpr bool fix_elem_orien = true;
   constexpr bool save_coloring = true;
   constexpr bool headless = false;

   // SDL2 window needs to be in the 'main' thread
   if (data.serial)
   {
      dbg("🟠 Serial mode 🟠");
      GLVisLibWindow(fix_elem_orien, save_coloring, headless,
                     data.streams[0], data.type);
   }
   else if (mpi_root)
   {
      dbg("🟣 Parallel Root 🟣");
      const auto to_istream_vector = [](std::vector<std::stringstream> &&streams)
                                     ->std::vector<std::unique_ptr<std::istream>>
      {
         std::vector<std::unique_ptr<std::istream>> result;
         result.reserve(streams.size());
         for (auto& s : streams)
         {
            // dbg("Adding stream of size {}", s.str().size());
            result.push_back(
               std::make_unique<std::stringstream>(std::move(s)));
         }
         return result;
      };
      GLVisLibWindow(fix_elem_orien, save_coloring, headless,
                     to_istream_vector(std::move(data.streams)));
   }
   else
   {
      dbg("🔴 Parallel None 🔴");
   }
   dbg("✅");
}

void glvis_stream::MpiExchange()
{
   dbg("MPI size: {}, rank: {}", data.mpi_size, data.mpi_rank);

   // Gather sizes from ALL ranks
   std::vector<uint64_t> all_sizes(data.mpi_size);
   uint64_t local_size_u = data.stream.tellp();
   int status = MPI_Allgather(&local_size_u, 1, MPI_UINT64_T,
                              all_sizes.data(), 1, MPI_UINT64_T,
                              MPI_COMM_WORLD);
   assert(status == MPI_SUCCESS);
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
      data.stream.clear();
      data.stream.seekg(0);
      data.stream.seekp(0);
      dbg("data.stream size before write: {}", (int)data.stream.tellp());
      data.stream.write(recvbuf.data(), total_size);
      dbg("data.stream size after write: {}", (int)data.stream.tellp());
   }

   MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace mfem
