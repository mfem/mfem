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
#define NVTX_COLOR ::nvtx::kOrchid

#include <cassert>

#include "../../general/glvis/data.hpp"
#include "../../general/glvis/exchange.hpp"

namespace mfem
{

GLVisExchanger::GLVisExchanger(const std::shared_ptr<GLVisData> &data):
   stream_size((assert(data), data->stream.tellp())),
   buffer_size(16*1024*1024),
   buffer(static_cast<char*>(std::malloc(buffer_size))),
   mpi_size((MPI_Comm_size(MPI_COMM_WORLD, &tmp), tmp)),
   mpi_rank((MPI_Comm_rank(MPI_COMM_WORLD, &tmp), tmp))
{
   dbg("MPI size: {}, rank: {}", mpi_size, mpi_rank);

   dbg("buffer_size: {}, stream_size: {}", buffer_size, stream_size);
   assert(stream_size <= buffer_size);

   dbg("\x1b[33m data stream:\n{}", data->stream.str());

   dbg("Split into shared-memory communicator");
   auto status = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                     0, MPI_INFO_NULL, &shared_comm);
   assert(status == MPI_SUCCESS);

   MPI_Comm_rank(shared_comm, &shared_rank);
   MPI_Comm_size(shared_comm, &shared_size);
   assert(shared_size == mpi_size);

   constexpr int size_of_type = sizeof(char);
   static_assert(size_of_type == 1, "size_of_type != 1");

   std::vector<uint64_t> all_sizes(shared_size);
   uint64_t local_size_u = stream_size;
   status = MPI_Allgather(&local_size_u, 1, MPI_UINT64_T, all_sizes.data(), 1,
                          MPI_UINT64_T, shared_comm);
   assert(status == MPI_SUCCESS);

   if (data)
   {
      for (int r = 0; r < shared_size; ++r)
      {
         dbg("Rank {} size: {}", r, all_sizes[r]);
      }
   }

   std::vector<uint64_t> offsets(shared_size);
   offsets[0] = 0;
   if (data) { dbg("Offsets[0]: {}", offsets[0]); }

   if (data) { data->mpi_size = 1; }
   if (data) { data->total_size = shared_size; }
   if (data) { data->offset[0] = 0; }
   for (int i = 1; i < shared_size; ++i)
   {
      offsets[i] = offsets[i - 1] + all_sizes[i - 1];
      if (data) { dbg("Offsets[{}]: {}", i, offsets[i]); }
      if (data) { data->offset[i] = offsets[i]; }
   }
   uint64_t total_size = (shared_size > 0) ? offsets.back() + all_sizes.back() : 0;
   if (data) { assert((size_t)shared_size < sizeof(data->offset)); }
   if (data) { data->offset[shared_size] = total_size; }
   if (data) { data->total_size = total_size; }
   if (data) { dbg("Total size: {}", total_size); }

   // Allocate shared window (full on rank 0)
   MPI_Aint alloc_size = (shared_rank == 0) ? total_size * size_of_type : 0;
   status = MPI_Win_allocate_shared(alloc_size, size_of_type,
                                    MPI_INFO_NULL, shared_comm,
                                    &base_ptr, &win);
   assert(status == MPI_SUCCESS);

   // Get shared buffer pointer
   char* shared_buf = nullptr;
   if (shared_rank == 0)
   {
      dbg("shared_rank == 0, base_ptr: {}, alloc_size: {}",
          (void*)base_ptr, alloc_size);
      shared_buf = static_cast<char*>(base_ptr);
   }
   else
   {
      // Query from rank 0
      int target_disp_unit;
      status = MPI_Win_shared_query(win, 0, &alloc_size, &target_disp_unit,
                                    &base_ptr);
      assert(status == MPI_SUCCESS);
      shared_buf = static_cast<char*>(base_ptr);
      dbg("shared_rank != 0, base_ptr: {}, alloc_size: {}, target_disp_unit: {}",
          (void*)base_ptr, alloc_size, target_disp_unit);
      assert(false);
   }

   // Start access
   MPI_Win_fence(0, win);
   MPI_Barrier(MPI_COMM_WORLD);

   // #ifdef MFEM_DEBUG
   //    for (int r = 0; r < shared_size; ++r)
   //    {
   //       if (r == shared_rank)
   //       {
   //          dbl("Rank {} data ({} bytes):\n", r, all_sizes[r]);
   //          for (size_t i = 0; i < all_sizes[r]; ++i) { dba("{}", buffer[i]); }
   //          dbc();
   //       }
   //       MPI_Barrier(MPI_COMM_WORLD);
   //    }
   // #endif

   // each rank copies to the MPI window
   dbg("Rank {} copying data to shared buffer at offset {}, size {}",
       mpi_rank, offsets[shared_rank], all_sizes[shared_rank]);
   std::memcpy(shared_buf + offsets[shared_rank], // dst
               data->stream.str().data(), // src
               all_sizes[shared_rank] * size_of_type);
   dbg("Rank {}, buf: {}", mpi_rank, std::string(data->stream.str().data(),
                                                 all_sizes[shared_rank]));

   // Sync writes
   MPI_Win_fence(0, win);

   // Wait for all
   MPI_Barrier(shared_comm);

   // Root rank copies to SharedData buffer
   if ((mpi_rank == 0) && data)
   {
      // #ifdef MFEM_DEBUG
      //       dbg("Rank 0 reads combined data");
      //       for (int r = 0; r < shared_size; ++r)
      //       {
      //          dbl("\x1b[33m(rank {}):\n", r);
      //          for (uint64_t i = 0; i < all_sizes[r]; ++i)
      //          {
      //             dba("{}", shared_buf[offsets[r] + i]);
      //          }
      //          dbc();
      //       }
      // #endif
      assert(total_size <= buffer_size);
      data->stream.clear();
      data->stream.seekg(0);
      data->stream.seekp(0);
      dbg("data->stream size before write: {}", (int)data->stream.tellp());
      data->stream.write(shared_buf, total_size);
      dbg("data->stream size after write: {}", (int)data->stream.tellp());
   }

   MPI_Barrier(shared_comm);
   MPI_Barrier(MPI_COMM_WORLD);

   MPI_Win_free(&win);
   MPI_Comm_free(&shared_comm);
}

} // namespace mfem
