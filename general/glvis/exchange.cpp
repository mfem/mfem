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

GLVisExchanger::GLVisExchanger(GLVisData &data): data(data)
{
   MpiDefaultExchange();
   // MpiSharedMemoryExchange();
}

void GLVisExchanger::MpiDefaultExchange()
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

///////////////////////////////////////////////////////////////////////////////
void GLVisExchanger::MpiSharedMemoryExchange()
{
   dbg("Split into shared-memory communicator");
   MPI_Comm shared_comm;
   auto status = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                     0, MPI_INFO_NULL, &shared_comm);
   assert(status == MPI_SUCCESS);

   int shared_rank, shared_size;
   MPI_Comm_rank(shared_comm, &shared_rank);
   MPI_Comm_size(shared_comm, &shared_size);
   dbg("shared_comm size: {}, rank: {}", shared_size, shared_rank);

   // Gather sizes to root and compute offsets
   std::vector<uint64_t> all_sizes(shared_size);
   const size_t stream_size = data.stream.tellp();
   uint64_t local_size_u = stream_size;
   status = MPI_Allgather(&local_size_u, 1, MPI_UINT64_T, all_sizes.data(), 1,
                          MPI_UINT64_T, shared_comm);
   assert(status == MPI_SUCCESS);

   const bool mpi_root = data.mpi_root;
   if (mpi_root)
   {
      for (int r = 0; r < shared_size; ++r)
      {
         dbg("Rank {} size: {}", r, all_sizes[r]);
      }
   }
   data.offsets.resize(shared_size + 1);
   std::vector<uint64_t> offsets(shared_size);
   offsets[0] = 0;
   if (mpi_root) { data.offsets[0] = 0; }
   for (int i = 1; i < shared_size; ++i)
   {
      offsets[i] = offsets[i - 1] + all_sizes[i - 1];
      if (mpi_root) { dbg("Offsets[{}]: {}", i, offsets[i]); }
      if (mpi_root) { data.offsets[i] = offsets[i]; }
   }
   assert(shared_size > 0);
   uint64_t total_size = (shared_size > 0) ? offsets.back() + all_sizes.back() : 0;
   if (mpi_root) { data.offsets[shared_size] = total_size; }
   if (mpi_root) { data.total_size = total_size; }
   if (mpi_root) { dbg("Total size: {}", total_size); }

   // Allocate shared window (total size on root, 0 on others)
   constexpr int size_of_type = sizeof(char);
   static_assert(size_of_type == 1, "size_of_type != 1");
   MPI_Win mpi_win;
   void* base_ptr = nullptr;
   MPI_Aint alloc_size = mpi_root ? total_size * size_of_type : 0;
   status = MPI_Win_allocate_shared(alloc_size, size_of_type,
                                    MPI_INFO_NULL, shared_comm,
                                    &base_ptr, &mpi_win);
   assert(status == MPI_SUCCESS);

   // Get shared buffer pointer
   char* shared_buf = nullptr;
   if (mpi_root)
   {
      dbg("shared_rank == 0, base_ptr: {}, alloc_size: {}",
          (void*)base_ptr, alloc_size);
      shared_buf = static_cast<char*>(base_ptr);
   }
   else
   {
      // Query from rank 0
      int target_disp_unit;
      status = MPI_Win_shared_query(mpi_win, 0, &alloc_size, &target_disp_unit,
                                    &base_ptr);
      assert(status == MPI_SUCCESS);
      shared_buf = static_cast<char*>(base_ptr);
      dbg("shared_rank != 0, base_ptr: {}, alloc_size: {}, target_disp_unit: {}",
          (void*)base_ptr, alloc_size, target_disp_unit);
   }

   // Start access
   MPI_Win_fence(0, mpi_win);
   MPI_Barrier(shared_comm);

   // each rank copies to the MPI window
   dbg("Rank {} copying data to shared buffer at offset {}, size {}",
       data.mpi_rank, offsets[shared_rank], all_sizes[shared_rank]);
   std::memcpy(shared_buf + offsets[shared_rank], // dst
               data.stream.str().data(), // src
               all_sizes[shared_rank] * size_of_type);
   assert(all_sizes[shared_rank] == stream_size);

   // Sync writes
   MPI_Win_fence(0, mpi_win);

   // Wait for all
   MPI_Barrier(shared_comm);

   // Root rank copies to SharedData buffer
   if (mpi_root)
   {
      data.stream.clear();
      data.stream.seekg(0);
      data.stream.seekp(0);
      dbg("data.stream size before write: {}", (int)data.stream.tellp());
      data.stream.write(shared_buf, total_size);
      dbg("data.stream size after write: {}", (int)data.stream.tellp());
   }

   MPI_Barrier(shared_comm);
   MPI_Win_free(&mpi_win);
   MPI_Comm_free(&shared_comm);
}

} // namespace mfem
