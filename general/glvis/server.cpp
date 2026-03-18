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
#define NVTX_COLOR ::nvtx::kPaleGreen

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "../../config/config.hpp" // IWYU pragma: keep dbg

#include "../../general/glvis/server.hpp"

#include "../../mesh/mesh.hpp"
#include "../../fem/gridfunc.hpp"

///////////////////////////////////////////////////////////////////////////////
static int Execute(const std::shared_ptr<GLVisData> &data)
{
   const int data_mpi_size = data->mpi_size;
   dbg("\x1b[37m data_mpi_size: {}", data_mpi_size);

   const bool serial = data->serial;
   dbg("\x1b[37m data->serial: {}", serial);

   if (serial)
   {
      dbg("\x1b[37m Serial mode");
      assert(data->mpi_size == 1);
   }
   else
   {
      dbg("\x1b[37m Parallel mode");
      assert(data->mpi_size >= 1);
   }

   std::unique_ptr<mfem::Mesh> serial_mesh;
   std::unique_ptr<mfem::GridFunction> serial_grid_f;

   data->streams.clear();
   data->type.clear();

   std::vector<mfem::Mesh*> parallel_mesh(data_mpi_size);
   std::vector<mfem::GridFunction*> parallel_gf(data_mpi_size);

   // loop over all input streams
   for (int k = 0; k < data_mpi_size; ++k)
   {
      const size_t offset = data->offset[k];
      const size_t size = data->offset[k+1] - data->offset[k];
      dbg("\x1b[37m Creating bufferstream #{}, size: {}, offset: {}",
          k, size, offset);

      // add a new stream for this rank's data
      data->streams.emplace_back();
      data->streams.back().write(data->stream.str().data() + offset, size);

      auto isock = &data->streams.back();
      if (!(*isock)) { dbg("\x1b[37mdone"); break; }

      dbg("\x1b[37m Get data_type");
      *isock >> std::ws >> data->type >> std::ws;
      dbg("\x1b[37m data_type: '{}'", data->type);
      // for (char &c : data->type) { dbg("{:02x}", c); }

      if (data->type == "parallel") // Handle parallel data
      {
         dbg("\x1b[37m<parallel>");

         int mpi_size, mpi_rank;
         *isock >> mpi_size >> mpi_rank;
         dbg("\x1b[37mmpi_size: {}, mpi_rank: {}", mpi_size, mpi_rank);
         assert(mpi_size == static_cast<int>(data->mpi_size));
         assert(mpi_rank == static_cast<int>(k));

#if 1 // do nothing with the stream 
         dbg("Nothing done with the streams");
         dbg("data->type: {}", data->type);
#else
         // "*_data" / "mesh" / "solution"
         *isock >> std::ws >> data->type >> std::ws;
         dbg("\x1b[37mdata_type: {}", data->type);
         parallel_mesh[k] = new mfem::Mesh(*isock, 1, 0, true);
         if (true)
         {
            // set element and boundary attributes to proc+1
            for (int i = 0; i < parallel_mesh[k]->GetNE(); i++)
            {
               parallel_mesh[k]->GetElement(i)->SetAttribute(k+1);
            }
            for (int i = 0; i < parallel_mesh[k]->GetNBE(); i++)
            {
               parallel_mesh[k]->GetBdrElement(i)->SetAttribute(k+1);
            }
         }
         parallel_gf[k] = new mfem::GridFunction(parallel_mesh[k], *isock);
#endif
      }
      else if (data->type == "mesh" || data->type == "solution")
      {
         bool fix_elem_orient = false;
         //  DataState tmp;
         if (data->type == "mesh")
         {
            dbg("\x1b[37m<mesh>");
            serial_mesh = std::make_unique<mfem::Mesh>(data->streams[0], 1, 0,
                                                       fix_elem_orient);
            if (!(data->streams[0])) { dbg("\x1b[37mdone"); break; }
         }
         else if (data->type == "solution")
         {
            dbg("\x1b[37m<solution>");
#if 1 // do nothing with the stream 
            dbg("Nothing done with the streams");
            dbg("data->type: {}", data->type);
#else
            dbg("\x1b[37mMesh");
            serial_mesh = std::make_unique<mfem::Mesh>(data->streams[0], 1, 0,
                                                       fix_elem_orient);
            if (!(data->streams[0])) { dbg("\x1b[37mdone"); break; }

            dbg("\x1b[37mGridFunction");
            serial_grid_f =
               std::make_unique<mfem::GridFunction>(serial_mesh.get(), data->streams[0]);
            if (!(data->streams[0])) { dbg("\x1b[37mdone"); break; }
#endif
         }
         else { MFEM_ABORT("Unknown identifier");}
      }
      else
      {
         dbg("\x1b[37mUnknown data_type: '{}'", data->type);
         MFEM_ABORT("\x1b[31mStream: unknown command: " << data->type);
      }
   }
   dbg("\x1b[37mStreams: end of input.");

   if (!serial)
   {
      dbg("\x1b[37m🚨🚨🚨 Parallel mode 🚨🚨🚨");
      dbg("\x1b[37m🚨🚨🚨   Skipping    🚨🚨🚨");
      /*const size_t nproc = data->streams.size();
      assert(data->mpi_size == nproc);
      serial_mesh = std::make_unique<mfem::Mesh>(parallel_mesh.data(), nproc);
      serial_grid_f = std::make_unique<mfem::GridFunction>
                      (serial_mesh.get(), parallel_gf.data(), nproc);*/
   }

   // glvis -m shm.mesh -g shm.gf

   if (serial_mesh)
   {
      dbg("\x1b[37mSaving SHM mesh");
      std::ofstream mesh_ofs("shm.mesh");
      mesh_ofs.precision(8);
      serial_mesh->Print(mesh_ofs);
   }

   if (serial_grid_f)
   {
      dbg("\x1b[37mSaving SHM grid function");
      std::ofstream sol_ofs("shm.gf");
      sol_ofs.precision(8);
      serial_grid_f->Save(sol_ofs);
   }

   dbg("\x1b[37m✅");
   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Server loop: Waits in place, notifies on ready (no copy)
static int GLVisThreadLoop(const std::shared_ptr<GLVisData> &data)
{
   dbg("\x1b[33m[GLVisThreadLoop] Starting GLVis server thread loop");
   assert(data);
   assert(data->running);
   const auto cleanup = [&]() { dbg("\x1b[33mCleanup"); data->running = false; };

   while (data->running)
   {
      constexpr auto TIMEOUT = 3600; // seconds
      // dbg("\x1b[33mWait for update (timed wait of {}s)", TIMEOUT);
      {
         std::unique_lock<std::mutex> lock(data->mutex);
         bool signaled = false;

         while (!data->update && !signaled)
         {
            const auto start = std::chrono::steady_clock::now();
            const auto timeout = start + std::chrono::seconds(TIMEOUT);
            if (data->cond.wait_until(lock, timeout) == std::cv_status::timeout)
            {
               dbg("\x1b[33mTimeout occurred after {} seconds", TIMEOUT);
               return (cleanup(), EXIT_FAILURE);
            }
            else { signaled = true; }
         } // End of waiting loop

         // timed out or killed
         if (data && !data->update) { dbg("\x1b[33m❌[1]"); return (cleanup(), EXIT_FAILURE); }
         if (!data) { dbg("\x1b[33m❌[2]"); return (cleanup(), EXIT_FAILURE); }
         dbg("\x1b[33mData ready");
      } // unlock mutex

      assert(data);
      assert(data->update);

      const size_t mpi_size = data->mpi_size;
      dbg("\x1b[33mmpi_size: {}", mpi_size);
      assert(mpi_size >= 0 &&
             mpi_size <= sizeof(data->offset) / sizeof(data->offset[0]) - 1);
      if (data->mpi_root)
      {
         for (size_t i = 0; i < mpi_size; ++i)
         {
            dbg("\x1b[33mdata->offset[{}]: {}", i, data->offset[i]);
         }
         dbg("\x1b[33mdata->total_size: {}", data->total_size);
         assert(data->offset[mpi_size] == data->total_size);
      }

      const size_t total_size = data->total_size;
      // assert(total_size <= SHM_DELTA_SIZE);
      dbg("\x1b[33m{}", total_size);

      Execute(data); // Execute

      dbg("Ack: Reset for next");
      // signal_for_update(data);
      data->update.store(false);

      // dbg("one-shot for now");
      // break; // one-shot for now
   }
   return (cleanup(), EXIT_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////
extern void SetUseHiDPI(bool);
thread_local mfem::GeometryRefiner GLVisGeometryRefiner;

///////////////////////////////////////////////////////////////////////////////
GLVisServer::GLVisServer(std::shared_ptr<GLVisData> data): data(data)
{
   if (data->mpi_root)
   {
      dbg("MPI Root");
   }
   else
   {
      dbg("MPI Non-root, return without starting server thread");
      return;
   }

   dbg();
   assert(data);
   SetUseHiDPI(true);

   if (data->running)
   {
      dbg("GLVis Server already running, stopping it first...");
      Stop();
      assert(!data->running);
      dbg("Now starting a new GLVis Server...");
   }

   dbg("Spawning server thread");
   signal_for_running(data);
   assert(data->running);
   glvis_thread = std::make_unique<std::thread>(GLVisThreadLoop, data);

   dbg("GLVis Server started ✅");
   signal_for_ready(data);
   assert(data->ready);
}

int GLVisServer::Wait()
{
   dbg("Waiting for data");
   assert(glvis_thread->joinable());
   glvis_thread->join();
   glvis_thread.reset(nullptr);
   dbg("✅");
   return EXIT_SUCCESS;
}

int GLVisServer::Stop()
{
   dbg();
   if (!data->running)
   {
      dbg("Server not running");
      return EXIT_FAILURE;
   }

   if (glvis_thread && glvis_thread->joinable())
   {
      dbg("Kill server thread: !update & cond.notify_one");
      if (data)
      {
         std::unique_lock<std::mutex> lock(data->mutex);
         data->update = false;
         data->cond.notify_one();
      }
      dbg("Waiting for server thread to finish...");
      glvis_thread->join();
      assert(!data->running);
      dbg("GLVis Server killed");
   }
   dbg("GLVis Server stopped");
   return EXIT_SUCCESS;
}
