// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <exception>
#include <istream>
#include <limits>
#include <new>

#include "../config/config.hpp"
#include "communication.hpp"
#include "error.hpp"

#ifdef MFEM_USE_GLVIS

#include "glvis_stream.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

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

glvis_stream::glvis_stream()
   : std::iostream(nullptr),
#ifdef MFEM_USE_MPI
     data(MPI_COMM_WORLD)
#else
     data()
#endif
{
   std::iostream::rdbuf(data.stream.rdbuf());
}

#ifdef MFEM_USE_MPI
glvis_stream::glvis_stream(MPI_Comm comm)
   : std::iostream(nullptr), data(comm)
{
   std::iostream::rdbuf(data.stream.rdbuf());
}
#endif

void glvis_stream::SyncComm()
{
#ifdef MFEM_USE_MPI
   // A serial example may link a parallel MFEM and never call Mpi::Init.
   // A stream built before init reads the communicator here, on flush.
   if (!Mpi::IsInitialized() || Mpi::IsFinalized())
   {
      data.serial = true;
      data.mpi_size = 1;
      data.mpi_root = true;
      return;
   }
   int size = 1, rank = 0;
   MFEM_VERIFY(MPI_Comm_size(data.comm, &size) == MPI_SUCCESS &&
               MPI_Comm_rank(data.comm, &rank) == MPI_SUCCESS,
               "MPI_Comm_size/rank failed");
   data.serial = (size == 1);
   data.mpi_size = size;
   data.mpi_root = (rank == 0);
#endif
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   pf(static_cast<std::ostream&>(*this));
   // std::endl calls basic_ostream::flush. Only std::flush starts a session.
   if (pf == static_cast<ostream_manipulator>(std::flush))
   {
      (*this)();
   }
   return *this;
}

void glvis_stream::flush()
{
   std::iostream::flush();
   (*this)();
}

// Every rank enters. One rank's failure aborts every rank after the reduce.
static void FailTogether(int failed, const std::string &why
#ifdef MFEM_USE_MPI
                         , const bool serial, MPI_Comm comm
#endif
                        )
{
#ifdef MFEM_USE_MPI
   if (!serial)
   {
      int any = 0;
      MFEM_VERIFY(MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX,
                                comm) == MPI_SUCCESS,
                  "MPI_Allreduce failed");
      failed = any;
   }
#endif
   if (failed)
   {
      std::string msg = why;
      if (msg.empty()) { msg = "GLVis stream failed"; }
      MFEM_ABORT(msg);
   }
}

void glvis_stream::operator()()
{
   SyncComm();

   if (data.serial)
   {
      const std::streamoff written = data.stream.tellp();
      MFEM_VERIFY(written >= 0 &&
                  static_cast<unsigned long long>(written) <=
                  static_cast<unsigned long long>
                  (std::numeric_limits<int>::max()),
                  "GLVis stream is too large");
      const int bytes = static_cast<int>(written);
      data.offsets = {0, bytes};
      data.total_size = bytes;
   }
   else
   {
      serialize();
   }

   int failed = 0;
   std::string why;

   try
   {
      if (data.mpi_root)
      {
         if (data.mpi_size < 0 ||
             static_cast<size_t>(data.mpi_size) + 1 != data.offsets.size())
         {
            failed = 1;
            why = "Invalid MPI size";
         }

         const std::string payload = data.stream.str();
         data.streams.clear();
         data.type.clear();

         for (int k = 0; k < data.mpi_size && !failed; ++k)
         {
            const size_t offset = static_cast<size_t>(data.offsets[k]);
            const size_t count = static_cast<size_t>(data.offsets[k + 1] -
                                                     data.offsets[k]);
            if (offset + count > payload.size())
            {
               failed = 1;
               why = "GLVis rank slice is out of range";
               break;
            }

            data.streams.emplace_back(std::make_unique<std::stringstream>());
            data.streams.back()->write(payload.data() + offset, count);

            auto stream = data.streams.back().get();
            if (!(*stream))
            {
               failed = 1;
               why = "GLVis rank slice could not be written";
               break;
            }

            std::string rank_type;
            *stream >> std::ws >> rank_type >> std::ws;
            if (rank_type.empty())
            {
               failed = 1;
               why = "GLVis payload has no type";
               break;
            }
            if (data.type.empty())
            {
               data.type = rank_type;
            }
            else if (rank_type != data.type)
            {
               failed = 1;
               why = "GLVis ranks sent different payload types";
               break;
            }

            if (data.type == "parallel")
            {
               int is_mpi_size = 0, is_mpi_rank = 0;
               *stream >> is_mpi_size >> is_mpi_rank;
               if (!(*stream) ||
                   is_mpi_size != data.mpi_size || is_mpi_rank != k)
               {
                  failed = 1;
                  why = "GLVis parallel header mismatch";
                  break;
               }
            }
            else if (data.type != "mesh" && data.type != "solution")
            {
               failed = 1;
               why = "Stream: unknown command: " + data.type;
               break;
            }
         }
      }

      reset();

      if (data.mpi_root && !failed && visualize)
      {
         constexpr bool fix_elem_orien = true;
         constexpr bool save_coloring = true;
         constexpr bool keep_attr = false;
         constexpr bool headless = false;
         const std::string plot_caption {};
         std::vector<std::unique_ptr<std::istream>> istreams;
         istreams.reserve(data.streams.size());
         for (auto &s : data.streams) { istreams.push_back(std::move(s)); }
         const int status = GLVisStreamSession(fix_elem_orien,
                                               save_coloring,
                                               keep_attr,
                                               headless,
                                               plot_caption,
                                               data.type,
                                               std::move(istreams));
         if (status != 0)
         {
            failed = 1;
            why = "GLVisStreamSession failed";
         }
      }
   }
   catch (const std::bad_alloc &)
   {
      failed = 1;
      if (why.empty()) { why = "GLVis stream allocation failed"; }
   }
#ifdef MFEM_USE_EXCEPTIONS
   catch (const std::exception &ex)
   {
      failed = 1;
      why = ex.what();
   }
#endif

   FailTogether(failed, why
#ifdef MFEM_USE_MPI
                , data.serial, data.comm
#endif
               );
}

void glvis_stream::serialize()
{
#ifdef MFEM_USE_MPI
   int failed = 0;
   std::string why;
   std::string local;
   std::streamoff written = 0;
   try
   {
      local = data.stream.str();
      written = data.stream.tellp();
   }
   catch (const std::exception &)
   {
      failed = 1;
      why = "GLVis stream allocation failed";
   }
   if (!failed)
   {
      const unsigned long long limit =
         static_cast<unsigned long long>(std::numeric_limits<int>::max());
      const bool too_big =
         written < 0 ||
         static_cast<unsigned long long>(written) > limit ||
         (written >= 0 && static_cast<size_t>(written) > local.size());
      if (too_big)
      {
         failed = 1;
         why = "GLVis stream is too large for MPI_Gatherv";
      }
   }
   FailTogether(failed, why, data.serial, data.comm);
   const int local_size = static_cast<int>(written);

   std::vector<int> sizes;
   try
   {
      sizes.resize(static_cast<size_t>(data.mpi_size));
   }
   catch (const std::exception &)
   {
      failed = 1;
      why = "GLVis stream allocation failed";
   }
   FailTogether(failed, why, data.serial, data.comm);

   MFEM_VERIFY(MPI_Allgather(&local_size, 1, MPI_INT,
                             sizes.data(), 1, MPI_INT,
                             data.comm) == MPI_SUCCESS,
               "MPI_Allgather failed");

   long long total = 0;
   for (int count : sizes)
   {
      MFEM_VERIFY(count >= 0, "GLVis rank size is negative");
      total += count;
      MFEM_VERIFY(total <= std::numeric_limits<int>::max(),
                  "GLVis stream is too large for MPI_Gatherv");
   }

   std::vector<char> recvbuf;
   try
   {
      if (data.mpi_root)
      {
         data.offsets.resize(static_cast<size_t>(data.mpi_size) + 1);
         data.offsets[0] = 0;
         for (int i = 0; i < data.mpi_size; ++i)
         {
            data.offsets[i + 1] = data.offsets[i] + sizes[i];
         }
         data.total_size = static_cast<int>(total);
         recvbuf.resize(static_cast<size_t>(total));
      }
   }
   catch (const std::exception &)
   {
      failed = 1;
      why = "GLVis stream allocation failed";
   }
   FailTogether(failed, why, data.serial, data.comm);

   MFEM_VERIFY(MPI_Gatherv(local.data(), local_size, MPI_CHAR,
                           data.mpi_root ? recvbuf.data() : nullptr,
                           data.mpi_root ? sizes.data() : nullptr,
                           data.mpi_root ? data.offsets.data() : nullptr,
                           MPI_CHAR, 0, data.comm) == MPI_SUCCESS,
               "MPI_Gatherv failed");

   if (data.mpi_root)
   {
      try
      {
         reset();
         data.stream.write(recvbuf.data(), data.total_size);
         if (!data.stream)
         {
            failed = 1;
            why = "GLVis rank slice could not be written";
         }
      }
      catch (const std::exception &)
      {
         failed = 1;
         why = "GLVis stream allocation failed";
      }
   }
   FailTogether(failed, why, data.serial, data.comm);
#endif // MFEM_USE_MPI
}

} // namespace mfem

#endif // MFEM_USE_GLVIS
