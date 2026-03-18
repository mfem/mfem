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

// #include "../../fem/pgridfunc.hpp"

#include "../../general/glvis/stream.hpp"

#ifdef MFEM_USE_MPI
#include "../../general/glvis/exchange.hpp"
#endif

extern int GLVisLibWindow(bool fix_elem_orient,
                          bool save_coloring, bool headless,
                          const std::string &plot_caption,
                          std::istream &stream,
                          std::string data_type);

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

///////////////////////////////////////////////////////////////////////////////
glvis_stream::NullImpl::null_streambuf glvis_stream::NullImpl::nullbuf {};

///////////////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream(const char *host, int port):
   glvis_stream::glvis_stream(host, port, -1)
{
   dbg("Serial glvis_stream");
}

///////////////////////////////////////////////////////////////////////////////
const auto IsMpiInitialized = []()
{
   int flag;
   MPI_Initialized(&flag);
   return flag != 0;
};

const auto GetImpl = [](const bool serial,
                        const std::shared_ptr<GLVisData> &data)
                     -> std::unique_ptr<glvis_stream::IBase>
{
   if (serial) { return std::make_unique<glvis_stream::SerialImpl>(data); }
   return std::make_unique<glvis_stream::ParallelRealImpl>(data);
};

/////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream(const char *, int, int myid):
   std::iostream((dbg(myid >= 0 ? "Parallel" : "Serial"), nullptr)),
   MpiInitialized(IsMpiInitialized),
   mpi_initialized(MpiInitialized()),
   mpi_size(MpiSize()),
   mpi_rank(MpiRank()),
   serial(myid < 0),
   data(std::make_shared<GLVisData>()),
   impl(GetImpl(serial, data)),
   glvis(data)
{
   if (serial) { assert(myid == -1); }
   data->serial = serial;

   dbg("Wait for GLVis server RUNNING...");
   wait_for_running(data);
   assert(data->running);

   dbg("Wait for GLVis server READY...");
   wait_for_ready(data);
   assert(data->ready);

   // link the stream buffer to the one provided by the implementation
   this->rdbuf(impl->get_buf());
}

/////////////////////////////////////////////////////////////////////
glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   dbg();
   dbg("\x1b[32m data stream:\n{}", data->stream.str());
   this->flush(); // will trigger the GLVis update
   std::iostream::operator<<(pf); // optional
   this->glvis_window();
   return *this;
}

void glvis_stream::flush() { impl->flush(); }

void glvis_stream::glvis_window()
{
   const auto size = impl->size();
   dbg("stream size: {}", size);

   if (size == 0) { return; }

   data->mpi_size = 1;
   data->offset[0] = 0, data->offset[1] = size;
   data->total_size = size;

   // reset the local buffer for reuse
   data->stream.clear();
   data->stream.seekg(0, std::ios::beg);
   data->stream.seekp(0, std::ios::beg);

   const size_t impl_size = impl->size();
   dbg("size: {}", impl_size);

   {
      dbg("Signal UPDATE");
      assert(!data->update);
      signal_for_update(data);
      assert(data->update);
   }

   {
      dbg("Waiting ACK");
      while (data->update.load())
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      assert(!data->update);
   }

   dbg("ACK");
   dbg("data type: {}", data->type);
   dbg("streams size: #{}", data->streams.size());
   assert(data->streams[0].good());
   dbg("streams[0] size: #{}", data->streams[0].str().size());

   constexpr bool fix_elem_orien = true;
   constexpr bool save_coloring = true;
   constexpr bool headless = false;

   // needs to be in 'main' thread
   GLVisLibWindow(fix_elem_orien, save_coloring, headless, "",
                  data->streams[0], data->type);

   dbg("✅");
}


#ifdef MFEM_USE_MPI
///////////////////////////////////////////////////////////////////////////////
glvis_stream::ParallelRealImpl::ParallelRealImpl(const
                                                 std::shared_ptr<GLVisData> &data):
   data(data)
{
   dbg("Master GLVis stream");
}

size_t glvis_stream::ParallelRealImpl::size() const
{
   assert(data);
   return data->stream.tellp();
}

std::streambuf* glvis_stream::ParallelRealImpl::get_buf()
{
   assert(data);
   return data->stream.rdbuf();
}

std::streamsize glvis_stream::ParallelRealImpl::precision() const
{
   assert(data);
   return data->stream.precision();
}

std::streamsize
glvis_stream::ParallelRealImpl::precision(std::streamsize new_prec)
{
   assert(data);
   data->stream.precision(new_prec);
   return new_prec;
}

void glvis_stream::ParallelRealImpl::reset()
{
   dbg();
   data->stream.clear();
   data->stream.seekg(0, std::ios::beg);
   data->stream.seekp(0, std::ios::beg);
}

void glvis_stream::ParallelRealImpl::flush()
{
   dbg();
   dbg("\x1b[31m data stream:\n{}", data->stream.str());
   const size_t ssize = data->stream.tellp();
   dbg("stream size: {}", ssize);
   GLVisExchanger mpi_exchange(data);
   dbg("✅");
}
#endif // MFEM_USE_MPI

} // namespace mfem
