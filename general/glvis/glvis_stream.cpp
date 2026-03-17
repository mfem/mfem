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

#include "mesh/mesh.hpp"
#include "fem/gridfunc.hpp"

#include "glvis_server.hpp"
#include "glvis_stream.hpp"

// #include NVTX_FMT_HPP // IWYU pragma: keep

extern int GLVisLibWindow(bool fix_elem_orient,
                          bool save_coloring, bool headless,
                          const std::string &plot_caption,
                          std::unique_ptr<std::istream> &&stream,
                          const std::string &data_type);

namespace mfem
{

////////////////////////////////////////////////////////////////////////////
glvis_stream::glvis_stream():
   std::iostream((dbg(), nullptr)),
   data(std::make_shared<GLVisData>()),
   impl(std::make_unique<glvis_stream::SerialImpl>(data)),
   glvis(data)
{
   dbg("Wait for GLVis server RUNNING...");
   wait_for_running(data);
   assert(data->running);

   dbg("Wait for GLVis server READY...");
   wait_for_ready(data);
   assert(data->ready);

   this->rdbuf(impl->get_buf());
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   dbg();
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

   data->serial = true, data->shared_size = 1;
   data->offset[0] = 0, data->offset[1] = size;
   data->total_size = size;

   // reset the local buffer for reuse
   data->stream.clear();
   data->stream.seekp(0);

   const size_t impl_size = impl->size();
   dbg("size: {} 🔥", impl_size);

   {
      dbg("Signal UPDATE");
      data->streamsize = impl_size;
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

   dbg("ACK ✅");

   dbg("data type: {}", data->type);
   dbg("streams: #{}", data->streams.size());
   assert(data->streams[0]->good());

   // needs to be in 'main' thread
   GLVisLibWindow(data->fix_elem_orien,
                  data->save_coloring, data->headless,
                  "",
                  std::move(data->streams[0]),
                  data->type);
   dbg("✅");
}

} // namespace mfem
