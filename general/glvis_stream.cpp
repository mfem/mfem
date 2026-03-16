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

glvis_stream::~glvis_stream()
{
   dbg("🚨");
   Flush();
   impl.reset(nullptr); // trigger impl destructor
   dbg("✅");
}

void glvis_stream::Flush()
{
   const size_t impl_size = impl->size();
   dbg("size: {} 🔥", impl_size);
   impl->flush();
   {
      dbg("Signal UPDATE");
      data->streamsize = impl_size;
      assert(!data->update);
      signal_for_update(data);
      assert(data->update);
   }
   {
      dbg("Waiting ACK");
      // std::unique_lock<std::mutex> lock(data->mutex);
      while (data->update.load()) { /*data->cond.wait(lock);*/ }
      assert(!data->update);
   }

   dbg("ACK ✅");
}

glvis_stream& glvis_stream::operator<<(ostream_manipulator pf)
{
   Flush();
   std::iostream::operator<<(pf);
   return *this;
}

// Specialization for mfem::GridFunction
template<>
glvis_stream& glvis_stream::operator<<(const GridFunction& gf)
{
   dbg("GridFunction specialization");
   static_cast<std::ostream&>(*this) << gf;
   return *this;
}

// Specialization for mfem::Mesh
template<>
glvis_stream& glvis_stream::operator<<(const Mesh& mesh)
{
   dbg("Mesh specialization");
   static_cast<std::ostream&>(*this) << mesh;
   return *this;
}

int glvis_stream::open(const char hostname[], int port) { dbg(); return 0; }

bool glvis_stream::is_open() const { dbg(); return true; }

int glvis_stream::close() { dbg(); return 0; }

} // namespace mfem
