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
#define NVTX_COLOR ::nvtx::kDodgerBlue

#include <cassert>

#include "config/config.hpp" // IWYU pragma: keep

#include "glvis_stream.hpp"

namespace mfem
{

glvis_stream::SerialImpl::SerialImpl(std::shared_ptr<GLVisData> data):
   data((assert(data), data)),
   stream((dbg(), std::make_unique<char_stream_t>(data->buffer, RNK_SIZE)))
{
   dbg("Serial stream ready");
   stream->clear();
}

glvis_stream::SerialImpl::~SerialImpl()
{
   dbg("🚨");
   flush();
   dbg("✅");
}

std::streamsize glvis_stream::SerialImpl::precision() const
{
   dbg();
   return stream->precision();
}

std::streamsize glvis_stream::SerialImpl::precision(std::streamsize new_prec)
{
   dbg();
   stream->precision(new_prec);
   return new_prec;
}

std::streambuf *glvis_stream::SerialImpl::get_buf()
{
   dbg();
   return stream->rdbuf();
}

size_t glvis_stream::SerialImpl::size() const
{
   dbg();
   return stream->tellp();
}

void glvis_stream::SerialImpl::flush()
{
   dbg();
   stream->flush(); // Optional

   const size_t ssize = size();
   dbg("stream size: {}", ssize);
   assert(ssize > 0 && ssize <= RNK_SIZE);
   data->serial = true, data->shared_size = 1;
   data->offset[0] = 0, data->offset[1] = ssize;
   data->total_size = ssize;

   // Optionally reset the local buffer for reuse.
   stream->clear();
   stream->seekp(0);  // Reset put position.
}

} // namespace mfem
