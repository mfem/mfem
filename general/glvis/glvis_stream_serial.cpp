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
   data(data)
{
   data->stream.clear();
}

std::streamsize glvis_stream::SerialImpl::precision() const
{
   dbg();
   return data->stream.precision();
}

std::streamsize glvis_stream::SerialImpl::precision(std::streamsize new_prec)
{
   dbg();
   data->stream.precision(new_prec);
   return new_prec;
}

std::streambuf *glvis_stream::SerialImpl::get_buf()
{
   dbg();
   return data->stream.rdbuf();
}

size_t glvis_stream::SerialImpl::size() const
{
   dbg();
   return data->stream.tellp();
}

void glvis_stream::SerialImpl::flush()
{
   dbg();
   data->stream.flush();
   assert(data->stream.str().size() == size());
}

} // namespace mfem
